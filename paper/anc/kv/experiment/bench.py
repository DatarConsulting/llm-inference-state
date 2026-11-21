import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
import torch
from backends import Dense,Packed,check_gpu_codec
from model_loop import Decoder,clone_state,load_model,prefill
from prompts import texts

def log(s):
    print(time.strftime('%H:%M:%S'),s,flush=True)

def quality(a,b):
    la,lb=a.log_softmax(-1),b.log_softmax(-1)
    diff=(a-b).norm()/a.norm().clamp_min(1e-15)
    metrics={'relative_logits_l2':float(diff),'kl_dense_compact':max(0.,float((la.exp()*(la-lb)).sum())),
             'top1_match':int(a.argmax(-1).item()==b.argmax(-1).item())}
    if not all(torch.isfinite(x).all().item() for x in (a,b)):raise ArithmeticError('Nonfinite logits')
    return metrics

def teacher(decoders,cache,seed,steps,eos):
    states=[clone_state(cache),clone_state(cache)]
    token=seed; metrics=[]
    for j in range(steps):
        if token.item() in eos:break
        a,b=(d.step(token,s,j) for d,s in zip(decoders,states))
        metrics.append(quality(a,b))
        token=a.argmax(-1)
    return metrics

def run_free(decoder,cache,seed,steps,eos):
    state=clone_state(cache)
    ids=[];milliseconds=[]
    token=seed
    ended=token.item() in eos
    torch.cuda.synchronize()
    start=time.perf_counter();last=start
    for j in range(steps):
        if ended:break
        logits=decoder.step(token,state,j)
        token=logits.argmax(-1)
        picked=token.item()  # streaming delivery fences GPU work for this token
        ids.append(picked)
        ended=picked in eos
        now=time.perf_counter()
        milliseconds.append(1000*(now-last));last=now
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    return dict(ids=ids,seconds=elapsed,tokens=len(ids),stopped_eos=ended,step_ms=milliseconds,
                tokens_s=len(ids)/elapsed if ids else None)

def comparison(a,b):
    x,y=a['ids'],b['ids']
    prefix=next((j for j,(i,k) in enumerate(zip(x,y)) if i!=k),min(len(x),len(y)))
    return dict(exact=x==y,common_prefix=prefix,first_difference=None if x==y else prefix+1,
                same_token_count=len(x)==len(y),same_positions=sum(i==k for i,k in zip(x,y)))

def percentile95(xs):
    return sorted(xs)[math.ceil(len(xs)*.95)-1]

def summarize(row):
    runs=row['runs']
    by={mode:[r for r in runs if r['method']==mode and r['tokens']>0] for mode in ('FI','compact')}
    stats={}
    for mode,rs in by.items():
        if not rs:continue
        times=[x for r in rs for x in r['step_ms']]
        stats[mode]=dict(median_tokens_s=statistics.median(r['tokens_s'] for r in rs),
                         pooled_tokens_s=sum(r['tokens'] for r in rs)/sum(r['seconds'] for r in rs),
                         median_itl_ms=statistics.median(times),p95_itl_ms=percentile95(times))
    pairs=[]
    for j in sorted({r['pair'] for r in runs}):
        group={r['method']:r for r in runs if r['pair']==j}
        if len(group)!=2:continue
        a,b=group['FI'],group['compact']
        ratio=b['tokens_s']/a['tokens_s'] if a['tokens_s'] and b['tokens_s'] else None
        pairs.append(dict(pair=j,**comparison(a,b),speed_ratio=ratio))
    equal=[p['speed_ratio'] for p in pairs if p['same_token_count'] and p['speed_ratio'] is not None]
    identical=[p['speed_ratio'] for p in pairs if p['exact'] and p['speed_ratio'] is not None]
    row.update(summary=stats,pairs=pairs,median_paired_speed_ratio_equal_length=statistics.median(equal) if equal else None,
               median_paired_speed_ratio_same_tokens=statistics.median(identical) if identical else None)

def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter()
    value=fn()
    torch.cuda.synchronize()
    return value,time.perf_counter()-start

def integration_probe(text,head,weights,limits,prompt):
    # Original Hugging Face text decode with its own rolling cache. Both
    # candidates are compared at identical positions, including across a 256 boundary.
    ids=prompt[:255][None].cuda();initial,cache=prefill(text,head,ids,lambda _s:None)
    stores=[Decoder(text,head,cache,255,4,c,weights,limits) for c in (Dense,Packed)]
    native_state=clone_state(cache);other_states=[clone_state(cache) for _ in stores]
    token=initial.argmax(-1);out=[]
    for j in range(4):
        result=text(input_ids=token.reshape(1,1),past_key_values=native_state,use_cache=True,return_dict=True)
        ref=head(result.last_hidden_state[:,-1]).float()
        line=[quality(ref,d.step(token,s,j)) for d,s in zip(stores,other_states)]
        if max(v['relative_logits_l2'] for v in line)>.08:raise ArithmeticError(f'Hugging Face integration mismatch : {line}')
        out.append(line);token=ref.argmax(-1)
    for d in stores:d.sizes(4)
    return out

def case(text,head,weights,limits,tokenizer,desc,ids,n,cfg,row,save):
    steps=cfg['steps']; eos=set(cfg['eos_ids'])
    prompt=ids[:n][None].cuda()
    row['prompt_ids_sha256']=hashlib.sha256(ids[:n].numpy().astype('<i8').tobytes()).hexdigest()
    (logits,cache),row['prefill_s']=timed(lambda:prefill(text,head,prompt,log))
    seed=logits.argmax(-1);row['seed_token']=int(seed.item())
    decoders={}; row['build_s']={}
    for mode,ctor in [('FI',Dense),('compact',Packed)]:
        log(f'{mode} : building the initial cache for the six layers')
        decoders[mode],row['build_s'][mode]=timed(lambda:Decoder(text,head,cache,n,steps,ctor,weights,limits))
    # Each engine has its own full-attention storage. Preserve the recurrent
    # prefix state to reset both trajectories; release the construction copy.
    for i in weights:cache.layers[i].keys=None;cache.layers[i].values=None
    torch.cuda.empty_cache()
    row['initial_kv']={name:d.sizes(0) for name,d in decoders.items()}
    save()
    log('Checking common-history logits outside the timer.')
    row['teacher']=teacher(list(decoders.values()),cache,seed,min(cfg['teacher'],steps),eos)
    # A complete unmeasured free run per backend covers every kernel shape
    # actually visited, including crossing a logical block boundary.
    for mode,d in decoders.items():
        log(f'Full warm-up : {mode}')
        warm=run_free(d,cache,seed,steps,eos)
        row.setdefault('warmup_tokens',{})[mode]=warm['tokens']
    save()
    for pair in range(cfg['pairs']):
        order=('FI','compact') if (pair+row['case_index'])%2==0 else ('compact','FI')
        for mode in order:
            log(f"{desc['title']} · {n:,} · pair {pair+1}/{cfg['pairs']} · {mode}")
            d=decoders[mode]
            run=run_free(d,cache,seed,steps,eos)
            run.update(method=mode,pair=pair+1,cache=d.sizes(run['tokens']),
                       generated_excerpt=tokenizer.decode(run['ids'][:48]))
            row['runs'].append(run);summarize(row);save()
            log(f"{run['tokens']} tokens · {round(run['tokens_s'],2) if run['tokens_s'] else 'EOS at prefill'} tok/s")
    row['status']='terminé';save()

@torch.inference_mode()
def main(root):
    cfg=json.loads((root/'settings.json').read_text())
    report=dict(protocol='generation-256-v1',stage='initialisation',settings=cfg,cases=[])
    path=root/'qwen_generation_256.json'
    def save():
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=1,allow_nan=False,ensure_ascii=False))
        tmp.replace(path)
    save()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        report['hardware']=dict(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            cuda=torch.version.cuda,torch=str(torch.__version__),triton=importlib.metadata.version('triton'),
            flashinfer=importlib.metadata.version('flashinfer-python'),transformers=importlib.metadata.version('transformers'))
        log('Checking GPU append of new keys and values')
        report['codec_test']=check_gpu_codec();save()
        log('Loading Qwen3.5-0.8B')
        tok,text,head,info=load_model();report['model']=info;cfg['eos_ids']=info['eos_ids']
        prompts=texts(tok,cfg['paths'],cfg['offset'],max(cfg['lengths']))
        report['texts']=[x for x,_ in prompts]
        weights={i:text.layers[i].self_attn.o_proj.weight.float().contiguous() for i in info['full_layers']}
        norms={i:w.double().abs().sum(-1).max().item() for i,w in weights.items()}
        limits={i:(1/128)/v*(1-1e-10) for i,v in norms.items()}
        report['w_infinity']=norms
        report['stage']='contrôle modèle HF';save()
        report['native_integration']=integration_probe(text,head,weights,limits,prompts[0][1]);save()
        for n in cfg['lengths']:
            for desc,ids in prompts:
                report['stage']=f"{desc['title']} · {n}";log(report['stage'])
                row=dict(case_index=len(report['cases']),title=desc['title'],prefix=n,runs=[],status='en cours')
                report['cases'].append(row);save()
                case(text,head,weights,limits,tok,desc,ids,n,cfg,row,save)
                gc.collect(); torch.cuda.empty_cache()
        report['stage']='terminé';save()
    except BaseException:
        report.update(stage='interrompu',error=traceback.format_exc());save();raise

if __name__=='__main__':
    main(Path(__file__).resolve().parent)
