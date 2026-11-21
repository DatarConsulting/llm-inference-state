"""Batch-one, text-only full model, native GDN, MLP, norm and vocabulary head."""
import copy
from contextlib import ExitStack, nullcontext
from unittest.mock import patch
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from fast_kernels import gate_project
from backends import Dense,Packed

MODEL='Qwen/Qwen3.5-0.8B'
REV='2fc06364715b967f1860aea9cf38778875588b17'

def load_model():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV)
    wrapper=hf.Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,revision=REV,dtype=torch.bfloat16,attn_implementation='sdpa')
    text=wrapper.model.language_model.cuda().eval(); head=wrapper.lm_head.cuda().eval()
    eos=wrapper.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    assert [i for i,l in enumerate(text.layers) if l.layer_type=='full_attention']==[3,7,11,15,19,23]
    info=dict(model=MODEL,revision=REV,full_layers=[3,7,11,15,19,23],linear_layers=18,
              eos_ids=sorted(e for e in eos if e is not None),
              gdn_decode=text.layers[0].linear_attn.recurrent_gated_delta_rule.__name__)
    return tok,text,head,info

def chunk_scan(original):
    def run(q,k,v,*,g,beta,initial_state=None,output_final_state=False,**kw):
        state=initial_state
        parts=[]
        for start in range(0,q.shape[1],2048):
            sl=slice(start,start+2048)
            out,state=original(q[:,sl].contiguous(),k[:,sl].contiguous(),v[:,sl].contiguous(),
                               g=g[:,sl].contiguous(),beta=beta[:,sl].contiguous(),initial_state=state,
                               output_final_state=True,**kw)
            parts.append(out)
        return torch.cat(parts,dim=1),state if output_final_state else None
    return run

def prefill(text,head,ids,log):
    # Whole-sequence causal SDPA. Only the GDN scan is split, carrying its state
    # across chunks. The convolution and position IDs cover the whole prompt.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with ExitStack() as scope:
        for i,l in enumerate(text.layers):
            if l.layer_type=='linear_attention' and ids.shape[-1]>2048:
                mixer=l.linear_attn
                scope.enter_context(patch.object(mixer,'chunk_gated_delta_rule',chunk_scan(mixer.chunk_gated_delta_rule)))
            if i%4==0:
                handle=l.register_forward_pre_hook(lambda _m,_a,i=i:log(f'Prefill layer {i+1}/{len(text.layers)}'))
                scope.callback(handle.remove)
        if ids.is_cuda:
            # One unpadded sequence, empty cache and pure causal text attention.
            # Let SDPA use is_causal=True without materializing an N-by-N mask.
            assert ids.shape[0]==1 and ids.shape[1]>1
            scope.enter_context(patch.object(hf,"create_causal_mask",return_value=None))
            scope.enter_context(sdpa_kernel(SDPBackend.FLASH_ATTENTION))
        out=text(input_ids=ids,use_cache=True,return_dict=True)
    return head(out.last_hidden_state[:,-1]).float(),out.past_key_values

def clone_state(cache):
    work=copy.copy(cache)
    work.layers=[copy.copy(l) for l in cache.layers]
    for l in work.layers:
        if hasattr(l,'conv_states'):
            l.conv_states=l.conv_states.clone()
            l.recurrent_states=l.recurrent_states.clone()
    return work

def projections(att,x,rot):
    q,g=att.q_proj(x).reshape(1,1,8,512).chunk(2,-1)
    q=att.q_norm(q).transpose(1,2)
    k=att.k_norm(att.k_proj(x).reshape(1,1,2,256)).transpose(1,2)
    v=att.v_proj(x).reshape(2,256)
    q,k=hf.apply_rotary_pos_emb(q,k,*rot)
    return q.reshape(8,256).contiguous(),k.reshape(2,256).contiguous(),v,g.sigmoid().reshape(8,256).float().contiguous()

def fused_projection(heads,gate,w):
    out=torch.empty((1024,),dtype=torch.float32,device=heads.device)
    gate_project[(256,)](heads,gate,w,out,4,num_warps=4,num_stages=1)
    return out.reshape(1,1,-1).to(torch.bfloat16)

class Decoder:
    def __init__(self,text,head,cache,prefix,tail,ctor,weights,limits,projection=fused_projection):
        self.text=text;self.head=head;self.stores={}; self.weights=weights;self.project=projection
        self.pos=torch.arange(prefix,prefix+tail,device=head.weight.device).reshape(1,1,-1).expand(3,1,-1)
        for i in weights:
            layer=cache.layers[i]
            assert layer.keys.shape==(1,2,prefix,256)
            store=ctor(layer.keys[0],layer.values[0],tail,limits[i])
            if isinstance(store,Packed):store.preflight(range(prefix+1,prefix+tail+1))
            self.stores[i]=store
    def step(self,token,state,j):
        x=self.text.embed_tokens(token.reshape(1,1))
        rot=self.text.rotary_emb(x,self.pos[:,:,j:j+1])
        for i,layer in enumerate(self.text.layers):
            if i not in self.stores:
                x=layer(x,position_embeddings=rot,attention_mask=None,past_key_values=state,use_cache=True)
            else:
                att=layer.self_attn
                q,k,v,gate=projections(att,layer.input_layernorm(x),rot)
                heads=self.stores[i].attention(q,k,v,j)
                x=x+self.project(heads,gate,self.weights[i])
                x=x+layer.mlp(layer.post_attention_layernorm(x))
        return self.head(self.text.norm(x))[:,0].float()
    def sizes(self,used):
        layers={str(i):s.sizes(used) for i,s in self.stores.items()}
        return dict(logical_kv_bytes=sum(d['logical_kv_bytes'] for d in layers.values()),
                    reserved_kv_bytes=sum(d['reserved_kv_bytes'] for d in layers.values()),layers=layers)
