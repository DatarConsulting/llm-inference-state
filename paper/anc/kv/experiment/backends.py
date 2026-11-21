import math
import numpy as np
import torch
import triton
from triton.runtime import driver
from compact import encode_parts
from fast_kernels import split,merge,inspect_words
from append_kv import append_dense,append_packed

class Dense:
    def __init__(self,k,v,tail,limit):
        self.prefix=k.shape[1]
        self.k=torch.empty((self.prefix+tail,2,256),dtype=k.dtype,device=k.device)
        self.v=torch.empty_like(self.k)
        self.k[:self.prefix].copy_(k.transpose(0,1))
        self.v[:self.prefix].copy_(v.transpose(0,1))
    def attention(self,q,k,v,step):
        import flashinfer
        pos=self.prefix+step
        append_dense[(1,)](k,v,self.k,self.v,pos)
        return flashinfer.single_decode_with_kv_cache(q,self.k[:pos+1],self.v[:pos+1],
                                                       kv_layout='NHD',use_tensor_cores=True)
    def sizes(self,used):
        return {'logical_kv_bytes':(self.prefix+used)*2048,
                'reserved_kv_bytes':self.k.nbytes+self.v.nbytes}

class Packed:
    def __init__(self,k,v,tail,limit):
        self.prefix=k.shape[1];self.cap=self.prefix+tail
        # Pass the next smaller representable FP32 number to the GPU.
        self.limit=float(np.nextafter(np.float32(limit),np.float32(0)))
        self.streams=[]; self.start=[]; self.prefix_bytes=0
        for kind,a in [('K',k),('V',v)]:
            words=a.contiguous().view(torch.uint16).cpu().numpy()
            h,p,_,_=encode_parts(words,kind,self.limit)
            arena=torch.empty(p.size+tail*1024,dtype=torch.uint8,device=k.device)
            arena[:p.size].copy_(torch.from_numpy(p))
            header=torch.empty((2,self.cap),dtype=torch.uint32,device=k.device)
            header[:,:self.prefix].copy_(torch.from_numpy(h))
            self.streams.extend([arena,header]); self.start.append(int(p.size))
            self.prefix_bytes+=h.nbytes+p.nbytes
        assert (self.streams[2].numel()-1)//32<2**23
        self.bad=torch.zeros((tail,2),dtype=torch.uint8,device=k.device)
        groups=triton.cdiv(self.cap,256)
        self.part=torch.empty((8,groups,256),device=k.device,dtype=torch.float32)
        self.stats=torch.empty((8,groups,2),device=k.device,dtype=torch.float32)
        self.out=torch.empty((8,256),device=k.device,dtype=k.dtype)
        self.checked=set()
    def _args(self,q,n):
        count=triton.cdiv(n,256)
        return (q,*self.streams,self.part,self.stats,n,self.cap,count,32,8,True)
    def preflight(self,queries):
        # JIT every block count appearing during generation, outside the timer.
        q=self.out
        limit=driver.active.utils.get_device_properties(q.device.index)['max_shared_mem']
        for n in queries:
            count=triton.cdiv(n,256)
            if count in self.checked: continue
            args=self._args(q,n)
            for fn,grid,argv in [(split,(count,2),args),(merge,(8,16),(self.part,self.stats,self.out,count,triton.next_power_of_2(count)))]:
                cubin=fn.warmup(*argv,grid=grid,num_warps=4,num_stages=1)
                if cubin.metadata.shared>limit:
                    raise RuntimeError(f"Frozen variant is incompatible with this GPU : {cubin.metadata.shared} > {limit} bytes.")
            self.checked.add(count)
    def attention(self,q,k,v,step):
        n=self.prefix+step+1
        append_packed[(2,)](k,v,*self.streams,self.bad,step,self.limit,
                            self.prefix,self.cap,*self.start,num_warps=4,num_stages=1)
        count=triton.cdiv(n,256)
        split[(count,2)](*self._args(q,n),num_warps=4,num_stages=1)
        merge[(8,16)](self.part,self.stats,self.out,count,triton.next_power_of_2(count),num_warps=4,num_stages=1)
        return self.out
    def sizes(self,used):
        if self.bad[:used].any().item(): raise ArithmeticError('Nonfinite K/V during append.')
        def raw(header):
            return int((header[:,self.prefix:self.prefix+used].to(torch.int64)&1).sum().item())
        kr,vr=raw(self.streams[1]),raw(self.streams[3])
        return {'logical_kv_bytes':self.prefix_bytes+used*2*(384+352+8)+kr*128+vr*160,
                'reserved_kv_bytes':sum(s.nbytes for s in self.streams),
                'appended_raw_k':kr,'appended_raw_v':vr}

def check_gpu_codec():
    # Read through the reader's own GPU decoder. Nonzero offset, alternating groups,
    # padding, exact and raw records, signed zero and subnormal BF16 inputs.
    from compact import pack_k,pack_v,unpack_v
    torch.manual_seed(147)
    prefix,tail=3,5
    keys=torch.randn(2,prefix+tail,256,device='cuda').bfloat16()
    vals=(.16*torch.randn_like(keys)).bfloat16()
    keys[1,-1,0]=64;keys[1,-1,1]=2**-25
    vals[1,-1,0]=32;vals[1,-1,1]=.002
    vals[0,-2].zero_(); vals[0,-2,1]=-0.; vals[0,-2,2]=2**-125
    store=Packed(keys[:,:prefix],vals[:,:prefix],tail,0.000128)
    for j in range(tail):
        k=keys[:,prefix+j].contiguous();v=vals[:,prefix+j].contiguous()
        append_packed[(2,)](k,v,*store.streams,store.bad,j,store.limit,prefix,store.cap,*store.start,num_warps=4,num_stages=1)
    ko=torch.empty_like(keys);vo=torch.empty_like(vals)
    inspect_words[(triton.cdiv(2*store.cap,16),)](*store.streams,ko,vo,2*store.cap,16)
    assert torch.equal(keys.view(torch.int16),ko.view(torch.int16))
    vw=vals.view(torch.uint16).cpu().numpy()
    header,p=pack_v(vw,store.limit); expected=unpack_v(header,p)
    np.testing.assert_array_equal(vo.float().cpu().numpy().view('uint32'),expected.view('uint32'))
    error=(vals.float()-vo.float()).abs().max().item()
    assert error<=store.limit
    sizes=store.sizes(tail)
    assert sizes['appended_raw_v']>0 and sizes['appended_raw_k']>0
    return dict(max_v_error=error,**sizes)
