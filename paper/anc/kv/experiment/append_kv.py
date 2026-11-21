"""One token, two KV groups. Prefix arenas and tail slots share the reader format."""
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["step"], do_not_specialize_on_alignment=["step"])
def append_packed(K,V,KP,KH,VP,VH,Bad,step,limit,
                  PREFIX:tl.constexpr,CAP:tl.constexpr,KSTART:tl.constexpr,VSTART:tl.constexpr):
    group=tl.program_id(0)
    d=tl.arange(0,256)
    slot=step*2+group
    hdr=group*CAP+PREFIX+step
    kb=KSTART+slot*512
    vb=VSTART+slot*512
    k=tl.load(K+group*256+d).to(tl.uint16,bitcast=True).to(tl.uint32)
    v=tl.load(V+group*256+d).to(tl.uint16,bitcast=True).to(tl.uint32)
    ke=((k>>7)&255).to(tl.int32)
    kt=tl.max(ke,0)
    kr=kt-tl.min(tl.where(ke==0,255,ke),0)>14
    ks=(k&127)|((k>>8)&128)
    tl.store(KP+kb+d,ks.to(tl.uint8))
    if kr:
        tl.store(KP+kb+256+d,ke.to(tl.uint8))
    else:
        nibble=tl.where(ke==0,0,kt+1-ke)<<((d%2)*4)
        pairs=tl.sum(tl.reshape(nibble,(128,2)),1)
        tl.store(KP+kb+256+tl.arange(0,128),pairs.to(tl.uint8))
    tl.store(KH+hdr,((kb//128)<<9)|(kt<<1)|kr.to(tl.int32))

    ve=((v>>7)&255).to(tl.int32)
    vt=tl.minimum(254,tl.maximum(32,tl.max(ve,0)))
    low=ve<vt-6
    mag=((v&32767)<<16).to(tl.float32,bitcast=True)
    grid=((vt-13).to(tl.uint32)<<23).to(tl.float32,bitcast=True)
    # All relevant ratios are small dyadic numbers. Round to nearest, ties to even.
    ratio=tl.where(low,mag,0.)/grid
    lower=ratio.to(tl.int32)
    fraction=ratio-lower.to(tl.float32)
    m=lower+((fraction>0.5)|((fraction==0.5)&((lower&1)!=0))).to(tl.int32)
    error=tl.max(tl.where(low,tl.abs(mag-m.to(tl.float32)*grid),0.),0)
    vr=error>limit
    e=tl.where(low,(m==128).to(tl.int32),ve-vt+7)
    sm=((v>>8)&128)|tl.where(vr,v&127,tl.where(low,m&127,v&127))
    tl.store(VP+vb+d,sm.to(tl.uint8))
    if vr:
        tl.store(VP+vb+256+d,ve.to(tl.uint8))
    else:
        for bit in tl.static_range(3):
            bits=((e>>bit)&1)<<(d%8)
            plane=tl.sum(tl.reshape(bits,(32,8)),1)
            tl.store(VP+vb+256+32*bit+tl.arange(0,32),plane.to(tl.uint8))
    tl.store(VH+hdr,((vb//32)<<9)|(vt<<1)|vr.to(tl.int32))
    tl.store(Bad+slot,(tl.maximum(kt,tl.max(ve,0))==255).to(tl.uint8))

@triton.jit(do_not_specialize=["pos"], do_not_specialize_on_alignment=["pos"])
def append_dense(K,V,KOUT,VOUT,pos):
    d=tl.arange(0,512)
    tl.store(KOUT+pos*512+d,tl.load(K+d))
    tl.store(VOUT+pos*512+d,tl.load(V+d))
