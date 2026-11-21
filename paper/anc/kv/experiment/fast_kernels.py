import triton
import triton.language as tl


@triton.jit
def expand_k(data,hdr,row,d,live):
    h=tl.load(hdr+row,live,0).to(tl.uint32)
    base=((h>>9).to(tl.int64)*128)[:,None]
    raw=(h&1)!=0
    sm=tl.load(data+base+d[None,:],live[:,None],0).to(tl.uint32)
    loc=tl.where(raw[:,None],d[None,:],d[None,:]//2)
    x=tl.load(data+base+256+loc,live[:,None],0).to(tl.int32)
    code=(x>>((d%2)*4)[None,:])&15
    top=((h>>1)&255).to(tl.int32)
    e=tl.where(raw[:,None],x,tl.where(code==0,0,top[:,None]+1-code))
    bits=(sm&127)|((sm&128)<<8)|(e.to(tl.uint32)<<7)
    return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)


@triton.jit
def expand_v(data,hdr,row,d,live):
    h=tl.load(hdr+row,live,0).to(tl.uint32)
    base=((h>>9).to(tl.int64)*32)[:,None]
    raw=(h&1)!=0
    sm=tl.load(data+base+d[None,:],live[:,None],0).to(tl.uint32)
    off=base+256
    b0=tl.load(data+off+tl.where(raw[:,None],d[None,:],d[None,:]//8),live[:,None],0).to(tl.uint32)
    small=(live&~raw)[:,None]
    b1=tl.load(data+off+32+d[None,:]//8,small,0).to(tl.uint32)
    b2=tl.load(data+off+64+d[None,:]//8,small,0).to(tl.uint32)
    shift=(d%8)[None,:]
    code=((b0>>shift)&1)|(((b1>>shift)&1)<<1)|(((b2>>shift)&1)<<2)
    top=((h>>1)&255).to(tl.int32)
    exp=tl.where(raw[:,None],b0,top[:,None]-7+code)
    bits=((sm&127)|(exp.to(tl.uint32)<<7)|((sm&128)<<8)).to(tl.uint16)
    # For the low bin the 7 mantissa bits encode a signed multiple of a power of two.
    scale=((tl.maximum(top,32)-13).to(tl.uint32)<<23).to(tl.float32,bitcast=True)
    sub=(sm&127).to(tl.float32)*scale[:,None]
    sub=tl.where((sm&128)!=0,-sub,sub)
    sub_bits=(sub.to(tl.uint32,bitcast=True)>>16).to(tl.uint16)
    return tl.where((~raw)[:,None]&(code==0),sub_bits,bits).to(tl.bfloat16,bitcast=True)


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def split(Q,K,KH,V,VH,Y,S,N,CAP:tl.constexpr,NP:tl.constexpr,B:tl.constexpr,
          TILES:tl.constexpr,PACK:tl.constexpr):
    block=tl.program_id(0);group=tl.program_id(1)
    heads=tl.arange(0,16); d=tl.arange(0,256)
    query=tl.load(Q+(group*4+heads[:,None])*256+d[None,:],heads[:,None]<4,0)
    m=tl.full((16,),-float('inf'),tl.float32)
    den=tl.full((16,),0,tl.float32)
    acc=tl.full((16,256),0,tl.float32)
    for tile in range(TILES):
        t=block*(B*TILES)+tile*B+tl.arange(0,B)
        live=t<N; row=group*CAP+t
        if PACK:
            key=expand_k(K,KH,row,d,live)
            val=expand_v(V,VH,row,d,live)
        else:
            key=tl.load(K+row[:,None]*256+d[None,:],live[:,None],0)
            val=tl.load(V+row[:,None]*256+d[None,:],live[:,None],0)
        score=tl.dot(query,tl.trans(key))*0.0625
        score=tl.where(live[None,:],score,-float('inf'))
        new=tl.maximum(m,tl.max(score,axis=1))
        alpha=tl.exp2((m-new)*1.4426950408889634)
        p=tl.exp2((score-new[:,None])*1.4426950408889634)
        den=den*alpha+tl.sum(p,axis=1)
        acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),val)
        m=new
    slot=(group*4+heads)*NP+block
    tl.store(Y+slot[:,None]*256+d[None,:],acc,heads[:,None]<4)
    tl.store(S+2*slot,m,heads<4);tl.store(S+2*slot+1,den,heads<4)


@triton.jit
def merge(Y,S,Out,NP:tl.constexpr,PB:tl.constexpr):
    head=tl.program_id(0); d=tl.program_id(1)*16+tl.arange(0,16)
    p=tl.arange(0,PB);slot=head*NP+p
    m=tl.load(S+2*slot,p<NP,-float('inf'))
    a=tl.exp2((m-tl.max(m,0))*1.4426950408889634)
    denom=tl.sum(a*tl.load(S+2*slot+1,p<NP,0),axis=0)
    acc=tl.load(Y+slot[:,None]*256+d[None,:],p[:,None]<NP,0.)
    tl.store(Out+head*256+d,tl.sum(a[:,None]*acc,axis=0)/denom)


@triton.jit
def inspect_words(K,KH,V,VH,KO,VO,ROWS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B);d=tl.arange(0,256);live=row<ROWS
    k=expand_k(K,KH,row,d,live); v=expand_v(V,VH,row,d,live)
    tl.store(KO+row[:,None]*256+d[None,:],k,live[:,None])
    tl.store(VO+row[:,None]*256+d[None,:],v,live[:,None])


@triton.jit
def gate_project(Heads,Gate,W,Out,BM:tl.constexpr):
    """Each program finishes BM rows of W_O. All intermediate math is FP32."""
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    d=tl.arange(0,2048)
    z=tl.load(Heads+d).to(tl.float32)*tl.load(Gate+d)
    weight=tl.load(W+row[:,None]*2048+d[None,:])
    tl.store(Out+row,tl.sum(weight*z[None,:],axis=1))
