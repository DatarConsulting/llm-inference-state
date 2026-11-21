"""Exact K and bounded V. Every 256-coordinate row is independently encodable."""
import numpy as np
from key_codec import pack as pack_k, unpack as unpack_k

def bf32(words):
    return (words.astype(np.uint32)<<16).view(np.float32)

def unpack_v(header,payload):
    h=header.reshape(-1).astype(np.uint32)
    base=(h.astype(np.int64)>>9)*32
    top=((h>>1)&255).astype(np.int32)
    raw=(h&1).astype(bool)
    d=np.arange(256)
    sm=payload[base[:,None]+d]
    start=base[:,None]+256
    e=np.zeros_like(sm)
    for bit in range(3):
        addr=start+d//8+bit*32
        e |= (((payload[addr]>>(d%8))&1)<<bit).astype(np.uint8)
    ee=np.where(raw[:,None],payload[start+np.where(raw[:,None],d,0)],top[:,None]-7+e.astype(np.int32))
    out=bf32(((sm.astype(np.uint16)&128)<<8)|(ee.astype(np.uint16)<<7)|(sm.astype(np.uint16)&127))
    low=(~raw[:,None])&(e==0)
    sub=np.ldexp((sm&127).astype(np.float64),top[:,None]-140)
    out=np.where(low,np.where(sm&128,-sub,sub),out).astype(np.float32)
    return out.reshape(*header.shape,256)

def pack_v(words,limit):
    if words.dtype!=np.uint16 or words.shape[-1]!=256 or not np.isfinite(limit) or limit<0:
        raise ValueError('Expected BF16 words and a finite nonnegative per-component tolerance')
    x=words.reshape(-1,256)
    exponent=((x>>7)&255).astype(np.int32)
    if (exponent==255).any():raise ValueError('Nonfinite native values')
    top=exponent.max(1).clip(32,254)
    floor=top[:,None]-6
    low=exponent<floor
    magnitude=bf32(x&0x7fff).astype(np.float64)
    grid=np.ldexp(np.ones(len(x)),top-140)[:,None]
    # Mask normals before casting. Only subnormal-format inputs use this integer.
    small=np.rint(np.where(low,magnitude,0)/grid).astype(np.int32)
    err=np.max(np.where(low,abs(magnitude-small*grid),0),axis=1)
    raw=err>limit
    ex=np.where(low,small==128,exponent-floor+1).astype(np.uint8)
    man=np.where(low,small&127,x&127).astype(np.uint8)
    man[raw]=(x[raw]&127).astype(np.uint8)
    sm=man|((x>>8)&128).astype(np.uint8)
    widths=np.where(raw,16,11)
    offsets=np.cumsum(widths,dtype=np.int64)-widths
    if offsets[-1]>=2**23:raise ValueError('Use separate arenas above 256 MiB.')
    header=((offsets<<9)|(top<<1)|raw).astype(np.uint32)
    base=offsets[:,None]*32
    out=np.empty(int(widths.sum())*32,np.uint8)
    out[base+np.arange(256)]=sm
    out[base[raw]+256+np.arange(256)]=exponent[raw].astype('uint8')
    for bit in range(3):
        plane=np.packbits((ex[~raw]>>bit)&1,axis=1,bitorder='little')
        out[base[~raw]+256+bit*32+np.arange(32)]=plane
    return header.reshape(words.shape[:-1]),out

def encode_parts(words,kind,limit):
    # Chunk temporary arrays and rebase headers. This is equivalent to appending
    # independent records; no calibration on future rows or target queries.
    flat=np.ascontiguousarray(words).reshape(-1,256); hs=[];ds=[];bytes_so_far=0
    unit=128 if kind=='K' else 32
    largest=0.; raw=0
    for start in range(0,len(flat),2048):
        x=flat[start:start+2048]
        if np.any(((x>>7)&255)==255):raise ValueError('Nonfinite inputs')
        if kind=='K':
            h,data=pack_k(x)
            if not np.array_equal(unpack_k(h,data),x):raise ArithmeticError('K roundtrip')
        else:
            h,data=pack_v(x,limit)
            largest=max(largest,float(np.max(np.abs(unpack_v(h,data).astype('float64')-bf32(x).astype('float64')))))
            if largest>limit:raise ArithmeticError('V exceeds tolerance')
        shifted=(h.astype(np.uint64)>>9)+bytes_so_far//unit
        if shifted.max()>=2**23:raise OverflowError('Packed arena offset overflow')
        hs.append(((shifted<<9)|(h&511).astype(np.uint64)).astype(np.uint32))
        ds.append(data);raw+=int((h&1).sum());bytes_so_far+=data.nbytes
    return np.concatenate(hs).reshape(words.shape[:-1]),np.concatenate(ds),largest,raw
