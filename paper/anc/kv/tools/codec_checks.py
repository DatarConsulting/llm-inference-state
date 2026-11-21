"""Exercise the archived CPU reference with finite BF16 bit patterns, including edge rows."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiment"))
from compact import pack_v, unpack_v, bf32
from key_codec import pack, unpack

def check():
    rng=np.random.default_rng(271828)
    # Narrow, wide and clipped exponent ranges exercise both payload layouts.
    exp=rng.integers(0,255,(240,256),dtype=np.uint16)
    for i in range(120):
        top=int(rng.integers(16,255))
        exp[i]=rng.integers(top-13,top+1,256,dtype=np.uint16)
    sign=rng.integers(0,2,exp.shape,dtype=np.uint16)<<15
    frac=rng.integers(0,128,exp.shape,dtype=np.uint16)
    words=sign|(exp<<7)|frac
    special=np.zeros((4,256),dtype=np.uint16)
    special[0,::2]=0x8000
    special[1]=np.arange(256,dtype=np.uint16)
    special[2]=np.array([0x7f7f,0xff7f,1,0x8001]*64,dtype=np.uint16)
    special[3]=np.array([0x3f80,0x3c7f,0x3c80,0x0000]*64,dtype=np.uint16)
    words=np.concatenate((special,words))
    kh,kp=pack(words)
    np.testing.assert_array_equal(unpack(kh,kp),words)
    native=bf32(words).astype(np.float64)
    for tol in [0.,1/8192,1/4096,1/128]:
        vh,vp=pack_v(words,tol)
        rec=unpack_v(vh,vp).astype(np.float64)
        assert np.isfinite(rec).all() and np.max(np.abs(rec-native))<=tol
        is_raw=(vh&1).astype(bool)
        np.testing.assert_array_equal(rec[is_raw],native[is_raw])
        assert 0<is_raw.sum()<len(words)
    print("Passed CPU key round trips and four rowwise value budgets on 244 × 256 BF16 coordinates.")
if __name__=="__main__":
    check()
