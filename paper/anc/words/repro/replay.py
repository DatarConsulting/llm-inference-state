"""Build the native auditor, verify all supplied words and aggregate by head."""
import argparse
import csv
import hashlib
import itertools
import subprocess
import zlib
from pathlib import Path

import numpy as np
import zstandard as zstd

from prepare_data import HEAD_BYTES

HERE = Path(__file__).resolve().parent
def table(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))
def write_table(path, rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def compressed_frame(block, transform="raw", backend="zstd"):
    """Actual bytes, fixed geometry. Four-byte length, one-byte encoding tag."""
    if transform=="bytes":
        source = np.frombuffer(block,np.uint8).reshape(-1,4).T.copy().tobytes()
    elif transform=="bits":
        groups=np.frombuffer(block,np.uint8).reshape(-1,256,4)
        bits=np.unpackbits(groups,axis=2,bitorder="little").transpose(0,2,1)
        source=np.packbits(bits,axis=2,bitorder="little").tobytes()
    else:
        source = block
    packed = zstd.ZstdCompressor(level=3).compress(source) if backend=="zstd" else zlib.compress(source,int(backend))
    if len(packed)<len(block):
        tag={"raw":1,"bytes":2,"bits":3}[transform] + (0 if backend=="zstd" else 3)
        payload=packed
    else:
        tag=0
        payload=block
    frame = (1+len(payload)).to_bytes(4,'little') + bytes([tag]) + payload
    assert len(frame)==4+int.from_bytes(frame[:4],'little')
    if frame[4]:
        restored = zstd.ZstdDecompressor().decompress(frame[5:]) if frame[4]<=3 else zlib.decompress(frame[5:])
        if frame[4] in (2,5): restored=np.frombuffer(restored,np.uint8).reshape(4,-1).T.copy().tobytes()
        if frame[4] in (3,6):
            planes=np.frombuffer(restored,np.uint8).reshape(-1,32,32)
            bits=np.unpackbits(planes,axis=2,bitorder="little").transpose(0,2,1)
            restored=np.packbits(bits,axis=2,bitorder="little").tobytes()
    else:
        restored=frame[5:]
    if restored!=block: raise ValueError("Zstandard round trip")
    return len(frame)

def replay(data, output):
    output.mkdir(exist_ok=True,parents=True)
    exe=output/'replay'
    subprocess.run(['g++','-O3','-std=c++17',str(HERE/'run.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe.resolve()),str(data/'heads.u32'),str(output/'rows.csv'),str(output/'local.csv')],check=True)
    summary=[]
    indices=table(data/'heads.csv')
    sizes=table(output/'local.csv')
    assert len(sizes)==len(indices)
    with open(output/'rows.csv') as rf, open(data/'heads.u32','rb') as bf:
        for h,(idx,size) in enumerate(zip(indices,sizes)):
            if h==0: reader=csv.DictReader(rf)
            block=bf.read(HEAD_BYTES)
            assert len(block)==HEAD_BYTES and hashlib.sha256(block).hexdigest()==idx['head_sha256']
            rows=[{k:int(v) for k,v in r.items()} for r in itertools.islice(reader,128)]
            assert len(rows)==128 and all(r['id']==h and r['row']==i for i,r in enumerate(rows))
            s={k:int(idx[k]) if k in ('token','layer','head','id') else idx[k] for k in ('id','trace','token','layer','head','head_sha256')}
            assert int(size['id'])==h
            s.update(raw=HEAD_BYTES,mid=int(size['mid']),median=int(size['median']),opt=int(size['opt']))
            s.update(zstd=compressed_frame(block),shuffle=compressed_frame(block,"bytes"),
                     bitshuffle=compressed_frame(block,"bits"),deflate=compressed_frame(block,"bytes","9"))
            for k in ('mid_res','median_res','opt_res','lower'):
                s[k]=sum(r[k] for r in rows)
            # Relax the residual mode to a distinct free reference per group. A second
            # relaxation also gives away signs and the stored reference word.
            ud=np.frombuffer(block,dtype="<u4").reshape(128,16,8)&0x7fffffff
            spans=ud.max(axis=2)-ud.min(axis=2)
            individual=np.searchsorted(1<<np.arange(31,dtype=np.int64),spans,side="right").sum(axis=1)
            overhead=4+(1 if any(r['mid_mode']!=4 for r in rows) else 0)
            if overhead==4: # empty all-positive-zero head payload
                s['free_groups']=s['free_signs']=4
            else:
                s['free_groups']=overhead+sum(min(r['alt'],r['lower']) for r in rows)
                s['free_signs']=overhead+sum(min(r['alt'],13+int(w)) for r,w in zip(rows,individual))
            for r,w in zip(rows,individual): assert r['lower']-int(w) in (17,21,25,29,33)
            s['improved']=sum(r['opt']<r['mid'] for r in rows)
            s['intrinsic']=sum(r['opt_res']<r['mid_res'] for r in rows)
            s['pruned']=sum(r['lower']>=r['alt'] for r in rows)
            s['mid_cert']=sum(r['lower']<r['alt'] and r['mid_res']==r['lower'] for r in rows)
            for r in rows:
                assert r['lower']<=r['opt_res'] and r['opt']==min(r['alt'],r['opt_res'])
                assert r['mid']-r['opt']==max(0,min(r['mid_res']-r['opt_res'],r['alt']-r['opt_res']))
            for k in ('mid','opt'):
                for i in range(5): s[f'{k}_tag{i}']=sum(r[k+'_mode']==i for r in rows)
            summary.append(s)
        assert bf.read(1)==b'' and next(reader,None) is None
    write_table(output/'heads_summary.csv',summary)
    return summary

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('data',type=Path);a.add_argument('output',type=Path)
    args=a.parse_args();replay(args.data,args.output)
