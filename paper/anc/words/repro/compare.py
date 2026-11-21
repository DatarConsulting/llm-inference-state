"""Compare regenerated head observations and isolate external compressor drift."""
import argparse
import csv
from pathlib import Path
def read(path):
    with open(path,newline="") as f: return list(csv.DictReader(f))
def compare(reference, observed):
    a,b=read(reference),read(observed)
    if len(a)!=len(b) or not a or set(a[0])!=set(b[0]): raise ValueError("Different census size or fields")
    external={'zstd','shuffle','bitshuffle','deflate'}
    changes={k:0 for k in sorted(external)}
    for x,y in zip(a,b):
        for k in x:
            if x[k]!=y[k]:
                if k not in external: raise ValueError(f"Unexpected {k} at head {x['id']}")
                changes[k]+=1
    print(f"{len(a):,} exact head records checked. External compressor count differences: {changes}")
    return changes
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference');p.add_argument('observed')
    a=p.parse_args();compare(a.reference,a.observed)
