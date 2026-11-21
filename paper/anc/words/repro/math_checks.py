"""Independent exact-integer audit for the single- and multi-reference theorems."""
from collections import Counter
from itertools import combinations, product
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

def zig(d):
    return 2*d if d>=0 else -2*d-1

def width(group,p):
    return max(zig(a-p) for a in group).bit_length()

def intervals(groups,b):
    top=(1<<b)-1
    for group in groups:
        lo,hi=min(group),max(group)
        if hi==lo: yield (hi,lo)
        for t in range(1,b+1):
            half=1<<(t-1)
            left,right=max(0,hi-half+1),min(top,lo+half)
            if left<=right: yield (left,right)

def optimal_width(groups,b,k=1):
    """Minimum sum of widths with at most k shared anchors; width ceiling b+1."""
    intervals_=list(intervals(groups,b))
    points=sorted({0}|{l for l,r in intervals_})
    best={j:sum(l<=p<=r for l,r in intervals_) for j,p in enumerate(points)}
    covered=max(best.values())
    for count in range(2,min(len(points),k)+1):
        nxt={}
        for j,p in enumerate(points):
            choices=[value+sum(points[i]<l<=p<=r for l,r in intervals_)
                     for i,value in best.items() if i<j]
            if choices: nxt[j]=max(choices)
        best=nxt
        covered=max(covered,max(best.values()))
    return len(groups)*(b+1)-covered

def brute(groups,b,k):
    return min(sum(min(width(g,p) for p in anchors) for g in groups)
               for anchors in combinations(range(1<<b),k))

def run():
    checks=Counter()
    # Distinct ordered data records, including ties and constant groups.
    for row in product(range(8),repeat=4):
        groups=[row[:2],row[2:]]
        for k in (1,2):
            assert optimal_width(groups,3,k)==brute(groups,3,k)
            checks[f'exhaustive_k{k}']+=1
    rng=random.Random(20261121)
    for _ in range(180):
        groups=[rng.choices(range(16),k=3) for i in range(4)]
        for k in (1,2,3):
            assert optimal_width(groups,4,k)==brute(groups,4,k)
            checks[f'random_k{k}']+=1
    for g in (2,3,8,16):
        for k in range(1,30):
            m=1<<26
            groups=[[m]*8 for j in range(g-1)]+[[m]*7+[m+(1<<k)]]
            exact=optimal_width(groups,31)
            mid=m+(1<<(k-1))
            assert exact==k+2
            assert sum(width(v,mid) for v in groups)-exact==(g-1)*k-1
            checks['gap_identities']+=1
    for points in ([0],[0,(1<<31)-1],[(1<<31)-1],[(1<<31)-2,(1<<31)-1]):
        assert optimal_width([points],31)==(max(points)-min(points)).bit_length()
        checks['boundary_cases']+=1
    return dict(checks)

if __name__=='__main__': print(json.dumps(run(),indent=2))
