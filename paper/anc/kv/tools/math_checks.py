"""Small exact witnesses for the algebra in the supplement (standard library only)."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from math import tanh

def exact_box():
    # One shared KV group, two heads, two source positions, two coordinates.
    p=((F(1,4),F(3,4)),(F(2,3),F(1,3)))
    A=(((F(2),F(-1)),(F(-2),F(3))),((F(1),F(2)),(F(-1),F(-2))))
    eta=(F(1,3),F(1,5),F(1,4),F(2,7))
    coeff=[[sum(A[j][h][c]*p[h][i] for h in range(2))
            for i,c in product(range(2),repeat=2)] for j in range(2)]
    sharp=max(sum(abs(x)*v for x,v in zip(row,eta)) for row in coeff)
    brute=F(0)
    for signs in product((-1,1), repeat=4):
        e=[[signs[2*i+c]*eta[2*i+c] for c in range(2)] for i in range(2)]
        outs=[sum(A[j][h][c]*sum(p[h][i]*e[i][c] for i in range(2))
                for h,c in product(range(2),repeat=2)) for j in range(2)]
        brute=max(brute,*map(abs,outs))
    assert brute==sharp
    return str(sharp)

def coefficients(keys,residuals,weights):
    result=defaultdict(F)
    for choice in product(*(range(len(k)) for k in keys)):
        s=sum(k[i] for k,i in zip(keys,choice))
        val=sum(r[i] for r,i in zip(residuals,choice))
        w=F(1)
        for ws,i in zip(weights,choice):
            w*=ws[i]
        result[s]+=w*val
    return dict(result)

def rational_probe(z,keys,residuals,weights):
    out=F(0);den=F(1)
    for ks,rs,ws in zip(keys,residuals,weights):
        terms=[w*z**k for w,k in zip(ws,ks)]
        den*=sum(terms)
        out+=sum(v*t for v,t in zip(rs,terms))/sum(terms)
    return out,den

def checks():
    keys=((0,1),(0,1));r=((1,-2),(-1,2));w=((1,2),(1,2))
    assert all(v==0 for v in coefficients(keys,r,w).values())
    keys=((0,1,1),(0,2));r=((2,-1,3),(-2,4));w=((1,1,2),(1,3))
    a=coefficients(keys,r,w)
    assert any(a.values())
    for m in range(len(a)):
        z=F(2)**m
        observed,den=rational_probe(z,keys,r,w)
        assert observed*den==sum(v*z**k for k,v in a.items())
    # One projected head with a duplicate key: 1*2 + 2*(-1) = 0.
    assert all(v==0 for v in coefficients(((0,0,1),),((2,-1,0),),((1,2,1),)).values())
    for z in (F(1,7), F(1), F(5,2)):
        assert z/(1+z)+(1/z)/(1+1/z)==1
    assert 0<tanh(0.0005)<0.0005
    assert (6*2*(512+512+8),6*2*(512+512))==(12384,12288)
    print("Exact shared-group box optimum:",exact_box())
    print("Passed exact cancellation, nonzero collision sums, finite probes, duplicate keys and byte identities.")
if __name__=="__main__":
    checks()
