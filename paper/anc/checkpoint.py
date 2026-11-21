"""Visualize the two-probe witness and the harmonic bit sum; check finite cases.

For uniform probability on the circle and beta=1, the convolution eigenvalues
are the Fourier coefficients I_l(1). The plotted values are theoretical shell
contributions, not measured file sizes or certified finite-epsilon bit budgets.
"""
from fractions import Fraction
from math import exp, log, factorial
from pathlib import Path
import csv
import numpy as np
from scipy.special import iv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def check_and_plot(figure_dir, evidence_dir):
    figure_dir.mkdir(exist_ok=True, parents=True)
    evidence_dir.mkdir(exist_ok=True, parents=True)
    steps = np.arange(13)
    heavy = [Fraction(int(j), 1) / (Fraction(13, 2) + int(j)) for j in steps]
    light = [Fraction(int(j), 1) / (Fraction(7, 2) + int(j)) for j in steps]
    assert light[1] - heavy[1] == Fraction(4, 45)
    degrees = np.arange(26)
    eigen = iv(degrees, 1.0)
    for ell, value in zip(degrees, eigen):
        ell = int(ell)
        assert 1/(2**ell * factorial(ell)) <= value*(1+1e-14)
        assert value <= exp(1)/factorial(ell)
    theta = 2*np.pi*np.arange(2048)/2048
    numeric = np.fft.rfft(np.exp(np.cos(theta)))/len(theta)
    np.testing.assert_allclose(numeric[:12], eigen[:12], rtol=0, atol=5e-16)
    # Test the inverse estimate on pairs of non-antipodal ordinary prefixes.
    rng = np.random.default_rng(20261121)
    max_ratio = 0.
    for _ in range(256):
        masses = []
        numerators = []
        for j in range(2):
            vals = rng.uniform(-1,1,8)
            weights = np.exp(np.cos(rng.uniform(-np.pi,np.pi,8)))
            masses.append(weights.mean())
            numerators.append(np.mean(weights*vals))
        z, x = np.array(masses), np.array(numerators)
        gaps = [abs(np.diff((x+sign*exp(1))/(z+exp(1)))[0]) for sign in (-1,1)]
        max_ratio = max(max_ratio, float(np.hypot(z[0]-z[1],x[0]-x[1])/max(gaps)))
    assert max_ratio <= 8*exp(1)
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig, (left,right) = plt.subplots(1,2,figsize=(7.0,2.55),layout="constrained")
    for values, color, marker, label in [(heavy,"#18557b","o","Mass 13/2"),(light,"#ba582e","s","Mass 7/2")]:
        left.plot(steps, list(map(float,values)),marker=marker,color=color,ms=3,label=label)
    left.set(xlabel="Common tokens appended",ylabel="Attention output",title="A   One suffix reveals mass",xticks=[0,3,6,9,12])
    left.legend(frameon=False,fontsize=8)
    rows=[]
    for digits,color,marker in [(3,"#26736f","^"),(6,"#18557b","o"),(12,"#ba582e","s")]:
        vals=np.where(degrees==0,1,2)*np.maximum(0,np.log2(eigen)+digits*np.log2(10))
        right.plot(degrees[:15],vals[:15],lw=1.4,marker=marker,ms=3,color=color,label=rf"$10^{{-{digits}}}$")
        rows.extend(zip([digits]*len(vals),degrees,eigen,vals))
    right.set(xlabel="Harmonic degree",ylabel="Shell contribution (bits)",title="B   Precision per degree",xticks=[0,3,6,9,12])
    right.legend(frameon=False,fontsize=8,title="Error",loc="upper right")
    for ax in (left,right): ax.grid(axis="y",alpha=.2)
    fig.savefig(figure_dir/"checkpoint.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(evidence_dir/"checkpoint.png",dpi=160)
    plt.close(fig)
    with (evidence_dir/"shells.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["decimal_digits","degree","eigenvalue","scalar_shell_bits"])
        writer.writerows(rows)
    print("Two-probe gap 4/45; Fourier eigenvalues and 256 non-antipodal probe pairs verified.")


if __name__=="__main__":
    import sys
    check_and_plot(Path(sys.argv[1]),Path(sys.argv[2]))

