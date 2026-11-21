from fractions import Fraction
from itertools import combinations, product
from math import comb, factorial, lgamma, log, log1p
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcdefaults()
INK, BLUE, ORANGE, GRAY = "#192b3b", "#16657b", "#974114", "#d6dcdf"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#87929b", "axes.spines.right": False,
    "axes.spines.top": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
    "axes.axisbelow": True, "lines.linewidth": 1.8,
})

def save_vector(fig, path):
    """A fixed canvas and no creation timestamp keep the figure portable."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06,
                metadata={"Creator": "Softmax denominator reproduction",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)

def prefix_read(prefix_mass, suffix_length):
    return Fraction(suffix_length, 1) / (prefix_mass + suffix_length)

def witness_data():
    rows = []
    for j in range(13):
        heavy = prefix_read(Fraction(13, 2), j)
        light = prefix_read(Fraction(7, 2), j)
        rows.append((j, heavy, light))
    assert rows[1][1:] == (Fraction(2, 15), Fraction(2, 9))
    assert (rows[1][2] - rows[1][1]) / 2 == Fraction(2, 45)
    return rows

def witness_figure(rows, target):
    fig, ax = plt.subplots(figsize=(5.25, 1.88))
    x = [r[0] for r in rows]
    ax.plot(x, [float(r[1]) for r in rows], "o-", color=BLUE, ms=3.3,
            label=r"Prefix mass $13/2$")
    ax.plot(x, [float(r[2]) for r in rows], "s--", color=ORANGE, ms=3.2,
            label=r"Prefix mass $7/2$")
    ax.set(xlabel="Number of common appended tokens", ylabel="Attention output",
           ylim=(-.025, .85), xlim=(-.15,12.15), xticks=[0,1,2,4,6,8,10,12])
    ax.grid(axis="y", color=GRAY, linewidth=.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_vector(fig, target)

def harmonic_dimension(d, degree):
    if d < 2 or degree < 0:
        raise ValueError("Use ambient dimension >= 2 and a nonnegative degree.")
    return comb(degree + d - 1, d - 1) + comb(degree + d - 2, d - 1)

def certified_degrees(d, digits, beta=1.0):
    """Largest certified lower degree and smallest sufficient upper degree."""
    log_eps = -digits * log(10)
    log_c = 0.0
    lower = 0
    for ell in range(1, 10001):
        log_c += log(beta) - log(d + 2 * ell - 2)
        dim = harmonic_dimension(d, ell)
        threshold = -3 * beta + min(0.0, log_c) - log(8) - .5 * log(dim)
        if threshold <= log_eps:
            break
        lower = ell
    else:
        raise ValueError("Lower degree search limit exceeded.")
    # delta <= eps / (2 + eps) is equivalent to 2 delta / (1-delta) <= eps.
    threshold = log_eps - log(2.0) - log1p(10.0**(-digits) / 2.0)
    for m in range(10001):
        log_tail_bound = (m + 1) * log(2 * beta) - lgamma(m + 2)
        if log_tail_bound <= threshold:
            return lower, m
    raise ValueError("Upper degree search limit exceeded.")

def finite_rows():
    rows = []
    p = 2
    for d in (2, 3, 4):
        for digits in range(1, 121):
            ell, m = certified_degrees(d, digits)
            lower = (p + 1)*harmonic_dimension(d, ell) - 1 if ell else 0
            upper = (p + 1)*harmonic_dimension(d, m)
            assert lower <= upper
            rows.append((d, digits, ell, m, lower, upper))
    return rows

def check_degree_selection(rows):
    # At beta = 1 the factorial test is exactly rational. Squaring the
    # lower test leaves a single exp(6), enclosed here by rational bounds.
    # A positive Taylor sum for e^6 and a geometric tail give that enclosure.
    last = Fraction(6**70, factorial(70))
    partial = sum(Fraction(6**j, factorial(j)) for j in range(71))
    exp6_min, exp6_max = partial, partial + last * Fraction(6, 71) / (1-Fraction(6,72))
    for d, digits, ell, m, _, _ in rows:
        if digits not in (1,3,10,40,80,120):
            continue
        eps = Fraction(1, 10**digits)
        bound = lambda k: Fraction(2**(k+1), factorial(k+1))
        assert bound(m) <= eps/(2+eps)
        assert m == 0 or bound(m-1) > eps/(2+eps)
        for k in ([ell, ell+1] if ell else [1]):
            denom = 1
            for j in range(k):
                denom *= d + 2*j
            comparison = Fraction(1, denom**2) / (64*harmonic_dimension(d,k)*eps**2)
            if k == ell:
                assert comparison > exp6_max
            else:
                assert comparison <= exp6_min

def dimension_figure(rows, target):
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.05), layout="constrained")
    for ax, d in zip(axes, (2, 3, 4)):
        selected = np.array([r for r in rows if r[0] == d])
        x, lo, hi = selected[:,1], selected[:,4], selected[:,5]
        # The log y-axis cannot display a trivial zero lower bound.
        keep = lo > 0
        ax.fill_between(x[keep],lo[keep],hi[keep],color="#e0edef",step="post")
        ax.step(x,hi,where="post",color=ORANGE,ls="--",label="Upper")
        ax.step(x[keep],lo[keep],where="post",color=BLUE,label="Lower")
        ax.set(yscale="log",xlim=(0,120),xticks=[0,60,120],title=f"$d={d}$",
               xlabel=r"$-\log_{10}\,\epsilon$")
        ax.grid(axis="y",which="major",color=GRAY,linewidth=.6)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Real coordinates")
    axes[0].legend(loc="lower right",frameon=False,fontsize=7)
    save_vector(fig, target)

def check_two_signs():
    # Exact componentwise algebra on a nonscalar example with nonzero cross term.
    A, b, z = Fraction(7,2), Fraction(3,2), Fraction(-2,3)
    X = [Fraction(1,5), Fraction(-1,3)]
    e = [Fraction(3,5), Fraction(4,5)]
    energy = Fraction(0)
    for sign in (-1,1):
        for xi, ei in zip(X,e):
            gap = (xi+sign*b*ei)/(A+z) - (-xi+sign*b*ei)/(A-z)
            assert gap == 2*(A*xi-sign*b*z*ei)/(A*A-z*z)
            energy += gap**2
    assert energy*(A*A-z*z)**2 == 8*(A*A*sum(x*x for x in X)+b*b*z*z)

def check_designs():
    reports = {}
    for d, ell in [(2,4),(3,1),(4,2)]:
        if d == 2:
            theta = 2*np.pi*np.arange(17)/17
            points = np.column_stack((np.cos(theta),np.sin(theta)))
            cols = [np.ones_like(theta)]
            for k in range(1,ell+1):
                cols.extend([np.sqrt(2)*np.cos(k*theta),np.sqrt(2)*np.sin(k*theta)])
            V = np.column_stack(cols)
        elif d == 3:
            points = np.vstack((np.eye(d),-np.eye(d)))
            V = np.column_stack((np.ones(len(points)),np.sqrt(d)*points))
        else:
            # The 24-cell in four dimensions is a 5-design. The analytic
            # degree-two basis comprises cross terms and traceless diagonals.
            pairs = list(combinations(range(d),2))
            vertices = []
            for i,j in pairs:
                for a,b in product((-1,1),repeat=2):
                    x = np.zeros(d)
                    x[[i,j]] = [a/np.sqrt(2),b/np.sqrt(2)]
                    vertices.append(x)
            points = np.array(vertices)
            cols = [np.ones(len(points)), *(np.sqrt(d)*points.T)]
            cols.extend(np.sqrt(d*(d+2))*points[:,i]*points[:,j] for i,j in pairs)
            for k in range(1,d):
                t = np.zeros(d)
                t[:k], t[k] = 1, -k
                t /= np.linalg.norm(t)
                cols.append(np.sqrt(d*(d+2)/2)*(points**2@t))
            V = np.column_stack(cols)
        H = len(points)
        assert np.max(np.abs(V.T@V/H-np.eye(V.shape[1]))) < 2e-14
        K = np.exp(points@points.T)/H
        eig = np.linalg.eigvalsh(V.T@K@V/H)
        c = 1.0
        for k in range(ell):
            c /= d+2*k
        assert eig.min() >= c - 3e-14
        reports[f"d={d},L={ell}"] = {"sampled_min_eigenvalue": float(eig.min()),
                                     "proved_lower_bound": c}
    return reports

def check_transport():
    # Complex Fourier coordinates implement the real harmonic recurrence on the circle. The grid extracts the coefficients
    # of the trigonometric polynomial exactly in real arithmetic.
    m, beta = 6, .8
    theta = 2*np.pi*np.arange(2*m+1)/(2*m+1)
    modes = np.arange(-m,m+1)
    def poly(s):
        return np.exp(-beta)*sum((beta*(1+s))**j/factorial(j) for j in range(m+1))
    coeff = np.array([np.mean(poly(np.cos(theta))*np.exp(-1j*k*theta)) for k in modes])
    moments = np.zeros((len(modes),3), complex)
    angles, vals = [], np.empty((0,2))
    error = 0.0
    for j in range(1,42):
        rot, key, query = .37*np.sin(j), .73*j, .17*j*j
        v = np.array([.7*np.sin(j),.4*np.cos(2*j)])
        rotation = np.array([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
        # Store e^{-ik angle} times the augmented value, in the current frame.
        moments *= np.exp(-1j*modes*rot)[:,None]
        moments[:,1:] = moments[:,1:]@rotation.T
        moments += np.exp(-1j*modes*key)[:,None]*np.r_[1,v]
        angles = [a+rot for a in angles]+[key]
        vals = np.vstack((vals@rotation.T,v))
        w = poly(np.cos(query-np.array(angles)))
        aggregate = (coeff*np.exp(1j*modes*query))@moments
        assert np.max(np.abs(aggregate.imag)) < 2e-12
        error = max(error, float(np.max(np.abs(aggregate.real[1:]/aggregate.real[0]-w@vals/w.sum()))))
    assert error < 2e-13
    return error

def reproduce(figure_dir, evidence_dir):
    figure_dir, evidence_dir = Path(figure_dir), Path(evidence_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    witness, bounds = witness_data(), finite_rows()
    # Supply the manuscript example from the same integer degrees as Figure 2.
    finite_example = next(row for row in bounds if row[:2] == (3, 6))
    macros = ("ExampleLowerDegree", "ExampleUpperDegree", "ExampleLower", "ExampleUpper")
    (figure_dir/"example.tex").write_text("".join(
        rf"\newcommand{{\{name}}}{{{value}}}" + "\n"
        for name,value in zip(macros,finite_example[2:])))
    check_two_signs()
    check_degree_selection(bounds)
    report = {"fifth_token": [str(x) for x in witness[1][1:]],
              "forced_error_if_merged": "2/45", "spectral_examples": check_designs(),
              "max_circle_recurrence_error": check_transport(),
              "bound_rows": len(bounds), "rational_degree_checks": 18}
    for name, header, rows in [
        ("witness", ["suffix_tokens","heavy_exact","light_exact"], witness),
        ("bounds", ["ambient_dimension","decimal_digits","lower_degree","upper_degree","lower","upper"], bounds)]:
        with (evidence_dir/(name+".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    (evidence_dir/"checks.json").write_text(json.dumps(report,indent=2)+"\n")
    witness_figure(witness,figure_dir/"witness.pdf")
    dimension_figure(bounds,figure_dir/"dimension.pdf")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("figs"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    print(json.dumps(reproduce(args.figures,args.evidence), indent=2))

