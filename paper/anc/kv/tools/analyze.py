"""Rebuild the paper's tables and figures from the archived generation-256-v1 record."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIGEST = "baeb7838f2c21ab7f7f17741bac85c459cf91960ebf21411ec62fa0b6f9beb1f"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LAYERS = [3, 7, 11, 15, 19, 23]
LENGTHS = [4096, 32768, 131072]
BOOKS = ["Emma", "Moby-Dick"]
METHODS = ["FI", "compact"]
BLUE, ORANGE, INK = "#126078", "#a64616", "#1b2630"
plt.rcParams.update({"font.size": 9, "font.family":"DejaVu Sans", "pdf.fonttype":42,
    "axes.spines.top":False, "axes.spines.right":False, "axes.labelcolor":INK,
    "axes.titleweight":"bold", "axes.titlelocation":"left", "axes.axisbelow":True})

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def pooled(runs):
    return sum(r["tokens"] for r in runs) / sum(r["seconds"] for r in runs)

def savefig(fig, directory, name):
    fig.savefig(directory/(name+".pdf"), metadata={"Creator":"Paper reproduction", "CreationDate":None, "ModDate":None})
    fig.savefig(directory/(name+".png"), dpi=175)
    plt.close(fig)

def analyze(source, directory, released=True):
    directory.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    if released:
        require(hashlib.sha256(raw).hexdigest()==DIGEST, "The paper is locked to its archived input.")
    data = json.loads(raw)
    require(data["protocol"]=="generation-256-v1" and data["stage"]=="terminé", "Incomplete or incompatible protocol.")
    require(data["model"]["revision"]==REVISION and data["model"]["full_layers"]==LAYERS, "Unexpected model.")
    cases = sorted(data["cases"], key=lambda c:(c["prefix"], BOOKS.index(c["title"])))
    require([(c["prefix"],c["title"]) for c in cases] == [(n,b) for n in LENGTHS for b in BOOKS], "Expected six cases.")
    details, pair_rows, groups, teacher, free_positions = [], [], [], [], 0
    for c in cases:
        n, book = c["prefix"], c["title"]
        runs = c["runs"]
        require(len(runs)==8 and c["status"]=="terminé", "Incomplete run collection.")
        by = {(r["pair"],r["method"]):r for r in runs}
        require(set(by)=={(p,m) for p in range(1,5) for m in METHODS}, "Invalid pairing.")
        first = by[1,"FI"]["ids"]
        for p in range(1,5):
            a,b = [by[p,m] for m in METHODS]
            for r in (a,b):
                require(len(r["ids"])==r["tokens"]==len(r["step_ms"])==64 and r["seconds"]>0, "Invalid timing record.")
                require(math.isclose(r["tokens_s"],64/r["seconds"],rel_tol=1e-10), "Stored rate mismatch.")
            if released:
                require(a["ids"]==b["ids"]==first, "Released free trajectory changed.")
            pair_rows.append(dict(book=book,n=n,pair=p,first=next(r["method"] for r in runs if r["pair"]==p),
                                  gain=100*(a["seconds"]/b["seconds"]-1),same=(a["ids"]==b["ids"])))
        free_positions+=sum(x==y for x,y in zip(first,by[1,"compact"]["ids"]))
        teacher.extend(c["teacher"])
        rows = {m:[r for r in runs if r["method"]==m] for m in METHODS}
        a,b=[rows[m][0]["cache"] for m in METHODS]
        for m in METHODS:
            for r in rows[m]:
                require(r["cache"]["reserved_kv_bytes"]==sum(v["reserved_kv_bytes"] for v in r["cache"]["layers"].values()), "Layer totals differ.")
                start = c["initial_kv"][m]
                if m=="FI":
                    require(start["reserved_kv_bytes"]==start["logical_kv_bytes"]+64*12288, "Dense allocation.")
                else:
                    require(start["reserved_kv_bytes"]==start["logical_kv_bytes"]+64*12384, "Compact allocation.")
                    for layer in LAYERS:
                        v=r["cache"]["layers"][str(layer)]
                        expected=128*744+128*v["appended_raw_k"]+160*v["appended_raw_v"]
                        require(v["logical_kv_bytes"]-start["layers"][str(layer)]["logical_kv_bytes"]==expected,"Tail byte identity.")
        reserve=a["reserved_kv_bytes"]
        kept=b["reserved_kv_bytes"]
        rec=dict(book=book,n=n,fi=pooled(rows["FI"]),compact=pooled(rows["compact"]),
                 fi_reserved=reserve/2**20,compact_reserved=kept/2**20,saved=(reserve-kept)/2**20,
                 percent=100*(1-kept/reserve),prefill=c["prefill_s"])
        for m in METHODS:
            rec[m+"_build"]=c["build_s"][m]
            rec[m+"_decode"]=statistics.mean(r["seconds"] for r in rows[m])
            rec[m+"_total"]=rec["prefill"]+rec[m+"_build"]+rec[m+"_decode"]
        details.append(rec)
    for n in LENGTHS:
        rs=[r for c in cases if c["prefix"]==n for r in c["runs"]]
        fa,pa=[pooled([r for r in rs if r["method"]==m]) for m in METHODS]
        ds=[d for d in details if d["n"]==n]
        groups.append(dict(n=n,fi=fa,compact=pa,gain=100*(pa/fa-1),low=min(d["saved"] for d in ds),high=max(d["saved"] for d in ds)))
    def csvout(name, rows):
        with (directory/(name+".csv")).open("w", newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvout("conditions",details);csvout("pairs",pair_rows);csvout("pooled",groups)
    nl=" "+chr(92)*2+"\n"
    tex="".join(f"{g['n']:,} & {g['fi']:.2f} & {g['compact']:.2f} & {g['gain']:+.2f} & {g['low']:.1f}--{g['high']:.1f}"+nl for g in groups)
    (directory/"pooled.tex").write_text(chr(92)+"newcommand{"+chr(92)+"PooledRows"+"}{%\n"+tex+"}\n")
    tex="".join(f"{d['book']} & {d['n']:,} & {d['prefill']:.3f} & {d['FI_build']*1e3:.2f} & {d['compact_build']:.3f} & {d['FI_total']:.2f} & {d['compact_total']:.2f}"+nl for d in details)
    (directory/"timing.tex").write_text(chr(92)+"newcommand{"+chr(92)+"TimingRows"+"}{%\n"+tex+"}\n")
    q={"matching_positions_across_six_continuations":free_positions,
       "teacher_matches":sum(x["top1_match"] for x in teacher),"teacher_steps":len(teacher),
       "max_relative_l2":max(x["relative_logits_l2"] for x in teacher),"max_kl":max(x["kl_dense_compact"] for x in teacher),
       "timed_seconds":sum(r["seconds"] for c in cases for r in c["runs"])}
    (directory/"quality.json").write_text(json.dumps(q, indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6.05,2.55), layout="constrained")
    for j,d in enumerate(details):
        rs=[r for r in pair_rows if (r["book"],r["n"])==(d["book"],d["n"])]
        y=5-j
        ax.plot([min(r["gain"] for r in rs),max(r["gain"] for r in rs)],[y,y],color="#96a4ac",lw=1)
        for k,r in enumerate(rs):
            densefirst=r["first"]=="FI"
            ax.scatter(r["gain"],y+(k-1.5)*.076,s=23,marker="o" if densefirst else "s",
                       color=BLUE if densefirst else ORANGE,zorder=3,
                       label=("Dense first" if densefirst else "Compact first") if j==0 and k<2 else None)
    ax.set_yticks(range(5,-1,-1),[f"{d['n']:,}   {d['book']}" for d in details])
    ax.axvline(0,color=INK,lw=1,ls="--");ax.grid(axis="x",alpha=.23)
    ax.set_xlabel("Compact throughput change relative to dense (%)")
    ax.legend(ncol=2,frameon=False,loc="lower right",bbox_to_anchor=(1.,1.0),fontsize=8)
    ax.set_ylim(-.55,5.5)
    savefig(fig,directory,"paired")
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.6),layout="constrained",width_ratios=[1,1.08])
    labels=["4K","32K","128K"]; x=np.arange(3)
    for book,marker,color in zip(BOOKS,["o","s"],[BLUE,ORANGE]):
        ds=[d for d in details if d["book"]==book]
        axes[0].plot(x,[d["percent"] for d in ds],marker=marker,color=color,label=book,lw=1.25)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,25),ylabel="Reserved KV bytes saved (%)",xlabel="Prefix length (tokens)")
    axes[0].legend(frameon=False,fontsize=7,loc="lower right")
    axes[0].set_title("A   Cache reservation",fontsize=10)
    labels2=["Dense","Compact","Dense","Compact"];x2=np.array([0,.8,2.05,2.85])
    long=[d for d in details if d["n"]==131072]
    pv=[d["prefill"] for d in long for m in METHODS]
    bv=[d[m+"_build"] for d in long for m in METHODS]
    dv=[d[m+"_decode"] for d in long for m in METHODS]
    for vals,base,col,name in [(pv,np.zeros(4),"#a6b4bc","Prefill"),(bv,pv,ORANGE,"Build"),(dv,np.add(pv,bv),BLUE,"Decode")]:
        axes[1].bar(x2,vals,bottom=base,width=.62,color=col,label=name)
    axes[1].set(xticks=x2,xticklabels=labels2,ylabel="Reconstructed time (s)",ylim=(0,30),xlabel="Emma                   Moby-Dick")
    axes[1].tick_params(axis="x",labelsize=7)
    axes[1].set_title("B   128K + 64 tokens",fontsize=10)
    axes[1].legend(ncol=3,frameon=False,fontsize=6.7,loc="upper center",handlelength=1,columnspacing=.9)
    for ax in axes:ax.grid(axis="y",alpha=.22)
    savefig(fig,directory,"budget")
    counts=[]
    for l in LAYERS:
        cnt=[sum(next(r["cache"]["layers"][str(l)]["appended_raw_"+k] for r in c["runs"] if r["method"]=="compact") for c in cases) for k in ("k","v")]
        counts.append(dict(layer=l,K=cnt[0],V=cnt[1],denominator=768))
    csvout("fallback",counts)
    fig,ax=plt.subplots(figsize=(6.05,2.08),layout="constrained")
    for j,(key,col) in enumerate([("K",BLUE),("V",ORANGE)]):
        ax.bar(np.arange(6)+(j-.5)*.33,[100*d[key]/768 for d in counts],width=.33,color=col,label=key)
    ax.set(xticks=np.arange(6),xticklabels=LAYERS,ylim=(0,108),xlabel="Full-attention layer (zero-based)",ylabel="Raw records (%)")
    ax.grid(axis="y",alpha=.22);ax.legend(frameon=False,ncol=2,loc="upper left")
    savefig(fig,directory,"fallback")
    return q

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--fresh",action="store_true",help="Analyze a fresh full-protocol trace in a separate output directory.")
    args=p.parse_args()
    print(json.dumps(analyze(args.input,args.output,not args.fresh),indent=2))
