"""Generate publication figures and TeX tables from per-head observations."""
import csv
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math_checks import width

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[1]
DATA=ROOT/'data'
NAVY='#214d6d'; TEAL='#167b79'; ORANGE='#bc591e'; LIGHT='#a8c5d3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                    'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
                    'axes.labelcolor':'#263b48','xtick.color':'#364751','ytick.color':'#364751'})
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def save(fig,name):
    fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.035,
                metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
def summ(rows,key): return sum(int(r[key]) for r in rows)
def table(name,lines):
    (PAPER/'figures'/(name+'.tex')).write_text('\n'.join(lines)+'\n')
def num(i): return f'{i:,}'
def percent(x): return f'{100*x:.2f}'+r'\%'
def row(*cells): return ' & '.join(map(str,cells))+r' \\'
def main():
    (PAPER/'figures').mkdir(exist_ok=True,parents=True)
    rows=read(DATA/'full_summary.csv'); timings=read(DATA/'timing.csv')
    assert len(rows)==2592 and len({(r['trace'],r['token'],r['layer'],r['head']) for r in rows})==2592
    methods=[('raw','Native FP32'),('zstd','Zstandard'),('shuffle','Byte shuffle + Zstandard'),
             ('mid','Local, midpoint'),('median','Local, lower median'),('opt','Local, optimum'),
             ('deflate','Byte shuffle + DEFLATE'),('bitshuffle','Bit shuffle + Zstandard')]
    raw=summ(rows,'raw')
    table('sizes',[row(label,num(summ(rows,k)),f'{32*summ(rows,k)/raw:.3f}',percent(1-summ(rows,k)/raw)) for k,label in methods])
    panel=[]
    for t in (1,428,460):
        sel=[r for r in rows if int(r['token'])==t]
        before=summ(sel,'mid_res')-summ(sel,'opt_res')
        after=summ(sel,'mid')-summ(sel,'opt')
        panel.append(row(t,num(before),num(after),percent(after/before),num(summ(sel,'improved'))))
    table('checkpoints',panel)
    floor_rows=[]
    for t in (1,428,460):
        sel=[r for r in rows if int(r['token'])==t]
        floor_rows.append(row(t,*[num(summ(sel,k)) for k in ('free_groups','free_signs','opt','bitshuffle')]))
    table('floors',floor_rows)
    labels={'mid':'Midpoint','median':'Lower median','opt':'Exact sweep','screened':'Screened sweep'}
    lines=[]
    for k,label in labels.items():
        sel=[r for r in timings if r['method']==k];t=[float(r['seconds'])*1000 for r in sel]
        assert len(t)==8
        lines.append(row(label,f'{statistics.median(t):.2f}',f'{min(t):.2f}--{max(t):.2f}',f'{int(sel[0]["raw_bytes"])/(1000*statistics.median(t)):.1f}'))
    table('timings',lines)
    # A five-group constructive example. The ordinate is bytes because groups contain eight words.
    groups=[[24]*8,[24]*8,[24]*8,[24]*8,[24]*7+[56]]
    ps=list(range(15,67)); C=[sum(width(g,p) for g in groups) for p in ps]
    fig,ax=plt.subplots(figsize=(6.35,2.0),layout='constrained')
    ax.step(ps,C,where='mid',c=NAVY,lw=1.8)
    for p,c,label in [(24,TEAL,'Optimum'),(40,ORANGE,'Midpoint')]:
        y=C[ps.index(p)];ax.scatter([p],[y],s=38,color=c,zorder=4)
        ax.annotate(f'{label}  ({p}, {y})',(p,y),xytext=(8,1 if p==40 else 5),textcoords='offset points',color=c,weight='bold',fontsize=9)
    ax.set(xlabel='Stored reference (integer magnitude word)',ylabel='Residual payload (bytes)',xlim=(15,66),ylim=(0,max(C)+4));ax.grid(axis='y',alpha=.22)
    save(fig,'reference_cost')
    # Reduction before and after competition; nine points in each panel, same data scopes.
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.42),layout='constrained')
    for ax,key1,key2,title in zip(axs,['mid_res','mid'],['opt_res','opt'],['Residual mode alone','Complete local format']):
        for trace,marker,colour,off in zip(('atelier','coast','ledger'),('o','s','^'),(NAVY,ORANGE,TEAL),(-.1,0,.1)):
            vals=[]
            for t in (1,428,460):
                sel=[r for r in rows if r['trace']==trace and int(r['token'])==t]
                vals.append((summ(sel,key1)-summ(sel,key2))/len(sel))
            ax.plot(np.arange(3)+off,vals,linestyle='none',marker=marker,color=colour,label=trace,ms=6)
        ax.set_xticks(range(3),['1','428','460']);ax.set_xlim(-.47,2.47);ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.22);ax.set_title(title,fontsize=10,loc='left',weight='bold');ax.set_xlabel('Token position')
        ax.yaxis.set_major_locator(MaxNLocator(5,integer=True))
    axs[0].set_ylabel('Bytes saved per head');axs[1].legend(loc='lower right',frameon=False,fontsize=7,ncol=1)
    save(fig,'mode_competition')
    layers=sorted({int(r['layer']) for r in rows});names=('atelier','coast','ledger')
    heat=[]
    for trace in names:
        for t in (1,428,460):
            v=[]
            for l in layers:
                sel=[r for r in rows if r['trace']==trace and int(r['token'])==t and int(r['layer'])==l]
                v.append((summ(sel,'mid')-summ(sel,'opt'))/len(sel))
            heat.append(v)
    fig,ax=plt.subplots(figsize=(6.9,2.8),layout='constrained')
    im=ax.imshow(heat,aspect='auto',interpolation='nearest',cmap='viridis',vmin=0)
    ax.set_xticks(range(len(layers)),layers,fontsize=8);ax.set_yticks(range(9),[f'{s}  /  {t}' for s in names for t in (1,428,460)],fontsize=7.5)
    ax.set_xlabel('Recurrent layer');fig.colorbar(im,ax=ax,label='Bytes saved per head',shrink=.93)
    for b in (2.5,5.5): ax.axhline(b,color='white',lw=1)
    save(fig,'layers')
    print('Wrote three vector figures and four evidence tables.')
if __name__=='__main__':main()
