"""Make a single-root arXiv source ZIP and compile all editions from its contents."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
NAME="Draga_LLM"
FIGURES=["checkpoint.pdf","dimension.pdf","reference_cost.pdf","budget.pdf","layers.pdf",
         "paired.pdf","fallback.pdf","mode_competition.pdf"]
TABLES=["pooled.tex","timing.tex","sizes.tex","floors.tex","checkpoints.tex","timings.tex"]

def members():
    yield from sorted(ROOT.glob("*.tex"))
    yield ROOT/"neurips_2026.sty"
    yield ROOT/"00README.json"
    for name in FIGURES+TABLES:
        yield ROOT/"figures"/name
    for path in sorted((ROOT/"anc").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc",".pyo"):
            yield path

def compile_edition(source, work, key, flags):
    work.mkdir(parents=True,exist_ok=True)
    command=["pdflatex","-no-shell-escape","-halt-on-error","-interaction=nonstopmode",
             "-file-line-error","-jobname="+key,"-output-directory="+str(work)]
    command+=[flags+r"\input{main.tex}" if flags else "main.tex"]
    for attempt in range(4):
        result=subprocess.run(command,cwd=source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode!=0:
            raise RuntimeError("LaTeX error:\n"+result.stdout[-4500:])
        log=(work/(key+".log")).read_text(errors="replace")
        if attempt and not any(s in log for s in ["There were undefined references","There were undefined citations","Rerun to get cross-references","Rerun to get outlines"]):
            break
    for bad in ["There were undefined","multiply defined","Overfull \\hbox","Overfull \\vbox","Font shape ","destination with the same identifier"]:
        if bad in log:
            raise RuntimeError(f"{key}: inspect LaTeX log for {bad!r}")
    pdf=work/(key+".pdf")
    if not pdf.is_file() or pdf.stat().st_size<50000:
        raise RuntimeError("Incomplete PDF")
    return pdf

def build(target):
    target=target.resolve()
    target.mkdir(parents=True,exist_ok=True)
    archive=target/(NAME+"_arXiv.zip")
    files=list(members())
    with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:
            z.write(f,f.relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix="checkpoint-publication-") as tmp:
        base=Path(tmp)
        clean=base/"source"
        with ZipFile(archive) as z:
            if z.testzip(): raise ValueError("ZIP CRC failure")
            z.extractall(clean)
        editions={"Paper":"","Math_Supplement":r"\def\SupplementCopy{1}","Anonymous_Review":r"\def\ReviewCopy{1}"}
        for suffix,flags in editions.items():
            key=NAME+"_"+suffix
            print("Compiling",key,flush=True)
            pdf=compile_edition(clean,base/suffix,key,flags)
            shutil.copy(pdf,target/(key+".pdf"))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive]+[target/(NAME+"_"+s+".pdf") for s in editions]}
    print(json.dumps(manifest,indent=2))
    return archive

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    build(parser.parse_args().output)

