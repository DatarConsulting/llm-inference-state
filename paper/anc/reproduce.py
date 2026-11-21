"""Run CPU evidence reconstruction for the unified manuscript.

Run from any directory. All transient output goes to the supplied working
directory. Figure PDFs and typeset table fragments are written under figures/.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ANC = Path(__file__).resolve().parent
ROOT = ANC.parent
DATA = ANC/"words/data"
REPRO = ANC/"words/repro"
FIG = ROOT/"figures"

def command(*args):
    subprocess.run([str(x) for x in args],check=True)

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(f.read()).hexdigest()

def theory(work):
    command(sys.executable, ANC/"sphere/calculations.py", "--figures", work/"sphere", "--evidence",work/"sphere/evidence")
    shutil.copy(work/"sphere/dimension.pdf", FIG/"dimension.pdf")
    command(sys.executable, ANC/"checkpoint.py", FIG, work/"entropy")
    command(sys.executable, ANC/"kv/tools/math_checks.py")
    command(sys.executable, REPRO/"math_checks.py")
    command("g++","-std=c++17","-O3",REPRO/"edge_cases.cpp","-o",work/"edge_cases")
    command(work/"edge_cases")

def gpu_record(work):
    command(sys.executable, ANC/"kv/tools/codec_checks.py")
    command(sys.executable, ANC/"kv/tools/analyze.py", ANC/"kv/data/qwen_generation_256.json",work/"kv")
    for name in ["paired.pdf","budget.pdf","fallback.pdf","pooled.tex","timing.tex"]:
        shutil.copy(work/"kv"/name,FIG/name)

def words(work, raw_zip=None):
    folder=work/("full" if raw_zip else "sample")
    folder.mkdir(parents=True,exist_ok=True)
    versions=json.loads((DATA/"versions.json").read_text())
    if raw_zip:
        with ZipFile(raw_zip) as z:
            for name in ["heads.csv","heads.u32"]:
                with z.open(name) as src,(folder/name).open("wb") as dst:
                    shutil.copyfileobj(src,dst)
        expected=versions["raw_full_sha256"]
        reference=DATA/"full_summary.csv"
    else:
        with gzip.open(DATA/"heads.u32.gz","rb") as src,(folder/"heads.u32").open("wb") as dst:
            shutil.copyfileobj(src,dst)
        shutil.copy(DATA/"heads.csv",folder/"heads.csv")
        expected=versions["raw_sample_sha256"]
        reference=DATA/"sample_summary.csv"
    if digest(folder/"heads.u32")!=expected:
        raise ValueError("Unexpected raw corpus SHA-256")
    command(sys.executable, REPRO/"replay.py", folder,folder/"audit")
    # The row layout and libzstd are pinned; report zlib-version drift separately.
    def records(file):
        with file.open() as f:
            return list(csv.DictReader(f))
    expected_rows, actual_rows = records(reference), records(folder/"audit/heads_summary.csv")
    if len(expected_rows) != len(actual_rows):
        raise ValueError("Census row count changed")
    zlib_differences = 0
    for ref, obs in zip(expected_rows,actual_rows):
        if ref.keys() != obs.keys() or any(ref[k] != obs[k] for k in ref if k != "deflate"):
            raise ValueError(f"Census differs at head {ref['id']}. Check the data, codec and pinned libzstd.")
        zlib_differences += int(ref["deflate"] != obs["deflate"])
    print("Heads whose independent DEFLATE control changed on this zlib build:",zlib_differences)
    (folder/"external_compressor_check.json").write_text(json.dumps({"deflate_differences":zlib_differences},indent=2)+"\n")
    command(sys.executable,REPRO/"mixed.py",folder,folder/"mixed",reference)
    command(sys.executable,REPRO/"figures.py")
    rows=records(DATA/"full_summary.csv")
    total=lambda key:sum(int(r[key]) for r in rows)
    stats=dict(heads=len(rows),native=total("raw"),local=total("opt"),shuffle=total("bitshuffle"),
               candidate_saved=total("mid_res")-total("opt_res"),complete_saved=total("mid")-total("opt"),
               fixed_floor=total("free_groups"),mixed=sum(1+min(int(r["opt"]),int(r["bitshuffle"])) for r in rows))
    assert list(stats.values())==[2592,169869312,148940720,142093897,2045888,106994,148335505,140735906]
    (work/"census.json").write_text(json.dumps(stats,indent=2)+"\n")
    print("Verified raw words, frame sizes and table totals.",stats)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["theory","gpu-record","words","all"])
    parser.add_argument("workspace",type=Path)
    parser.add_argument("--full-data",type=Path,help="Optional original Bitwise_State_Full_Data.zip")
    args=parser.parse_args()
    path=args.workspace.resolve()
    path.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(exist_ok=True)
    if args.stage in ["all","theory"]: theory(path)
    if args.stage in ["all","gpu-record"]: gpu_record(path)
    if args.stage in ["all","words"]: words(path,args.full_data)

