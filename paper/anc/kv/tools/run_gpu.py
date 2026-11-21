"""Run an independent, full generation-256-v1 replication on a compatible GPU."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path,help="A new directory for the full run and its checkpointed JSON.")
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA GPU runtime. CPU replay is available in analyze.py.")
    print("Device:",torch.cuda.get_device_name(),torch.cuda.get_device_capability(),flush=True)
    print("Published reference: SM120, CUDA 13.0, torch 2.14.1+cu130, Triton 3.8.0.",flush=True)
    # Use a fresh directory. The benchmark emits intermediate JSON checkpoints.
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parents[1]/"experiment"
    for f in src.glob("*.py"):
        shutil.copyfile(f,out/f.name)
    settings=dict(lengths=[4096,32768,131072],steps=64,pairs=4,paths=[],teacher=8,offset=4096)
    (out/"settings.json").write_text(json.dumps(settings))
    subprocess.run([sys.executable,"-u",str(out/"bench.py")],cwd=out,check=True)
    print("Fresh trace:",out/"qwen_generation_256.json")
if __name__=="__main__":
    main()
