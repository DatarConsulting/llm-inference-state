"""Extract canonical samples from the archived evidence ZIP without executing its code."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

LAYERS = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
TRACES = ["atelier", "coast", "ledger"]
TOKENS = [1,428,460]
HEAD_BYTES = 65536

def digest(raw): return hashlib.sha256(raw).hexdigest()

def extract(archive, data_dir, full=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with ZipFile(archive) as source, (data_dir/"heads.u32").open("wb") as output:
        for trace_no, trace in enumerate(TRACES):
            for token in TOKENS:
                for layer_no, layer in enumerate(LAYERS):
                    member = f"qwen_test/lossless_full_state/runs/{trace}/native/states/token{token:04d}/layer{layer:02d}.f32"
                    raw = source.read(member)
                    if len(raw) != 16*HEAD_BYTES: raise ValueError(member)
                    source_hash = digest(raw)
                    for head in (range(16) if full else [(layer_no+5*trace_no)%16]):
                        block = raw[head*HEAD_BYTES:(head+1)*HEAD_BYTES]
                        output.write(block)
                        entries.append(dict(id=len(entries),trace=trace,token=token,layer=layer,head=head,member=member,source_sha256=source_hash,head_sha256=digest(block)))
    with (data_dir/"heads.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=entries[0])
        w.writeheader()
        w.writerows(entries)

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive"); p.add_argument("directory")
    p.add_argument("--full",action="store_true")
    a=p.parse_args();extract(a.archive,a.directory,a.full)
