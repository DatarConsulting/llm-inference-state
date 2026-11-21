"""Encode, serialize and independently decode the cheaper of two head frames.

Wire layout: one selector byte, then the existing length-prefixed local or
bit-shuffle/Zstandard frame. A tie goes to bit shuffle. Trusted research inputs.
"""
import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import zstandard as zstd

SIZE = 65536
CAPACITY = 4 + 1 + 128*513
HERE = Path(__file__).resolve().parent

def read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Truncated frame")
    return data

def run(data, output, census):
    if sys.byteorder != "little":
        raise RuntimeError("The native replay needs a little-endian host.")
    output.mkdir(parents=True,exist_ok=True)
    libfile = (output/"local_frame.so").resolve()
    subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(HERE/"local_frame.cpp"),"-o",str(libfile)],check=True)
    lib = ctypes.CDLL(str(libfile))
    lib.write_local.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
    lib.write_local.restype = ctypes.c_size_t
    lib.read_local.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    lib.read_local.restype = ctypes.c_int
    with census.open() as f:
        rows = list(csv.DictReader(f))
    packed = output/"heads.mixed"
    buffer = np.empty(CAPACITY,dtype=np.uint8)
    compressor = zstd.ZstdCompressor(level=3)
    counts = [0,0]
    with (data/"heads.u32").open("rb") as raw, packed.open("wb") as sink:
        for row in rows:
            block = read_exact(raw,SIZE)
            native = np.frombuffer(block,dtype="<u4")
            local_length = lib.write_local(native.ctypes.data,buffer.ctypes.data)
            groups = np.frombuffer(block,np.uint8).reshape(-1,256,4)
            bits = np.unpackbits(groups,axis=2,bitorder="little").transpose(0,2,1)
            transformed = np.packbits(bits,axis=2,bitorder="little").tobytes()
            compressed = compressor.compress(transformed)
            body = bytes([3])+compressed if len(compressed)<SIZE else bytes([0])+block
            shuffled = len(body).to_bytes(4,"little")+body
            assert local_length == int(row["opt"]) and len(shuffled) == int(row["bitshuffle"])
            selector = int(local_length >= len(shuffled))
            counts[selector]+=1
            sink.write(bytes([selector]))
            sink.write(shuffled if selector else buffer[:local_length].tobytes())
        assert raw.read(1) == b""
    decoder = zstd.ZstdDecompressor()
    with packed.open("rb") as source, (data/"heads.u32").open("rb") as original:
        for row in rows:
            tag = read_exact(source,1)[0]
            prefix = read_exact(source,4)
            payload = read_exact(source,int.from_bytes(prefix,"little"))
            if tag==0:
                frame=np.frombuffer(prefix+payload,np.uint8)
                restored=np.empty(SIZE//4,dtype="<u4")
                if lib.read_local(frame.ctypes.data,len(frame),restored.ctypes.data):
                    raise ValueError("Invalid local head frame")
                block=restored.tobytes()
            elif tag==1 and payload[0]==3:
                planes=np.frombuffer(decoder.decompress(payload[1:]),np.uint8).reshape(-1,32,32)
                bits=np.unpackbits(planes,axis=2,bitorder="little").transpose(0,2,1)
                block=np.packbits(bits,axis=2,bitorder="little").tobytes()
            elif tag==1 and payload[0]==0:
                block=payload[1:]
            else:
                raise ValueError("Unknown head tag")
            assert block==read_exact(original,SIZE) and hashlib.sha256(block).hexdigest()==row["head_sha256"]
        assert source.read(1)==original.read(1)==b""
    expected=sum(1+min(int(r["opt"]),int(r["bitshuffle"])) for r in rows)
    assert packed.stat().st_size==expected
    result=dict(heads=len(rows),local_heads=counts[0],shuffle_heads=counts[1],encoded_bytes=expected)
    print(json.dumps(result))
    (output/"mixed_result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["data","output","census"]:
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    run(args.data,args.output,args.census)

