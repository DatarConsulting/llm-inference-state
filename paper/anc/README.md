# Reproducing Draga LLM

**What Must an Inference State Remember?**  
Sharp Memory Bounds for Softmax Attention and Exact Recurrent Checkpoints

All code for this study will be released at [DatarConsulting/llm-inference-state](https://github.com/DatarConsulting/llm-inference-state).

The public preprint uses the NeurIPS 2026 preprint style. The anonymous PDF is a separate edition. No conference acceptance is asserted. All statistical figures use the released measurements and all mathematical curves use the equations specified in the manuscript.

## Compile the source archive

Extract the ZIP into an empty directory. Run pdfLaTeX on `main.tex` three times from its directory. The figures are vector PDFs and the bibliography is plain LaTeX. The archive has a single top-level TeX document and a compiler directive in `00README.json`. No shell escape or Python is needed by arXiv. The proof supplement is included in the article and is also distributed separately for reading.

## Recompute on a CPU

From the submission root, install Python 3.10 or later, NumPy, SciPy, Matplotlib, zstandard 0.25.0, a C++17 compiler, and TeX Live with the recommended fonts and latex-extra packages. Then run

```sh
python anc/reproduce.py all ../replay
python anc/build.py ../publication
```

The first command recreates **every figure used in the paper** and the data-derived LaTeX table fragments. It replays 162 raw heads (one of sixteen per trace, position and layer), verifies their SHA-256 and frame sizes, and uses the full 2,592-head census for publication totals. It also checks 244 BF16 key rows and value budgets, the 24 GPU timing pairs, the exact five-token witness, harmonic examples, and native integer-codec oracles. The build creates the source ZIP and compiles three PDFs from a fresh extraction. The English notebook runs the same steps and downloads the uploadable ZIP from Colab.

## Full recurrent corpus

The separate original input is `Bitwise_State_Full_Data.zip` (about 161 MB, decimal). It contains `heads.csv` and `heads.u32`. The uncompressed word stream has SHA-256 `76a24c1dc52756cd645aee01562dfd9a5293566e5d068f067de713cac2d2ba16`. To recompute all four reference policies, the external compressor controls and the new mixed-head stream on every word, use

```sh
python anc/reproduce.py words ../full-replay --full-data /path/to/Bitwise_State_Full_Data.zip
```

The reference byte counts use python-zstandard 0.25.0 with libzstd 1.5.7 and zlib 1.3.2. If a local zlib build changes a byte-shuffle/DEFLATE control, the replay reports its difference count separately. All native fields and the pinned Zstandard controls are checked exactly. The plots continue to use the released census. Original timings are reported as archived measurements. CPU replay time depends on the machine.

## Inputs, scripts and outputs

| Directory or file | Role |
|---|---|
| `sphere/calculations.py`, `checkpoint.py` | Finite dimension certificates, numerical checks, two-probe and Fourier shell plots |
| `sphere/lean` | Previously checked Lean sources and transcript for selected probe identities; the bit and dimension asymptotics have written proofs |
| `kv/data/qwen_generation_256.json` | Original recorded BF16 experiment across two books and three contexts |
| `kv/experiment`, `kv/tools`, `kv/GPU_PROTOCOL.md` | GPU kernels, optional independent hardware replication and reproducible trace analysis |
| `words/data` | Full census, portable raw words, selection indices, timing records, environment and input hashes |
| `words/repro` | Lossless codec, exact interval optimization, alternative frames and readback of the mixed byte stream |
| `inputs.sha256` | Hashes of bundled input data, Lean sources and archived transcript |
| `verification.json` | Results of the complete CPU recurrent replay used for this edition |

The CPU runs leave derived CSVs, JSON checks, native binaries and the mixed stream in the replay directory. The arXiv source includes the release inputs, scripts and typeset figures. It omits transient build files, model weights, books and the full raw-word companion. GPU results rely on the original Blackwell run. The questions of repeated online quantization of the harmonic checkpoint and combined live deployment of both codecs are open.

## Attribution

The model and runtime are identified in the paper and the GPU protocol. Mathematical and experimental references are collected in `references.tex`. The style file is the 2026 conference template. Generative AI assisted the derivation, integration, source writing and reproduction code. The main theorems have written mathematical proofs; the finite checks are executable examples.

