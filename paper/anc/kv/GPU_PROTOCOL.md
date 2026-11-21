# Exact Keys, Bounded Values, and the Cost of a Smaller KV Cache

This companion reproduces the figures and numerical checks for the generation-256-v1 paper. All reported GPU measurements come from data/qwen_generation_256.json. Its SHA-256 is baeb7838f2c21ab7f7f17741bac85c459cf91960ebf21411ec62fa0b6f9beb1f. A CPU is sufficient for the reference analysis. The independent GPU experiment is optional.

## CPU replay

From this directory, use Python 3.10 or later with NumPy and Matplotlib installed. Run the following commands.

    python tools/math_checks.py
    python tools/codec_checks.py
    python tools/analyze.py data/qwen_generation_256.json output

The final command creates three plots in vector PDF and preview PNG formats, two LaTeX table fragments, three timing/condition CSVs, the fallback-count CSV and the quality summary. The input digest locks the published analysis to the reported trace. The theoretical checks use exact fractions for the box support and multihead cancellation examples. The codec checks use the numerical format of the original CPU encoder. Use --fresh only to analyze a separate complete six-condition trace, with a new output directory.

## Optional GPU replication

The recorded GPU was an NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120. The recorded binary stack was torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0, flashinfer-python 0.7.5.post1, transformers 5.5.0. Use a GPU runtime with compatible CUDA drivers, a compiler/toolkit for FlashInfer and Triton JIT, and enough device and host memory for 131072-token prefill. Exact peak allocation was not captured. CPU replay does not install these dependencies. In a suitable runtime with the required torch and Triton already available, install compatible NumPy and the remaining packages, then run

    python -m pip install transformers==5.5.0 flashinfer-python==0.7.5.post1 ninja
    python tools/run_gpu.py independent_run
    python tools/analyze.py independent_run/qwen_generation_256.json independent_analysis --fresh

The runner uses a new directory, downloads the pinned Hugging Face revision and verifies both Gutenberg corpus-file hashes. It performs a codec check, compares four positions with the native model, builds a dense and a compact cache, and runs eight common-history probes, two warm-ups and four pairs per condition. Local model/corpus access and a working Internet connection for initial downloads may be needed. Each full GPU trace has its own hardware record. The Colab retains the fixed trace for the published text even if a new device or software version gives a different outcome.

## Sources and attribution

- Qwen Team, Qwen3.5-0.8B (Apache-2.0), revision 2fc06364715b967f1860aea9cf38778875588b17. https://huggingface.co/Qwen/Qwen3.5-0.8B
- Austen, Emma; Melville, Moby-Dick. NLTK Gutenberg corpus, classified as public domain by NLTK. Upstream details: https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md
- FlashInfer: https://github.com/flashinfer-ai/flashinfer ; Transformers: https://github.com/huggingface/transformers ; Triton: https://github.com/triton-lang/triton ; PyTorch: https://github.com/pytorch/pytorch
- Official template: https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
- The companion includes locally written scripts, the recorded experiment and its short generated excerpts. It retrieves upstream model weights and texts separately.

