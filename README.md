# Draga LLM

**What Must an Inference State Remember?**  
Sharp Memory Bounds for Softmax Attention and Exact Recurrent Checkpoints

This repository contains the publication notebook, LaTeX source, Lean project, and reproduction code for Draga LLM. The study covers three settings: the information needed to resume softmax attention, the projected error of an exact-key KV cache, and lossless compression of recurrent FP32 checkpoints.

## Reproduce the article

Open the [publication notebook](notebooks/Draga_LLM_Colab.ipynb) in Google Colab and choose **Runtime → Run all** on a CPU runtime. The notebook regenerates eight figures and builds the public article, standalone mathematical supplement, anonymous review PDF, and the uploadable `Draga_LLM_arXiv.zip`.

For a local run, start in [paper](paper/) and follow the [environment and data instructions](paper/anc/README.md).

```sh
cd paper
python anc/reproduce.py all ../replay
python anc/build.py ../publication
```

The default run includes raw words for 162 heads, observations for all 2,592 heads, and the recorded GPU trace. The optional all-head raw replay takes a separate input archive described in the reproduction instructions. CPU analysis reconstructs published timing statistics from the recorded GPU session.

## Check the Lean proofs

The [Lean project](paper/anc/sphere/lean/) uses Lean and mathlib v4.19.0. With elan installed, run the following commands from the repository root.

```sh
cd paper/anc/sphere/lean
lake update
lake exe cache get
lake build
```

[TwoProbes.lean](paper/anc/sphere/lean/DRAGA/TwoProbes.lean) formalizes the two-probe identification argument, the two-sign energy identity, finite distinguishing examples and consequences for exact states. The [archived checker transcript](paper/anc/sphere/lean/evidence/lean_check.txt) identifies the checked declarations, axioms and source digests. The sharp dimension and checkpoint-bit bounds have written proofs in [dimension.tex](paper/dimension.tex) and [checkpoint.tex](paper/checkpoint.tex).

## Technical material

| Material | Entry point |
|---|---|
| Public manuscript and appendix | [main.tex](paper/main.tex) |
| Harmonic calculations and finite checks | [Sphere calculations](paper/anc/sphere/calculations.py) and [checkpoint calculations](paper/anc/checkpoint.py) |
| KV implementation, CUDA/Triton experiment and recorded trace | [GPU protocol and analysis](paper/anc/kv/GPU_PROTOCOL.md) |
| Recurrent codec, reference search and mixed-frame readback | [Python and C++ sources](paper/anc/words/repro/) |
| Complete offline validation totals | [Verification record](paper/anc/verification.json) |
