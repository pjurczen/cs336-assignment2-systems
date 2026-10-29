> **My work on this fork (self-study, 2025–26).** Sections 2 (tokenizer) and 3 (model) are complete; the training loop (
> §4–5) is in progress.
>
> - **BPE tokenizer from scratch** with parallel, `mmap`-based pretokenization across all CPU cores and a bucketed pair
    counter that replaced the naive rescan — **merge loop 66 s → 5.2 s (12.7×)** on TinyStories; a 10K vocabulary trains
    in 34.8 s at ~1.5 GB peak RSS.
> - **32K-vocabulary BPE trained on the full 11.9 GB OpenWebText corpus** in 61 min at ~3.6 GB peak; both corpora
    tokenized to `uint16` id arrays. Measured compression 4.0 bytes/token (TinyStories) vs 4.5 (OpenWebText), and a 31%
    degradation from cross-domain tokenizer mismatch.
> - **Transformer components in PyTorch without `torch.nn` layers:** `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`,
    softmax, scaled dot-product attention, RoPE, causal multi-head attention, the transformer block and the full
    `TransformerLM`, with FLOPs accounting annotated inline.
> - ~35 tests of my own (`tests/test_pretoken_counter.py`, `test_pretokenization.py`, `test_bpe_tokenizer.py`) on top of
    the course suite; 80 of 87 tests pass, the 7 open ones are the §4–5 stubs.
> - Written answers with measurements: [`assignment_1_answers.md`](./assignment_1_answers.md).
>
> The original assignment README follows.

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/
`brew install uv`.
We recommend reading a bit about managing projects in
`uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

