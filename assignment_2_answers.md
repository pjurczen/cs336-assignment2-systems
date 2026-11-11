# CS336 Assignment 2 — Written Answers

## 2.1.3 Problem (benchmarking_script): Benchmarking Script

**(a)** Script in cs336_systems/benchmark.py

**(b)**

| B200 (fp32, TF32 matmuls), batch 4, context 512 | small      | medium     | large      | xl           | 10B |
|-------------------------------------------------|------------|------------|------------|--------------|-----|
| forward (ms)                                    | 14.9 ± 0.4 | 25.0 ± 2.4 | 34.2 ± 0.7 | 52.4 ± 0.4   | OOM |
| backward (ms)                                   | 21.6 ± 0.8 | 37.7 ± 2.7 | 67.2 ± 0.3 | 125.5 ± 1.6  | OOM |
| optimizer step (ms)                             | 10.0 ± 0.4 | 17.1 ± 1.3 | 24.9 ± 0.5 | 70.35 ± 0.04 | OOM |
| backward / forward                              | 1.45×      | 1.51×      | 1.96×      | 2.39×        | –   |

Backward pass is 1.5-2.4x forward pass depending on the model size, which is expected, low variability in both forward pass and optimizer step, significant variability in backward pass.

**(c)**

I ran experiments on small and xl models:

| small: warmup steps | forward (ms)   | backward (ms) | optimizer step (ms) |
|---------------------|----------------|---------------|---------------------|
| 0                   | 60.08 ± 121.45 | 32.05 ± 31.63 | 9.84 ± 1.25         |
| 1                   | 17.58 ± 5.81   | 28.40 ± 9.72  | 11.43 ± 3.58        |
| 2                   | 12.60 ± 0.14   | 19.60 ± 0.44  | 8.87 ± 0.06         |
| 5 (part b)          | 14.89 ± 0.42   | 21.62 ± 0.80  | 9.95 ± 0.38         |

| xl: warmup steps | forward (ms)    | backward (ms)  | optimizer step (ms) |
|------------------|-----------------|----------------|---------------------|
| 0                | 103.90 ± 118.73 | 135.13 ± 16.19 | 73.91 ± 11.80       |
| 1                | 65.98 ± 43.09   | 129.85 ± 14.53 | 70.13 ± 0.11        |
| 2                | 53.53 ± 0.46    | 126.70 ± 1.03  | 70.57 ± 0.06        |
| 5 (part b)       | 52.39 ± 0.40    | 125.45 ± 1.58  | 70.35 ± 0.04        |

| no warmup (ms)       | step 1 | step 2 | steps 3–10 (median) |
|----------------------|--------|--------|---------------------|
| small forward        | 399.6  | 87.0   | 14.1                |
| small backward       | 121.9  | 24.9   | 20.9                |
| small optimizer step | 12.9   | 9.4    | 9.1                 |
| xl forward           | 411.4  | 208.9  | 52.4                |
| xl backward          | 161.7  | 125.7  | 125.3               |
| xl optimizer step    | 107.5  | 70.1   | 70.1                |

We can observe that with no warmup the times stabilize only from 3rd step onward and at the same time that with 2 and 5 warmup iters the standard deviation and mean were very similar.

This is most likely caused by two things:

1. How CUDA loads kernels into the GPU lazily on first use and initializes cuBLAS on first matmul
2. How pytorch expands reserved memory on the GPU, only if its already reserved (and freed) pool won't fit the data:

- Initially it only allocates memory for weights (W),
- On the first forward pass it allocates memory for activations (A),
- Then on the backward pass it allocates memory for gradients (G) and at the same time frees memory for activations, but it expanded reserved memory pool to W + A + G
- Then on optimizer pass it allocates memory for momentum state (M = 2G) => W + A + G + 2G - A = W + 3 G, because it only had to grow memory by some portion 2G - A, that wasn't reserved yet
- Then after the optimizer pass the gradients are freed (but kept in the reserved pool by pytorch), so the used memory stays as W + 2G, while reserved memory is W + 3G

This means, that in the first pass, it only had to grow the memory to W + 3G.

Then during second pass, it needs to allocate memory for activations again, so it grows to W + 2G + A, (A is typically bigger than G in our case - depends on input size, so W + 2G + A > W + 3G - A).
That's when the memory pool stabilizes and it expanded to the maximum, which is W + 2G + A. In the next steps there will be no new memory allocation needed, so that's why the steps are quicker from
this point on.

## 2.1.4 Problem (nsys_profile): Nsight Systems Profiling

**(a)**

| model, context (batch 4, fp32/TF32) | GPU       | `timeit` forward (ms) | NVTX `forward` range, CPU (ms) | NVTX `forward` range, GPU projection (ms) | `timeit` backward (ms) | `timeit` optimizer step (ms) |
|-------------------------------------|-----------|-----------------------|--------------------------------|-------------------------------------------|------------------------|------------------------------|
| small, 512                          | A100 SXM4 | 30.13 ± 1.36          | 30.11                          | 29.91                                     | 49.22 ± 2.90           | 22.56 ± 1.09                 |
| small, 1024                         | A100 SXM4 | 47.26 ± 0.35          | 47.24                          | 46.99                                     | 100.56 ± 0.33          | 26.87 ± 0.70                 |
| small, 2048                         | A100 SXM4 | 144.97 ± 0.29         | 144.93                         | 144.77                                    | 301.96 ± 0.23          | 22.54 ± 0.49                 |
| large, 256                          | A100 SXM4 | 89.31 ± 4.88          | 89.29                          | 89.12                                     | 139.30 ± 4.30          | 85.08 ± 0.10                 |
| large, 512                          | A100 PCIe | 101.00 ± 2.42         | 100.97                         | 100.80                                    | 227.31 ± 0.87          | 89.23 ± 0.45                 |
| large, 1024                         | A100 SXM4 | 243.41 ± 0.29         | 243.28                         | 242.98                                    | 528.08 ± 0.74          | 85.69 ± 0.39                 |

Our forward pass measured both by timeit and NVTX is almost the same, under 0.5% difference. Total time spent for 10 iterations was around 301ms on small model 512 context length.

**(b)**

During the small models 512 context length forward pass following kernel takes most time:

| Time  | Total Time | Instances | Avg       | Med       | Min       | Max       | StdDev    | Name                                                                                      |
|-------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-------------------------------------------------------------------------------------------|
| 18.8% | 3.012 ms   | 36        | 83.680 μs | 90.848 μs | 68.672 μs | 91.520 μs | 10.472 μs | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)` |

It was invoked 36 times in a single forward pass.

For both forward and backward passes together the kernel that takes most time is "void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::
binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)"

**(c)**

vectorized_elementwise_kernel and elementwise_kernel which are PyTorch's generic element-wise kernels, not matrix multiplications, like: add, multiply, divide, exp, copy and so on.
They are memory bound, not compute, that's why they take significant amount of time even though contributing almost no FLOPs.

**(d)**

| small, context 512 | matmul | element-wise | reduction | copy | other |
|--------------------|--------|--------------|-----------|------|-------|
| forward only       | 36.3%  | 49.6%        | 8.7%      | 4.1% | 1.3%  |
| full training step | 28.4%  | 60.3%        | 5.2%      | 5.7% | 0.4%  |

When we compare forward vs. full training step, then we can notice that matmul fraction decreases from 36.3% to 28.4%, while element-wise fraction increases from 49.6% to 60.3%. Reduction decreases
from 8.7% to 5.2% and copy increases from 4.1% to 5.7%.

**(e)**

| small, context 512, one attention call | FLOPs       | time     | achieved throughput |
|----------------------------------------|-------------|----------|---------------------|
| matmuls (QKᵀ + ·V)                     | 3.22 GFLOP  | 99.3 µs  | ~32 TFLOP/s         |
| softmax                                | 0.063 GFLOP | 364.1 µs | ~0.17 TFLOP/s       |

Softmax contributes ~50x less FLOPs but takes ~3.7x more time. What's interesting is also that in QKᵀ / √dₖ the element-wise division takes more time (54.6%) than the matmul itself (45.4%).

## 2.1.5 Problem (mixed_precision_accumulation): Mixed-Precision Accumulation

So first thing to notice is, by adding from 0.01 1000 times, we cross multiple float exponents, resulting in different spacing.
This results in different accumulating error depending on the accumulator and the addend.

The results are as following:

accumulator=fp32, addend=fp32:                  tensor(10.0001)
Spacing eg. in 8-16 range for fp32 is 0.00000095, adding 0.01 (0.0099999998 in fp32) produces error, since accumulator is of same precision,
these small rounding error get dragged through different exponent ranges, so it accumulates and we receive 10.0001.

accumulator=fp16, addend=fp16:                  tensor(9.9531, dtype=torch.float16)
Spacing eg. in 8-16 range for fp16 is 0.0078125, adding 0.01 (0.0100021 in fp16) 250 times (we reach ~8 after 750 additions already because of 0.039 spacing in [4, 8) range of fp16,
resulting in rounding of 0.01 to 0.0117) accumulates error of −0.547, resulting in total error of −0.049. Previous coarser ranges in lower exponents made the value higher than expected,
the rounding error was positive of +0.498 and went to a negative one when arriving at 1000 additions.

accumulator=fp32, addend=fp16:                  tensor(10.0021)
In this case we're adding each time 0.0100021 (fp16 representation of 0.01) 1000 times, but fp32 is capable of storing finer grained values, so this error simply accumulates 1000 times and we land at
10.0021.
Pytorch is upcasting the value 0.0100021 to fp32 for us when we're doing the addition.

accumulator=fp32, addend=fp16->upcast fp32:     tensor(10.0021)
Here it does not matter if we upcast manually or not, pytorch would upcast automatically based on the accumulator type, we already created 0.0100021 as our representation of 0.01 in fp16,
and we're storing it in higher precision fp32, also resulting in same error accumulating 1000 times.

## 2.1.5 Problem (benchmarking_mixed_precision): Benchmarking Mixed Precision

**(a)**

1. fp32, model parameters are never autocasted
2. fp16, linear layer output can be autocasted to fp16
3. fp32, layer_norm can only be autocasted to fp32
4. fp16, linear layer output can be autocasted fo tp16
5. fp32, cross_entropy can only be autocasted to fp32
6. fp32, grad has always same dtype as parameters

**(b)**

So LayerNorm is:
mean = (1/d) · Σ xᵢ
var = (1/d) · Σ (xᵢ − mean)²
y = (x − mean) / √(var + ε) · γ + β,

mean:

- adding small values accumulates to a significant rounding error over many elements with fp16 precision - precision problem
- sum can overflow before dividing

var

- subtracting mean from every element for large values results in cancellation - precision problem
- sum can overflow
- squaring it can result in value exceeding range of fp16 - overflow

y

- subtracting mean can result in cancellation - precision problem
- adding epsilon can result in value not changing at all (absorption) - precision problem
- dividing can result in overflow if the denominator is small

With BF16 the range issues (overflow/underflow) are solved, since its same range as fp32, but the precision problems are even bigger, so to fix those we would need to upcast those operations to fp32
(the accumulator needs to be in fp32 like in our experiment in mixed_precision_accumulation problem).

**(c)**

| model (ctx 512, batch 4) | GPU       | forward fp32 (ms) | forward bf16 (ms) | speedup | backward fp32 (ms) | backward bf16 (ms) | speedup |
|--------------------------|-----------|-------------------|-------------------|---------|--------------------|--------------------|---------|
| small                    | A100 SXM4 | 22.97 ± 0.19      | 24.84 ± 0.47      | 0.92×   | 40.02 ± 0.34       | 35.89 ± 2.19       | 1.12×   |
| medium                   | A100 SXM4 | 49.32 ± 1.14      | 45.95 ± 0.29      | 1.07×   | 110.55 ± 0.91      | 92.85 ± 0.74       | 1.19×   |
| large                    | A100 SXM4 | 92.42 ± 0.11      | 101.22 ± 5.37     | 0.91×   | 212.39 ± 0.24      | 170.58 ± 0.48      | 1.25×   |
| xl                       | A100 SXM4 | 198.24 ± 2.66     | 156.26 ± 0.21     | 1.27×   | 466.42 ± 4.35      | 339.73 ± 1.06      | 1.37×   |

We can see a steady speedup in backward pass. Forward pass speedup is not so steady and is varying depending on the model size, that is likely caused by overhead added by the extra kernels from
autocast
while only giving speedups in minority of time that matmuls take in the forward pass itself.

## 2.1.6 Problem (memory_profiling): Memory Profiling

**(a)**

For 128 context length the forward pass memory is not very interesting and mostly low and flat, only barely moving above weights memory.

Training run memory of 128 context length looks as on the image:

![training_run_memory](training_run_memory.png)

We can see clear phases of forward pass, then backward pass, then optimizer step and then zero grad releasing memory, repeated twice since I ran 2 iterations.

For 2048 context length we can see big memory peaks from the attention blocks since its memory depends on squared context length:

![forward_pass_memory_2048](forward_pass_memory_2048.png)

Training run with 2048 context crashes with OutOfMemoryError during forward pass, since activations with such big context length exceed our maximum memory.

**(b)**

| Context length | Forward pass peak memory (GiB) | Full training peak memory (GiB) |
|----------------|--------------------------------|---------------------------------|
| 128            | 12.8                           | 51.3                            |
| 2048           | 21.6                           | 78.5                            |

I ran my experiments on A100-80GiB, therefore it crashed around 80GiB

**(c)**

| xl, batch 4                      | fp32 peak                          | bf16 (autocast) peak               |
|----------------------------------|------------------------------------|------------------------------------|
| context 128, inference           | 12.79 GiB                          | 19.02 GiB                          |
| context 128, full training step  | 51.29 GiB                          | 51.28 GiB                          |
| context 2048, inference          | 21.58 GiB                          | 25.40 GiB                          |
| context 2048, full training step | OOM in forward, at layer ~10 of 32 | OOM in forward, at layer ~14 of 32 |.

So for inference it increases memory usage, because autocast copies weights into bf16 and keeps them until the block ends.
On short context the memory peak with bf16 stays same as with fp32, because the activations in attention are small,
with long context this changes dramatically and activations inside attention layers cause bigger peaks than the baseline
from weights, gradients and momentums, meaning storing those attention activations in bf16 would reduce the effective peak memory.

**(d)**

Residual stream is a tensor of shape (batch_size, sequence_length, d_model), so it's size for XL model and context length 2048 would be:
4 × 2048 × 2560 × 4 bytes = 83'886'080 bytes = 80 MiB

**(e)**

That's 2GiB on 2048 context length, there are multiple allocations of this size inside the the scaled dot product attention block,
for example the QK matmul, the softmax division, the max subtraction inside softmax and more.

**(f)**

Operations allocating memory:

| t (ms) | MiB  | kind     | op              | module                                     |
|--------|------|----------|-----------------|--------------------------------------------|
| 0.28   | 5.0  | temp     | `aten::pow`     | `TransformerLM.layers.10.ln1`              |
| 0.79   | 5.0  | RESIDUAL | `aten::mul`     | `TransformerLM.layers.10.ln1`              |
| 1.02   | 5.0  | RESIDUAL | `aten::div`     | `TransformerLM.layers.10.ln1`              |
| 1.74   | 15.0 | temp     | `aten::bmm`     | `TransformerLM.layers.10.attn.qkv_proj`    |
| 2.45   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 2.65   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 2.83   | 2.5  | temp     | `aten::sub`     | `TransformerLM.layers.10.attn.rope`        |
| 3.29   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 3.48   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 3.67   | 2.5  | temp     | `aten::add`     | `TransformerLM.layers.10.attn.rope`        |
| 4.17   | 5.0  | RESIDUAL | `aten::cat`     | `TransformerLM.layers.10.attn.rope`        |
| 5.0    | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 5.2    | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 5.4    | 2.5  | temp     | `aten::sub`     | `TransformerLM.layers.10.attn.rope`        |
| 5.81   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 6.01   | 2.5  | temp     | `aten::mul`     | `TransformerLM.layers.10.attn.rope`        |
| 6.2    | 2.5  | temp     | `aten::add`     | `TransformerLM.layers.10.attn.rope`        |
| 6.64   | 5.0  | RESIDUAL | `aten::cat`     | `TransformerLM.layers.10.attn.rope`        |
| 7.39   | 8.0  | temp     | `aten::bmm`     | `TransformerLM.layers.10.attn`             |
| 7.7    | 8.0  | temp     | `aten::div`     | `TransformerLM.layers.10.attn`             |
| 8.17   | 8.0  | temp     | `aten::empty`   | `TransformerLM.layers.10.attn`             |
| 8.66   | 8.0  | temp     | `aten::sub`     | `TransformerLM.layers.10.attn`             |
| 8.86   | 8.0  | RESIDUAL | `aten::exp`     | `TransformerLM.layers.10.attn`             |
| 9.15   | 8.0  | RESIDUAL | `aten::div`     | `TransformerLM.layers.10.attn`             |
| 9.77   | 5.0  | RESIDUAL | `aten::empty`   | `TransformerLM.layers.10.attn`             |
| 10.01  | 5.0  | temp     | `aten::bmm`     | `TransformerLM.layers.10.attn`             |
| 10.35  | 5.0  | RESIDUAL | `aten::empty`   | `TransformerLM.layers.10.attn`             |
| 10.86  | 5.0  | temp     | `aten::bmm`     | `TransformerLM.layers.10.attn.output_proj` |
| 11.61  | 5.0  | RESIDUAL | `aten::add`     | `TransformerLM.layers.10`                  |
| 12.02  | 5.0  | temp     | `aten::pow`     | `TransformerLM.layers.10.ln2`              |
| 12.52  | 5.0  | RESIDUAL | `aten::mul`     | `TransformerLM.layers.10.ln2`              |
| 12.72  | 5.0  | RESIDUAL | `aten::div`     | `TransformerLM.layers.10.ln2`              |
| 13.16  | 20.0 | RESIDUAL | `aten::bmm`     | `TransformerLM.layers.10.ffn.w1`           |
| 13.47  | 20.0 | RESIDUAL | `aten::sigmoid` | `TransformerLM.layers.10.ffn`              |
| 13.68  | 20.0 | RESIDUAL | `aten::mul`     | `TransformerLM.layers.10.ffn`              |
| 14.05  | 20.0 | RESIDUAL | `aten::bmm`     | `TransformerLM.layers.10.ffn.w3`           |
| 14.37  | 20.0 | RESIDUAL | `aten::mul`     | `TransformerLM.layers.10.ffn`              |
| 14.73  | 5.0  | temp     | `aten::bmm`     | `TransformerLM.layers.10.ffn.w2`           |
| 15.07  | 5.0  | RESIDUAL | `aten::add`     | `TransformerLM.layers.10`                  |

Top 5 operations creating residual memory:

| op              | tensors | MiB  | % of block residuals |
|-----------------|---------|------|----------------------|
| `aten::mul`     | 4       | 50.0 | 30.1                 |
| `aten::bmm`     | 2       | 40.0 | 24.1                 |
| `aten::sigmoid` | 1       | 20.0 | 12.0                 |
| `aten::div`     | 3       | 18.0 | 10.8                 |
| `aten::empty`   | 2       | 10.0 | 6.0                  |

Data was exported by querying nsys sqlite files. UI does not support aggregating memory usage.

Memory at start of forward pass (2nd iteration): 38.16 GiB
Memory at end of forward pass: 43.34 GiB

Memory at start of backward pass: 43.38 GiB
Memory at end of backward pass: 50.82 GiB

Memory difference in backward pass for one transformer layer (layer 11): 48.22 - 48.09 = +0.13 GiB

## 3.2.1 Problem (gradient_checkpointing): Memory-Optimal Gradient Checkpointing

**(a)**

The strategy is to recompute residuals of every layer when going down the backward graph by creating recursive checkpoints from layer N-1 to layer 1.
We can save residuals of layer Nth, since we're gonna need it immediately when calling loss.backward() and it doesn't influence peak memory, since we're gonna
be holding residuals of one layer when going down the graph anyway, just one at a time instead of all layers (as it would be without the checkpoints).
This means that asymptotic memory with this solution is O(1) - we're holding residuals of one layer at a time only.
This will cost us O(n²) compute though, since for all the layers from N-1 to 1 we will be essentially doing a forward pass to arrive at that layer's input to be
able to recompute that layer's residuals.

def checkpointed_blocks(layers: int, x: torch.Tensor) -> torch.Tensor:
block = self.layers[layers - 1]
if layers == 1:
return block(x)
h = checkpoint(checkpointed_blocks, layers - 1, x, use_reentrant=False)
return block(h)

**(b)**

peak memory ≈ (N/k) × input size + k × one block's residuals

where k is selected block size for checkpointing, eg. we could create a checkpoint around 4 layers out of 32 for the XL model.

We can measure one block's residual using PyTorch's MemoryViz:

![Ctx 2048 residual memory](assets/ctx2048_residual_memory.png)

which read from that tool is around 4.6 GiB when using bf16.

The input size can be calculated as:
batch size × sequence length × d_model × bytes per element = 4 × 2048 × 2560 × 4 bytes = 83'886'080 bytes
= 80 MiB

plugging this into our formula for peak memory:

peak memory ≈ (32/k) × 0.078125 GiB + k × 4.6 GiB

we can see that we will get minimal value when using low k value.

For k=1 we get:

peak memory ≈ 32 × 0.078125 GiB + 4.6 GiB = 2.5 GiB + 4.6 GiB = 7.1 GiB

For k=2 we get:

peak memory ≈ 32/2 × 0.078125 GiB + 2 × 4.6 GiB = 1.25 GiB + 9.2 GiB = 10.45 GiB

Showing that k=1 minimizes the sum and gives minimal peak memory, meaning we would create one checkpoint around all 32 blocks.

For k=1 the measured peak memory in memory viz during the backward pass is 63.3 GiB

For k=2 the measured peak memory in memory viz during the backward pass is 67.5 GiB

Confirming that k=1 is the best choice for lowest possible peak memory.

## 4.1.1 Problem (pytorch_attention): PyTorch Attention Benchmarking

| d_model | seq_len | forward (ms)  | backward (ms) | memory before backward (MiB) | added by forward (MiB) |
|---------|---------|---------------|---------------|------------------------------|------------------------|
| 16      | 256     | 0.47 ± 0.12   | 0.93 ± 0.15   | 21.40                        | 4.09                   |
| 16      | 1024    | 0.70 ± 0.03   | 1.71 ± 0.14   | 86.34                        | 65.09                  |
| 16      | 4096    | 7.73 ± 0.04   | 17.83 ± 0.15  | 1088.62                      | 1040.38                |
| 16      | 8192    | 28.92 ± 0.10  | 65.03 ± 0.24  | 4273.00                      | 4160.75                |
| 16      | 16384   | 115.22 ± 0.25 | 259.56 ± 0.27 | 16977.75                     | 16641.50               |
| 16      | 32768   | OOM           | OOM           | OOM                          | OOM                    |
| 32      | 256     | 0.43 ± 0.04   | 0.90 ± 0.17   | 22.40                        | 4.09                   |
| 32      | 1024    | 0.70 ± 0.12   | 1.72 ± 0.16   | 90.34                        | 65.09                  |
| 32      | 4096    | 7.94 ± 0.02   | 18.06 ± 0.20  | 1104.62                      | 1040.38                |
| 32      | 8192    | 29.91 ± 0.04  | 66.20 ± 0.19  | 4305.00                      | 4160.75                |
| 32      | 16384   | 119.62 ± 0.29 | 263.74 ± 0.29 | 17041.75                     | 16641.50               |
| 32      | 32768   | OOM           | OOM           | OOM                          | OOM                    |
| 64      | 256     | 0.43 ± 0.03   | 0.88 ± 0.12   | 24.40                        | 4.09                   |
| 64      | 1024    | 0.72 ± 0.03   | 1.74 ± 0.13   | 98.34                        | 65.09                  |
| 64      | 4096    | 8.40 ± 0.03   | 18.54 ± 0.16  | 1136.62                      | 1040.38                |
| 64      | 8192    | 31.93 ± 0.07  | 68.58 ± 0.22  | 4369.00                      | 4160.75                |
| 64      | 16384   | 127.42 ± 0.25 | 271.88 ± 0.36 | 17169.75                     | 16641.50               |
| 64      | 32768   | OOM           | OOM           | OOM                          | OOM                    |
| 128     | 256     | 0.46 ± 0.14   | 0.93 ± 0.21   | 28.40                        | 4.09                   |
| 128     | 1024    | 0.78 ± 0.02   | 1.83 ± 0.16   | 114.34                       | 65.09                  |
| 128     | 4096    | 9.37 ± 0.08   | 19.69 ± 0.17  | 1200.62                      | 1040.38                |
| 128     | 8192    | 35.80 ± 0.13  | 72.89 ± 0.30  | 4497.00                      | 4160.75                |
| 128     | 16384   | 142.97 ± 0.27 | 288.28 ± 0.37 | 17425.75                     | 16641.50               |
| 128     | 32768   | OOM           | OOM           | OOM                          | OOM                    |

I ran out of memory on A100-80GB with d_model=16 and seq_len=32768. For this configuration for a single head activations (stored in fp32) would require:

- 𝑄𝐾⊤ matrix multiply: batch_size × num_heads × context_length²
- softmax: batch_size × num_heads × context_length²
  => 2 × batch_size × context_length² × 4 bytes = 2 × 8 × 32768² × 4 = 68'719'476'736 bytes = 64 GiB

The memory saved for backward is quadratic in context_length, when seq_len doubles, memory from forward pass grows around 4x.

To avoid saving this memory, we need to break down what forward pass saves exactly in shape of L × L for the backward pass:

- the result of exp in softmax calculation
- the inputs of attention output matmul (output of softmax @ Vᵀ)

This points us to trick that flash attention uses: avoid saving the L × L matrices for the backward pass and recompute missing parts during the backward pass itself.
We would also need to avoid materializing the L × L matrices during the forward pass as well.

## 4.2 Problem (torch_compile): Torch Compile

**(a)**

| d_model | seq_len | forward (ms)  | forward compiled (ms) | backward (ms) | backward compiled (ms) | memory before backward (MiB) | memory before backward compiled (MiB) | added by forward (MiB) | added by forward compiled (MiB) |
|---------|---------|---------------|-----------------------|---------------|------------------------|------------------------------|---------------------------------------|------------------------|---------------------------------|
| 16      | 256     | 0.33 ± 0.05   | 0.25 ± 0.06           | 0.71 ± 0.35   | 0.52 ± 0.32            | 21.40                        | 21.34                                 | 4.09                   | 4.03                            |
| 16      | 1024    | 0.65 ± 0.01   | 0.39 ± 0.01           | 1.62 ± 0.15   | 1.09 ± 0.10            | 86.34                        | 85.38                                 | 65.09                  | 64.12                           |
| 16      | 4096    | 9.07 ± 14.86  | 3.66 ± 0.02           | 17.61 ± 0.30  | 10.21 ± 0.39           | 1088.62                      | 1072.75                               | 1040.38                | 1024.50                         |
| 16      | 8192    | 28.72 ± 0.06  | 12.24 ± 0.08          | 64.98 ± 0.27  | 35.48 ± 0.47           | 4273.00                      | 4209.25                               | 4160.75                | 4097.00                         |
| 16      | 16384   | 115.90 ± 0.40 | 52.26 ± 0.57          | 258.44 ± 0.39 | 146.73 ± 0.62          | 16977.75                     | 16722.25                              | 16641.50               | 16386.00                        |
| 16      | 32768   | OOM           | OOM                   | OOM           | OOM                    | OOM                          | OOM                                   | OOM                    | OOM                             |
| 32      | 256     | 0.34 ± 0.03   | 0.23 ± 0.03           | 0.72 ± 0.17   | 0.49 ± 0.11            | 22.40                        | 22.34                                 | 4.09                   | 4.03                            |
| 32      | 1024    | 0.66 ± 0.02   | 0.40 ± 0.01           | 1.64 ± 0.24   | 1.11 ± 0.12            | 90.34                        | 89.38                                 | 65.09                  | 64.12                           |
| 32      | 4096    | 7.85 ± 0.03   | 3.86 ± 0.09           | 17.95 ± 0.27  | 10.41 ± 0.39           | 1104.62                      | 1088.75                               | 1040.38                | 1024.50                         |
| 32      | 8192    | 29.95 ± 0.09  | 13.33 ± 0.13          | 66.65 ± 0.29  | 36.89 ± 0.22           | 4305.00                      | 4241.25                               | 4160.75                | 4097.00                         |
| 32      | 16384   | 120.36 ± 0.28 | 56.08 ± 0.34          | 263.26 ± 0.46 | 150.77 ± 0.45          | 17041.75                     | 16786.25                              | 16641.50               | 16386.00                        |
| 32      | 32768   | OOM           | OOM                   | OOM           | OOM                    | OOM                          | OOM                                   | OOM                    | OOM                             |
| 64      | 256     | 0.35 ± 0.04   | 0.24 ± 0.04           | 0.74 ± 0.19   | 0.51 ± 0.14            | 24.40                        | 24.34                                 | 4.09                   | 4.03                            |
| 64      | 1024    | 0.70 ± 0.01   | 0.43 ± 0.01           | 1.71 ± 0.31   | 1.15 ± 0.11            | 98.34                        | 97.38                                 | 65.09                  | 64.12                           |
| 64      | 4096    | 8.32 ± 0.03   | 4.30 ± 0.06           | 18.46 ± 0.31  | 10.86 ± 0.36           | 1136.62                      | 1120.75                               | 1040.38                | 1024.50                         |
| 64      | 8192    | 31.82 ± 0.08  | 15.17 ± 0.12          | 68.53 ± 0.24  | 38.82 ± 0.29           | 4369.00                      | 4305.25                               | 4160.75                | 4097.00                         |
| 64      | 16384   | 127.44 ± 0.31 | 63.98 ± 0.28          | 270.47 ± 0.36 | 158.39 ± 0.27          | 17169.75                     | 16914.25                              | 16641.50               | 16386.00                        |
| 64      | 32768   | OOM           | OOM                   | OOM           | OOM                    | OOM                          | OOM                                   | OOM                    | OOM                             |
| 128     | 256     | 0.34 ± 0.01   | 0.24 ± 0.01           | 0.70 ± 0.08   | 0.51 ± 0.11            | 28.40                        | 28.34                                 | 4.09                   | 4.03                            |
| 128     | 1024    | 0.75 ± 0.01   | 0.49 ± 0.01           | 1.75 ± 0.24   | 1.22 ± 0.11            | 114.34                       | 113.38                                | 65.09                  | 64.12                           |
| 128     | 4096    | 9.23 ± 0.03   | 5.22 ± 0.03           | 19.45 ± 0.28  | 11.91 ± 0.37           | 1200.62                      | 1184.75                               | 1040.38                | 1024.50                         |
| 128     | 8192    | 35.49 ± 0.12  | 18.95 ± 0.13          | 72.34 ± 0.28  | 43.02 ± 0.51           | 4497.00                      | 4433.25                               | 4160.75                | 4097.00                         |
| 128     | 16384   | 141.98 ± 0.30 | 78.76 ± 0.36          | 285.93 ± 0.52 | 173.69 ± 0.68          | 17425.75                     | 17170.25                              | 16641.50               | 16386.00                        |
| 128     | 32768   | OOM           | OOM                   | OOM           | OOM                    | OOM                          | OOM                                   | OOM                    | OOM                             |

**(b)**

xl, context 512, batch 4, fp32 (TF32 matmuls), A100 80GB PCIe, 5 warmup + 10 measured steps:

| xl                                         | vanilla (ms)  | compiled (ms)  | speedup |
|--------------------------------------------|---------------|----------------|---------|
| forward                                    | 214.68 ± 0.81 | 163.12 ± 13.93 | 1.32×   |
| backward                                   | 510.31 ± 1.32 | 374.04 ± 1.42  | 1.36×   |
| optimizer step                             | 281.69 ± 0.09 | 281.76 ± 0.38  | 1.00×   |
| full step (forward + backward + optimizer) | 1006.68       | 818.92         | 1.23×   |

