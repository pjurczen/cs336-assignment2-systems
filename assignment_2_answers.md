# CS336 Assignment 2 — Written Answers

## 2.1.3 Problem (benchmarking_script): Benchmarking Script

**(a)** Script in cs336_systems/benchmarking.py

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