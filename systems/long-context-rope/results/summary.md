### Zero-shot: trained on 128-byte windows, evaluated longer (bits/byte, 64 held-out windows)

| Method | 128 bytes | 256 bytes | 512 bytes | 1024 bytes |
|---|---|---|---|---|
| plain RoPE (extrapolate) | 1.381 | 2.275 | 3.687 | 4.598 |
| position interpolation | 1.381 | 3.253 | 4.335 | 4.943 |
| NTK-aware | 1.381 | 1.569 | 2.481 | 3.448 |
| YaRN | 1.381 | 2.101 | 2.876 | 3.348 |

### After 200 fine-tuning steps on 512-byte windows (each method active during fine-tuning)

| Fine-tuned with | 512 bytes | 1024 bytes |
|---|---|---|
| plain RoPE (extrapolate) | 1.341 | 1.475 |
| position interpolation | 1.436 | 2.209 |
| NTK-aware | 1.331 | 1.701 |
| YaRN | 1.349 | 1.543 |
| control: 200 more steps at 128 bytes, evaluated with plain RoPE (extrapolate) | 3.692 | 4.605 |
| control: 200 more steps at 128 bytes, evaluated with position interpolation | 4.375 | 4.981 |
| control: 200 more steps at 128 bytes, evaluated with NTK-aware | 2.461 | 3.447 |
| control: 200 more steps at 128 bytes, evaluated with YaRN | 2.864 | 3.354 |

Same fine-tuned models with the scale factor kept at the fine-tuning value at every length:

| Fine-tuned with | 512 bytes | 1024 bytes |
|---|---|---|
| plain RoPE (extrapolate) | 1.341 | 1.475 |
| position interpolation | 1.436 | 2.764 |
| NTK-aware | 1.331 | 1.881 |
| YaRN | 1.349 | 2.745 |

Paired differences vs plain RoPE fine-tuned the same way (95 % bootstrap interval over windows):

- dynamic 512 bytes: position interpolation +0.095 (+0.080, +0.111); NTK-aware -0.010 (-0.014, -0.006); YaRN +0.009 (+0.002, +0.016)
- dynamic 1024 bytes: position interpolation +0.734 (+0.663, +0.809); NTK-aware +0.226 (+0.207, +0.243); YaRN +0.068 (+0.045, +0.090)
- fixed 512 bytes: position interpolation +0.095 (+0.080, +0.111); NTK-aware -0.010 (-0.014, -0.006); YaRN +0.009 (+0.002, +0.016)
- fixed 1024 bytes: position interpolation +1.289 (+1.222, +1.352); NTK-aware +0.406 (+0.373, +0.435); YaRN +1.269 (+1.211, +1.324)
