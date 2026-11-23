| Model | Max tokens | Accuracy (%) | Mean reward | Correct / partial / failed | Avg output tokens |
|---|---|---|---|---|---|
| Qwen3-1.7B zero-shot | L256 | 0.00 | 0.000 | 0 / 0 / 1024 | 256 |
| Qwen3-1.7B zero-shot | L512 | 8.98 | 0.092 | 92 / 21 / 911 | 498 |
| Qwen3-1.7B zero-shot | L1024 | 8.69 | 0.093 | 89 / 60 / 875 | 888 |
| train_L256_dr_grpo (step 80) | L256 | 60.74 | 0.611 | 622 / 36 / 366 | 161 |
|   (step 0 = zero-shot in the same run: 0.10%; best eval 61.13% at step 70) | | | | | |
| train_L256_grpo (step 80) | L256 | 58.50 | 0.615 | 599 / 310 / 115 | 139 |
|   (step 0 = zero-shot in the same run: 0.10%; best eval 58.50% at step 80) | | | | | |
| train_L512_dr_grpo (step 80) | L512 | 55.86 | 0.560 | 572 / 10 / 442 | 356 |
|   (step 0 = zero-shot in the same run: 7.62%; best eval 55.86% at step 80) | | | | | |
| train_L512_grpo (step 80) | L512 | 52.05 | 0.521 | 533 / 10 / 481 | 374 |
|   (step 0 = zero-shot in the same run: 7.62%; best eval 52.05% at step 70) | | | | | |
