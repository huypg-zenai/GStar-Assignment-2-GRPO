| Policy | Exploitability ↓ (chips/hand) | vs equilibrium | vs random | vs always-call | EV reward | seen | held-out | Regret (chips) ↓ | Valid answers | Avg tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| equilibrium (CFR+) (bot) | 0.000 | +0.000 | +0.699 | +0.633 | – | – | – | – | – | – |
| uniform random (bot) | 2.374 | -0.699 | +0.000 | +0.000 | – | – | – | – | – | – |
| always check/call (bot) | 1.467 | -0.633 | +0.000 | +0.000 | – | – | – | – | – | – |
| always bet/raise (bot) | 2.367 | -0.335 | +1.899 | +0.000 | – | – | – | – | – | – |
| Qwen3-1.7B zero-shot L512 (k=8) | 3.125 | -0.556 | -0.079 | +0.274 | 0.641 | 0.652 | 0.617 | 1.553 | 0.958 | 140 |
| train_leduc_ev step 0 (k=2) | 3.523 | -0.535 | -0.090 | +0.500 | 0.640 | 0.652 | 0.613 | 1.658 | 0.957 | 137 |
| **train_leduc_ev step 80** (k=8) | 2.184 | -0.151 | +0.070 | +0.824 | 0.860 | 0.882 | 0.811 | 0.291 | 0.999 | 64 |
| train_leduc_outcome step 0 (k=2) | 3.523 | -0.535 | -0.090 | +0.500 | 0.640 | 0.652 | 0.613 | 1.658 | 0.957 | 137 |
| **train_leduc_outcome step 80** (k=8) | 3.760 | -0.150 | +0.692 | +0.004 | 0.756 | 0.801 | 0.657 | 0.388 | 0.999 | 54 |
