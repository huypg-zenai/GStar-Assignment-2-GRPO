# Report sources

`report.tex` → `../report.pdf` (build: `tectonic -X compile report.tex && cp report.pdf ..`).

Numbers and figures are regenerated from the downloaded Kaggle outputs (`outputs_L256/`, `outputs_L512/`,
`outputs_L256_dr/`, `outputs_L512_dr/`, `outputs_p2_ev/`, `outputs_p2_outcome/` in one folder, e.g. `~/Downloads`):

| Script | Produces |
|---|---|
| `analyze_p1.py DIR --out p1_analysis.json` | Problem 1 failure-mode breakdown, 3- vs 4-number accuracy, subset-of-numbers answers |
| `make_figures.py` | `figures/p1_accuracy.pdf`, `figures/p1_failures.pdf` |
| `grad_reductions.py` | `grad_reductions.txt` (conceptual question: per-token gradients of the two reductions) |
| `analyze_p2.py DIR --out p2_analysis.json` | Problem 2 playing style, per-seat exploitability, regret by situation, reasoning-trace keywords |
| `analyze_p2_reasoning.py DIR --out p2_reasoning.json` | Problem 2: are the stated hand-strength claims correct (vs showdown equity)? invented poker terms |

`figures/p1_curves.png` and `figures/p2_curves.png` come from `kaggle/report.py` and `problem2/report_leduc.py`;
`p1_results.md` / `p2_results.md` are their tables.
