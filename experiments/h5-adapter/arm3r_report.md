# arm-3r report (regenerated from raw artifacts)

* `experiments/h5-adapter/summary.json` — every accuracy, CI and paired test below.
* `experiments/h5-adapter/step0_identity.json` — the step-0 identity assertion.
* `experiments/h5-adapter/probe_arm3r.json` — the branch learning-rate probe.
* `experiments/h5-adapter/runs/<arm>/seed<k>/training.json` — the per-epoch curves.


Regenerate: `env/venv/bin/python experiments/h5-adapter/report_arm3r.py`


## Per-cell accuracy, n=200, label-balanced (oracle beside every number)

| cell | oracle | arm1_frozen | arm2_shipped_init | arm2long_shipped_init | arm3_xattn | arm3r_residual | arm3r_residual_ablated |
|---|---|---|---|---|---|---|---|
| L0 | 0.2500 | 0.640 [0.575, 0.705] | 0.620 [0.550, 0.690] | 0.600 [0.530, 0.665] | 0.225 [0.170, 0.285] | 0.595 [0.530, 0.660] | 0.595 [0.530, 0.660] |
| L4000-p000 | 0.2500 | 0.580 [0.510, 0.650] | 0.590 [0.520, 0.660] | 0.530 [0.460, 0.600] | 0.270 [0.210, 0.335] | 0.715 [0.650, 0.780] | 0.715 [0.650, 0.775] |
| L4000-p025 | 0.2500 | 0.295 [0.235, 0.360] | 0.430 [0.360, 0.500] | 0.545 [0.475, 0.615] | 0.230 [0.175, 0.290] | 0.520 [0.450, 0.590] | 0.525 [0.455, 0.595] |
| L4000-p050 | 0.2500 | 0.335 [0.270, 0.400] | 0.500 [0.430, 0.570] | 0.555 [0.485, 0.625] | 0.250 [0.190, 0.310] | 0.590 [0.520, 0.660] | 0.590 [0.520, 0.660] |
| L4000-p075 | 0.2500 | 0.280 [0.220, 0.345] | 0.425 [0.360, 0.495] | 0.545 [0.475, 0.615] | 0.250 [0.190, 0.310] | 0.555 [0.485, 0.625] | 0.565 [0.495, 0.635] |
| L4000-p100 | 0.2500 | 0.415 [0.345, 0.485] | 0.495 [0.425, 0.565] | 0.465 [0.395, 0.535] | 0.255 [0.195, 0.315] | 0.560 [0.490, 0.630] | 0.560 [0.490, 0.630] |
| L7000-p000 | 0.2500 | 0.595 [0.525, 0.665] | 0.585 [0.515, 0.655] | 0.540 [0.470, 0.610] | 0.245 [0.185, 0.305] | 0.680 [0.615, 0.745] | 0.680 [0.615, 0.745] |
| L7000-p100 | 0.2500 | 0.365 [0.300, 0.435] | 0.455 [0.385, 0.525] | 0.410 [0.340, 0.480] | 0.240 [0.185, 0.300] | 0.485 [0.415, 0.555] | 0.500 [0.430, 0.570] |

## Paired McNemar, recomputed here from the raw per-item JSONL

Holm column: `summary.json`'s `comparisons` entry for the same pair where it exists (the correction is over that file's family), `—` where the pair is not in the family.

| candidate | baseline | cell | acc cand | acc base | Δ pp | wrong→right | right→wrong | p exact | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| arm3r_residual | arm2_shipped_init | L0 | 0.595 | 0.620 | -2.5 | 23 | 18 | 0.533 | 1 |
| arm3r_residual | arm2_shipped_init | L4000-p000 | 0.715 | 0.590 | +12.5 | 13 | 38 | 0.000621 | 0.0354 |
| arm3r_residual | arm2_shipped_init | L4000-p025 | 0.520 | 0.430 | +9.0 | 25 | 43 | 0.0385 | 1 |
| arm3r_residual | arm2_shipped_init | L4000-p050 | 0.590 | 0.500 | +9.0 | 25 | 43 | 0.0385 | 1 |
| arm3r_residual | arm2_shipped_init | L4000-p075 | 0.555 | 0.425 | +13.0 | 19 | 45 | 0.00156 | 0.0844 |
| arm3r_residual | arm2_shipped_init | L4000-p100 | 0.560 | 0.495 | +6.5 | 25 | 38 | 0.13 | 1 |
| arm3r_residual | arm2_shipped_init | L7000-p000 | 0.680 | 0.585 | +9.5 | 6 | 25 | 0.000878 | 0.0492 |
| arm3r_residual | arm2_shipped_init | L7000-p100 | 0.485 | 0.455 | +3.0 | 26 | 32 | 0.512 | 1 |
| arm3r_residual | arm2long_shipped_init | L0 | 0.595 | 0.600 | -0.5 | 11 | 10 | 1 | 1 |
| arm3r_residual | arm2long_shipped_init | L4000-p000 | 0.715 | 0.530 | +18.5 | 9 | 46 | 4.34e-07 | 3.47e-05 |
| arm3r_residual | arm2long_shipped_init | L4000-p025 | 0.520 | 0.545 | -2.5 | 26 | 21 | 0.56 | 1 |
| arm3r_residual | arm2long_shipped_init | L4000-p050 | 0.590 | 0.555 | +3.5 | 23 | 30 | 0.41 | 1 |
| arm3r_residual | arm2long_shipped_init | L4000-p075 | 0.555 | 0.545 | +1.0 | 17 | 19 | 0.868 | 1 |
| arm3r_residual | arm2long_shipped_init | L4000-p100 | 0.560 | 0.465 | +9.5 | 25 | 44 | 0.0295 | 1 |
| arm3r_residual | arm2long_shipped_init | L7000-p000 | 0.680 | 0.540 | +14.0 | 14 | 42 | 0.000234 | 0.015 |
| arm3r_residual | arm2long_shipped_init | L7000-p100 | 0.485 | 0.410 | +7.5 | 26 | 41 | 0.0864 | 1 |
| arm3r_residual_ablated | arm2_shipped_init | L0 | 0.595 | 0.620 | -2.5 | 23 | 18 | 0.533 | 1 |
| arm3r_residual_ablated | arm2_shipped_init | L4000-p000 | 0.715 | 0.590 | +12.5 | 12 | 37 | 0.00047 | 0.0282 |
| arm3r_residual_ablated | arm2_shipped_init | L4000-p025 | 0.525 | 0.430 | +9.5 | 22 | 41 | 0.0226 | 1 |
| arm3r_residual_ablated | arm2_shipped_init | L4000-p050 | 0.590 | 0.500 | +9.0 | 24 | 42 | 0.0356 | 1 |
| arm3r_residual_ablated | arm2_shipped_init | L4000-p075 | 0.565 | 0.425 | +14.0 | 17 | 45 | 0.000497 | 0.0293 |
| arm3r_residual_ablated | arm2_shipped_init | L4000-p100 | 0.560 | 0.495 | +6.5 | 24 | 37 | 0.124 | 1 |
| arm3r_residual_ablated | arm2_shipped_init | L7000-p000 | 0.680 | 0.585 | +9.5 | 5 | 24 | 0.000546 | 0.0317 |
| arm3r_residual_ablated | arm2_shipped_init | L7000-p100 | 0.500 | 0.455 | +4.5 | 24 | 33 | 0.289 | 1 |
| arm3r_residual | arm3r_residual_ablated | L0 | 0.595 | 0.595 | +0.0 | 0 | 0 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L4000-p000 | 0.715 | 0.715 | +0.0 | 1 | 1 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L4000-p025 | 0.520 | 0.525 | -0.5 | 3 | 2 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L4000-p050 | 0.590 | 0.590 | +0.0 | 1 | 1 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L4000-p075 | 0.555 | 0.565 | -1.0 | 2 | 0 | 0.5 | — |
| arm3r_residual | arm3r_residual_ablated | L4000-p100 | 0.560 | 0.560 | +0.0 | 1 | 1 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L7000-p000 | 0.680 | 0.680 | +0.0 | 1 | 1 | 1 | — |
| arm3r_residual | arm3r_residual_ablated | L7000-p100 | 0.485 | 0.500 | -1.5 | 3 | 0 | 0.25 | — |
| arm2long_shipped_init | arm2_shipped_init | L0 | 0.600 | 0.620 | -2.0 | 22 | 18 | 0.636 | — |
| arm2long_shipped_init | arm2_shipped_init | L4000-p000 | 0.530 | 0.590 | -6.0 | 27 | 15 | 0.0884 | — |
| arm2long_shipped_init | arm2_shipped_init | L4000-p025 | 0.545 | 0.430 | +11.5 | 21 | 44 | 0.00592 | — |
| arm2long_shipped_init | arm2_shipped_init | L4000-p050 | 0.555 | 0.500 | +5.5 | 40 | 51 | 0.294 | — |
| arm2long_shipped_init | arm2_shipped_init | L4000-p075 | 0.545 | 0.425 | +12.0 | 23 | 47 | 0.00558 | — |
| arm2long_shipped_init | arm2_shipped_init | L4000-p100 | 0.465 | 0.495 | -3.0 | 45 | 39 | 0.586 | — |
| arm2long_shipped_init | arm2_shipped_init | L7000-p000 | 0.540 | 0.585 | -4.5 | 27 | 18 | 0.233 | — |
| arm2long_shipped_init | arm2_shipped_init | L7000-p100 | 0.410 | 0.455 | -4.5 | 43 | 34 | 0.362 | — |
| arm2_shipped_init | arm1_frozen | L0 | 0.620 | 0.640 | -2.0 | 6 | 2 | 0.289 | — |
| arm2_shipped_init | arm1_frozen | L4000-p000 | 0.590 | 0.580 | +1.0 | 5 | 7 | 0.774 | — |
| arm2_shipped_init | arm1_frozen | L4000-p025 | 0.430 | 0.295 | +13.5 | 4 | 31 | 3.47e-06 | — |
| arm2_shipped_init | arm1_frozen | L4000-p050 | 0.500 | 0.335 | +16.5 | 3 | 36 | 3.61e-08 | — |
| arm2_shipped_init | arm1_frozen | L4000-p075 | 0.425 | 0.280 | +14.5 | 3 | 32 | 4.18e-07 | — |
| arm2_shipped_init | arm1_frozen | L4000-p100 | 0.495 | 0.415 | +8.0 | 5 | 21 | 0.00249 | — |
| arm2_shipped_init | arm1_frozen | L7000-p000 | 0.585 | 0.595 | -1.0 | 8 | 6 | 0.791 | — |
| arm2_shipped_init | arm1_frozen | L7000-p100 | 0.455 | 0.365 | +9.0 | 7 | 25 | 0.0021 | — |

## Convergence (per-epoch curves, from the training records)


### arm2long_shipped_init — 40 epochs, lr=0.0005, lr_cross=0.003, 3337 s, 16160 steps

| epoch | train loss | train acc | lr | branch \|Δlogit\| | \|W_out\| | s |
|---|---|---|---|---|---|---|
| 0 | 1.4417 | 0.3373 | 2.50e-04 | — | — | 78 |
| 1 | 1.3274 | 0.3472 | 5.00e-04 | — | — | 157 |
| 2 | 1.3586 | 0.3413 | 4.99e-04 | — | — | 242 |
| 3 | 1.3325 | 0.3562 | 4.97e-04 | — | — | 330 |
| 4 | 1.3362 | 0.3353 | 4.92e-04 | — | — | 416 |
| 5 | 1.3480 | 0.3423 | 4.86e-04 | — | — | 505 |
| 6 | 1.3370 | 0.3482 | 4.79e-04 | — | — | 593 |
| 7 | 1.3140 | 0.3750 | 4.70e-04 | — | — | 680 |
| 8 | 1.3203 | 0.3760 | 4.59e-04 | — | — | 790 |
| 9 | 1.3061 | 0.3720 | 4.47e-04 | — | — | 874 |
| 10 | 1.3088 | 0.3839 | 4.34e-04 | — | — | 955 |
| 11 | 1.3119 | 0.3621 | 4.19e-04 | — | — | 1037 |
| 12 | 1.3118 | 0.3671 | 4.04e-04 | — | — | 1116 |
| 13 | 1.2984 | 0.3631 | 3.87e-04 | — | — | 1194 |
| 14 | 1.3009 | 0.3700 | 3.69e-04 | — | — | 1274 |
| 15 | 1.2937 | 0.3819 | 3.50e-04 | — | — | 1353 |
| 16 | 1.3026 | 0.3661 | 3.31e-04 | — | — | 1451 |
| 17 | 1.2992 | 0.3859 | 3.11e-04 | — | — | 1528 |
| 18 | 1.2909 | 0.3700 | 2.91e-04 | — | — | 1607 |
| 19 | 1.2725 | 0.3988 | 2.71e-04 | — | — | 1685 |
| 20 | 1.2719 | 0.3760 | 2.50e-04 | — | — | 1763 |
| 21 | 1.2696 | 0.3770 | 2.29e-04 | — | — | 1841 |
| 22 | 1.2484 | 0.3948 | 2.09e-04 | — | — | 1918 |
| 23 | 1.2435 | 0.4087 | 1.89e-04 | — | — | 1995 |
| 24 | 1.2203 | 0.4286 | 1.69e-04 | — | — | 2095 |
| 25 | 1.2397 | 0.4196 | 1.50e-04 | — | — | 2175 |
| 26 | 1.2093 | 0.4296 | 1.31e-04 | — | — | 2254 |
| 27 | 1.2275 | 0.4157 | 1.13e-04 | — | — | 2334 |
| 28 | 1.2106 | 0.4226 | 9.65e-05 | — | — | 2414 |
| 29 | 1.1848 | 0.4514 | 8.07e-05 | — | — | 2495 |
| 30 | 1.1771 | 0.4524 | 6.61e-05 | — | — | 2576 |
| 31 | 1.1622 | 0.4702 | 5.27e-05 | — | — | 2656 |
| 32 | 1.1523 | 0.4683 | 4.07e-05 | — | — | 2758 |
| 33 | 1.1553 | 0.4950 | 3.02e-05 | — | — | 2837 |
| 34 | 1.1413 | 0.4831 | 2.11e-05 | — | — | 2916 |
| 35 | 1.1331 | 0.4921 | 1.36e-05 | — | — | 2995 |
| 36 | 1.1162 | 0.4980 | 7.66e-06 | — | — | 3075 |
| 37 | 1.1012 | 0.4980 | 3.42e-06 | — | — | 3155 |
| 38 | 1.1158 | 0.4960 | 8.58e-07 | — | — | 3235 |
| 39 | 1.1134 | 0.4861 | 5.23e-12 | — | — | 3315 |

Last 5 epochs: loss 1.1413 → 1.1134 (Δ -0.0279), train acc 0.4831 → 0.4861. First→last: loss 1.4417 → 1.1134.


### arm3r_residual — 40 epochs, lr=0.0005, lr_cross=0.003, 3968 s, 16160 steps

| epoch | train loss | train acc | lr | branch \|Δlogit\| | \|W_out\| | s |
|---|---|---|---|---|---|---|
| 0 | 1.4396 | 0.3373 | 2.50e-04 | 0.0994 | 7.36 | 88 |
| 1 | 1.3288 | 0.3502 | 5.00e-04 | 0.12 | 25.3 | 186 |
| 2 | 1.3651 | 0.3313 | 4.99e-04 | 0.714 | 53.1 | 291 |
| 3 | 1.3684 | 0.3105 | 4.97e-04 | 0.318 | 61.1 | 394 |
| 4 | 1.3388 | 0.3462 | 4.92e-04 | 0.72 | 64.8 | 498 |
| 5 | 1.3637 | 0.3323 | 4.86e-04 | 0.301 | 70.1 | 601 |
| 6 | 1.3708 | 0.3145 | 4.79e-04 | 0.228 | 75.1 | 705 |
| 7 | 1.3600 | 0.3373 | 4.70e-04 | 0.341 | 78 | 808 |
| 8 | 1.3565 | 0.3274 | 4.59e-04 | 0.192 | 86.1 | 938 |
| 9 | 1.3407 | 0.3502 | 4.47e-04 | 0.224 | 89.7 | 1036 |
| 10 | 1.3123 | 0.3562 | 4.34e-04 | 0.256 | 94.8 | 1137 |
| 11 | 1.3241 | 0.3522 | 4.19e-04 | 0.398 | 101 | 1239 |
| 12 | 1.3099 | 0.3899 | 4.04e-04 | 0.266 | 104 | 1341 |
| 13 | 1.3166 | 0.3770 | 3.87e-04 | 0.163 | 108 | 1436 |
| 14 | 1.3032 | 0.3700 | 3.69e-04 | 0.251 | 113 | 1530 |
| 15 | 1.3054 | 0.3710 | 3.50e-04 | 0.249 | 118 | 1631 |
| 16 | 1.2872 | 0.4038 | 3.31e-04 | 0.362 | 121 | 1768 |
| 17 | 1.3071 | 0.3720 | 3.11e-04 | 0.314 | 125 | 1871 |
| 18 | 1.3321 | 0.3373 | 2.91e-04 | 1.3 | 131 | 1971 |
| 19 | 1.3287 | 0.3552 | 2.71e-04 | 1.01 | 136 | 2071 |
| 20 | 1.3193 | 0.3710 | 2.50e-04 | 1.14 | 139 | 2170 |
| 21 | 1.2893 | 0.3899 | 2.29e-04 | 1.17 | 141 | 2267 |
| 22 | 1.2897 | 0.3849 | 2.09e-04 | 0.896 | 142 | 2368 |
| 23 | 1.2828 | 0.3849 | 1.89e-04 | 0.66 | 142 | 2465 |
| 24 | 1.2747 | 0.3810 | 1.69e-04 | 0.717 | 142 | 2588 |
| 25 | 1.2424 | 0.4187 | 1.50e-04 | 0.603 | 142 | 2677 |
| 26 | 1.2398 | 0.4087 | 1.31e-04 | 0.608 | 142 | 2767 |
| 27 | 1.2497 | 0.3909 | 1.13e-04 | 0.554 | 143 | 2856 |
| 28 | 1.2212 | 0.4296 | 9.65e-05 | 0.541 | 143 | 2945 |
| 29 | 1.2177 | 0.4325 | 8.07e-05 | 0.627 | 143 | 3032 |
| 30 | 1.2245 | 0.4425 | 6.61e-05 | 0.689 | 143 | 3122 |
| 31 | 1.2007 | 0.4544 | 5.27e-05 | 0.608 | 143 | 3210 |
| 32 | 1.2037 | 0.4365 | 4.07e-05 | 0.517 | 143 | 3326 |
| 33 | 1.1975 | 0.4296 | 3.02e-05 | 0.539 | 142 | 3413 |
| 34 | 1.1850 | 0.4415 | 2.11e-05 | 0.544 | 142 | 3500 |
| 35 | 1.1823 | 0.4544 | 1.36e-05 | 0.519 | 142 | 3589 |
| 36 | 1.1761 | 0.4593 | 7.66e-06 | 0.514 | 142 | 3676 |
| 37 | 1.1733 | 0.4683 | 3.42e-06 | 0.513 | 142 | 3764 |
| 38 | 1.1704 | 0.4583 | 8.58e-07 | 0.516 | 142 | 3852 |
| 39 | 1.1588 | 0.4722 | 5.23e-12 | 0.514 | 142 | 3940 |

Last 5 epochs: loss 1.1850 → 1.1588 (Δ -0.0262), train acc 0.4415 → 0.4722. First→last: loss 1.4396 → 1.1588.


## Step-0 identity and liveness

* **shipped_init: arm3r == arm2 (all cells)** — n=2200, bitwise-identical logits: `True`, max |Δlogit| 0.0, items with a different prediction: **0**
* **arm2's trained head in both: arm3r == arm2 (3 cells)** — n=600, bitwise-identical logits: `True`, max |Δlogit| 0.0, items with a different prediction: **0**
* **liveness: a non-zero branch must NOT equal arm2 (L7000-p100)** — n=200, bitwise-identical logits: `False`, max |Δlogit| 0.03125, items with a different prediction: **0**
* trainability: out_proj weight grad sum 50.230445861816406; inner-branch max grad sum at step 0 0.0 (24 tensors, zero by construction while W=0)
* verdict: **VALID -- arm3r is arm2 at step 0 (bitwise, 0/2200 items differ) and the branch moves the logits when non-zero**


## Branch learning-rate probe

| cross lr | loss per epoch | acc per epoch | branch \|Δlogit\| per epoch | \|W_out\| per epoch |
|---|---|---|---|---|
| 0.0005 | 1.3660 1.3266 1.3040 | 0.331 0.339 0.349 | 0.0507 0.0897 0.0819 | 5.35 6.64 6.65 |
| 0.003 | 1.3651 1.3172 1.3026 | 0.341 0.343 0.375 | 0.01 0.204 0.193 | 25.9 32.6 32.5 |
| 0.01 | 1.3639 1.3311 1.3248 | 0.333 0.341 0.330 | 0.129 0.711 0.657 | 80.8 102 102 |

## Verdict cells as `stats.py` computes them (oracle included)

```json
{
 "L0": {
  "arm1_frozen": 0.64,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.62,
  "arm2long_shipped_init": 0.6,
  "arm3_beats_position_only_oracle": false,
  "arm3_minus_arm2_shipped_pp": -39.5,
  "arm3_xattn": 0.225,
  "arm3r_beats_position_only_oracle": true,
  "arm3r_minus_arm2_pp": -2.500000000000002,
  "arm3r_minus_arm2long_pp": -0.5000000000000004,
  "arm3r_residual": 0.595,
  "arm4_xattn_long": null,
  "cell": "L0",
  "inside_arm2_seed_spread": false,
  "inside_arm2long_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "inside_arm3r_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L4000-p000": {
  "arm1_frozen": 0.58,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.59,
  "arm2long_shipped_init": 0.53,
  "arm3_beats_position_only_oracle": true,
  "arm3_minus_arm2_shipped_pp": -31.999999999999996,
  "arm3_xattn": 0.27,
  "arm3r_beats_position_only_oracle": true,
  "arm3r_minus_arm2_pp": 12.5,
  "arm3r_minus_arm2long_pp": 18.499999999999993,
  "arm3r_residual": 0.715,
  "arm4_xattn_long": null,
  "cell": "L4000-p000",
  "inside_arm2_seed_spread": false,
  "inside_arm2long_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "inside_arm3r_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L4000-p050": {
  "arm1_frozen": 0.335,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.5,
  "arm2long_shipped_init": 0.555,
  "arm3_beats_position_only_oracle": false,
  "arm3_minus_arm2_shipped_pp": -25.0,
  "arm3_xattn": 0.25,
  "arm3r_beats_position_only_oracle": true,
  "arm3r_minus_arm2_pp": 8.999999999999996,
  "arm3r_minus_arm2long_pp": 3.499999999999992,
  "arm3r_residual": 0.59,
  "arm4_xattn_long": null,
  "cell": "L4000-p050",
  "inside_arm2_seed_spread": false,
  "inside_arm2long_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "inside_arm3r_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L4000-p100": {
  "arm1_frozen": 0.415,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.495,
  "arm2long_shipped_init": 0.465,
  "arm3_beats_position_only_oracle": true,
  "arm3_minus_arm2_shipped_pp": -24.0,
  "arm3_xattn": 0.255,
  "arm3r_beats_position_only_oracle": true,
  "arm3r_minus_arm2_pp": 6.500000000000005,
  "arm3r_minus_arm2long_pp": 9.500000000000004,
  "arm3r_residual": 0.56,
  "arm4_xattn_long": null,
  "cell": "L4000-p100",
  "inside_arm2_seed_spread": false,
  "inside_arm2long_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "inside_arm3r_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L7000-p100": {
  "arm1_frozen": 0.365,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.455,
  "arm2long_shipped_init": 0.41,
  "arm3_beats_position_only_oracle": false,
  "arm3_minus_arm2_shipped_pp": -21.500000000000004,
  "arm3_xattn": 0.24,
  "arm3r_beats_position_only_oracle": true,
  "arm3r_minus_arm2_pp": 2.9999999999999973,
  "arm3r_minus_arm2long_pp": 7.500000000000001,
  "arm3r_residual": 0.485,
  "arm4_xattn_long": null,
  "cell": "L7000-p100",
  "inside_arm2_seed_spread": false,
  "inside_arm2long_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "inside_arm3r_seed_spread": false,
  "position_only_oracle": 0.25
 }
}
```

* oracle: `summary.json` → `position_only_oracle`; n=200 per cell, 50 per class. Any arm that does not beat it has learned nothing about content and is reported in those words.

* seed spread: `summary.json` → `seed_spread`.

