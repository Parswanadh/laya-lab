# arm-3r report (regenerated from raw artifacts)

* `experiments/h5-adapter/summary.json` — every accuracy, CI and paired test below.
* `experiments/h5-adapter/step0_identity.json` — the step-0 identity assertion.
* `experiments/h5-adapter/probe_arm3r.json` — the branch learning-rate probe.
* `experiments/h5-adapter/runs/<arm>/seed<k>/training.json` — the per-epoch curves.


Regenerate: `env/venv/bin/python experiments/h5-adapter/report_arm3r.py`


## Per-cell accuracy, n=200, label-balanced (oracle beside every number)

| cell | oracle | arm1_frozen | arm2_shipped_init | arm2long_shipped_init | arm3_xattn | arm3r_residual | arm3r_residual_ablated |
|---|---|---|---|---|---|---|---|
| L0 | 0.2500 | 0.640 [0.575, 0.705] | 0.620 [0.550, 0.690] | — | 0.225 [0.170, 0.285] | — | — |
| L4000-p000 | 0.2500 | 0.580 [0.510, 0.650] | 0.590 [0.520, 0.660] | — | 0.270 [0.210, 0.335] | — | — |
| L4000-p025 | 0.2500 | 0.295 [0.235, 0.360] | 0.430 [0.360, 0.500] | — | 0.230 [0.175, 0.290] | — | — |
| L4000-p050 | 0.2500 | 0.335 [0.270, 0.400] | 0.500 [0.430, 0.570] | — | 0.250 [0.190, 0.310] | — | — |
| L4000-p075 | 0.2500 | 0.280 [0.220, 0.345] | 0.425 [0.360, 0.495] | — | 0.250 [0.190, 0.310] | — | — |
| L4000-p100 | 0.2500 | 0.415 [0.345, 0.485] | 0.495 [0.425, 0.565] | — | 0.255 [0.195, 0.315] | — | — |
| L7000-p000 | 0.2500 | 0.595 [0.525, 0.665] | 0.585 [0.515, 0.655] | — | 0.245 [0.185, 0.305] | — | — |
| L7000-p100 | 0.2500 | 0.365 [0.300, 0.435] | 0.455 [0.385, 0.525] | — | 0.240 [0.185, 0.300] | — | — |

## Paired McNemar, recomputed here from the raw per-item JSONL

Holm column: `summary.json`'s `comparisons` entry for the same pair where it exists (the correction is over that file's family), `—` where the pair is not in the family.

| candidate | baseline | cell | acc cand | acc base | Δ pp | wrong→right | right→wrong | p exact | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| arm2_shipped_init | arm1_frozen | L0 | 0.620 | 0.640 | -2.0 | 6 | 2 | 0.289 | — |
| arm2_shipped_init | arm1_frozen | L4000-p000 | 0.590 | 0.580 | +1.0 | 5 | 7 | 0.774 | — |
| arm2_shipped_init | arm1_frozen | L4000-p025 | 0.430 | 0.295 | +13.5 | 4 | 31 | 3.47e-06 | — |
| arm2_shipped_init | arm1_frozen | L4000-p050 | 0.500 | 0.335 | +16.5 | 3 | 36 | 3.61e-08 | — |
| arm2_shipped_init | arm1_frozen | L4000-p075 | 0.425 | 0.280 | +14.5 | 3 | 32 | 4.18e-07 | — |
| arm2_shipped_init | arm1_frozen | L4000-p100 | 0.495 | 0.415 | +8.0 | 5 | 21 | 0.00249 | — |
| arm2_shipped_init | arm1_frozen | L7000-p000 | 0.585 | 0.595 | -1.0 | 8 | 6 | 0.791 | — |
| arm2_shipped_init | arm1_frozen | L7000-p100 | 0.455 | 0.365 | +9.0 | 7 | 25 | 0.0021 | — |

## Convergence (per-epoch curves, from the training records)


`arm2long_shipped_init`: no training record.


`arm3r_residual`: no training record.


## Step-0 identity and liveness

`step0_identity.json`: missing.


## Branch learning-rate probe

`probe_arm3r.json`: missing.


## Verdict cells as `stats.py` computes them (oracle included)

```json
{
 "L0": {
  "arm1_frozen": 0.64,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.62,
  "arm3_beats_position_only_oracle": false,
  "arm3_minus_arm2_shipped_pp": -39.5,
  "arm3_xattn": 0.225,
  "arm4_xattn_long": null,
  "cell": "L0",
  "inside_arm2_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L4000-p000": {
  "arm1_frozen": 0.58,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.59,
  "arm3_beats_position_only_oracle": true,
  "arm3_minus_arm2_shipped_pp": -31.999999999999996,
  "arm3_xattn": 0.27,
  "arm4_xattn_long": null,
  "cell": "L4000-p000",
  "inside_arm2_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L4000-p100": {
  "arm1_frozen": 0.415,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.495,
  "arm3_beats_position_only_oracle": true,
  "arm3_minus_arm2_shipped_pp": -24.0,
  "arm3_xattn": 0.255,
  "arm4_xattn_long": null,
  "cell": "L4000-p100",
  "inside_arm2_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "position_only_oracle": 0.25
 },
 "L7000-p100": {
  "arm1_frozen": 0.365,
  "arm2_random_init": null,
  "arm2_shipped_init": 0.455,
  "arm3_beats_position_only_oracle": false,
  "arm3_minus_arm2_shipped_pp": -21.500000000000004,
  "arm3_xattn": 0.24,
  "arm4_xattn_long": null,
  "cell": "L7000-p100",
  "inside_arm2_seed_spread": false,
  "inside_arm3_seed_spread": false,
  "position_only_oracle": 0.25
 }
}
```

* oracle: `summary.json` → `position_only_oracle`; n=200 per cell, 50 per class. Any arm that does not beat it has learned nothing about content and is reported in those words.

* seed spread: `summary.json` → `seed_spread`.

