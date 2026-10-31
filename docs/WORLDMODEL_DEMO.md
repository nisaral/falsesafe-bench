# World-model panel ("Learned from pixels") — plain-language guide and data schema

> **Status: exploratory · dev seeds · gates not passed.** Nothing in this panel shows that the
> world model works. It shows *where* and *why* it currently fails, so the numbers are a diagnostic, not a result.

## What you are looking at (in plain words)

A **world model** learns to predict what happens next from camera images rather than from the true equations.
It has three parts:

1. **Encoder.** It squeezes each image into 2 numbers (the *latent state*).
2. **Latent dynamics.** It predicts how those 2 numbers change when the robot acts.
3. **Learned safety margin.** It decides which latent states count as "safe".

We compute safety (an HJ value) inside that learned space and then map it back onto the real states.
Wherever the learned value says **safe** but the exact value says **unsafe**, the state is **false-safe**
(red in the panel). The panel shows this on the real position × velocity map, with a dashed band where the
physics are hidden from the camera (the "gap").

* **Full world model:** all three parts are learned.
* **Encoder only:** we keep the true dynamics and margin and only learn the encoder. This shows how much of the problem the encoder causes.
* **Exact answer:** the true answer, for reference.

The **breakdown bars** split the false-safe volume: encoder + dynamics + margin = total.

Conditions:

| | gap visible in images | gap covered by training data |
|---|---|---|
| QA | yes | yes |
| QB | yes | no (withheld) |
| QC | no (hidden) | yes |
| QD | no | no |

## Where to open it

* 3D viewer: the **World model** button, or press `W`.
* Gym: the **World model** button in the tools bar. The simulation pauses while the panel is open.
* If the build has no data, the panel says so and shows the command below (this is the empty state).

## Producing the data

```bash
python scripts/export_worldmodel_demo.py   (default: sweep_main + sweep_sup; sweep_h6 skipped as confounded) \
    --src /path/to/results_exploratory/worldmodel --out demos/common/worldmodel
python scripts/build_site.py
```

The exporter only reads the result files. It writes `demos/common/worldmodel/<tag>__<cond>_s<seed>.js`
and `index.js`. `gates_passed` is true only if the results markdown literally contains
`gates_passed: true`; today it never does.

## Schema `falsesafe.worldmodel_demo/v1`

Each file is `(window.FSB_WM=window.FSB_WM||[]).push({...});`. `index.js` sets `window.FSB_WM_FILES=[...]`.

| field | meaning |
|---|---|
| `schema` | `"falsesafe.worldmodel_demo/v1"` |
| `id`, `tag`, `condition`, `condition_text`, `seed` | the dataset identity |
| `status` | `{exploratory:true, dev_seeds:true, gates_passed:false, label}` |
| `provenance` | the source paths and the git commit of the producing tree |
| `gap` | `{axis:0, lo, hi, source}`: the hidden-physics band in position (parsed read-only from `data.py`) |
| `metrics.full/enc/dyn` | FSV, FUV, gapFSV, outFSV, cert_frac on the **full** evaluation grid (Python) |
| `breakdown` | `{encoder, dynamics, margin, total}`: these sum to `total` |
| `curves`, `diagnostics`, `privileged` | numeric-only copies of the source JSON |
| `closed_loop` | the per-arm crash rate and interventions, if present |
| `state` | `{n:[nx,nv], lo, hi, order:"x-major"}`: the display grid (stride 2 → 101²) |
| `grids.vstar/v_latent/v_enc` | `{b64, scale}` int8. `value ≈ q*scale`, and the quantiser **preserves sign**, so the classes are exact |
| `grids.inhull` | uint8 mask: the state is inside the training-data hull |
| `display_counts` | TS/FS/FU/TU counts and FSV/FUV on the display grid (they can differ slightly from `metrics`) |
| `latent` | `dz`, `lo/hi`, `points` (uint16 coordinates at stride 4, with their `class_b64`), and `V0` (int8 latent value grid) |
| `samples` | up to 11 example 32×32 uint8 camera images with their true states |

Classes: TS = true safe, FS = false-safe (learned ≥ 0, exact < 0), FU = false-unsafe, TU = true unsafe.

Tests: `tests/test_worldmodel_demo.py` checks the exporter, sign preservation, Python/JS class-count parity
(using node) and the honesty labels.
