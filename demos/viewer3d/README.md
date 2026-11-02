# 3D viewer (three.js r147, vendored — opens offline from file://)

`index.html` renders framework exports (`lcert.framework`, schema `falsesafe.framework/v1`):
V_true (HJ ground truth) and V_hat (certifier) as 3D surfaces over the first two state axes,
false-safe cells in magenta on the surface and on the error floor, the gap box, zero level sets,
and an exported true-dynamics rollout (marker on the surface + robot in the world view).
Sliders scrub the **precomputed** sweep (no live approximation: every number is read from the JSON;
the metrics panel cites the exact `cases[i].metrics` source). 3D envs (Dubins) show heading slices.

Data: `data/bundle.js` is regenerated on every `python -m lcert.framework run ...` from all
`data/*.json`; you can also drag-drop any exported JSON. Vendored: `vendor/three.min.js`,
`vendor/OrbitControls.js` (MIT, see `vendor/THREE_LICENSE`).
