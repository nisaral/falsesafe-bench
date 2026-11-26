# Roadmap — future work

Nothing on this page has been run. These are the registered or planned items the
current branch deliberately did not attempt. Each is one sentence; no results
exist for any of them.

## Dubins experiments (E1–E3, incl. H4)

The Dubins car has a ground-truth HJ solver and learned-model code
(`src/lcert/hj_dubins.py`, `hj_dubins_learned.py`), but no Dubins experiment
has been run, so H4 (separated vs blended uncertainty) remains untested.

## Latent world model

`PREREGISTRATION_v2.md` registers a latent world-model path on new confirmatory
seeds 3000-3029, but no encoder or latent training code exists yet and
`configs/latent_di.yaml` has not been created.

## PREREGISTRATION_v2 checklist

The remaining open items in `PREREGISTRATION_v2.md` are still unchecked: filling
the author and commit hash, creating `configs/latent_di.yaml`, implementing the
shared confirmatory guard with its coverage test, verifying the failure region
lies outside `G`, running gates G1–G3 on development seeds 0–9, and tagging
`prereg-v2` before touching seeds 3000-3029.

## Data-sampling variability

The n=30 P1 replicates share `split_seed=42`; a `--vary-split-seed` flag now
exists but has not been run, so a split-seed sensitivity study is future work.
