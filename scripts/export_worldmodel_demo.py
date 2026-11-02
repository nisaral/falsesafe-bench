#!/usr/bin/env python
"""Export latent world-model ("learned from pixels") dev results into the demo format.

Reads (READ ONLY) ``<src>/<tag>/{QA,QB,QC,QD}_s<seed>.json`` plus the matching ``npz/`` grids and the
optional ``closed_loop/<cond>_s<seed>_cl_s*.json`` written by the latent pipeline, and writes one
small JS file per (tag, condition, seed) to ``demos/common/worldmodel/`` plus an ``index.js``.
The JS files only push a JSON payload onto ``window.FSB_WM`` so they also work from ``file://``.

Schema ``falsesafe.worldmodel_demo/v1`` is documented in docs/WORLDMODEL_DEMO.md.

Honesty: these results are exploratory, from dev seeds, and the preregistered gates were NOT
passed.  The exporter always writes ``status.gates_passed = false`` unless
/workspace/worldmodel-dev-results.md exists AND contains the line ``gates_passed: true`` (it does not
today); there is no command-line flag to override that.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

SCHEMA = "falsesafe.worldmodel_demo/v1"
DEFAULT_SRC = "/workspace/fsb-wm/results_exploratory/worldmodel"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "demos", "common", "worldmodel")
RESULTS_NOTE = "/workspace/worldmodel-dev-results.md"
CONDITIONS = {
    "QA": "gap visible in the images, gap covered by training data",
    "QB": "gap visible in the images, gap transitions withheld from training",
    "QC": "gap hidden (not drawn), gap covered by training data",
    "QD": "gap hidden (not drawn), gap transitions withheld from training",
}
# Defaults mirror lcert/latent/data.py DI (pc=0.7, w=0.6 -> p in [0.4, 1.0]); parsed when available.
GAP_DEFAULT = (0.7, 0.6)
# Arms of the dev sweep (see /workspace/worldmodel-dev-results.md).  Never pooled with each other.
ARMS = {
    "sweep_main": {"arm": "main", "label": "camera-only encoder (main arm)",
                   "note": "unsupervised: the encoder learns from the images alone"},
    "sweep_sup": {"arm": "sup", "label": "different encoder: trained with the true state",
                  "note": "adds an auxiliary head trained on the privileged true state (lam_state=1); this is NOT "
                          "pixel-only learning and is never mixed with the main arm"},
    "probe_llf_base": {"arm": "probe", "label": "early probe run", "note": "pre-sweep probe"},
}
CONFOUNDED = {"sweep_h6": "single frame (main arm uses 3 frames) AND no blur, so the effect of blur is confounded "
                          "with the frame count; not shown"}
DEFAULT_TAGS = ["sweep_main", "sweep_sup"]
# Gate definitions, copied from fsb-wm scripts/aggregate_latent_dev.py (applied per seed here).
GATE_DEFS = {
    "G1": {"text": "few wrong answers: false-safe <= 3% and false-unsafe <= 3%",
           "f": lambda r: r["FSV"] <= .03 and r["FUV"] <= .03},
    "G4": {"text": "the latent keeps the order of states (Spearman >= 0.9) and the encoder adds <= 1% false-safe",
           "f": lambda r: r["spearman"] >= .9 and r["FSV_enc"] <= .01},
    "C2.5": {"text": "the latent map does not fold over (fold fraction <= 5%)", "f": lambda r: r["fold"] <= .05},
}
COLLAPSE_FSV = 0.25   # degenerate "certify everything" runs sit at FSV ~0.30 (cert_frac ~1)
FOLD_FLAG = 0.9
TUNING = {"seed": 0, "condition": "QA",
          "text": "tuning seed — not representative (encoder settings were chosen on QA seed 0)"}
METRIC_KEYS = ("FSV", "FUV", "gapFSV", "outFSV", "gapFUV", "cert_frac")


def q8(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Sign-preserving int8 quantisation: q >= 0 iff v >= 0 (so safe/unsafe classes are exact)."""
    v = np.asarray(v, dtype=np.float64)
    m = float(np.nanmax(np.abs(v))) if v.size else 1.0
    scale = (m / 127.0) if m > 0 else 1.0
    q = np.clip(np.rint(v / scale), -127, 127)
    q = np.where(v >= 0, np.maximum(q, 0), np.minimum(q, -1))
    return q.astype(np.int8), scale


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def classes(vstar: np.ndarray, vhat: np.ndarray) -> np.ndarray:
    """0 truly safe, 1 FALSE-SAFE (vhat>=0, vstar<0), 2 false-unsafe, 3 truly unsafe."""
    s, h = vstar >= 0, vhat >= 0
    return np.where(s & h, 0, np.where(~s & h, 1, np.where(s & ~h, 2, 3))).astype(np.int8)


def gate_status() -> bool:
    try:
        txt = open(RESULTS_NOTE).read()
    except OSError:
        return False
    return bool(re.search(r"^\s*gates_passed:\s*true\s*$", txt, re.M | re.I))


def read_gap(src_root: str) -> tuple[float, float, str]:
    """Parse pc and w from <repo>/src/lcert/latent/data.py (read only); fall back to defaults."""
    repo = os.path.abspath(os.path.join(src_root, "..", ".."))
    p = os.path.join(repo, "src", "lcert", "latent", "data.py")
    try:
        txt = open(p).read()
        pc = float(re.search(r"pc=([-0-9.]+)", txt).group(1))
        w = float(re.search(r"\bw=([-0-9.]+)", txt).group(1))
        return pc - w / 2, pc + w / 2, os.path.relpath(p, repo)
    except Exception:
        pc, w = GAP_DEFAULT
        return pc - w / 2, pc + w / 2, "default (lcert/latent/data.py DI)"


def git_head(src_root: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", src_root, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def metrics(d: dict | None) -> dict | None:
    if not isinstance(d, dict):
        return None
    return {k: float(d[k]) for k in METRIC_KEYS if k in d and d[k] is not None}


def numeric(d, depth=0):
    """Keep only finite numbers (and nested dicts of them) for the small diagnostics blocks."""
    if isinstance(d, dict) and depth < 2:
        out = {k: numeric(v, depth + 1) for k, v in d.items()}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(d, (int, float)) and not isinstance(d, bool) and np.isfinite(d):
        return float(d)
    return None


def export_one(json_path: str, tag: str, src_root: str, stride: int = 2) -> dict:
    d = json.load(open(json_path))
    base = os.path.basename(json_path)[:-5]  # e.g. QA_s0
    cond, seed = d.get("condition", base.split("_")[0]), int(d.get("seed", 0))
    npz_path = os.path.join(os.path.dirname(json_path), "npz", base + ".npz")
    out: dict = {
        "schema": SCHEMA, "id": f"{tag}__{base}", "tag": tag, "condition": cond, "seed": seed,
        "condition_text": CONDITIONS.get(cond.split("-")[0], cond),
        "status": {"exploratory": True, "dev_seeds": True, "gates_passed": gate_status(),
                   "label": "exploratory · dev seeds · gates not passed"},
        "provenance": {"source": os.path.relpath(json_path, os.path.dirname(os.path.dirname(src_root))),
                       "source_commit": git_head(src_root), "cfg": d.get("cfg", {}),
                       "n_train": d.get("n_train"), "vae_loss_final": d.get("vae_loss_final")},
    }
    if out["status"]["gates_passed"]:
        out["status"]["label"] = "exploratory · dev seeds"
    lo, hi, gsrc = read_gap(src_root)
    out["gap"] = {"axis": 0, "lo": lo, "hi": hi, "source": gsrc}
    dec = d.get("decomposition", {})
    full = metrics(dec.get("V_full")) or metrics((d.get("curves", {}).get("E") or [{}])[0])
    out["metrics"] = {"full": full, "enc": metrics(dec.get("V_enc")), "dyn": metrics(dec.get("V_dyn")),
                      "grid": "full eval grid (Python)"}
    out["breakdown"] = {
        "encoder": dec.get("FSV_enc"), "dynamics": dec.get("FSV_dyn_term"), "margin": dec.get("FSV_margin_term"),
        "total": (full or {}).get("FSV"),
        "gap": {"encoder": dec.get("gap_enc"), "dynamics": dec.get("gap_dyn_term"), "margin": dec.get("gap_margin_term")},
        "margin_head_rmse": dec.get("margin_head_rmse_eval"),
    }
    out["curves"] = {k: [{"knob": float(r.get("knob", i)), **(metrics(r) or {})} for i, r in enumerate(v)]
                     for k, v in (d.get("curves") or {}).items() if isinstance(v, list)}
    arm = ARMS.get(tag, {"arm": tag, "label": tag, "note": ""})
    geo, dec2 = d.get("geometry") or {}, d.get("decomposition") or {}
    fold, sp = geo.get("fold_frac_C25"), geo.get("spearman_state_latent")
    fsv = (full or {}).get("FSV")
    out["arm"] = {"tag": tag, **arm}
    out["seed_info"] = {"fold": fold, "spearman": sp, "FSV_enc": dec2.get("FSV_enc"),
                        "collapsed": bool(fsv is not None and fsv >= COLLAPSE_FSV),
                        "folded": bool(fold is not None and fold > FOLD_FLAG),
                        "tuning_seed": seed == TUNING["seed"],
                        "tuning_text": TUNING["text"] if seed == TUNING["seed"] else None}
    out["diagnostics"] = {k: numeric(d.get(k)) for k in ("fiber", "geometry", "encoder_ood", "calib") if d.get(k)}
    out["privileged"] = {k: metrics(v) for k, v in (d.get("privileged") or {}).items() if isinstance(v, dict)}
    cl = sorted(glob.glob(os.path.join(os.path.dirname(json_path), "closed_loop", base + "_cl_s*.json")))
    if cl:
        c = json.load(open(cl[0]))
        out["closed_loop"] = {"seed": c.get("seed"), "arms": {k: numeric(v) for k, v in (c.get("arms") or {}).items()}}
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        X = z["Xe"]
        n = int(round(np.sqrt(X.shape[0])))
        assert n * n == X.shape[0], "state grid must be square"
        xs, vs = X[::n, 0], X[:n, 1]
        if not (np.allclose(X[:n, 0], X[0, 0]) and np.allclose(X[:, 1].reshape(n, n), vs[None, :])):
            raise ValueError("unexpected Xe ordering (expected x-major, v-minor)")
        sl = (slice(None, None, stride), slice(None, None, stride))
        grid = lambda k: np.asarray(z[k], dtype=np.float64).reshape(n, n)[sl]  # noqa: E731
        vstar, vlat = grid("Vstar"), grid("v0x")
        venc = grid("venc_x") if "venc_x" in z else None
        m = vstar.shape[0]
        st = {"names": ["position p", "velocity v"], "lo": [float(xs[0]), float(vs[0])], "hi": [float(xs[-1]), float(vs[-1])],
              "n": [m, m], "order": "x-major (index = i*n1 + j, i over p, j over v)", "stride": stride, "full_n": [n, n]}
        out["state"] = st
        grids = {}
        for k, a in (("vstar", vstar), ("v_latent", vlat), ("v_enc", venc)):
            if a is None:
                continue
            q, s = q8(a)
            grids[k] = {"b64": b64(q.ravel()), "dtype": "int8", "scale": s}
        if "inhull" in z:
            grids["inhull"] = {"b64": b64(grid("inhull").ravel().astype(np.uint8)), "dtype": "uint8", "scale": 1}
        out["grids"] = grids
        cls = classes(q8(vstar)[0], q8(vlat)[0])
        cnt = np.bincount(cls.ravel(), minlength=4)
        out["display_counts"] = {"grid": f"{m}x{m} (every {stride}th point of the {n}x{n} state grid)" if stride > 1 else f"{m}x{m}", "n": int(cls.size),
                                 "TS": int(cnt[0]), "FS": int(cnt[1]), "FU": int(cnt[2]), "TU": int(cnt[3]),
                                 "FSV": float(cnt[1] / cls.size), "FUV": float(cnt[2] / cls.size)}
        # latent picture: where the (sub-sampled) state grid lands in the 2-D latent, plus the latent V0 grid
        Z = np.asarray(z["Zx"]).reshape(n, n, -1)[::stride * 2, ::stride * 2].reshape(-1, z["Zx"].shape[1])
        zl, zh = np.asarray(z["lo"], float), np.asarray(z["hi"], float)
        lat = {"dz": int(Z.shape[1]), "lo": zl.tolist(), "hi": zh.tolist()}
        if Z.shape[1] == 2:
            zq = np.clip(np.rint((Z - zl) / (zh - zl) * 65535), 0, 65535).astype(np.uint16)
            sub = (slice(None, None, stride * 2), slice(None, None, stride * 2))
            ccls = classes(np.asarray(z["Vstar"]).reshape(n, n)[sub], np.asarray(z["v0x"]).reshape(n, n)[sub])
            lat["points"] = {"b64": b64(zq.ravel()), "dtype": "uint16", "n": int(Z.shape[0]),
                             "class_b64": b64(ccls.ravel().astype(np.uint8))}
            if np.asarray(z["V0"]).ndim == 2:
                q, s = q8(z["V0"])
                lat["V0"] = {"b64": b64(q.ravel()), "dtype": "int8", "scale": s, "n": list(q.shape),
                             "order": "z0-major"}
        out["latent"] = lat
        if "sample_imgs" in z:
            im = np.clip(np.rint(np.asarray(z["sample_imgs"]) * 255), 0, 255).astype(np.uint8)
            out["samples"] = {"n": int(im.shape[0]), "h": int(im.shape[1]), "w": int(im.shape[2]), "b64": b64(im.ravel()),
                              "states": np.asarray(z["sample_states"], float).round(4).tolist()}
    return out


def write_js(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("// generated by scripts/export_worldmodel_demo.py; EXPLORATORY dev results, gates not passed\n")
        f.write("(window.FSB_WM=window.FSB_WM||[]).push(" + json.dumps(obj, separators=(",", ":")) + ");\n")


def read_js(path: str) -> dict:
    """Inverse of write_js (used by tests)."""
    txt = open(path).read()
    i = txt.index(".push(") + 6
    return json.loads(txt[i:txt.rindex(");")])


def doc_gates(path: str = RESULTS_NOTE) -> dict:
    """Parse per-seed QA gate counts and seed-mean gate blocks from the results doc, per sweep section."""
    try:
        txt = open(path).read()
    except OSError:
        return {}
    out = {}
    secs = re.split(r"^## (sweep_\w+) \(", txt, flags=re.M)
    for tag, body in zip(secs[1::2], secs[2::2]):
        r = {}
        m = re.search(r"Per-seed QA gate passes: G1 (\d+)/(\d+), G4 (\d+)/\d+ \(FSV_enc part only: (\d+)/\d+\), "
                      r"C2\.5 (\d+)/\d+", body)
        if m:
            g1, n, g4, enc, c25 = map(int, m.groups())
            r["per_seed"] = {"G1": {"pass": g1, "n": n}, "G4": {"pass": g4, "n": n}, "C2.5": {"pass": c25, "n": n},
                             "G4_FSV_enc_part": {"pass": enc, "n": n}}
        m = re.search(r"### Gates on seed means.*?```\s*(\{.*?\})\s*```", body, re.S)
        if m:
            try:
                g = json.loads(m.group(1))
                r["seed_mean"] = {k: g[k] for k in ("G1", "G4", "C2.5") if k in g}
            except ValueError:
                pass
        out[tag] = r
    return out


def seed_row(obj: dict) -> dict:
    f, si = obj["metrics"]["full"] or {}, obj["seed_info"]
    return {"FSV": f.get("FSV"), "FUV": f.get("FUV"), "cert_frac": f.get("cert_frac"), **{k: si[k] for k in
            ("fold", "spearman", "FSV_enc", "collapsed", "folded", "tuning_seed")}}


def gate_counts(rows: list[dict]) -> dict:
    ok = [r for r in rows if None not in (r["FSV"], r["FUV"], r["fold"], r["spearman"], r["FSV_enc"])]
    return {g: {"pass": int(sum(bool(G["f"](r)) for r in ok)), "n": len(ok)} for g, G in GATE_DEFS.items()} | \
        {"G4_FSV_enc_part": {"pass": int(sum(r["FSV_enc"] <= .01 for r in ok)), "n": len(ok)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", action="append", default=None, help="result tag(s); default sweep_main + sweep_sup")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args(argv)
    tags = a.tag or DEFAULT_TAGS
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, "*__*.js")):  # stale generated files only
        os.remove(old)
    entries, excluded, by = [], [], {}
    for tag in tags:
        if tag in CONFOUNDED:
            excluded.append({"tag": tag, "reason": CONFOUNDED[tag]})
            print(f"[skip] {tag}: confounded ({CONFOUNDED[tag]})", file=sys.stderr)
            continue
        files = sorted(glob.glob(os.path.join(a.src, tag, "Q*_s*.json")),
                       key=lambda p: (os.path.basename(p).split("_")[0], int(re.search(r"_s(\d+)", p).group(1))))
        if not files:
            print(f"[skip] no results for tag {tag!r} in {a.src}", file=sys.stderr)
            continue
        for jp in files:
            obj = export_one(jp, tag, a.src, a.stride)
            fn = obj["id"] + ".js"
            write_js(obj, os.path.join(a.out, fn))
            row = seed_row(obj)
            ent = {"file": obj["id"], "tag": tag, "arm": obj["arm"]["arm"], "condition": obj["condition"],
                   "seed": obj["seed"], "title": f"{tag} · {obj['condition']} seed {obj['seed']}",
                   "condition_text": obj["condition_text"], "has_grids": "grids" in obj, **row,
                   "kb": round(os.path.getsize(os.path.join(a.out, fn)) / 1024)}
            entries.append(ent)
            by.setdefault(obj["arm"]["arm"], {}).setdefault(obj["condition"], []).append(row)
            print(f"wrote {fn} ({ent['kb']} KB)")
    for tag in CONFOUNDED:
        if tag not in tags and os.path.isdir(os.path.join(a.src, tag)):
            excluded.append({"tag": tag, "reason": CONFOUNDED[tag]})
    dg = doc_gates()
    for t, v in dg.items():
        if t in ARMS and ARMS[t]["arm"] in by and "per_seed" in v:
            got = gate_counts(by[ARMS[t]["arm"]]["QA"])
            if got != v["per_seed"]:
                print(f"[WARN] {t}: per-seed QA gates from data {got} != doc {v['per_seed']}", file=sys.stderr)
    meta = {"schema": SCHEMA + "#index", "gates_passed": gate_status(),
            "label": "exploratory · dev seeds · gates not passed",
            "arms": {v["arm"]: {"tag": k, "label": v["label"], "note": v["note"]} for k, v in ARMS.items()
                     if any(e["tag"] == k for e in entries)},
            "gate_defs": {k: v["text"] for k, v in GATE_DEFS.items()},
            "gates_per_seed": {arm: {c: gate_counts(rows) for c, rows in cs.items()} for arm, cs in by.items()},
            "collapse_rule": f"collapsed = full-model false-safe >= {COLLAPSE_FSV} (certifies almost everything); "
                             f"folded = fold fraction > {FOLD_FLAG}",
            "doc_gates": {ARMS[t]["arm"]: v for t, v in doc_gates().items() if t in ARMS},
            "tuning": TUNING, "excluded": excluded,
            "source_note": "numbers from " + RESULTS_NOTE}
    with open(os.path.join(a.out, "index.js"), "w") as f:
        f.write("// generated by scripts/export_worldmodel_demo.py; EXPLORATORY dev results, gates not passed\n")
        f.write("window.FSB_WM_FILES=" + json.dumps(entries, separators=(",", ":")) + ";\n")
        f.write("window.FSB_WM_META=" + json.dumps(meta, separators=(",", ":")) + ";\n")
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
