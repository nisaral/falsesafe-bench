"""World-model ("learned from pixels") demo exporter + JS panel parity.

* scripts/export_worldmodel_demo.py on a synthetic result dir (no dependency on the other worktree):
  schema, honesty status, sign-preserving quantisation, breakdown sums, class counts.
* node: demos/common/worldmodel.js decodes the same grids and gets the same class counts as Python.
* optional: the real sweep_main / sweep_sup tags (skipped if /workspace/fsb-wm results are absent).
* the exported demo files shipped in demos/common/worldmodel/ are labelled exploratory / gates not passed.
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
_spec = importlib.util.spec_from_file_location("export_wm", ROOT / "scripts" / "export_worldmodel_demo.py")
X = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(X)
REAL = Path("/workspace/fsb-wm/results_exploratory/worldmodel/probe_llf_base")


def _fake_tag(root: Path, n=41, cond="QA", seed=3):
    d = root / "worldmodel" / "faketag"
    (d / "npz").mkdir(parents=True)
    (d / "closed_loop").mkdir()
    p, v = np.meshgrid(np.linspace(-1, 3, n), np.linspace(-2, 2, n), indexing="ij")
    Xe = np.stack([p.ravel(), v.ravel()], 1)
    vstar = (p + 0.5 * v - 0.3).ravel()                     # exact-ish value
    vlat = vstar + 0.25 * np.exp(-((p - 0.7) ** 2) / 0.05).ravel()  # optimistic inside the band -> false-safe
    venc = vstar + 0.1 * np.exp(-((p - 0.7) ** 2) / 0.05).ravel()
    vlat[5] = -1e-4  # tiny negative must stay negative after quantisation
    Zx = np.stack([p.ravel() * 2, v.ravel() - p.ravel()], 1)
    np.savez(d / "npz" / f"{cond}_s{seed}.npz", Xe=Xe, Vstar=vstar, v0x=vlat, venc_x=venc, Zx=Zx,
             V0=np.random.default_rng(0).normal(size=(21, 21)), lo=Zx.min(0), hi=Zx.max(0),
             inhull=(np.abs(v.ravel()) < 1.8).astype(float), sample_imgs=np.random.default_rng(1).random((3, 8, 8)),
             sample_states=Xe[:3])
    m = lambda f: {"FSV": f, "FUV": 0.01, "gapFSV": f / 2, "outFSV": f / 2, "gapFUV": 0, "cert_frac": 0.7}  # noqa: E731
    js = {"condition": cond, "seed": seed, "cfg": {"dz": 2}, "exploratory": True,
          "decomposition": {"V_enc": m(0.05), "V_dyn": m(0.06), "V_full": {"knob": 0.0, **m(0.07)}, "FSV_enc": 0.05,
                            "FSV_dyn_term": 0.01, "FSV_margin_term": 0.01, "gap_enc": 0.02, "gap_dyn_term": 0, "gap_margin_term": 0},
          "curves": {"E": [{"knob": 0.0, **m(0.07)}, {"knob": 1.0, **m(0.05)}]}, "fiber": {"FSV_fib": 0.1, "note": "x"}}
    (d / f"{cond}_s{seed}.json").write_text(json.dumps(js))
    (d / "closed_loop" / f"{cond}_s{seed}_cl_s0.json").write_text(json.dumps(
        {"seed": 0, "arms": {"V_full": {"crash_rate": 0.1, "mean_interventions": 0.4}}}))
    return root / "worldmodel", vstar.reshape(n, n), vlat.reshape(n, n)


def test_q8_preserves_sign():
    v = np.array([-3.0, -1e-9, 0.0, 1e-9, 2.0, -0.004, 0.004])
    q, s = X.q8(v)
    assert ((q >= 0) == (v >= 0)).all()
    assert np.max(np.abs(q * s - v)) <= s + 1e-12


def test_export_synthetic(tmp_path):
    src, vstar, vlat = _fake_tag(tmp_path)
    out = tmp_path / "out"
    assert X.main(["--tag", "faketag", "--src", str(src), "--out", str(out)]) == 0
    obj = X.read_js(out / "faketag__QA_s3.js")
    idx = (out / "index.js").read_text()
    assert "faketag__QA_s3" in idx
    assert obj["schema"] == "falsesafe.worldmodel_demo/v1"
    assert obj["status"]["exploratory"] and obj["status"]["dev_seeds"] and obj["status"]["gates_passed"] is False
    assert "gates not passed" in obj["status"]["label"]
    b = obj["breakdown"]
    assert b["encoder"] + b["dynamics"] + b["margin"] == pytest.approx(b["total"])
    assert obj["closed_loop"]["arms"]["V_full"]["crash_rate"] == 0.1
    assert obj["diagnostics"]["fiber"] == {"FSV_fib": 0.1}          # strings dropped
    st = obj["state"]
    assert st["n"] == [21, 21] and st["lo"] == [-1.0, -2.0] and st["hi"] == [3.0, 2.0]
    # classes on the displayed (stride-2) grid equal the classes of the raw values
    raw = X.classes(vstar[::2, ::2], vlat[::2, ::2])
    cnt = np.bincount(raw.ravel(), minlength=4)
    dc = obj["display_counts"]
    assert [dc["TS"], dc["FS"], dc["FU"], dc["TU"]] == cnt.tolist()
    assert dc["FS"] > 0
    assert obj["gap"]["lo"] == pytest.approx(0.4) and obj["gap"]["hi"] == pytest.approx(1.0)


def test_classes_definition():
    c = X.classes(np.array([1, -1, 1, -1.0]), np.array([1, 1, -1, -1.0]))
    assert c.tolist() == [0, 1, 2, 3]


def _node_decode(js_file: Path):
    code = f"""
const W=require({json.dumps(str(ROOT / 'demos/common/worldmodel.js'))});
const fs=require('fs');const t=fs.readFileSync({json.dumps(str(js_file))},'utf8');
const d=JSON.parse(t.slice(t.indexOf('.push(')+6,t.lastIndexOf(');')));
const o=W.decode(d);
console.log(JSON.stringify({{counts:o.counts,enc:o.counts_enc,vs:Array.from(o.vstar.slice(0,50)),
  nz:o.zpts?o.zpts.length:0,ncls:o.zcls?o.zcls.length:0}}));"""
    r = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_panel_parity(tmp_path):
    src, _, _ = _fake_tag(tmp_path)
    out = tmp_path / "out"
    X.main(["--tag", "faketag", "--src", str(src), "--out", str(out)])
    obj = X.read_js(out / "faketag__QA_s3.js")
    j = _node_decode(out / "faketag__QA_s3.js")
    dc = obj["display_counts"]
    for k in ("TS", "FS", "FU", "TU", "n"):
        assert j["counts"][k] == dc[k]
    # dequantised values agree with Python
    q = np.frombuffer(__import__("base64").b64decode(obj["grids"]["vstar"]["b64"]), np.int8)[:50] * obj["grids"]["vstar"]["scale"]
    assert np.allclose(j["vs"], q, atol=1e-5)
    assert j["nz"] == 2 * j["ncls"] == 2 * obj["latent"]["points"]["n"]


DOC = Path("/workspace/worldmodel-dev-results.md")
REAL_MAIN = REAL.parent / "sweep_main"


def _index(out: Path):
    t = (out / "index.js").read_text()
    files = json.loads(t.split("window.FSB_WM_FILES=", 1)[1].split(";\n", 1)[0])
    meta = json.loads(t.split("window.FSB_WM_META=", 1)[1].rsplit(";", 1)[0])
    return files, meta


@pytest.mark.skipif(not REAL_MAIN.exists() or not DOC.exists(), reason="world-model dev results not present")
def test_real_sweep_main_and_sup(tmp_path):
    out = tmp_path / "out"
    assert X.main(["--src", str(REAL.parent), "--out", str(out)]) == 0
    files, meta = _index(out)
    main = [f for f in files if f["arm"] == "main"]
    sup = [f for f in files if f["arm"] == "sup"]
    assert sorted((f["condition"], f["seed"]) for f in main) == [(c, s) for c in ("QA", "QB", "QC", "QD") for s in range(10)]
    assert len(sup) == 15 and {f["seed"] for f in sup} == set(range(5))
    assert not any("h6" in f["file"] for f in files)
    assert [e["tag"] for e in meta["excluded"]] == ["sweep_h6"] and "confounded" in meta["excluded"][0]["reason"]
    assert meta["gates_passed"] is False and "gates not passed" in meta["label"]
    # every metric equals the source JSON
    for f in main + sup:
        d = json.loads((REAL.parent / f["tag"] / f"{f['condition']}_s{f['seed']}.json").read_text())
        assert f["FSV"] == pytest.approx(d["decomposition"]["V_full"]["FSV"])
        assert f["fold"] == pytest.approx(d["geometry"]["fold_frac_C25"])
        assert f["spearman"] == pytest.approx(d["geometry"]["spearman_state_latent"])
        assert f["tuning_seed"] == (f["seed"] == 0)
    # QA FSV per seed matches the doc table
    doc_qa = [.0020, .0532, .0043, .0572, .0677, .0437, .0204, .0003, .0927, .0163]
    qa = sorted((f for f in main if f["condition"] == "QA"), key=lambda f: f["seed"])
    assert [round(f["FSV"], 4) for f in qa] == doc_qa
    # collapsed seeds named in the doc
    assert {(f["condition"], f["seed"]) for f in main if f["collapsed"]} == {("QB", 1), ("QB", 2), ("QC", 0)}
    assert {(f["condition"], f["seed"]) for f in main if f["folded"]} == {("QB", 0), ("QB", 1), ("QC", 1), ("QD", 8)}
    # gates: recomputed from data == parsed from doc == the doc's literal sentence
    assert meta["gates_per_seed"]["main"]["QA"] == meta["doc_gates"]["main"]["per_seed"]
    assert meta["gates_per_seed"]["sup"]["QA"] == meta["doc_gates"]["sup"]["per_seed"]
    txt = DOC.read_text()
    assert "G1 5/10, G4 0/10 (FSV_enc part only: 3/10), C2.5 2/10" in txt
    g = meta["doc_gates"]["main"]["per_seed"]
    assert (g["G1"]["pass"], g["G4"]["pass"], g["G4_FSV_enc_part"]["pass"], g["C2.5"]["pass"]) == (5, 0, 3, 2)
    for arm in ("main", "sup"):
        assert all(v["pass"] is False for v in meta["doc_gates"][arm]["seed_mean"].values())
    assert meta["doc_gates"]["main"]["seed_mean"]["G1"]["FSV"] == pytest.approx(0.03577386698349051)
    # size budget (lazy-loaded, one file per run)
    total = sum(p.stat().st_size for p in out.glob("*.js"))
    assert total < 14e6 and max(p.stat().st_size for p in out.glob("*__*.js")) < 200e3
    # full-grid FSV recomputed from the npz equals the stored metric; display grid tracks it
    for f in qa[:3]:
        obj = X.read_js(out / (f["file"] + ".js"))
        z = np.load(REAL_MAIN / "npz" / f"QA_s{f['seed']}.npz")
        assert np.mean((z["v0x"] >= 0) & (z["Vstar"] < 0)) == pytest.approx(obj["metrics"]["full"]["FSV"])
        assert abs(obj["display_counts"]["FSV"] - obj["metrics"]["full"]["FSV"]) < 0.01


@pytest.mark.skipif(NODE is None or not REAL_MAIN.exists(), reason="node or dev results missing")
def test_real_js_python_agreement(tmp_path):
    out = tmp_path / "out"
    X.main(["--src", str(REAL.parent), "--out", str(out)])
    files, _ = _index(out)
    for f in files[::7]:                        # a spread of conditions/seeds/arms incl. collapsed ones
        obj = X.read_js(out / (f["file"] + ".js"))
        j = _node_decode(out / (f["file"] + ".js"))
        for k in ("TS", "FS", "FU", "TU", "n"):
            assert j["counts"][k] == obj["display_counts"][k], (f["file"], k)
    # JS median used by the variability strip == numpy median
    qa = [f["FSV"] for f in files if f["arm"] == "main" and f["condition"] == "QA"]
    code = "console.log(require(%s).median(%s))" % (json.dumps(str(ROOT / "demos/common/worldmodel.js")), json.dumps(qa))
    r = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert float(r.stdout) == pytest.approx(float(np.median(qa)))


def test_shipped_demo_files_are_labelled():
    d = ROOT / "demos" / "common" / "worldmodel"
    files = sorted(d.glob("*__*.js"))
    if not files:
        pytest.skip("no exported world-model demo data (panel shows its empty state)")
    for f in files:
        obj = X.read_js(f)
        assert obj["schema"] == "falsesafe.worldmodel_demo/v1"
        assert obj["status"]["gates_passed"] is False and obj["status"]["exploratory"] is True
        assert "EXPLORATORY" in f.read_text()[:200]


def test_panel_wired_and_has_empty_state():
    js = (ROOT / "demos/common/worldmodel.js").read_text()
    assert "No world-model data in this build" in js and "gates not passed" in js
    for page in ("demos/gym/index.html", "demos/viewer3d/index.html"):
        h = (ROOT / page).read_text()
        assert "../common/worldmodel.js" in h and 'id="wmBtn"' in h


def test_panel_honesty_labels():
    js = (ROOT / "demos/common/worldmodel.js").read_text()
    assert js.count("exploratory · dev seeds · gates not passed") >= 1 and "header()" in js
    assert "tuning seed — not representative" in js
    assert "different encoder: trained with the true state" in js.lower()
    assert "The camera-based model works on some random seeds and fails on others." in js
    assert "certifies (almost) everything" in js or "calls (almost) every state safe" in js
    assert "none passed" in js and "Nothing here shows that the world model works" in js
    assert "!f.tuning_seed" in js                    # default view never opens on the tuning seed
    idx = ROOT / "demos/common/worldmodel/index.js"
    if idx.exists():
        files, meta = _index(idx.parent)
        assert meta["gates_passed"] is False
        assert not any("h6" in f["file"] for f in files)
        assert all(f["tuning_seed"] == (f["seed"] == 0) for f in files)
