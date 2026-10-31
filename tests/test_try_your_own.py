"""The gym's "Try your own" snippet: the copy-paste code shown in demos/gym/index.html must be exactly
examples/framework/my_certifier.py, and that file must load as a plugin and build a working scenario."""
import dataclasses, html, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIP = ROOT / "examples/framework/my_certifier.py"


def test_snippet_in_gym_matches_example_file():
    page = (ROOT / "demos/gym/index.html").read_text()
    m = re.search(r'<pre id="tyoCode"[^>]*>(.*?)</pre>', page, re.S)
    assert m, "Try-your-own snippet missing from the gym help"
    assert html.unescape(m.group(1)).strip() == SNIP.read_text().strip()
    assert "docs/FRAMEWORK.md" in page and "--plugin examples/framework/my_certifier.py" in page


def test_snippet_runs_and_cushion_is_more_cautious():
    from lcert.framework.__main__ import _load_plugin
    from lcert.framework import gym
    _load_plugin(str(SNIP))
    s = gym.SCENARIOS["my_cart"]
    s = dataclasses.replace(s, env_params={**s.env_params, "grid_size": 41})
    doc = gym.build_scenario(s, verbose=False)
    naive, mine = (c["metrics"]["fsv"] for c in doc["cases"])
    assert doc["user_defined"] and mine < naive
