"""First-run gym copy has to explain the idea without jargon."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "demos/gym/index.html").read_text(encoding="utf-8")
JS = (ROOT / "demos/gym/gym.js").read_text(encoding="utf-8")


def test_story_overlay_has_the_three_beats():
    assert "A safety guard that can be wrong" in PAGE
    assert "incomplete data" in PAGE
    assert "false-safe" in PAGE
    assert "Watch it fail" in PAGE
    assert 'id="startDemo"' in PAGE
    assert 'id="helpSkip"' in PAGE
    assert 'id="tyoCode"' in PAGE


def test_sidebar_leads_with_english():
    assert "The idea, in three lines" in PAGE
    assert "id=\"sceneStory\"" in PAGE
    assert "id=\"metPlain\"" in PAGE
    assert "Which guard is in charge?" in PAGE


def test_hud_uses_coach_not_just_symbols():
    assert "function coachLine" in JS
    assert "TRAP. The guard still says SAFE" in JS
    assert "function friendlyCert" in JS
    assert "Perfect guard" in JS
