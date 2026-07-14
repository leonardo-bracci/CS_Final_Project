"""Unit tests for the pure-logic functions of the emotion detector."""

import emotion_detector as ed


class TestGetCurrentScene:

    def test_start_of_first_scene(self):
        assert ed.get_current_scene(0) == "threatening scene"

    def test_middle_of_a_scene(self):
        assert ed.get_current_scene(7) == "joyful scene"

    def test_boundary_is_exclusive(self):
        assert ed.get_current_scene(5) == "joyful scene"

    def test_after_last_scene_returns_end(self):
        assert ed.get_current_scene(20) == "end"


class TestSummariseTimeline:

    def test_single_emotion(self):
        timeline = [(1.0, "joyful scene", "happy"), (2.0, "joyful scene", "happy")]
        assert ed.summarise_timeline(timeline) == {"joyful scene": ("happy", 2)}

    def test_picks_most_common(self):
        timeline = [
            (1.0, "sad scene", "sad"),
            (2.0, "sad scene", "sad"),
            (3.0, "sad scene", "neutral"),
        ]
        assert ed.summarise_timeline(timeline)["sad scene"] == ("sad", 2)

    def test_empty_timeline(self):
        assert ed.summarise_timeline([]) == {}