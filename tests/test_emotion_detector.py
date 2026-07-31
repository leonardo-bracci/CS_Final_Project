"""Unit tests for the pure-logic functions of the emotion detector.

get_current_scene() and get_current_stimulus() are tested against a fixed,
hand-built sequence passed explicitly, rather than the real module-level
video_sequence (which is built from whatever images happen to be in
samples/stimuli/<scene>/ on disk). This keeps the tests deterministic and
independent of how many OASIS images are present locally, and means they
still run correctly on a machine/CI environment with no stimuli folder at
all - see build_video_sequence()'s own docstring in emotion_detector.py for
the same reasoning.
"""

from pathlib import Path

from src import emotion_detector as ed

# A small fixed schedule mirroring the real four-scene structure, but with
# round numbers and dummy paths so test expectations are easy to read and
# don't depend on real image counts. Three images per scene at 5s each:
#   threatening scene: 0-15
#   joyful scene:      15-30
#   sad scene:         30-45
#   relaxing scene:    45-60
def _make_sequence():
    sequence = []
    elapsed = 0.0
    for label in ["threatening scene", "joyful scene", "sad scene", "relaxing scene"]:
        for i in range(3):
            start, end = elapsed, elapsed + 5
            sequence.append((start, end, label, Path(f"/fake/{label}/{i}.jpg")))
            elapsed = end
    return sequence


class TestBuildVideoSequence:

    def test_builds_expected_number_of_entries(self, tmp_path):
        """One entry per image file found, across all four scene folders."""
        for scene, folder in ed.SCENE_FOLDERS.items():
            scene_dir = tmp_path / folder
            scene_dir.mkdir()
            (scene_dir / "a.jpg").touch()
            (scene_dir / "b.png").touch()

        sequence = ed.build_video_sequence(seconds_per_image=2, stimuli_root=tmp_path)
        assert len(sequence) == 8  # 4 scenes x 2 images

    def test_scene_duration_scales_with_image_count(self, tmp_path):
        scene_dir = tmp_path / ed.SCENE_FOLDERS["threatening scene"]
        scene_dir.mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (scene_dir / name).touch()
        (tmp_path / ed.SCENE_FOLDERS["joyful scene"]).mkdir()  # empty, produces a warning not a crash
        (tmp_path / ed.SCENE_FOLDERS["sad scene"]).mkdir()
        (tmp_path / ed.SCENE_FOLDERS["relaxing scene"]).mkdir()

        sequence = ed.build_video_sequence(seconds_per_image=4, stimuli_root=tmp_path)
        threatening_entries = [s for s in sequence if s[2] == "threatening scene"]
        assert len(threatening_entries) == 3
        assert threatening_entries[0][0] == 0
        assert threatening_entries[-1][1] == 12  # 3 images x 4s

    def test_missing_stimuli_root_returns_empty_sequence(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert ed.build_video_sequence(stimuli_root=missing) == []

    def test_non_image_files_are_ignored(self, tmp_path):
        scene_dir = tmp_path / ed.SCENE_FOLDERS["threatening scene"]
        scene_dir.mkdir()
        (scene_dir / "a.jpg").touch()
        (scene_dir / "notes.txt").touch()
        for label in ["joyful scene", "sad scene", "relaxing scene"]:
            (tmp_path / ed.SCENE_FOLDERS[label]).mkdir()

        sequence = ed.build_video_sequence(stimuli_root=tmp_path)
        assert len(sequence) == 1


class TestGetCurrentScene:

    def test_start_of_first_scene(self):
        assert ed.get_current_scene(0, sequence=_make_sequence()) == "threatening scene"

    def test_middle_of_a_scene(self):
        assert ed.get_current_scene(20, sequence=_make_sequence()) == "joyful scene"

    def test_boundary_is_exclusive(self):
        assert ed.get_current_scene(15, sequence=_make_sequence()) == "joyful scene"

    def test_after_last_scene_returns_end(self):
        assert ed.get_current_scene(60, sequence=_make_sequence()) == "end"

    def test_defaults_to_module_level_sequence_when_none_given(self, monkeypatch):
        """Calling without a sequence argument should fall back to the
        module-level video_sequence, not raise - this is what run_session()
        relies on in real use."""
        monkeypatch.setattr(ed, "video_sequence", _make_sequence())
        assert ed.get_current_scene(0) == "threatening scene"


class TestGetCurrentStimulus:

    def test_returns_label_and_image_path(self):
        sequence = _make_sequence()
        label, path = ed.get_current_stimulus(0, sequence=sequence)
        assert label == "threatening scene"
        assert path == Path("/fake/threatening scene/0.jpg")

    def test_returns_correct_image_within_scene(self):
        sequence = _make_sequence()
        # second image of the threatening scene: elapsed 5-10
        label, path = ed.get_current_stimulus(7, sequence=sequence)
        assert label == "threatening scene"
        assert path == Path("/fake/threatening scene/1.jpg")

    def test_after_last_scene_returns_none_none(self):
        label, path = ed.get_current_stimulus(60, sequence=_make_sequence())
        assert label is None
        assert path is None


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


class TestComposeFrame:
    """compose_frame() itself needs real numpy/cv2 image arrays and is
    exercised via manual/feature testing (see the Implementation chapter's
    account of the live run), not here - these tests stick to the
    deterministic label/geometry logic that doesn't need a display or a
    webcam."""

    def test_pip_position_bottom_right_default(self):
        x, y = ed._pip_position(960, 720, 160, 120, "bottom-right", 15)
        assert x == 960 - 160 - 15
        assert y == 720 - 120 - 15

    def test_pip_position_top_left(self):
        x, y = ed._pip_position(960, 720, 160, 120, "top-left", 15)
        assert (x, y) == (15, 15)
