import array
import copy
import io
import subprocess
import sys
from pathlib import Path

import cairosvg
import chess
import pytest
from PIL import Image

import chess_video.renderer as renderer_module
from chess_video.renderer import Renderer, _face, _face_base, render_frame, render_video, write_showcase


def timeline(**overrides):
    data = {
        "version": 1,
        "width": 240,
        "height": 426,
        "fps": 10,
        "duration": 4.0,
        "initial_fen": chess.STARTING_FEN,
        "events": [],
    }
    data.update(overrides)
    return data


def opening_moves():
    board = chess.Board()
    events = []
    for uci, start, end in (("e2e4", 0.8, 1.5), ("e7e5", 2.0, 2.7)):
        event = {"kind": "move", "start": start, "end": end, "uci": uci, "before_fen": board.fen()}
        board.push_uci(uci)
        event["fen"] = board.fen()
        events.append(event)
    events.extend([
        {"kind": "arrow", "start": 2.9, "end": 4.0, "from_square": "d8", "to_square": "h4",
         "automatic": True, "tactical_group": "threat"},
        {"kind": "zoom", "start": 0.1, "end": 0.7, "square": "e2", "scale": 1.3},
    ])
    return events


def test_default_presenter_preserves_explicit_male_frames():
    data = timeline(events=opening_moves())
    original = copy.deepcopy(data)
    with Renderer(data) as default, Renderer(dict(data, presenter="male")) as male:
        assert default.presenter == "male"
        for time in (0.0, 1.0, 2.3, 3.1):
            assert default.frame(time).tobytes() == male.frame(time).tobytes()
    assert data == original
    assert _face("neutral", False, 0, 600, 480).tobytes() == _face(
        "neutral", False, 0, 600, 480, presenter="male"
    ).tobytes()


def test_male_base_remains_original_asset():
    png = cairosvg.svg2png(
        url=str(renderer_module._ASSETS / "face.svg"), output_width=600, output_height=480
    )
    original = Image.open(io.BytesIO(png)).convert("RGBA")
    assert _face_base().tobytes() == original.tobytes()
    assert _face_base(presenter="male").tobytes() == original.tobytes()


@pytest.mark.parametrize("presenter", ["male", "female"])
@pytest.mark.parametrize("orientation", ["white", "black"])
@pytest.mark.parametrize("player_color", ["white", "black"])
def test_presenter_is_independent_of_orientation_and_player_color(presenter, orientation, player_color):
    data = timeline(
        presenter=presenter, orientation=orientation, player_color=player_color, events=opening_moves()
    )
    with Renderer(data) as renderer, Renderer(dict(data, presenter="male")) as male:
        assert renderer.presenter == presenter
        assert renderer.player_color == (player_color == "white")
        assert (renderer.square_center("a1")[1] > renderer.square_center("a8")[1]) == (orientation == "white")
        assert renderer._cursor_keys == male._cursor_keys
        assert renderer._camera_sequences == male._camera_sequences
        for time in (0.0, 1.0, 2.3, 3.1):
            frame = renderer.frame(time)
            expected = male.frame(time)
            assert frame.crop((0, 0, renderer.width, renderer.face_y)).tobytes() == expected.crop(
                (0, 0, male.width, male.face_y)
            ).tobytes()
            face = frame.crop((0, renderer.face_y, renderer.width, renderer.height)).tobytes()
            male_face = expected.crop((0, male.face_y, male.width, male.height)).tobytes()
            assert (face == male_face) == (presenter == "male")


@pytest.mark.parametrize("presenter", [None, "", "Female", "other", "../face", 1, True, [], {}, {"female"}])
@pytest.mark.parametrize("entrypoint", ["renderer", "frame", "video"])
def test_invalid_presenter_is_rejected_before_audio_or_resources(monkeypatch, tmp_path, presenter, entrypoint):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid presenter reached audio or resource work")

    monkeypatch.setattr(Renderer, "_load_audio", forbidden)
    monkeypatch.setattr(renderer_module, "_wood_board", forbidden)
    monkeypatch.setattr(renderer_module, "_face_base", forbidden)
    monkeypatch.setattr(renderer_module.subprocess, "run", forbidden)
    data = timeline(presenter=presenter, audio_path=str(tmp_path / "absent.wav"))
    output = tmp_path / "must-not-exist" / "video.mp4"
    with pytest.raises(ValueError, match="presenter.*male.*female"):
        if entrypoint == "renderer":
            Renderer(data)
        elif entrypoint == "frame":
            render_frame(data, 0)
        else:
            render_video(data, output)
    assert not output.parent.exists()


def test_alternating_presenter_caches_remain_distinct():
    _face_base.cache_clear()
    _face.cache_clear()
    bases = {profile: _face_base(presenter=profile).tobytes() for profile in ("male", "female")}
    faces = {
        profile: _face("thinking", False, 4, 240, 192, presenter=profile).tobytes()
        for profile in ("male", "female")
    }
    assert bases["male"] != bases["female"]
    assert faces["male"] != faces["female"]
    for profile in ("female", "male", "female", "male"):
        assert _face_base(presenter=profile).tobytes() == bases[profile]
        assert _face("thinking", False, 4, 240, 192, presenter=profile).tobytes() == faces[profile]
    assert _face_base.cache_info().maxsize == 2
    assert _face_base.cache_info().currsize == 2


def test_female_asset_has_distinct_hair_face_and_outfit():
    male = _face_base(presenter="male")
    female = _face_base(presenter="female")
    assert female.size == male.size == (600, 480)
    for region in ((190, 80, 400, 190), (240, 240, 360, 335), (100, 355, 510, 480)):
        assert female.crop(region).tobytes() != male.crop(region).tobytes()


def test_female_emotions_have_distinct_brows_and_mouths():
    emotions = ("neutral", "worried", "thinking", "surprised", "happy")
    faces = [_face(emotion, False, 0, 600, 480, presenter="female") for emotion in emotions]
    assert len({face.tobytes() for face in faces}) == len(emotions)
    assert len({face.crop((240, 182, 365, 213)).tobytes() for face in faces}) == len(emotions)
    assert len({face.crop((270, 272, 332, 330)).tobytes() for face in faces}) == len(emotions)


@pytest.mark.parametrize("emotion", ["neutral", "worried", "thinking", "surprised", "happy"])
def test_female_panel_routes_explicit_emotions_to_female_rig(emotion):
    events = [{"kind": "face", "start": 0.0, "end": 4.0, "emotion": emotion}]
    with Renderer(timeline(presenter="female", events=events)) as renderer:
        height = renderer.height - renderer.face_y
        assert renderer._face_panel(1.0).tobytes() == _face(
            emotion, False, 0, renderer.width, height, presenter="female"
        ).tobytes()


@pytest.mark.parametrize("emotion", ["neutral", "worried", "thinking", "surprised", "happy"])
def test_female_blink_and_mouth_share_the_original_rig_coordinates(emotion):
    still = _face(emotion, False, 0, 600, 480, presenter="female")
    blinking = _face(emotion, True, 0, 600, 480, presenter="female")
    speaking = _face(emotion, False, 7, 600, 480, presenter="female")
    for cx in (264, 338):
        region = (cx - 19, 209, cx + 19, 242)
        assert still.crop(region).tobytes() != blinking.crop(region).tobytes()
        assert still.crop(region).tobytes() == speaking.crop(region).tobytes()
    mouth = (270, 273, 332, 330)
    assert still.crop(mouth).tobytes() != speaking.crop(mouth).tobytes()
    assert still.crop(mouth).tobytes() == blinking.crop(mouth).tobytes()


def test_female_panel_uses_decoded_rms_mouth_and_timed_blinks(monkeypatch, tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"decoder-fixture")
    samples = array.array("h", [0] * 160 + [12000] * 160)
    if sys.byteorder != "little":
        samples.byteswap()

    def decoded(command, **kwargs):
        assert str(audio) in command
        return subprocess.CompletedProcess(command, 0, stdout=samples.tobytes(), stderr=b"")

    monkeypatch.setattr(renderer_module.subprocess, "run", decoded)
    with Renderer(timeline(presenter="female", audio_path=str(audio))) as renderer:
        height = renderer.height - renderer.face_y
        assert renderer._mouth(0.0) == 0
        assert renderer._mouth(0.02) == 7
        assert renderer._mouth(0.04) == 0
        assert renderer._face_panel(0.0).tobytes() == _face(
            "neutral", False, 0, renderer.width, height, presenter="female"
        ).tobytes()
        assert renderer._face_panel(0.02).tobytes() == _face(
            "neutral", False, 7, renderer.width, height, presenter="female"
        ).tobytes()
        assert renderer._face_panel(3.65).tobytes() == _face(
            "neutral", True, 0, renderer.width, height, presenter="female"
        ).tobytes()


def test_female_mate_reaction_preserves_zoom_transform():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(uci)
    with Renderer(timeline(presenter="female", initial_fen=board.fen())) as renderer:
        time = 2.0
        height = renderer.height - renderer.face_y
        face = _face("happy", False, 0, renderer.width, height, presenter="female")
        assert renderer._reaction(time) == 1.0
        scale = 1.25
        expected = face.transform(
            face.size,
            Image.Transform.AFFINE,
            (1 / scale, 0, (renderer.width - renderer.width / scale) / 2,
             0, 1 / scale, (height - height / scale) * 0.35),
            resample=Image.Resampling.BICUBIC,
        )
        assert renderer._face_panel(time).tobytes() == expected.tobytes()
        assert expected.tobytes() != face.tobytes()


def test_render_frame_cache_separates_presenters():
    male = render_frame(timeline(), 1.0).tobytes()
    female = render_frame(timeline(presenter="female"), 1.0).tobytes()
    assert male != female
    assert render_frame(timeline(), 1.0).tobytes() == male
    assert render_frame(timeline(presenter="female"), 1.0).tobytes() == female


@pytest.mark.parametrize("presenter", ["male", "female"])
def test_render_video_passes_presenter_to_renderer(monkeypatch, tmp_path, presenter):
    profiles = []

    def encode(renderer, output):
        profiles.append(renderer.presenter)
        assert renderer._face_panel(0).size == (renderer.width, renderer.height - renderer.face_y)
        output.write_bytes(b"encoded-fixture")

    monkeypatch.setattr(renderer_module, "_encode", encode)
    output = tmp_path / "video.mp4"
    assert render_video(timeline(presenter=presenter), output) == output
    assert profiles == [presenter]
    assert output.read_bytes() == b"encoded-fixture"


@pytest.mark.parametrize("presenter", [None, "male", "female"])
def test_showcase_propagates_presenter_to_every_mini_timeline(monkeypatch, tmp_path, presenter):
    timelines = []

    def render_segment(data, output):
        timelines.append(copy.deepcopy(data))
        Path(output).write_bytes(b"segment-fixture")
        return Path(output)

    def assemble(command, **kwargs):
        Path(command[-1]).write_bytes(b"showcase-fixture")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(renderer_module, "_showcase_sound", lambda *args: None)
    monkeypatch.setattr(renderer_module, "render_video", render_segment)
    monkeypatch.setattr(renderer_module.subprocess, "run", assemble)
    output = tmp_path / "showcase.mp4"
    options = {} if presenter is None else {"presenter": presenter}
    assert write_showcase(output, width=240, height=426, fps=10, **options) == output
    assert len(timelines) == 5
    assert {data["presenter"] for data in timelines} == {presenter or "male"}
    assert {data["orientation"] for data in timelines} == {"white", "black"}
    assert output.read_bytes() == b"showcase-fixture"


@pytest.mark.parametrize("presenter", [None, "Female", [], {}])
def test_showcase_rejects_invalid_presenter_before_creating_resources(monkeypatch, tmp_path, presenter):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid showcase presenter reached resource work")

    monkeypatch.setattr(renderer_module, "_showcase_sound", forbidden)
    monkeypatch.setattr(renderer_module, "render_video", forbidden)
    output = tmp_path / "must-not-exist" / "showcase.mp4"
    with pytest.raises(ValueError, match="presenter.*male.*female"):
        write_showcase(output, presenter=presenter)
    assert not output.parent.exists()
