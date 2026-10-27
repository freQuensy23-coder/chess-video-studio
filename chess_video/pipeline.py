import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from chess_video.analysis import analyze_game
from chess_video.game import load_game, validate_game, validate_script
from chess_video.providers import generate_script, synthesize
from chess_video.renderer import render_video
from chess_video.timeline import compile_timeline
from chess_video.video_options import validate_video_options


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def generate_video(
    pgn: str,
    output_dir: str | Path,
    settings,
    *,
    width: int = 608,
    height: int = 1080,
    fps: int = 30,
    script_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    validate_video_options(width, height, fps)
    presenter = getattr(settings, 'presenter', 'male')
    if presenter not in ('male', 'female'):
        raise ValueError('Presenter must be male or female')
    voice_id = getattr(settings, 'resolved_voice_id', None)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)

    def announce(stage: str) -> None:
        if progress is not None:
            progress(stage)

    announce("parsing")
    game = load_game(pgn)
    engine_path = getattr(settings, "stockfish_path", None) or os.environ.get("STOCKFISH_PATH")
    game_key = hashlib.sha256(json.dumps({"pgn": pgn, "engine": str(engine_path)}, sort_keys=True).encode()).hexdigest()
    inputs_path = output_dir / "game-input.json"
    if engine_path:
        announce("analysis")
        existing_key = json.loads(inputs_path.read_text()).get("key") if inputs_path.is_file() else None
        if existing_key == game_key and (output_dir / "game.json").is_file():
            game = validate_game(json.loads((output_dir / "game.json").read_text(encoding="utf-8")))
        else:
            game = analyze_game(game, engine_path=str(engine_path))
    if hasattr(settings, "model_copy"):
        settings = settings.model_copy(update={"generation_dir": output_dir})
    (output_dir / "game.pgn").write_text(pgn, encoding="utf-8")
    write_json(output_dir / "game.json", game)
    write_json(inputs_path, {"key": game_key})
    announce("script")
    if script_path is not None:
        script = json.loads(Path(script_path).read_text(encoding="utf-8"))
    else:
        script = generate_script(game, settings)
    script = validate_script(game, script)
    write_json(output_dir / "script.json", script)
    (output_dir / "transcript.txt").write_text(
        "\n\n".join(beat["text"] for beat in script["beats"]), encoding="utf-8"
    )
    announce("speech")
    speech = synthesize(script, output_dir, settings)
    write_json(output_dir / "speech.json", speech)
    announce("timing")
    timeline = compile_timeline(game, script, speech, width=width, height=height, fps=fps, presenter=presenter)
    write_json(output_dir / "timeline.json", timeline)
    announce("rendering")
    video_path = render_video(timeline, output_dir / "video.mp4")
    result = {
        "video_path": str(video_path),
        "duration": timeline["duration"],
        "width": width,
        "height": height,
        "fps": fps,
        "presenter": presenter,
        "voice_id": voice_id,
        "title": script["title"],
        "plies": len(game["positions"]) - 1,
        "providers": {"llm": "provided_script" if script_path is not None else getattr(settings, "llm_provider", None), "tts": "elevenlabs"},
        "artifacts": ["game.pgn", "game.json", "script.json", "transcript.txt", "narration.mp3", "speech.json", "timeline.json", "video.mp4"],
    }
    write_json(manifest_path, result)
    announce("completed")
    return result
