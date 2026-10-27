import argparse
import json
import sys
from pathlib import Path

from chess_video import __version__
from chess_video.video_options import validate_video_options


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="chess-video")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate a narrated video from a PGN")
    generate.add_argument("--pgn", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--script", type=Path)
    generate.add_argument("--language", choices=["en", "ru"])
    script = commands.add_parser("script", help="Generate a chess-grounded commentary script")
    script.add_argument("--pgn", type=Path, required=True)
    script.add_argument("--output", type=Path, required=True)
    script.add_argument("--language", choices=["en", "ru"])
    render = commands.add_parser("render", help="Render an existing aligned timeline without provider calls")
    render.add_argument("--timeline", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    showcase = commands.add_parser("showcase", help="Render the complete animation catalogue without API keys")
    showcase.add_argument("--output", type=Path, required=True)
    for command in [generate, showcase]:
        command.add_argument("--width", type=int, default=608)
        command.add_argument("--height", type=int, default=1080)
        command.add_argument("--fps", type=int, default=30)
    serve = commands.add_parser("serve", help="Start the authenticated single-worker local API")
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--jobs", type=Path, default=None)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command in {"generate", "showcase"}:
            validate_video_options(arguments.width, arguments.height, arguments.fps)
        if arguments.command == "showcase":
            from chess_video.renderer import write_showcase

            result = write_showcase(
                arguments.output, width=arguments.width, height=arguments.height, fps=arguments.fps
            )
            print(json.dumps({"video_path": str(result)}))
        elif arguments.command == "render":
            from chess_video.renderer import render_video

            timeline = json.loads(arguments.timeline.read_text(encoding="utf-8"))
            path = render_video(timeline, arguments.output)
            print(json.dumps({"video_path": str(path)}))
        elif arguments.command in {"generate", "script"}:
            from chess_video.game import load_game, validate_script
            from chess_video.pipeline import generate_video, write_json
            from chess_video.providers import generate_script
            from chess_video.settings import Settings

            settings = Settings()
            if arguments.language:
                settings = settings.model_copy(update={"language": arguments.language})
            pgn = arguments.pgn.read_text(encoding="utf-8")
            if arguments.command == "script":
                game = load_game(pgn)
                generated = validate_script(game, generate_script(game, settings))
                write_json(arguments.output, generated)
                print(json.dumps({"script_path": str(arguments.output), "title": generated["title"]}))
            else:
                result = generate_video(
                    pgn,
                    arguments.output,
                    settings,
                    width=arguments.width,
                    height=arguments.height,
                    fps=arguments.fps,
                    script_path=arguments.script,
                    progress=lambda stage: print(stage, file=sys.stderr),
                )
                print(json.dumps(result, ensure_ascii=False))
        elif arguments.command == "serve":
            import uvicorn

            from chess_video.api import create_app

            if not 1 <= arguments.port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            from chess_video.settings import Settings

            settings = Settings()
            if not settings.api_token:
                raise ValueError("Set CHESS_VIDEO_API_TOKEN to a private random token")
            app = create_app(settings=settings, job_root=arguments.jobs)
            uvicorn.run(app, host=arguments.host, port=arguments.port, workers=1)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Generation stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
