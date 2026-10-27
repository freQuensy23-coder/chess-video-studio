# Chess Video Studio

Generate vertical chess commentary videos from a PGN: LLM-written narration, ElevenLabs speech, character-aligned animation timing, and an original SVG presenter placeholder.

## Run

Requires Python 3.11+, FFmpeg, Cairo, and a font package such as DejaVu Sans. Install dependencies in the project environment with `uv sync`. Copy `.env.example` to `.env` and set your provider credentials; generated media and local credentials are ignored by Git.

`uv run chess-video generate --pgn examples/legal-mate.pgn --output output/my-video`

`uv run chess-video showcase --output output/animations.mp4`

`uv run chess-video script --pgn examples/legal-mate.pgn --output output/script.json`

`uv run chess-video generate --pgn examples/legal-mate.pgn --script output/script.json --output output/my-video`

`uv run chess-video render --timeline output/my-video/timeline.json --output output/rerender.mp4`

Default video size is 608×1080 at 30 fps. `generate` and `showcase` accept `--width`, `--height`, and `--fps`; generated narration supports `--language en` or `--language ru`.

## Generation model

The PGN is replayed by python-chess. The LLM receives actual positions and moves; its structured script must reference valid plies and exact spoken phrases. The board follows the PGN, never model-generated moves. ElevenLabs returns actual speech plus character timing. Moves, annotations, zooms, and expressions are scheduled from those timestamps, not estimated reading speed.

A finished job contains the original PGN, parsed game, editable script, transcript, speech alignment, timeline, and H.264/AAC MP4. An existing script can be edited and reused. Existing timelines can be rendered again without provider calls. Provider and alignment failures stop generation instead of substituting fake speech or fabricated timing.

## Animation surface

Board and presenter split-screen; animated pieces and captures; castling, en passant, promotion picker; last-move and selected-square highlights; legal destination dots; smooth cursor movement and clicks; cumulative straight, diagonal, and knight-shaped arrows; colored cell marks; glowing accents; labels such as `!!`; board-only focus zooms; expression changes, blinking and speech mouth motion; actual check/mate/result overlays and final presenter emphasis.

The presenter is an original vector placeholder. Photorealistic avatars, voice cloning, and likeness replication are outside this release. A decorative move badge does not establish engine-certified brilliance.

## Local API

Set `CHESS_VIDEO_API_TOKEN` to a private random value in `.env` or the process environment, then run:

`uv run chess-video serve --host 127.0.0.1 --port 8787`

Interactive OpenAPI documentation is at `http://127.0.0.1:8787/docs`. All job and artifact endpoints require `Authorization: Bearer <your token>`.

`POST /v1/jobs` accepts `pgn`, optional `width`, `height`, and `fps`, and returns a job ID. `GET /v1/jobs/{id}` returns status and stage. `GET /v1/jobs/{id}/files/video.mp4` downloads a completed video. `GET /v1/jobs` lists local jobs.

Jobs are persisted under `CHESS_VIDEO_JOB_ROOT` from `.env` or the process environment (default `.jobs/`); `serve --jobs PATH` overrides that setting. One worker owns the directory; queue length and input size are bounded. Interrupted jobs are marked failed on restart and are not silently submitted to paid providers again. Only allowlisted finished artifacts are exposed. The CLI binds the API to loopback and does not install services or alter server settings.

## Costs and verification

Script and speech generation call paid external providers. Review the configured models and character budget before starting long games. The animation showcase needs no provider credentials.

Push/PR-triggered GitHub Actions runs lint, tests, an actual animation render, and full FFmpeg decoding. The rendered catalogue is attached as a workflow artifact. Mocked provider tests validate contracts and error handling; they do not demonstrate paid-provider availability or speech quality.

GPL-3.0-or-later. No video, face, recording, or graphical asset from the reference creator is distributed with this project.
