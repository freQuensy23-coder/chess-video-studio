import asyncio
import fcntl
import hmac
import json
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chess_video.pipeline import generate_video, write_json
from chess_video.settings import Settings


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pgn: str = Field(min_length=1, max_length=200_000)
    width: int = Field(default=608, ge=320, le=2160)
    height: int = Field(default=1080, ge=480, le=3840)
    fps: int = Field(default=30, ge=12, le=60)

    @model_validator(mode="after")
    def valid_video_shape(self):
        if self.width % 2 or self.height % 2 or self.width >= self.height:
            raise ValueError("Use even dimensions with width smaller than height")
        return self


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    settings: Settings | None = None,
    *,
    job_root: str | Path | None = None,
    api_token: str | None = None,
    pipeline: Callable = generate_video,
    max_pending_jobs: int = 8,
) -> FastAPI:
    effective_settings = settings if settings is not None else Settings()
    root = Path(job_root or effective_settings.job_root).resolve()
    configured_token = effective_settings.api_token
    token = api_token if api_token is not None else (configured_token.get_secret_value() if configured_token else "")
    jobs: dict[str, dict] = {}
    lock = threading.RLock()
    work: queue.Queue = queue.Queue(maxsize=max_pending_jobs)
    stopping = threading.Event()
    files = {"video.mp4", "narration.mp3", "script.json", "timeline.json", "transcript.txt", "game.pgn", "manifest.json"}

    def store(job: dict) -> None:
        write_json(root / job["id"] / "job.json", job)

    def update(job_id: str, **changes) -> None:
        with lock:
            jobs[job_id].update(changes, updated_at=timestamp())
            store(jobs[job_id])

    def worker() -> None:
        while not stopping.is_set():
            try:
                job_id = work.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                update(job_id, status="running", stage="starting")
                directory = root / job_id
                request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
                options = JobRequest.model_validate(request)
                result = pipeline(
                    pgn=options.pgn,
                    output_dir=directory,
                    settings=effective_settings,
                    width=options.width,
                    height=options.height,
                    fps=options.fps,
                    progress=lambda stage: update(job_id, stage=stage),
                )
                available = [name for name in sorted(files) if (directory / name).is_file()]
                update(
                    job_id,
                    status="completed",
                    stage="completed",
                    duration=result.get("duration"),
                    files=available,
                )
            except Exception as error:
                update(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=f"Generation failed ({type(error).__name__}); inspect private job files locally",
                )
            finally:
                work.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not token:
            raise RuntimeError("Set CHESS_VIDEO_API_TOKEN before starting the API")
        root.mkdir(parents=True, exist_ok=True)
        owner = (root / ".owner.lock").open("a+")
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            owner.close()
            raise RuntimeError("Another worker already owns this job directory") from None
        for record in root.glob("*/job.json"):
            if record.parent.is_symlink():
                continue
            try:
                job = json.loads(record.read_text(encoding="utf-8"))
                parsed = str(uuid.UUID(job["id"]))
                if parsed != record.parent.name:
                    continue
                if job["status"] in {"running", "queued"}:
                    job.update(status="failed", stage="failed", error="Interrupted; not retried to avoid duplicate billing")
                    store(job)
                jobs[parsed] = job
            except (ValueError, KeyError, TypeError, OSError):
                continue
        thread = threading.Thread(target=worker, name="chess-video-worker", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopping.set()
            await asyncio.to_thread(thread.join)
            fcntl.flock(owner, fcntl.LOCK_UN)
            owner.close()

    app = FastAPI(title="Chess Video Studio", version="0.1.0", lifespan=lifespan)
    security = HTTPBearer(auto_error=False)

    def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]) -> None:
        if credentials is None or not hmac.compare_digest(credentials.credentials, token):
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        if request.method == "POST":
            from fastapi.responses import JSONResponse

            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid content length"})
            if length > 250_000:
                return JSONResponse(status_code=413, content={"detail": "Request too large"})
            chunks = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > 250_000:
                    return JSONResponse(status_code=413, content={"detail": "Request too large"})
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    def find_job(job_id: str) -> dict:
        try:
            parsed = str(uuid.UUID(job_id))
        except ValueError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        with lock:
            if parsed not in jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            return dict(jobs[parsed])

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def submit(request: JobRequest):
        from chess_video.game import load_game

        try:
            load_game(request.pgn)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid PGN") from None
        job_id = str(uuid.uuid4())
        with lock:
            if work.full():
                raise HTTPException(status_code=429, detail="Generation queue is full")
            directory = root / job_id
            directory.mkdir(mode=0o700)
            write_json(directory / "request.json", request.model_dump())
            job = {"id": job_id, "status": "queued", "stage": "queued", "created_at": timestamp()}
            jobs[job_id] = job
            store(job)
            work.put_nowait(job_id)
        return dict(job)

    @app.get("/v1/jobs", dependencies=[Depends(authorize)])
    def list_jobs():
        with lock:
            return {"jobs": [dict(value) for value in reversed(list(jobs.values()))]}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str):
        return find_job(job_id)

    @app.get("/v1/jobs/{job_id}/files/{name}", dependencies=[Depends(authorize)])
    def get_file(job_id: str, name: str):
        job = find_job(job_id)
        if job["status"] != "completed" or name not in files:
            raise HTTPException(status_code=404, detail="File not found")
        path = root / job["id"] / name
        if path.is_symlink() or not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, filename=name)

    return app
