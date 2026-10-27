from __future__ import annotations

import array
import bisect
import copy
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import cairosvg
import chess
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

_ASSETS = Path(__file__).with_name("assets")
_ANNOTATIONS = {"arrow", "highlight", "badge"}
_KINDS = _ANNOTATIONS | {"move", "zoom", "cursor", "clear", "face", "result"}
_EMOTIONS = {"neutral", "worried", "thinking", "surprised", "happy"}
_CYAN = (65, 230, 232)
_GOLD = (255, 197, 63)
_ARROW_GROWTH = 0.28


def _number(value: Any, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def _mix(a: float, b: float, progress: float) -> float:
    return a + (b - a) * progress


def _motion(a: tuple[float, float], b: tuple[float, float], progress: float, bend: float = 0.06) -> tuple[float, float]:
    progress = _smooth(progress)
    dx, dy = b[0] - a[0], b[1] - a[1]
    arc = math.sin(math.pi * progress) * bend
    return _mix(a[0], b[0], progress) - dy * arc, _mix(a[1], b[1], progress) + dx * arc


def _color(value: Any, fallback: tuple[int, int, int] = _GOLD) -> tuple[int, int, int]:
    if value is None:
        return fallback
    try:
        return ImageColor.getrgb(str(value))[:3]
    except ValueError as error:
        raise ValueError(f"Invalid annotation color: {value}") from error


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    stem = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (Path("/usr/share/fonts/truetype/dejavu") / stem, Path(stem)):
        try:
            return ImageFont.truetype(str(path), max(8, size))
        except OSError:
            pass
    return ImageFont.load_default(size=max(8, size))


@lru_cache(maxsize=96)
def _piece(symbol: str, size: int) -> Image.Image:
    root = ElementTree.fromstring((_ASSETS / "pieces.svg").read_text())
    shape = next(element for element in root.iter() if element.attrib.get("id") == symbol.lower())
    body = ElementTree.tostring(shape, encoding="unicode")
    white = symbol.isupper()
    main, light, edge = ("#e0ceb0", "#fff8e5", "#463d38") if white else ("#272d38", "#68717e", "#121a26")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><linearGradient id="piece" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{light}"/><stop offset=".6" stop-color="{main}"/><stop offset="1" stop-color="{main}"/></linearGradient></defs><g fill="url(#piece)" stroke="{edge}">{body}</g></svg>'
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size * 2, output_height=size * 2)
    return Image.open(io.BytesIO(png)).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)


@lru_cache(maxsize=96)
def _piece_shadow(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size * 0.14, size * 0.70, size * 0.92, size * 0.94), fill=(12, 13, 22, 135))
    return image.filter(ImageFilter.GaussianBlur(max(1, size * 0.04)))


@lru_cache(maxsize=8)
def _wood_board(side: int, orientation: str) -> Image.Image:
    rng = random.Random(17291)
    image = Image.new("RGB", (side, side))
    pixels = image.load()
    for y in range(side):
        row = min(7, y * 8 // side)
        for x in range(side):
            col = min(7, x * 8 // side)
            base = (208, 181, 133) if (row + col) % 2 == 0 else (137, 98, 64)
            grain = 2.9 * math.sin(x * 0.25 + math.sin(y * 0.014) * 1.8)
            grain += 1.8 * math.sin(x * 0.83 + y * 0.005) + rng.uniform(-1.4, 1.4)
            pixels[x, y] = tuple(int(max(0, min(255, component + grain))) for component in base)
    draw = ImageDraw.Draw(image)
    cell = side / 8
    font = _font(round(cell * 0.16), True)
    for index in range(8):
        file = chr(97 + (index if orientation == "white" else 7 - index))
        rank = str(8 - index if orientation == "white" else index + 1)
        draw.text((index * cell + cell - cell * 0.18, side - cell * 0.20), file, font=font, fill=(77, 58, 39) if index % 2 else (238, 213, 166))
        draw.text((cell * 0.04, index * cell + cell * 0.025), rank, font=font, fill=(77, 58, 39) if index % 2 == 0 else (238, 213, 166))
    return image.convert("RGBA")


@lru_cache(maxsize=1)
def _face_base() -> Image.Image:
    png = cairosvg.svg2png(url=str(_ASSETS / "face.svg"), output_width=600, output_height=480)
    return Image.open(io.BytesIO(png)).convert("RGBA")


@lru_cache(maxsize=96)
def _face(emotion: str, blink: bool, mouth: int, width: int, height: int) -> Image.Image:
    image = _face_base().copy()
    draw = ImageDraw.Draw(image)
    ink = (65, 48, 49)
    for cx in (264, 338):
        if blink:
            draw.arc((cx - 16, 220, cx + 16, 233), 0, 180, fill=ink, width=4)
        else:
            radius = 15 if emotion == "surprised" else 11
            draw.ellipse((cx - 12, 224 - radius, cx + 12, 224 + radius), fill=(255, 249, 230), outline=(133, 88, 67), width=2)
            pupil_x = cx + (4 if emotion == "thinking" else 0)
            draw.ellipse((pupil_x - 5, 218, pupil_x + 5, 230), fill=(44, 51, 62))
            draw.ellipse((pupil_x - 3, 218, pupil_x, 221), fill="white")
    if emotion == "worried":
        brows = [(248, 204, 278, 195), (324, 195, 355, 204)]
    elif emotion == "thinking":
        brows = [(248, 195, 278, 199), (324, 205, 355, 208)]
    elif emotion == "surprised":
        brows = [(248, 190, 278, 188), (324, 188, 355, 190)]
    else:
        brows = [(248, 203, 278, 201), (324, 201, 355, 203)]
    for brow in brows:
        draw.line(brow, fill=(84, 53, 42), width=5)
    opening = mouth * 2.8
    if emotion == "surprised":
        opening = max(18, opening)
        draw.ellipse((286, 278, 315, 290 + opening), fill=(83, 42, 46), outline=(169, 93, 77), width=3)
    elif emotion == "happy":
        draw.rounded_rectangle((276, 279, 325, 294 + max(5, opening)), radius=14, fill=(88, 43, 46))
        draw.rounded_rectangle((280, 280, 321, 288), radius=3, fill=(255, 249, 229))
        draw.arc((272, 266, 329, 305), 5, 175, fill=(151, 79, 64), width=3)
    elif opening > 3:
        draw.ellipse((281, 281, 321, 288 + opening), fill=(91, 44, 47), outline=(174, 95, 77), width=2)
        draw.arc((284, 288, 319, 292 + opening), 0, 180, fill=(219, 116, 116), width=4)
    elif emotion == "worried":
        draw.arc((282, 283, 321, 307), 195, 345, fill=(143, 77, 68), width=4)
    elif emotion == "thinking":
        draw.line((286, 290, 316, 286), fill=(143, 77, 68), width=4)
    else:
        draw.arc((282, 275, 321, 294), 10, 170, fill=(143, 77, 68), width=3)
    return image.resize((width, height), Image.Resampling.LANCZOS)


class Renderer:
    def __init__(self, timeline: dict):
        if not isinstance(timeline, dict):
            raise TypeError("timeline must be a dictionary")
        self.timeline = copy.deepcopy(timeline)
        if timeline.get("version", 1) != 1:
            raise ValueError("Unsupported timeline version")
        self.width = self._dimension(timeline.get("width", 608), "width")
        self.height = self._dimension(timeline.get("height", 1080), "height")
        if self.width < 160 or self.height < self.width * 1.35 or self.width > 4096 or self.height > 8192:
            raise ValueError("Dimensions must describe an even portrait frame between 160 and 4096 pixels wide")
        fps = _number(timeline.get("fps", 30), "fps", 1)
        if not fps.is_integer() or fps > 120:
            raise ValueError("fps must be an integer from 1 to 120")
        self.fps = int(fps)
        self.duration = _number(timeline.get("duration"), "duration", 0.001)
        self.orientation = timeline.get("orientation", "white")
        if self.orientation not in {"white", "black"}:
            raise ValueError("orientation must be white or black")
        try:
            self.initial = chess.Board(timeline.get("initial_fen", chess.STARTING_FEN))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid initial FEN") from error
        if not self.initial.is_valid():
            raise ValueError("Invalid chess position in initial FEN")
        self.margin = max(3, round(self.width * 0.009))
        self.panel_height = max(self.width, round(self.height * 0.565))
        self.panel_height = min(self.panel_height, self.height - round(self.width * 0.5))
        self.side = min(self.width - self.margin * 2, self.panel_height - self.margin * 2)
        self.board_x = (self.width - self.side) // 2
        self.board_y = (self.panel_height - self.side) // 2
        self.cell = self.side / 8
        self.sprite_size = round(self.cell * 0.96)
        self.face_y = self.panel_height + max(2, self.margin // 2)
        self.events = self._events(timeline.get("events", []))
        self.moves = [event for event in self.events if event["kind"] == "move"]
        self.move_ends = [event["end"] for event in self.moves]
        self._validate_moves()
        self._validate_results()
        self._cursor_keys = self._make_cursor_keys()
        self._cursor_times = [item[0] for item in self._cursor_keys]
        self._static = Image.new("RGBA", (self.width, self.panel_height), (23, 30, 41, 255))
        self._static.alpha_composite(_wood_board(self.side, self.orientation), (self.board_x, self.board_y))
        self._position_cache: dict[str, Image.Image] = {}
        self._audio_levels: list[float] = []
        self.audio_duration = 0.0
        self.audio_path: Path | None = None
        if timeline.get("audio_path") is not None:
            self.audio_path = Path(timeline["audio_path"]).expanduser().resolve()
            if not self.audio_path.is_file():
                raise ValueError(f"Audio file does not exist: {self.audio_path}")
            self._load_audio()
            self.duration = max(self.duration, self.audio_duration)
        self._closed = False

    @staticmethod
    def _dimension(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
            raise ValueError(f"{name} must be a positive even integer")
        return value

    def _events(self, events: Any) -> list[dict]:
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        result = []
        for index, original in enumerate(events):
            if not isinstance(original, dict) or original.get("kind") not in _KINDS:
                raise ValueError("Unknown timeline event")
            event = copy.deepcopy(original)
            event["start"] = _number(event.get("start"), "event start")
            event["end"] = _number(event.get("end", self.duration), "event end")
            if event["end"] < event["start"] or event["start"] > self.duration or event["end"] > self.duration + 0.001:
                raise ValueError("Event times must be ordered and inside the timeline")
            event["_order"] = index
            for key in ("square", "from_square", "to_square"):
                if event.get(key) is not None:
                    try:
                        chess.parse_square(event[key])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Invalid {key}") from error
            kind = event["kind"]
            required = ("from_square", "to_square") if kind == "arrow" else ("square",) if kind in {"cursor", "zoom", "highlight", "badge"} else ()
            if any(not event.get(key) for key in required):
                raise ValueError(f"Missing square in {kind} event")
            if kind in _ANNOTATIONS:
                event["_color"] = _color(event.get("color"), _CYAN if kind == "badge" else _GOLD)
            if kind == "zoom":
                scale = _number(event.get("scale", 1.6), "zoom scale", 1)
                if scale > 4:
                    raise ValueError("Zoom scale must be between 1 and 4")
                event["scale"] = scale
            if kind == "face" and event.get("emotion") not in _EMOTIONS:
                raise ValueError("Invalid face emotion")
            if kind == "move" and event["end"] <= event["start"]:
                raise ValueError("Move duration must be positive")
            result.append(event)
        return sorted(result, key=lambda event: (event["start"], 0 if event["kind"] == "clear" else 1, event["_order"]))

    def _validate_moves(self) -> None:
        board = self.initial.copy()
        previous_end = 0.0
        for event in self.moves:
            if event["start"] < previous_end - 1e-8:
                raise ValueError("Move events overlap")
            try:
                before = chess.Board(event["before_fen"])
                after = chess.Board(event["fen"])
                move = chess.Move.from_uci(event["uci"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Move requires valid before_fen, fen and uci") from error
            if before.fen() != board.fen() or move not in board.legal_moves:
                raise ValueError("Illegal move or discontinuous before_fen")
            event["_before"] = board.copy()
            event["_move"] = move
            event["_piece"] = board.piece_at(move.from_square)
            event["_capture"] = board.is_capture(move)
            event["_ep"] = board.is_en_passant(move)
            event["_castle"] = board.is_castling(move)
            event["_san"] = board.san(move)
            board.push(move)
            if after.fen() != board.fen():
                raise ValueError("Move fen does not match legal replay")
            event["_after"] = board.copy()
            previous_end = event["end"]

    def _validate_results(self) -> None:
        declared = self.timeline.get("game_result", "*")
        if not isinstance(declared, str) or declared not in {"*", "1-0", "0-1", "1/2-1/2"}:
            raise ValueError("Invalid declared game result")
        final_outcome = self.board_at(self.duration).outcome()
        if declared != "*" and final_outcome and declared != final_outcome.result():
            raise ValueError("Declared game result contradicts actual terminal board")
        for event in self.events:
            if event["kind"] == "result":
                if self.moves and event["start"] < self.moves[-1]["end"]:
                    raise ValueError("A result event requires the completed final move")
                board = self.board_at(event["start"])
                outcome = board.outcome(claim_draw=declared == "*")
                if outcome is None and declared == "*":
                    raise ValueError("A result event requires an actual terminal position")
                label = str(event.get("label", "")).casefold().replace("ё", "е")
                if re.search(r"\b(?:checkmate|checkmated|mate|мат)\b", label) and not board.is_checkmate():
                    raise ValueError("Result cannot claim checkmate for this board")
                if re.search(r"\b(?:stalemate|пат)\b", label) and not board.is_stalemate():
                    raise ValueError("Result cannot claim stalemate for this board")
                actual = outcome.result() if outcome else declared
                claims = (
                    (r"\b(?:white|белые|белых)\b|(?<![\w/-])1-0(?![\w/-])", "1-0"),
                    (r"\b(?:black|черные|черных)\b|(?<![\w/-])0-1(?![\w/-])", "0-1"),
                    (r"\b(?:draw|drawn|stalemate|ничья|пат)\b|(?<![\w/-])1/2-1/2(?![\w/-])", "1/2-1/2"),
                )
                if any(re.search(pattern, label) and actual != result for pattern, result in claims):
                    raise ValueError("Result label contradicts actual winner")

    def _load_audio(self) -> None:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(self.audio_path), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"]
        try:
            decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = error.stderr.decode(errors="replace")[-2000:] if isinstance(error, subprocess.CalledProcessError) else str(error)
            raise RuntimeError(f"ffmpeg audio decoding failed: {detail}") from error
        samples = array.array("h")
        samples.frombytes(decoded.stdout)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            raise ValueError("Audio file contains no decodable samples")
        self.audio_duration = len(samples) / 8000
        window = 160
        levels = [math.sqrt(sum(sample * sample for sample in samples[index:index + window]) / len(samples[index:index + window])) / 32768 for index in range(0, len(samples), window)]
        peak = max(levels, default=1)
        scale = max(peak * 0.7, 0.015)
        self._audio_levels = [min(1.0, level / scale) for level in levels]

    def square_center(self, square: str | int) -> tuple[float, float]:
        square = chess.parse_square(square) if isinstance(square, str) else square
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (file, 7 - rank) if self.orientation == "white" else (7 - file, rank)
        return self.board_x + (col + 0.5) * self.cell, self.board_y + (row + 0.5) * self.cell

    def _box(self, square: str | int, inset: float = 0) -> tuple[float, float, float, float]:
        x, y = self.square_center(square)
        half = self.cell / 2 - inset
        return x - half, y - half, x + half, y + half

    def _validate_time(self, time: float) -> float:
        time = _number(time, "frame time")
        if time > self.duration + 1e-8:
            raise ValueError("Frame time is past the timeline duration")
        return min(time, self.duration)

    def board_at(self, time: float) -> chess.Board:
        index = bisect.bisect_right(self.move_ends, time)
        return (self.moves[index - 1]["_after"] if index else self.initial).copy()

    def _active_move(self, time: float) -> dict | None:
        index = bisect.bisect_right(self.move_ends, time)
        if index < len(self.moves) and self.moves[index]["start"] <= time < self.moves[index]["end"]:
            return self.moves[index]
        return None

    def _last_move(self, time: float) -> dict | None:
        index = bisect.bisect_right(self.move_ends, time)
        return self.moves[index - 1] if index else None

    def legal_destinations(self, square: str, time: float) -> list[str]:
        board = self.board_at(time)
        source = chess.parse_square(square)
        return sorted({chess.square_name(move.to_square) for move in board.legal_moves if move.from_square == source})

    def annotations_at(self, time: float) -> list[dict]:
        visible = []
        for event in self.events:
            if event["start"] > time:
                break
            if event["kind"] == "clear":
                visible.clear()
            elif event["kind"] in _ANNOTATIONS and time < event["end"]:
                visible.append(event)
        return visible

    def promotion_choices_at(self, time: float) -> tuple[str, ...]:
        event = self._active_move(time)
        if event and event["_move"].promotion:
            progress = (time - event["start"]) / (event["end"] - event["start"])
            if progress >= 0.63:
                return "q", "r", "b", "n"
        return ()

    def result_at(self, time: float) -> str | None:
        board = self.board_at(time)
        active = [event for event in self.events if event["kind"] == "result" and event["start"] <= time <= event["end"]]
        label = str(active[-1].get("label", "")) if active else ""
        russian = self.timeline.get("language") == "ru" or bool(re.search("[А-Яа-яЁё]", label))
        if board.is_checkmate():
            if russian:
                return "МАТ · " + ("ЧЁРНЫЕ ПОБЕДИЛИ" if board.turn == chess.WHITE else "БЕЛЫЕ ПОБЕДИЛИ")
            return "CHECKMATE · " + ("BLACK WINS" if board.turn == chess.WHITE else "WHITE WINS")
        if active:
            if label:
                return label
            outcome = board.outcome(claim_draw=True)
            result = self.timeline.get("game_result", "*")
            if result == "*" and outcome:
                result = outcome.result()
            if result in {"1-0", "0-1"}:
                if russian:
                    return "БЕЛЫЕ ПОБЕДИЛИ" if result == "1-0" else "ЧЁРНЫЕ ПОБЕДИЛИ"
                return "WHITE WINS" if result == "1-0" else "BLACK WINS"
            if russian:
                return "НИЧЬЯ · ПАТ" if board.is_stalemate() else "НИЧЬЯ"
            return "DRAW · " + ("STALEMATE" if board.is_stalemate() else "GAME OVER")
        return None

    def _make_cursor_keys(self) -> list[tuple[float, tuple[float, float]]]:
        actions = []
        previous_end = 0.0
        for event in self.moves:
            duration = event["end"] - event["start"]
            a = self.square_center(event["_move"].from_square)
            b = self.square_center(event["_move"].to_square)
            approach = max(previous_end, event["start"] - 0.4)
            contact = max(approach, event["start"] - 0.08)
            points = []
            if contact > approach:
                points.append((contact, a))
            points.extend([(event["start"] + duration * 0.18, a), (event["start"] + duration * (0.63 if event["_move"].promotion else 0.82), b)])
            if event["_move"].promotion:
                left, top, down = self._promotion_geometry(event)
                choice = ("q", "r", "b", "n").index(chess.piece_symbol(event["_move"].promotion))
                row = choice if down else 3 - choice
                selected = (left + self.cell / 2, top + (row + 0.5) * self.cell)
                points.extend([(event["start"] + duration * 0.9, selected), (event["end"], selected)])
            else:
                points.append((event["end"], b))
            actions.append({"approach": approach, "end": event["end"], "event": event, "keys": points})
            previous_end = event["end"]
        self._cursor_strokes: list[dict] = []
        for event in self.events:
            kind = event["kind"]
            if kind not in _ANNOTATIONS | {"cursor"}:
                continue
            start = event["start"]
            if kind == "cursor":
                draw_end = min(event["end"], start + 0.4)
                approach, end = start, draw_end
                points = [(draw_end, self.square_center(event["square"]))]
            else:
                if kind == "arrow" and event["from_square"] == event["to_square"]:
                    continue
                draw_end = start + (_ARROW_GROWTH if kind == "arrow" else 0.08)
                clear = next((item["start"] for item in self.events if item["kind"] == "clear" and item["start"] > start), self.duration)
                end = min(draw_end + 0.12, event["end"], clear, self.duration)
                if end <= draw_end:
                    continue
                approach = max(0.0, start - 0.4)
                a = self.square_center(event["from_square"] if kind == "arrow" else event["square"])
                b = self.square_center(event["to_square"]) if kind == "arrow" else a
                points = [(start, a), (draw_end, b), (end, b)]
            if end <= approach or any(approach < action["end"] and end > action["approach"] for action in actions):
                continue
            action = {"approach": approach, "start": start, "draw_end": draw_end, "end": end, "event": event, "keys": points}
            actions.append(action)
            if kind in _ANNOTATIONS:
                self._cursor_strokes.append(action)
        actions.sort(key=lambda action: (action["approach"], action["event"]["_order"]))
        source = self.moves[0]["_move"].from_square if self.moves else chess.D4
        initial = self.square_center(source)
        if actions and actions[0]["keys"][0][0] == 0:
            initial = actions[0]["keys"][0][1]
        keys = [(0.0, initial)]
        for action in actions:
            keys.append((action["approach"], keys[-1][1]))
            keys.extend(action["keys"])
        ordered: dict[float, tuple[float, float]] = {}
        for time, point in sorted(keys, key=lambda key: key[0]):
            if not (self.board_x <= point[0] <= self.board_x + self.side and self.board_y <= point[1] <= self.board_y + self.side):
                raise ValueError("Cursor control points must remain inside the board")
            ordered[time] = point
        return sorted(ordered.items())

    def _promotion_geometry(self, event: dict) -> tuple[float, float, bool]:
        x, y = self.square_center(event["_move"].to_square)
        down = y < self.board_y + self.side / 2
        top = y - self.cell / 2 if down else y + self.cell / 2 - self.cell * 4
        top = min(self.board_y + self.side - self.cell * 4, max(self.board_y, top))
        left = max(self.board_x, min(self.board_x + self.side - self.cell, x - self.cell / 2))
        return left, top, down

    def cursor_at(self, time: float) -> tuple[float, float]:
        for stroke in self._cursor_strokes:
            if stroke["event"]["kind"] == "arrow" and stroke["start"] <= time < stroke["draw_end"]:
                return self._arrow_visible_points(stroke["event"], time)[-1]
        index = bisect.bisect_right(self._cursor_times, time)
        if index == 0:
            return self._cursor_keys[0][1]
        if index >= len(self._cursor_keys):
            return self._cursor_keys[-1][1]
        start, a = self._cursor_keys[index - 1]
        end, b = self._cursor_keys[index]
        return _motion(a, b, (time - start) / (end - start))

    def _cursor_gesture(self, time: float) -> tuple[str, str, float] | None:
        for stroke in self._cursor_strokes:
            if stroke["start"] <= time < stroke["end"]:
                press_end = min(stroke["draw_end"], stroke["start"] + 0.04) if stroke["event"]["kind"] == "arrow" else stroke["draw_end"]
                if time < press_end:
                    return "right", "press", (time - stroke["start"]) / (press_end - stroke["start"])
                if time < stroke["draw_end"]:
                    return "right", "drag", (time - press_end) / (stroke["draw_end"] - press_end)
                return "right", "release", (time - stroke["draw_end"]) / (stroke["end"] - stroke["draw_end"])
        return None

    def _selection(self, time: float) -> str | None:
        for event in self.moves:
            duration = event["end"] - event["start"]
            if max(0, event["start"] - 0.3) <= time < event["start"] + duration * 0.38:
                return chess.square_name(event["_move"].from_square)
        active = [event for event in self.events if event["kind"] == "cursor" and event["start"] <= time < event["end"]]
        return active[-1]["square"] if active else None

    def _position_layer(self, board: chess.Board) -> Image.Image:
        key = board.board_fen()
        if key not in self._position_cache:
            image = Image.new("RGBA", self._static.size)
            for square, piece in board.piece_map().items():
                self._put_piece(image, piece.symbol(), self.square_center(square))
            if len(self._position_cache) >= 6:
                self._position_cache.pop(next(iter(self._position_cache)))
            self._position_cache[key] = image
        return self._position_cache[key]

    def _put_piece(self, image: Image.Image, symbol: str, point: tuple[float, float], lifted: bool = False) -> None:
        size = self.sprite_size
        x, y = round(point[0] - size / 2), round(point[1] - size / 2)
        image.alpha_composite(_piece_shadow(size), (x + (3 if lifted else 0), y + (4 if lifted else 0)))
        image.alpha_composite(_piece(symbol, size), (x, y - (round(self.cell * 0.055) if lifted else 0)))

    def _square_accent(self, draw: ImageDraw.ImageDraw, square: str | int, color: tuple[int, int, int], opacity: float = 1) -> None:
        box = self._box(square, 1.5)
        draw.rounded_rectangle(box, radius=max(2, self.cell * 0.045), fill=(*color, round(36 * opacity)), outline=(*color, round(235 * opacity)), width=max(2, round(self.cell * 0.038)))
        inner = self._box(square, 4)
        draw.rounded_rectangle(inner, radius=2, outline=(*color, round(55 * opacity)), width=1)

    def _draw_move(self, panel: Image.Image, event: dict, time: float) -> None:
        board = event["_before"].copy()
        move = event["_move"]
        progress = (time - event["start"]) / (event["end"] - event["start"])
        board.remove_piece_at(move.from_square)
        travel_end = 0.63 if move.promotion else 0.82
        drag = _smooth((progress - 0.18) / (travel_end - 0.18))
        if event["_capture"] and drag >= 0.75:
            captured = move.to_square - (8 if event["_piece"].color else -8) if event["_ep"] else move.to_square
            board.remove_piece_at(captured)
        rook_data = None
        if event["_castle"]:
            rank = chess.square_rank(move.from_square)
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            rook = board.remove_piece_at(rook_from)
            rook_data = (rook_from, rook_to, rook)
        panel.alpha_composite(self._position_layer(board))
        a, b = self.square_center(move.from_square), self.square_center(move.to_square)
        point = _motion(a, b, (progress - 0.18) / (travel_end - 0.18))
        self._put_piece(panel, event["_piece"].symbol(), point, 0.18 < progress < travel_end)
        if rook_data and rook_data[2]:
            rook_from, rook_to, rook = rook_data
            point = _motion(self.square_center(rook_from), self.square_center(rook_to), (progress - 0.2) / 0.64, -0.04)
            self._put_piece(panel, rook.symbol(), point, progress < 0.84)
        if self.promotion_choices_at(time):
            self._promotion_menu(panel, event, progress)

    def _promotion_menu(self, panel: Image.Image, event: dict, progress: float) -> None:
        move = event["_move"]
        left, top, down = self._promotion_geometry(event)
        draw = ImageDraw.Draw(panel)
        shadow = (left + 3, top + 4, left + self.cell + 4, top + self.cell * 4 + 4)
        draw.rounded_rectangle(shadow, radius=6, fill=(12, 23, 35, 135))
        draw.rounded_rectangle((left, top, left + self.cell, top + self.cell * 4), radius=5, fill=(247, 242, 230, 255), outline=(*_CYAN, 255), width=2)
        for index, symbol in enumerate(("q", "r", "b", "n")):
            row = index if down else 3 - index
            py = top + row * self.cell
            if chess.piece_symbol(move.promotion) == symbol:
                draw.rectangle((left + 2, py + 2, left + self.cell - 2, py + self.cell - 2), fill=(77, 217, 212, 100 if progress < 0.89 else 185))
            self._put_piece(panel, symbol.upper() if event["_piece"].color else symbol, (left + self.cell / 2, py + self.cell / 2))

    def _arrow_visible_points(self, event: dict, time: float) -> list[tuple[float, float]]:
        source, target = event["from_square"], event["to_square"]
        a, b = self.square_center(source), self.square_center(target)
        if a == b:
            return [a]
        sf, sr = chess.square_file(chess.parse_square(source)), chess.square_rank(chess.parse_square(source))
        tf, tr = chess.square_file(chess.parse_square(target)), chess.square_rank(chess.parse_square(target))
        points = [a]
        if sorted((abs(sf - tf), abs(sr - tr))) == [1, 2]:
            points.append((a[0], b[1]) if abs(sr - tr) == 2 else (b[0], a[1]))
        points.append(b)
        lengths = [math.dist(p, q) for p, q in zip(points, points[1:])]
        growth = _smooth((time - event["start"]) / _ARROW_GROWTH)
        total = sum(lengths) * growth
        visible = [points[0]]
        for index, length in enumerate(lengths):
            if total >= length:
                visible.append(points[index + 1])
                total -= length
                if total <= 0:
                    break
            else:
                fraction = total / max(0.001, length)
                visible.append((_mix(points[index][0], points[index + 1][0], fraction), _mix(points[index][1], points[index + 1][1], fraction)))
                break
        return visible

    def _arrow(self, layer: Image.Image, event: dict, time: float) -> None:
        visible = self._arrow_visible_points(event, time)
        if len(visible) < 2 or sum(math.dist(a, b) for a, b in zip(visible, visible[1:])) < 0.5:
            return
        growth = _smooth((time - event["start"]) / _ARROW_GROWTH)
        fade = min(1.0, max(0.0, (event["end"] - time) / 0.23))
        color = (*event["_color"], round(220 * fade * growth))
        draw = ImageDraw.Draw(layer)
        shaft = max(4, round(self.cell * 0.13))
        draw.line(visible, fill=color, width=shaft, joint="curve")
        tip = visible[-1]
        previous = next(point for point in reversed(visible[:-1]) if math.dist(point, tip) > 1e-8)
        angle = math.atan2(tip[1] - previous[1], tip[0] - previous[0])
        head = self.cell * 0.32 * min(1, growth * 2)
        base = (tip[0] - head * math.cos(angle), tip[1] - head * math.sin(angle))
        wing = head * 0.55
        draw.polygon([tip, (base[0] + wing * math.sin(angle), base[1] - wing * math.cos(angle)), (base[0] - wing * math.sin(angle), base[1] + wing * math.cos(angle))], fill=color)

    def _badge(self, layer: Image.Image, event: dict, time: float) -> None:
        x, y = self.square_center(event["square"])
        pop = _smooth((time - event["start"]) / 0.2)
        radius = self.cell * 0.225 * pop
        if radius < 1:
            return
        x += self.cell * 0.31
        y -= self.cell * 0.31
        x = min(self.board_x + self.side - radius, max(self.board_x + radius, x))
        y = max(self.board_y + radius, y)
        draw = ImageDraw.Draw(layer)
        draw.ellipse((x - radius + 2, y - radius + 3, x + radius + 2, y + radius + 3), fill=(8, 27, 39, 110))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*event["_color"], 255), outline=(229, 255, 248, 220), width=1)
        label = str(event.get("label", "!!"))[:6]
        draw.text((x, y), label, anchor="mm", font=_font(round(self.cell * 0.245), True), fill=(13, 42, 53, 255))

    def _zoom(self, time: float) -> tuple[float, tuple[float, float]]:
        active = [event for event in self.events if event["kind"] == "zoom" and event["start"] <= time < event["end"]]
        if active:
            event = active[-1]
            ramp = min(0.36, (event["end"] - event["start"]) * 0.27)
            amount = min(_smooth((time - event["start"]) / max(0.001, ramp)), _smooth((event["end"] - time) / max(0.001, ramp)))
            return 1 + (event["scale"] - 1) * amount, self.square_center(event["square"])
        center = (self.width / 2, self.panel_height / 2)
        intro_scale = 1 + 0.45 * (1 - _smooth(time))
        for event in self.moves:
            start, end = event["start"] - 0.12, event["end"] + 0.5
            if start <= time < end:
                amount = min(_smooth((time - start) / 0.3), _smooth((end - time) / 0.45))
                target = self.square_center(event["_move"].to_square)
                combined = intro_scale - 1 + 0.16 * amount
                progress = 0.16 * amount / combined if combined else 1.0
                focus = (_mix(center[0], target[0], progress), _mix(center[1], target[1], progress))
                return max(intro_scale, 1 + 0.16 * amount), focus
        return intro_scale, center

    def _camera(self, panel: Image.Image, time: float) -> Image.Image:
        scale, focus = self._zoom(time)
        if scale <= 1.0001:
            return panel
        crop_w, crop_h = self.width / scale, self.panel_height / scale
        x = min(self.width - crop_w, max(0, focus[0] - crop_w / 2))
        y = min(self.panel_height - crop_h, max(0, focus[1] - crop_h / 2))
        return panel.transform(panel.size, Image.Transform.AFFINE, (1 / scale, 0, x, 0, 1 / scale, y), resample=Image.Resampling.BICUBIC)

    def _draw_cursor(self, layer: Image.Image, time: float) -> None:
        x, y = self.cursor_at(time)
        event = self._active_move(time)
        pulse = 0.0
        if event:
            progress = (time - event["start"]) / (event["end"] - event["start"])
            pulse = max(0, 1 - abs(progress - 0.16) / 0.16)
        draw = ImageDraw.Draw(layer)
        if pulse:
            radius = self.cell * (0.12 + pulse * 0.18)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(*_CYAN, round(170 * (1 - pulse * 0.5))), width=2)
        size = self.cell * 0.27
        gesture = self._cursor_gesture(time)
        if gesture:
            _, phase, progress = gesture
            radius = self.cell * (0.14 + (progress * 0.18 if phase == "release" else 0.04))
            opacity = round(210 * (1 - progress)) if phase == "release" else 210
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(*_GOLD, opacity), width=2)
            left = min(self.board_x + self.side - size * 1.7, max(self.board_x, x + size))
            top = min(self.board_y + self.side - size, max(self.board_y, y - size * 0.4))
            draw.rounded_rectangle((left, top, left + size * 0.7, top + size), radius=size * 0.3, fill=(24, 35, 43, 235), outline=(254, 254, 244, 235), width=1)
            if phase != "release":
                draw.rounded_rectangle((left + size * 0.37, top + size * 0.08, left + size * 0.6, top + size * 0.46), radius=size * 0.08, fill=(*_GOLD, 255))
            draw.line((left + size * 0.35, top, left + size * 0.35, top + size * 0.5), fill=(254, 254, 244, 235), width=1)
        points = [(x, y), (x + size * 0.1, y + size), (x + size * 0.35, y + size * 0.72), (x + size * 0.6, y + size * 1.15), (x + size * 0.79, y + size * 1.04), (x + size * 0.53, y + size * 0.62), (x + size * 0.94, y + size * 0.58)]
        draw.polygon([(px + 1.5, py + 2) for px, py in points], fill=(5, 12, 23, 100))
        draw.polygon(points, fill=(254, 254, 244, 255), outline=(27, 43, 52, 255), width=1)

    def _emotion(self, time: float) -> str:
        active = [event for event in self.events if event["kind"] == "face" and event["start"] <= time < event["end"]]
        if active:
            return active[-1]["emotion"]
        board = self.board_at(time)
        if board.is_checkmate():
            return "happy"
        move = self._active_move(time)
        if move and (move["_capture"] or move["_move"].promotion):
            return "surprised"
        return "thinking" if self._selection(time) else "neutral"

    def _mouth(self, time: float) -> int:
        if self._audio_levels:
            index = time * 50
            lo = int(index)
            if lo >= len(self._audio_levels):
                return 0
            a = self._audio_levels[lo]
            b = self._audio_levels[min(lo + 1, len(self._audio_levels) - 1)]
            return round(_mix(a, b, index - lo) * 7)
        activities = self.timeline.get("speech_activity", [])
        for activity in activities:
            if activity.get("start", 0) <= time < activity.get("end", 0):
                return round(2 + 4 * abs(math.sin(time * 12.7)))
        return 0

    def _reaction(self, time: float) -> float:
        mate = next((event["end"] for event in self.moves if event["_after"].is_checkmate()), None)
        if mate is None and self.initial.is_checkmate():
            mate = 0.0
        return _smooth((time - mate - 0.35) / 1.2) if mate is not None else 0.0

    def _face_panel(self, time: float) -> Image.Image:
        height = self.height - self.face_y
        blink_phase = (time + 0.7) % 4.3
        blink = blink_phase < 0.13 or 2.18 < blink_phase < 2.25
        face = _face(self._emotion(time), blink, self._mouth(time), self.width, height)
        reaction = self._reaction(time)
        if reaction:
            scale = 1 + 0.25 * reaction
            crop_w, crop_h = self.width / scale, height / scale
            face = face.transform(face.size, Image.Transform.AFFINE, (1 / scale, 0, (self.width - crop_w) / 2, 0, 1 / scale, (height - crop_h) * 0.35), resample=Image.Resampling.BICUBIC)
        return face

    def frame(self, time: float) -> Image.Image:
        if self._closed:
            raise RuntimeError("Renderer is closed")
        time = self._validate_time(time)
        board = self.board_at(time)
        panel = self._static.copy()
        under = Image.new("RGBA", panel.size)
        draw = ImageDraw.Draw(under)
        last = self._active_move(time) or self._last_move(time)
        if last:
            self._square_accent(draw, last["_move"].from_square, _CYAN, 0.65)
            self._square_accent(draw, last["_move"].to_square, _CYAN, 0.95)
        if board.is_check():
            king = board.king(board.turn)
            if king is not None:
                glow = Image.new("RGBA", panel.size)
                glow_draw = ImageDraw.Draw(glow)
                glow_draw.ellipse(self._box(king, -self.cell * 0.06), fill=(255, 54, 65, round(145 + 40 * math.sin(time * 6))))
                under.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(2, self.cell * 0.14))))
                draw = ImageDraw.Draw(under)
                self._square_accent(draw, king, (255, 83, 90))
        annotations = self.annotations_at(time)
        for event in annotations:
            if event["kind"] == "highlight":
                fade = min(_smooth((time - event["start"]) / 0.18), min(1, (event["end"] - time) / 0.2))
                self._square_accent(draw, event["square"], event["_color"], fade)
        selection = self._selection(time)
        if selection:
            self._square_accent(draw, selection, _CYAN)
            for destination in self.legal_destinations(selection, time):
                x, y = self.square_center(destination)
                radius = self.cell * 0.105
                if board.piece_at(chess.parse_square(destination)):
                    draw.ellipse(self._box(destination, self.cell * 0.08), outline=(250, 244, 224, 150), width=max(3, round(self.cell * 0.06)))
                else:
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(250, 244, 224, 175))
        panel.alpha_composite(under)
        move = self._active_move(time)
        if move:
            self._draw_move(panel, move, time)
        else:
            panel.alpha_composite(self._position_layer(board))
        over = Image.new("RGBA", panel.size)
        for event in annotations:
            if event["kind"] == "arrow":
                self._arrow(over, event, time)
            elif event["kind"] == "badge":
                self._badge(over, event, time)
        self._draw_cursor(over, time)
        panel.alpha_composite(over)
        panel = self._camera(panel, time)
        result = self.result_at(time)
        if result:
            draw = ImageDraw.Draw(panel)
            font = _font(round(self.width * 0.028), True)
            text_width = draw.textlength(result, font=font)
            padding = self.width * 0.04
            x = (self.width - text_width) / 2
            y = self.panel_height * 0.49
            draw.rounded_rectangle((x - padding, y - padding * 0.65, x + text_width + padding, y + padding * 1.5), radius=9, fill=(13, 32, 43, 240), outline=(*_CYAN, 255), width=2)
            draw.text((self.width / 2, y + padding * 0.4), result, anchor="mm", fill=(236, 254, 246, 255), font=font)
        output = Image.new("RGB", (self.width, self.height), (8, 17, 27))
        output.paste(panel.convert("RGB"), (0, 0))
        face = self._face_panel(time)
        output.paste(face.convert("RGB"), (0, self.face_y))
        reaction = self._reaction(time)
        if reaction:
            extra = round(self.height * 0.12 * reaction)
            enlarged = face.resize((self.width, self.height - self.face_y + extra), Image.Resampling.LANCZOS)
            output.paste(enlarged.convert("RGB"), (0, self.face_y - extra))
        return output

    def close(self) -> None:
        self._position_cache.clear()
        self._closed = True

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@lru_cache(maxsize=3)
def _cached_renderer(serialized: str) -> Renderer:
    return Renderer(json.loads(serialized))


def render_frame(timeline: dict, time: float) -> Image.Image:
    return _cached_renderer(json.dumps(timeline, sort_keys=True, allow_nan=False)).frame(time)


def clear_render_cache() -> None:
    _cached_renderer.cache_clear()
    _piece.cache_clear()
    _piece_shadow.cache_clear()
    _wood_board.cache_clear()
    _face.cache_clear()
    _face_base.cache_clear()


def _encode(renderer: Renderer, output: Path) -> None:
    frames = math.ceil(renderer.duration * renderer.fps)
    encoded_duration = frames / renderer.fps
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{renderer.width}x{renderer.height}", "-r", str(renderer.fps), "-i", "pipe:0"]
    if renderer.audio_path:
        command.extend(["-i", str(renderer.audio_path)])
    else:
        command.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
    command.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "aac", "-b:a", "128k", "-af", "apad", "-t", f"{encoded_duration:.9f}", "-movflags", "+faststart", "-f", "mp4", str(output)])
    with tempfile.TemporaryFile() as error_log:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
        except OSError as error:
            raise RuntimeError(f"ffmpeg encoder could not start: {error}") from error
        try:
            if process.stdin is None:
                raise RuntimeError("ffmpeg has no input pipe")
            for index in range(frames):
                process.stdin.write(renderer.frame(min(index / renderer.fps, renderer.duration)).tobytes())
            process.stdin.close()
            returncode = process.wait()
            if returncode:
                error_log.seek(0)
                raise RuntimeError(f"ffmpeg encoding failed ({returncode}): {error_log.read().decode(errors='replace')[-4000:]}")
        except BaseException as error:
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait()
            if isinstance(error, BrokenPipeError):
                error_log.seek(0)
                raise RuntimeError(f"ffmpeg encoding failed: {error_log.read().decode(errors='replace')[-4000:]}") from error
            raise


def render_video(timeline: dict, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with Renderer(timeline) as renderer:
            descriptor, name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent)
            os.close(descriptor)
            temporary = Path(name)
            _encode(renderer, temporary)
            if temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced an empty video")
            os.replace(temporary, output)
            temporary = None
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _showcase_move(board: chess.Board, uci: str, start: float, end: float, ply: int = 1) -> dict:
    move = chess.Move.from_uci(uci)
    event = {"kind": "move", "start": start, "end": end, "before_fen": board.fen(), "uci": uci, "san": board.san(move), "ply": ply, "is_capture": board.is_capture(move), "is_castling": board.is_castling(move), "is_en_passant": board.is_en_passant(move), "promotion": chess.piece_symbol(move.promotion) if move.promotion else None}
    board.push(move)
    event.update(fen=board.fen(), is_check=board.is_check(), is_checkmate=board.is_checkmate())
    return event


def _showcase_sound(path: Path, duration: float, section: int) -> None:
    sample_rate = 16000
    samples = array.array("h")
    for index in range(round(duration * sample_rate)):
        time = index / sample_rate
        phase = time % 0.74
        envelope = max(0, math.sin(math.pi * min(1, phase / 0.36))) ** 2 if phase < 0.36 else 0
        frequency = (220, 277.18, 329.63, 440)[(int(time / 0.74) + section) % 4]
        value = math.sin(math.tau * frequency * time) + 0.24 * math.sin(math.tau * frequency * 2 * time)
        samples.append(round(2600 * value * envelope))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


def write_showcase(output_path: str | Path, *, width: int = 608, height: int = 1080, fps: int = 30) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scenarios = [
        (chess.STARTING_FEN, ["e2e4", "e7e5", "g1f3"], "worried", 7.0, "white"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"], "thinking", 5.8, "black"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"], "surprised", 4.8, "white"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8r"], "surprised", 5.4, "white"),
        ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", ["e1e8"], "happy", 5.6, "white"),
    ]
    temporary: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="chess-showcase-", dir=output.parent) as directory:
            workspace = Path(directory)
            segments = []
            for index, (fen, moves, emotion, duration, orientation) in enumerate(scenarios):
                board = chess.Board(fen)
                events: list[dict] = [
                    {"kind": "face", "start": 0, "end": duration, "emotion": emotion},
                    {"kind": "cursor", "start": 0.05, "end": 0.8, "square": moves[0][:2]},
                ]
                start = 1.05
                for ply, uci in enumerate(moves, 1):
                    end = start + (1.65 if len(uci) == 5 else 1.05)
                    events.append(_showcase_move(board, uci, start, end, ply))
                    events.append({"kind": "zoom", "start": start - 0.25, "end": min(duration - 0.1, end + 0.55), "square": uci[2:4], "scale": 1.7 if len(uci) == 5 else 1.28})
                    start = end + 0.62
                if index == 0:
                    events.extend([
                        {"kind": "arrow", "start": 0.18, "from_square": "g1", "to_square": "f3", "color": "#ffc53f"},
                        {"kind": "arrow", "start": 0.45, "end": 6.8, "from_square": "f1", "to_square": "c4", "color": "#ffc53f"},
                        {"kind": "highlight", "start": 0.6, "end": 6.8, "square": "e4", "color": "#ff6b73"},
                        {"kind": "badge", "start": 4.95, "end": 6.8, "square": "f3", "label": "!!", "color": "#41e6e8"},
                        {"kind": "clear", "start": 6.3, "end": 6.3},
                    ])
                elif index == 2:
                    events.extend([
                        {"kind": "highlight", "start": 0.2, "end": 3.4, "square": "d5", "color": "#ff6b73"},
                        {"kind": "arrow", "start": 0.2, "end": 3.2, "from_square": "e5", "to_square": "d6", "color": "#ffc53f"},
                    ])
                elif index == 4:
                    events.extend([
                        {"kind": "arrow", "start": 0.1, "end": 1.0, "from_square": "e1", "to_square": "e8", "color": "#ffc53f"},
                        {"kind": "result", "start": 2.25, "end": duration, "label": "White wins"},
                        {"kind": "badge", "start": 2.3, "end": 4.6, "square": "e8", "label": "!!", "color": "#41e6e8"},
                    ])
                audio = workspace / f"sound-{index}.wav"
                _showcase_sound(audio, duration, index)
                data = {"version": 1, "width": width, "height": height, "fps": fps, "duration": duration, "audio_path": str(audio), "initial_fen": fen, "orientation": orientation, "title": "Original animation catalogue", "events": events}
                segment = workspace / f"segment-{index}.mp4"
                render_video(data, segment)
                segments.append(segment)
            manifest = workspace / "concat.txt"
            manifest.write_text("".join(f"file '{segment.name}'\n" for segment in segments))
            descriptor, name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent)
            os.close(descriptor)
            temporary = Path(name)
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(manifest), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags", "+faststart", str(temporary)]
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except (OSError, subprocess.CalledProcessError) as error:
                detail = error.stderr.decode(errors="replace")[-4000:] if isinstance(error, subprocess.CalledProcessError) else str(error)
                raise RuntimeError(f"ffmpeg showcase assembly failed: {detail}") from error
            if not temporary.stat().st_size:
                raise RuntimeError("ffmpeg produced an empty showcase")
            os.replace(temporary, output)
            temporary = None
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
