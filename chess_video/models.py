import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Emotion = Literal['neutral', 'worried', 'thinking', 'surprised', 'happy']
Square = str
RESULTS = {'*', '1-0', '0-1', '1/2-1/2'}
COLORS = {'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'white', 'black'}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Position(StrictModel):
    ply: int = Field(ge=0, le=2000)
    fen: str
    before_fen: str
    san: str
    uci: str
    is_capture: bool
    is_check: bool
    is_checkmate: bool
    is_castling: bool
    is_en_passant: bool
    promotion: str | None = None
    checked_king_square: str | None = None
    checking_squares: list[str] = Field(default_factory=list)
    analysis: dict | None = None


class Game(StrictModel):
    headers: dict[str, str]
    initial_fen: str
    positions: list[Position] = Field(min_length=2, max_length=2001)
    result: Literal['*', '1-0', '0-1', '1/2-1/2']
    analysis: dict | None = None


class Cue(StrictModel):
    kind: Literal['arrow', 'highlight', 'zoom', 'cursor', 'badge', 'clear', 'face', 'result']
    anchor: str = Field(min_length=1, max_length=4000)
    from_square: Square | None = None
    to_square: Square | None = None
    square: Square | None = None
    color: str | None = None
    label: str | None = Field(default=None, min_length=1, max_length=80)
    scale: float | None = Field(default=None, ge=1.0, le=3.0)
    duration: float | None = Field(default=None, gt=0, le=30)

    @field_validator('from_square', 'to_square', 'square')
    @classmethod
    def valid_square(cls, value):
        if value is not None and not re.fullmatch('[a-h][1-8]', value):
            raise ValueError('Squares must be lowercase a1 through h8')
        return value

    @field_validator('color')
    @classmethod
    def valid_color(cls, value):
        if value is not None and value not in COLORS and not re.fullmatch('#[0-9a-fA-F]{6}', value):
            raise ValueError('Color must be a supported name or #RRGGBB')
        return value

    @model_validator(mode='after')
    def valid_payload(self):
        fields = {
            'arrow': ({'from_square', 'to_square'}, {'color'}),
            'highlight': ({'square'}, {'color'}),
            'zoom': ({'square', 'scale'}, set()),
            'cursor': ({'square'}, set()),
            'badge': ({'square', 'label'}, {'color'}),
            'clear': (set(), set()),
            'face': ({'label'}, set()),
            'result': ({'label'}, set()),
        }
        required, optional = fields[self.kind]
        present = {key for key in ('from_square', 'to_square', 'square', 'color', 'label', 'scale')
                   if getattr(self, key) is not None}
        if not required <= present or present - required - optional:
            raise ValueError(f'Invalid payload for {self.kind}')
        if self.kind == 'face' and self.label not in {'neutral', 'worried', 'thinking', 'surprised', 'happy'}:
            raise ValueError('Face label must be an emotion')
        if self.kind == 'arrow' and self.from_square == self.to_square:
            raise ValueError('Arrow endpoints must differ')
        return self


class Beat(StrictModel):
    ply: int = Field(ge=0, le=2000)
    text: str = Field(min_length=1, max_length=4000)
    emotion: Emotion
    move_anchor: str | None = Field(default=None, min_length=1, max_length=4000)
    cues: list[Cue] = Field(default_factory=list, max_length=30)

    @field_validator('text')
    @classmethod
    def nonempty_text(cls, value):
        if not value.strip() or any(ord(char) < 32 and char not in '\n\t' for char in value):
            raise ValueError('Narration must contain printable text')
        return value

    @model_validator(mode='after')
    def exact_anchors(self):
        if self.move_anchor is not None and self.move_anchor not in self.text:
            raise ValueError('Move anchor must be an exact substring of its beat')
        if any(cue.anchor not in self.text for cue in self.cues):
            raise ValueError('Cue anchor must be an exact substring of its beat')
        return self


class Script(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    language: Literal['en', 'ru']
    perspective: Literal['white', 'black']
    beats: list[Beat] = Field(min_length=2, max_length=2001)

    @field_validator('title')
    @classmethod
    def nonempty_title(cls, value):
        if not value.strip():
            raise ValueError('Title cannot be blank')
        return value

    @model_validator(mode='after')
    def bounded_text(self):
        if sum(len(beat.text) for beat in self.beats) > 100_000:
            raise ValueError('Total narration exceeds 100000 characters')
        return self


class Speech(StrictModel):
    audio_path: str = Field(min_length=1, max_length=4096)
    duration: float = Field(gt=0, le=7200)
    text: str = Field(min_length=1, max_length=104_000)
    characters: list[str] = Field(min_length=1, max_length=104_000)
    character_start_times_seconds: list[float] = Field(min_length=1, max_length=104_000)
    character_end_times_seconds: list[float] = Field(min_length=1, max_length=104_000)

    @model_validator(mode='after')
    def exact_alignment(self):
        starts, ends = self.character_start_times_seconds, self.character_end_times_seconds
        if not (len(self.characters) == len(starts) == len(ends) == len(self.text)):
            raise ValueError('Character alignment arrays must have equal counts')
        if any(len(char) != 1 for char in self.characters) or ''.join(self.characters) != self.text:
            raise ValueError('Alignment characters must exactly roundtrip Unicode narration')
        previous_start = previous_end = 0.0
        for start, end in zip(starts, ends):
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError('Alignment timings must be finite')
            if not (previous_start <= start <= end <= self.duration and previous_end <= end):
                raise ValueError('Alignment must be nonnegative, monotonic and within audio duration')
            previous_start, previous_end = start, end
        return self


def narration_text(script: dict) -> str:
    return '\n\n'.join(beat['text'] for beat in script['beats'])


def validate_speech(script: dict, speech: dict) -> dict:
    result = Speech.model_validate(speech).model_dump(exclude_none=True)
    if result['text'] != narration_text(script):
        raise ValueError('Speech text differs from the exact canonical narration')
    return result
