from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_ignore_empty=True, extra='ignore', allow_inf_nan=False, hide_input_in_errors=True, populate_by_name=True)

    llm_provider: Literal['gemini', 'openai', 'vertex'] = 'gemini'
    llm_model: str | None = None
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices('GEMINI_API_KEY', 'GEMINI_FALLBACK_API_KEY', 'gemini_api_key'))
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_response_format: Literal['json_schema', 'json_object'] = 'json_schema'
    llm_max_output_tokens: int = Field(default=24000, ge=512, le=64000)
    llm_max_input_characters: int = Field(default=180000, ge=1000, le=1000000)
    llm_max_repairs: int = Field(default=1, ge=0, le=2)
    llm_temperature: float = Field(default=0.65, ge=0, le=2)
    language: Literal['en', 'ru'] = 'en'
    perspective: Literal['white', 'black'] = 'white'
    vertex_project: str | None = None
    vertex_location: str = 'global'
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str = 'pNInz6obpgDQGcFmaJgB'
    elevenlabs_model_id: Literal['eleven_multilingual_v2', 'eleven_v3'] = 'eleven_multilingual_v2'
    elevenlabs_stability: float = Field(default=0.4, ge=0, le=1)
    elevenlabs_similarity_boost: float = Field(default=0.75, ge=0, le=1)
    elevenlabs_style: float = Field(default=0.35, ge=0, le=1)
    elevenlabs_speed: float = Field(default=1.0, ge=0.7, le=1.2)
    elevenlabs_use_speaker_boost: bool = True
    max_speech_characters: int = Field(default=6000, ge=1, le=10000)
    provider_timeout_seconds: float = Field(default=180, gt=0, le=600)
    provider_connect_timeout_seconds: float = Field(default=15, gt=0, le=60)
    retry_uncertain_requests: bool = False
    provider_429_retries: int = Field(default=2, ge=0, le=3)
    retry_backoff_seconds: float = Field(default=1, ge=0, le=10)
    max_retry_after_seconds: float = Field(default=30, ge=0, le=60)
    max_provider_response_bytes: int = Field(default=64000000, ge=1024, le=128000000)
    ffprobe_executable: str = 'ffprobe'
    ffprobe_timeout_seconds: float = Field(default=30, gt=0, le=60)
    cache_dir: Path = Path('output/.cache/providers')
    generation_dir: Path | None = None
    stockfish_path: str | None = None
    api_token: SecretStr | None = Field(default=None, validation_alias=AliasChoices('CHESS_VIDEO_API_TOKEN', 'API_TOKEN', 'api_token'))
    job_root: Path = Field(default=Path('.jobs'), validation_alias=AliasChoices('CHESS_VIDEO_JOB_ROOT', 'JOB_ROOT', 'job_root'))

    @field_validator('llm_base_url')
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('LLM_BASE_URL must be an explicit HTTP(S) API base URL')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('LLM_BASE_URL cannot contain credentials, query, or fragment')
        if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Remote LLM_BASE_URL requires HTTPS')
        return value.rstrip('/')

    @field_validator('llm_model')
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 200 or any(not (character.isalnum() or character in '-_.:/@') for character in value)):
            raise ValueError('Model identifier contains unsupported characters')
        return value

    @field_validator('vertex_project', 'vertex_location', 'elevenlabs_voice_id')
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is not None and (not value or any(not (character.isalnum() or character in '-_.:') for character in value)):
            raise ValueError('Provider identifier contains unsupported characters')
        return value

    @model_validator(mode='after')
    def default_model(self) -> 'Settings':
        if self.llm_model is None and self.llm_provider != 'openai':
            self.llm_model = 'gemini-3.1-pro-preview' if self.llm_provider == 'vertex' else 'gemini-2.5-flash'
        if self.elevenlabs_model_id == 'eleven_v3' and self.elevenlabs_stability not in {0, 0.5, 1}:
            raise ValueError('eleven_v3 stability must be 0, 0.5, or 1')
        return self
