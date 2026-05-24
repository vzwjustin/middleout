from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BLOCKED_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "PROXY_ANTHROPIC_API_KEY",
    "PROXY_AUTH_MODE",
    "PROXY_FORCE_API_KEY",
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    # Local listener
    host: str = os.getenv("MIDDLEOUT_HOST", "127.0.0.1")
    port: int = _int_env("MIDDLEOUT_PORT", 8787)
    reload: bool = _bool_env("MIDDLEOUT_RELOAD", False)

    # Never use ANTHROPIC_BASE_URL here; Claude Code points that to this proxy.
    upstream_base_url: str = os.getenv("PROXY_UPSTREAM_BASE_URL", "https://api.anthropic.com")

    # Fixed mode: no API-key injection, no API-key fallback, OAuth passthrough only.
    auth_mode: str = "subscription_oauth_passthrough_only"

    # Input compaction. This is intentionally conservative: normal prompts pass through unchanged.
    input_compression_enabled: bool = _bool_env("MIDDLEOUT_INPUT_COMPRESSION", True)
    max_text_chars: int = _int_env("MIDDLEOUT_MAX_TEXT_CHARS", 12_000)
    min_omission_chars: int = _int_env("MIDDLEOUT_MIN_OMISSION_CHARS", 1_500)
    head_fraction: float = _float_env("MIDDLEOUT_HEAD_FRACTION", 0.55)
    compress_system: bool = _bool_env("MIDDLEOUT_COMPRESS_SYSTEM", False)
    compress_tool_results: bool = _bool_env("MIDDLEOUT_COMPRESS_TOOL_RESULTS", True)

    # JL-style sign sketch dedupe. It only removes near-duplicates already present in the same request.
    jl_dedupe_enabled: bool = _bool_env("MIDDLEOUT_JL_DEDUPE", True)
    jl_dims: int = _int_env("MIDDLEOUT_JL_DIMS", 512)
    jl_shingle_tokens: int = _int_env("MIDDLEOUT_JL_SHINGLE_TOKENS", 5)
    jl_similarity_threshold: float = _float_env("MIDDLEOUT_JL_SIMILARITY", 0.985)
    jl_min_chars: int = _int_env("MIDDLEOUT_JL_MIN_CHARS", 4_000)

    # Output compaction is off by default. It can break agent tooling if the client expects exact output.
    output_compression_enabled: bool = _bool_env("MIDDLEOUT_OUTPUT_COMPRESSION", True)
    output_max_text_chars: int = _int_env("MIDDLEOUT_OUTPUT_MAX_TEXT_CHARS", 20_000)

    # Caveman: terse-text engine — drops articles, filler, pleasantries.
    caveman_enabled: bool = _bool_env("MIDDLEOUT_CAVEMAN", False)
    caveman_level: str = os.getenv("MIDDLEOUT_CAVEMAN_LEVEL", "standard")

    # RTK: dictionary-based phrase abbreviation.
    rtk_enabled: bool = _bool_env("MIDDLEOUT_RTK", False)
    rtk_level: str = os.getenv("MIDDLEOUT_RTK_LEVEL", "minimal")

    # Anthropic prompt cache preservation. Never touched blocks at-or-before a
    # cache_control marker (mutating them would invalidate the upstream cache).
    preserve_anthropic_cache: bool = _bool_env("MIDDLEOUT_PRESERVE_ANTHROPIC_CACHE", True)

    # Local LRU cache for deterministic post-JL compression output. Independent of
    # Anthropic's native cache; only avoids local CPU work on repeated text.
    compression_cache_enabled: bool = _bool_env("MIDDLEOUT_COMPRESSION_CACHE", True)
    compression_cache_size: int = _int_env("MIDDLEOUT_COMPRESSION_CACHE_SIZE", 256)

    # Observability. Content is never logged unless explicitly enabled.
    audit_enabled: bool = _bool_env("MIDDLEOUT_AUDIT", True)
    audit_log_dir: Path = Path(os.getenv("MIDDLEOUT_AUDIT_DIR", ".middleout-logs"))
    log_text_samples: bool = _bool_env("MIDDLEOUT_LOG_TEXT_SAMPLES", False)
    log_json: bool = _bool_env("MIDDLEOUT_LOG_JSON", False)
    timeseries_minutes: int = _int_env("MIDDLEOUT_TIMESERIES_MINUTES", 60)
    recent_max: int = _int_env("MIDDLEOUT_RECENT_MAX", 200)

    # HTTP behavior
    timeout_connect_s: float = _float_env("MIDDLEOUT_CONNECT_TIMEOUT_S", 30.0)
    timeout_read_s: float = _float_env("MIDDLEOUT_READ_TIMEOUT_S", 600.0)
    default_anthropic_version: str = os.getenv("MIDDLEOUT_ANTHROPIC_VERSION", "2023-06-01")

    @property
    def upstream(self) -> str:
        return self.upstream_base_url.rstrip("/")


def load_settings() -> Settings:
    blocked = [name for name in BLOCKED_AUTH_ENV_VARS if os.getenv(name)]
    if blocked:
        names = ", ".join(blocked)
        raise ValueError(
            "Strict subscription-only mode refuses proxy-side auth environment variables. "
            f"Unset these before starting middleout-proxy: {names}"
        )

    settings = Settings()
    if settings.max_text_chars < 512:
        raise ValueError("MIDDLEOUT_MAX_TEXT_CHARS must be at least 512")
    if not 0.05 <= settings.head_fraction <= 0.95:
        raise ValueError("MIDDLEOUT_HEAD_FRACTION must be between 0.05 and 0.95")
    if settings.jl_dims < 16:
        raise ValueError("MIDDLEOUT_JL_DIMS must be at least 16")
    if settings.caveman_level not in {"lite", "standard", "aggressive", "ultra"}:
        raise ValueError(
            "MIDDLEOUT_CAVEMAN_LEVEL must be one of lite/standard/aggressive/ultra, "
            f"got {settings.caveman_level!r}"
        )
    if settings.rtk_level not in {"minimal", "standard", "aggressive"}:
        raise ValueError(
            "MIDDLEOUT_RTK_LEVEL must be one of minimal/standard/aggressive, "
            f"got {settings.rtk_level!r}"
        )
    if settings.compression_cache_size < 0:
        raise ValueError("MIDDLEOUT_COMPRESSION_CACHE_SIZE must be >= 0")
    return settings
