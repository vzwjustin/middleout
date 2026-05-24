from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BLOCKED_AUTH_ENV_VARS = (
    # Anthropic-native API-key / token env vars.
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BEARER_TOKEN",
    # Proxy-side overrides that would force API-key mode.
    "PROXY_ANTHROPIC_API_KEY",
    "PROXY_AUTH_MODE",
    "PROXY_FORCE_API_KEY",
    # Claude Code helper hooks / Bedrock / Vertex switches.
    "CLAUDE_CODE_API_KEY_HELPER",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_BEDROCK_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "VERTEX_API_KEY",
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

    # LLMLingua-2: cache-aware compression of the volatile tail. Opt-in; needs
    # the [lingua] install extra and downloads a ~200MB BERT model on first use.
    lingua_enabled: bool = _bool_env("BRAIN_LINGUA_ENABLED", False)
    lingua_ratio: float = _float_env("BRAIN_LINGUA_RATIO", 0.5)
    lingua_model_id: str = os.getenv(
        "BRAIN_LINGUA_MODEL",
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    )
    # When the incoming request has NO cache_control marker, the proxy can stamp
    # one after [system][tools] so the prefix becomes cacheable upstream. The
    # mutation is documented in the response header `x-brain-wall-inserted`.
    auto_insert_cache_wall: bool = _bool_env("BRAIN_AUTO_INSERT_WALL", True)

    # L1 exact-match response cache (Phase 2). SHA-256 of the normalized
    # post-compression payload keys a SQLite store of full responses. Off by
    # default — opt-in until response-replay semantics are vetted in production.
    # Streaming requests are intentionally not cached in this phase.
    l1_cache_enabled: bool = _bool_env("BRAIN_L1_CACHE_ENABLED", False)
    l1_cache_db_path: str = os.getenv("BRAIN_L1_CACHE_DB", ":memory:")
    l1_cache_max_entries: int = _int_env("BRAIN_L1_CACHE_MAX_ENTRIES", 10_000)
    l1_cache_max_body_bytes: int = _int_env("BRAIN_L1_CACHE_MAX_BODY_BYTES", 5 * 1024 * 1024)

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
    # `is not None` catches empty-string env vars too. An empty string is still
    # an explicit user intent to *set* the variable, even if downstream code
    # treats it as unset; we refuse to start in either case.
    # The .upper() comparison defends against POSIX case-sensitive env on macOS/Linux
    # where a user might set `anthropic_api_key=...` and slip through an exact-case lookup.
    blocked = []
    for env_name in os.environ:
        if env_name.upper() in BLOCKED_AUTH_ENV_VARS and os.environ.get(env_name) is not None:
            blocked.append(env_name)
    # Also check exact-case matches even if not in os.environ (defensive — handles
    # any future code path that explicitly sets one of these names with empty value).
    for name in BLOCKED_AUTH_ENV_VARS:
        if name not in blocked and os.getenv(name) is not None:
            blocked.append(name)
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
    if not 0.05 <= settings.lingua_ratio <= 0.95:
        raise ValueError("BRAIN_LINGUA_RATIO must be between 0.05 and 0.95")
    return settings
