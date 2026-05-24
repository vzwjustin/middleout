"""Runtime configuration: TOML file + environment variables + sane defaults.

Precedence (highest → lowest):

1. Environment variables (per-shell overrides, ops-friendly).
2. TOML config file (long-form deployment config).
3. Hard-coded defaults in :class:`Settings`.

TOML lookup order (first match wins):

1. The path in ``MIDDLEOUT_CONFIG`` if set.
2. ``./middleout.toml`` in the proxy's working directory.
3. ``~/.config/middleout/middleout.toml``.

The TOML schema mirrors the env var surface but namespaced into sections —
see :func:`_apply_toml` for the field map. Unknown TOML keys are tolerated
(forward compat) but logged. Invalid value types raise immediately.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


def _load_toml_defaults() -> dict[str, Any]:
    """Locate and parse a TOML config file. Return a flat field→value map.

    Returns an empty dict if no file is found or tomllib isn't available
    (Python <3.11). The flat map uses the same field names as :class:`Settings`
    so :func:`load_settings` can apply it uniformly.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.debug("tomllib not available; skipping TOML config")
            return {}

    explicit = os.getenv("MIDDLEOUT_CONFIG")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / "middleout.toml")
    candidates.append(Path.home() / ".config" / "middleout" / "middleout.toml")

    for path in candidates:
        if path.is_file():
            try:
                with path.open("rb") as f:
                    raw = tomllib.load(f)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ValueError(
                    f"Failed to read MiddleOut TOML config at {path}: {exc}"
                ) from exc
            flat = _flatten_toml(raw)
            logger.info("MiddleOut config loaded from %s (%d fields)", path, len(flat))
            return flat
    return {}


# Maps a TOML (section, key) to a Settings field name. TOML uses snake_case
# sections + snake_case keys; we collapse to the flat Settings field name. Any
# (section, key) not in this map is logged at debug level and ignored.
_TOML_FIELD_MAP: dict[tuple[str, str], str] = {
    ("server", "host"): "host",
    ("server", "port"): "port",
    ("server", "reload"): "reload",
    ("server", "upstream_base_url"): "upstream_base_url",
    ("server", "connect_timeout_s"): "timeout_connect_s",
    ("server", "read_timeout_s"): "timeout_read_s",
    ("server", "anthropic_version"): "default_anthropic_version",
    ("compression", "input_enabled"): "input_compression_enabled",
    ("compression", "output_enabled"): "output_compression_enabled",
    ("compression", "max_text_chars"): "max_text_chars",
    ("compression", "output_max_text_chars"): "output_max_text_chars",
    ("compression", "min_omission_chars"): "min_omission_chars",
    ("compression", "head_fraction"): "head_fraction",
    ("compression", "compress_system"): "compress_system",
    ("compression", "compress_tool_results"): "compress_tool_results",
    ("compression", "preserve_anthropic_cache"): "preserve_anthropic_cache",
    ("compression", "cache_enabled"): "compression_cache_enabled",
    ("compression", "cache_size"): "compression_cache_size",
    ("jl", "enabled"): "jl_dedupe_enabled",
    ("jl", "dims"): "jl_dims",
    ("jl", "shingle_tokens"): "jl_shingle_tokens",
    ("jl", "similarity_threshold"): "jl_similarity_threshold",
    ("jl", "min_chars"): "jl_min_chars",
    ("caveman", "enabled"): "caveman_enabled",
    ("caveman", "level"): "caveman_level",
    ("rtk", "enabled"): "rtk_enabled",
    ("rtk", "level"): "rtk_level",
    ("json_aware", "enabled"): "json_aware_enabled",
    ("json_aware", "level"): "json_aware_level",
    ("lsh", "enabled"): "lsh_enabled",
    ("lsh", "level"): "lsh_level",
    ("adaptive", "enabled"): "adaptive_enabled",
    ("lingua", "enabled"): "lingua_enabled",
    ("lingua", "ratio"): "lingua_ratio",
    ("lingua", "model_id"): "lingua_model_id",
    ("cache_wall", "auto_insert"): "auto_insert_cache_wall",
    ("l1_cache", "enabled"): "l1_cache_enabled",
    ("l1_cache", "db_path"): "l1_cache_db_path",
    ("l1_cache", "max_entries"): "l1_cache_max_entries",
    ("l1_cache", "max_body_bytes"): "l1_cache_max_body_bytes",
    ("l2_cache", "enabled"): "l2_cache_enabled",
    ("l2_cache", "similarity_threshold"): "l2_similarity_threshold",
    ("audit", "enabled"): "audit_enabled",
    ("audit", "log_dir"): "audit_log_dir",
    ("audit", "log_text_samples"): "log_text_samples",
    ("audit", "log_json"): "log_json",
    ("audit", "timeseries_minutes"): "timeseries_minutes",
    ("audit", "recent_max"): "recent_max",
}


def _flatten_toml(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a nested TOML doc into a flat ``{settings_field: value}`` map."""
    out: dict[str, Any] = {}
    for section, body in raw.items():
        if not isinstance(body, dict):
            logger.debug("TOML section %r is not a table; ignored", section)
            continue
        for key, value in body.items():
            mapped = _TOML_FIELD_MAP.get((section, key))
            if mapped is None:
                logger.debug("TOML key [%s].%s not recognized; ignored", section, key)
                continue
            out[mapped] = value
    return out


# Internal: the TOML defaults loaded once at module-import time. Set to an
# empty dict when no file exists. `load_settings()` reads from this and from
# the live environment so per-test monkeypatching of env vars works without
# re-parsing TOML.
_TOML_DEFAULTS: dict[str, Any] = {}


def _toml_default(field_name: str, fallback: Any) -> Any:
    """Resolve `field_name` from TOML, falling back to `fallback`."""
    return _TOML_DEFAULTS.get(field_name, fallback)


def _resolve_path(value: Any, fallback: Path) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    return fallback


@dataclass(frozen=True)
class Settings:
    # Local listener
    host: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_HOST", _toml_default("host", "127.0.0.1")
    ))
    port: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_PORT", _toml_default("port", 8787)
    ))
    reload: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_RELOAD", _toml_default("reload", False)
    ))

    # Never use ANTHROPIC_BASE_URL here; Claude Code points that to this proxy.
    upstream_base_url: str = field(default_factory=lambda: os.getenv(
        "PROXY_UPSTREAM_BASE_URL",
        _toml_default("upstream_base_url", "https://api.anthropic.com"),
    ))

    # Fixed mode: no API-key injection, no API-key fallback, OAuth passthrough only.
    auth_mode: str = "subscription_oauth_passthrough_only"

    # Input compaction. This is intentionally conservative: normal prompts pass through unchanged.
    input_compression_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_INPUT_COMPRESSION", _toml_default("input_compression_enabled", True)
    ))
    max_text_chars: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_MAX_TEXT_CHARS", _toml_default("max_text_chars", 12_000)
    ))
    min_omission_chars: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_MIN_OMISSION_CHARS", _toml_default("min_omission_chars", 1_500)
    ))
    head_fraction: float = field(default_factory=lambda: _float_env(
        "MIDDLEOUT_HEAD_FRACTION", _toml_default("head_fraction", 0.55)
    ))
    compress_system: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_COMPRESS_SYSTEM", _toml_default("compress_system", False)
    ))
    compress_tool_results: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_COMPRESS_TOOL_RESULTS", _toml_default("compress_tool_results", True)
    ))

    # JL-style sign sketch dedupe. It only removes near-duplicates already present in the same request.
    jl_dedupe_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_JL_DEDUPE", _toml_default("jl_dedupe_enabled", True)
    ))
    jl_dims: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_JL_DIMS", _toml_default("jl_dims", 512)
    ))
    jl_shingle_tokens: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_JL_SHINGLE_TOKENS", _toml_default("jl_shingle_tokens", 5)
    ))
    jl_similarity_threshold: float = field(default_factory=lambda: _float_env(
        "MIDDLEOUT_JL_SIMILARITY", _toml_default("jl_similarity_threshold", 0.985)
    ))
    jl_min_chars: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_JL_MIN_CHARS", _toml_default("jl_min_chars", 4_000)
    ))

    # Output compaction is off by default. It can break agent tooling if the client expects exact output.
    output_compression_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_OUTPUT_COMPRESSION", _toml_default("output_compression_enabled", True)
    ))
    output_max_text_chars: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_OUTPUT_MAX_TEXT_CHARS", _toml_default("output_max_text_chars", 20_000)
    ))

    # Caveman: terse-text engine — drops articles, filler, pleasantries.
    caveman_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_CAVEMAN", _toml_default("caveman_enabled", False)
    ))
    caveman_level: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_CAVEMAN_LEVEL", _toml_default("caveman_level", "standard")
    ))

    # RTK: dictionary-based phrase abbreviation.
    rtk_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_RTK", _toml_default("rtk_enabled", False)
    ))
    rtk_level: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_RTK_LEVEL", _toml_default("rtk_level", "minimal")
    ))

    # JSON-aware engine: structural compression that preserves JSON validity.
    # "safe" only drops conservative noise; "aggressive" rewrites large arrays.
    json_aware_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_JSON_AWARE", _toml_default("json_aware_enabled", False)
    ))
    json_aware_level: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_JSON_AWARE_LEVEL", _toml_default("json_aware_level", "safe")
    ))

    # LSH dedupe: MinHash/SimHash near-duplicate block elimination across the
    # whole request. Stronger than the JL sign-sketch for very large payloads.
    lsh_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_LSH", _toml_default("lsh_enabled", False)
    ))
    lsh_level: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_LSH_LEVEL", _toml_default("lsh_level", "standard")
    ))

    # Adaptive policy: pick engine levels at runtime from payload size +
    # estimated context window. Off by default — uses static config.
    adaptive_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_ADAPTIVE", _toml_default("adaptive_enabled", False)
    ))

    # Anthropic prompt cache preservation. Never touched blocks at-or-before a
    # cache_control marker (mutating them would invalidate the upstream cache).
    preserve_anthropic_cache: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_PRESERVE_ANTHROPIC_CACHE", _toml_default("preserve_anthropic_cache", True)
    ))

    # LLMLingua-2: cache-aware compression of the volatile tail. Opt-in; needs
    # the [lingua] install extra and downloads a ~200MB BERT model on first use.
    lingua_enabled: bool = field(default_factory=lambda: _bool_env(
        "BRAIN_LINGUA_ENABLED", _toml_default("lingua_enabled", False)
    ))
    lingua_ratio: float = field(default_factory=lambda: _float_env(
        "BRAIN_LINGUA_RATIO", _toml_default("lingua_ratio", 0.5)
    ))
    lingua_model_id: str = field(default_factory=lambda: os.getenv(
        "BRAIN_LINGUA_MODEL",
        _toml_default(
            "lingua_model_id",
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        ),
    ))
    # When the incoming request has NO cache_control marker, the proxy can stamp
    # one after [system][tools] so the prefix becomes cacheable upstream. The
    # mutation is documented in the response header `x-brain-wall-inserted`.
    auto_insert_cache_wall: bool = field(default_factory=lambda: _bool_env(
        "BRAIN_AUTO_INSERT_WALL", _toml_default("auto_insert_cache_wall", True)
    ))

    # L1 exact-match response cache (Phase 2). SHA-256 of the normalized
    # post-compression payload keys a SQLite store of full responses. Off by
    # default — opt-in until response-replay semantics are vetted in production.
    # Streaming requests are intentionally not cached in this phase.
    l1_cache_enabled: bool = field(default_factory=lambda: _bool_env(
        "BRAIN_L1_CACHE_ENABLED", _toml_default("l1_cache_enabled", False)
    ))
    l1_cache_db_path: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L1_CACHE_DB", _toml_default("l1_cache_db_path", ":memory:")
    ))
    l1_cache_max_entries: int = field(default_factory=lambda: _int_env(
        "BRAIN_L1_CACHE_MAX_ENTRIES", _toml_default("l1_cache_max_entries", 10_000)
    ))
    l1_cache_max_body_bytes: int = field(default_factory=lambda: _int_env(
        "BRAIN_L1_CACHE_MAX_BODY_BYTES",
        _toml_default("l1_cache_max_body_bytes", 5 * 1024 * 1024),
    ))

    # L2 semantic cache (Phase 2b). Always behind the operator wiring an
    # embedding client + vector store — the toggle alone is not enough.
    l2_cache_enabled: bool = field(default_factory=lambda: _bool_env(
        "BRAIN_L2_CACHE_ENABLED", _toml_default("l2_cache_enabled", False)
    ))
    l2_similarity_threshold: float = field(default_factory=lambda: _float_env(
        "BRAIN_L2_SIMILARITY", _toml_default("l2_similarity_threshold", 0.97)
    ))
    # L2 backend selection. "in_memory" is stdlib-only, bounded by
    # l2_max_entries. "qdrant" requires the qdrant-client package + URL/key.
    l2_backend: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_BACKEND", _toml_default("l2_backend", "in_memory")
    ))
    l2_max_entries: int = field(default_factory=lambda: _int_env(
        "BRAIN_L2_MAX_ENTRIES", _toml_default("l2_max_entries", 10_000)
    ))
    l2_qdrant_url: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_QDRANT_URL", _toml_default("l2_qdrant_url", "")
    ))
    l2_qdrant_collection: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_QDRANT_COLLECTION", _toml_default("l2_qdrant_collection", "brain_proxy_l2")
    ))
    l2_qdrant_api_key: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_QDRANT_API_KEY", _toml_default("l2_qdrant_api_key", "")
    ))
    # L2 embedder selection. "hash" is stdlib, deterministic, NOT semantic.
    # "openai" calls the OpenAI Embeddings API and needs OPENAI_API_KEY.
    l2_embedder: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_EMBEDDER", _toml_default("l2_embedder", "hash")
    ))
    l2_embedding_dim: int = field(default_factory=lambda: _int_env(
        "BRAIN_L2_EMBEDDING_DIM", _toml_default("l2_embedding_dim", 3072)
    ))
    l2_openai_model: str = field(default_factory=lambda: os.getenv(
        "BRAIN_L2_OPENAI_MODEL", _toml_default("l2_openai_model", "text-embedding-3-large")
    ))

    # Local LRU cache for deterministic post-JL compression output. Independent of
    # Anthropic's native cache; only avoids local CPU work on repeated text.
    compression_cache_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_COMPRESSION_CACHE", _toml_default("compression_cache_enabled", True)
    ))
    compression_cache_size: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_COMPRESSION_CACHE_SIZE", _toml_default("compression_cache_size", 256)
    ))

    # Observability. Content is never logged unless explicitly enabled.
    audit_enabled: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_AUDIT", _toml_default("audit_enabled", True)
    ))
    audit_log_dir: Path = field(default_factory=lambda: _resolve_path(
        os.getenv("MIDDLEOUT_AUDIT_DIR")
        or _toml_default("audit_log_dir", None),
        Path(".middleout-logs"),
    ))
    log_text_samples: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_LOG_TEXT_SAMPLES", _toml_default("log_text_samples", False)
    ))
    log_json: bool = field(default_factory=lambda: _bool_env(
        "MIDDLEOUT_LOG_JSON", _toml_default("log_json", False)
    ))
    timeseries_minutes: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_TIMESERIES_MINUTES", _toml_default("timeseries_minutes", 60)
    ))
    recent_max: int = field(default_factory=lambda: _int_env(
        "MIDDLEOUT_RECENT_MAX", _toml_default("recent_max", 200)
    ))

    # HTTP behavior
    timeout_connect_s: float = field(default_factory=lambda: _float_env(
        "MIDDLEOUT_CONNECT_TIMEOUT_S", _toml_default("timeout_connect_s", 30.0)
    ))
    timeout_read_s: float = field(default_factory=lambda: _float_env(
        "MIDDLEOUT_READ_TIMEOUT_S", _toml_default("timeout_read_s", 600.0)
    ))
    default_anthropic_version: str = field(default_factory=lambda: os.getenv(
        "MIDDLEOUT_ANTHROPIC_VERSION",
        _toml_default("default_anthropic_version", "2023-06-01"),
    ))

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

    # (Re)load the TOML defaults so a test that wrote middleout.toml in
    # tmp_path picks them up. The cost is one syscall + parse per
    # load_settings() call, which is once per process in normal use.
    global _TOML_DEFAULTS
    _TOML_DEFAULTS = _load_toml_defaults()

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
    if settings.json_aware_level not in {"safe", "standard", "aggressive"}:
        raise ValueError(
            "MIDDLEOUT_JSON_AWARE_LEVEL must be one of safe/standard/aggressive, "
            f"got {settings.json_aware_level!r}"
        )
    if settings.lsh_level not in {"conservative", "standard", "aggressive"}:
        raise ValueError(
            "MIDDLEOUT_LSH_LEVEL must be one of conservative/standard/aggressive, "
            f"got {settings.lsh_level!r}"
        )
    if settings.compression_cache_size < 0:
        raise ValueError("MIDDLEOUT_COMPRESSION_CACHE_SIZE must be >= 0")
    if not 0.05 <= settings.lingua_ratio <= 0.95:
        raise ValueError("BRAIN_LINGUA_RATIO must be between 0.05 and 0.95")
    if not 0.0 <= settings.l2_similarity_threshold <= 1.0:
        raise ValueError("BRAIN_L2_SIMILARITY must be between 0.0 and 1.0")
    if settings.l2_backend not in {"in_memory", "qdrant"}:
        raise ValueError(
            "BRAIN_L2_BACKEND must be one of in_memory/qdrant, "
            f"got {settings.l2_backend!r}"
        )
    if settings.l2_embedder not in {"hash", "openai"}:
        raise ValueError(
            "BRAIN_L2_EMBEDDER must be one of hash/openai, "
            f"got {settings.l2_embedder!r}"
        )
    if settings.l2_embedding_dim < 16:
        raise ValueError("BRAIN_L2_EMBEDDING_DIM must be >= 16")
    return settings


# Initialize the TOML defaults lazily on first import. This keeps the
# pre-existing import-only behavior (Settings() with hard-coded defaults works
# even when no TOML exists) and lets `load_settings()` refresh on demand.
_TOML_DEFAULTS = _load_toml_defaults()


__all__ = ["BLOCKED_AUTH_ENV_VARS", "Settings", "load_settings"]
