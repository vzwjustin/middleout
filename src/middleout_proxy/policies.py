"""Per-model + per-endpoint policy overrides for compression settings.

A :class:`PolicyRouter` is given an ordered list of :class:`PolicyMatch`
rules and a default :class:`CompressionPolicy`. On each request the integration
layer calls :meth:`PolicyRouter.resolve` with the payload's ``model`` and the
URL path; the first matching rule wins (or the default if nothing matches).

Glob matching uses :func:`fnmatch.fnmatch` on ``model_glob`` so callers can
write ``"claude-opus-*"``, ``"claude-haiku-*"``, ``"*"`` etc. Endpoints are
matched literally with a single ``"*"`` meaning "any endpoint".

Policies can be loaded from a JSON document in the ``MIDDLEOUT_POLICIES``
environment variable. Invalid JSON raises :class:`ValueError`.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field, fields, replace
from typing import Any
from collections.abc import Iterable


@dataclass(frozen=True)
class CompressionPolicy:
    """Resolved compression configuration for one (model, endpoint) pair.

    Fields mirror the per-request knobs exposed in ``server._runtime``. Where
    ``max_text_chars`` is ``None`` the integration layer should fall back to
    :attr:`Settings.max_text_chars`.
    """

    input_compression: bool = True
    jl_dedupe: bool = True
    caveman_enabled: bool = False
    caveman_level: str = "standard"
    rtk_enabled: bool = False
    rtk_level: str = "minimal"
    output_compression: bool = False
    max_text_chars: int | None = None


@dataclass(frozen=True)
class PolicyMatch:
    """A single rule: model+endpoint pattern → policy."""

    model_glob: str = "*"
    endpoint: str = "*"
    policy: CompressionPolicy = field(default_factory=CompressionPolicy)


_POLICY_FIELDS = {f.name for f in fields(CompressionPolicy)}


def _coerce_max_text_chars(value: Any) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_text_chars must be an integer or null, got {value!r}") from exc
    if coerced < 0:
        raise ValueError(f"max_text_chars must be non-negative, got {coerced}")
    return coerced


def _parse_policy(raw: Any) -> CompressionPolicy:
    """Build a CompressionPolicy from a JSON dict, validating field names."""
    if raw is None:
        return CompressionPolicy()
    if not isinstance(raw, dict):
        raise ValueError(f"policy must be a JSON object, got {type(raw).__name__}")

    unknown = set(raw) - _POLICY_FIELDS
    if unknown:
        raise ValueError(
            f"unknown policy fields: {sorted(unknown)}; expected subset of {sorted(_POLICY_FIELDS)}"
        )

    kwargs: dict[str, Any] = {}
    for name in _POLICY_FIELDS:
        if name not in raw:
            continue
        value = raw[name]
        if name == "max_text_chars":
            kwargs[name] = _coerce_max_text_chars(value)
        elif name in {
            "input_compression",
            "jl_dedupe",
            "caveman_enabled",
            "rtk_enabled",
            "output_compression",
        }:
            kwargs[name] = bool(value)
        elif name in {"caveman_level", "rtk_level"}:
            kwargs[name] = str(value)

    return replace(CompressionPolicy(), **kwargs)


def _model_match(glob: str, model: str | None) -> bool:
    if model is None:
        # "*" matches the absence of a model field; anything more specific does not.
        return glob == "*"
    return fnmatch.fnmatch(model, glob)


def _endpoint_match(rule_endpoint: str, endpoint: str) -> bool:
    if rule_endpoint == "*":
        return True
    return rule_endpoint == endpoint


class PolicyRouter:
    """First-match-wins router over a list of :class:`PolicyMatch` rules."""

    def __init__(
        self,
        rules: Iterable[PolicyMatch],
        default: CompressionPolicy | None = None,
    ) -> None:
        self.rules: list[PolicyMatch] = list(rules)
        self.default: CompressionPolicy = default if default is not None else CompressionPolicy()

    def resolve(self, *, model: str | None, endpoint: str) -> CompressionPolicy:
        """Return the first matching policy, or :attr:`default` if no rule matches."""
        for rule in self.rules:
            if not _endpoint_match(rule.endpoint, endpoint):
                continue
            if not _model_match(rule.model_glob, model):
                continue
            return rule.policy
        return self.default

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> PolicyRouter:
        """Build a router from the ``MIDDLEOUT_POLICIES`` JSON env var.

        An empty / unset variable yields an empty router with the default
        :class:`CompressionPolicy`.
        """
        environ = env if env is not None else os.environ
        raw = environ.get("MIDDLEOUT_POLICIES")
        if not raw or not raw.strip():
            return cls(rules=[], default=CompressionPolicy())
        return cls.from_json(raw)

    @classmethod
    def from_json(cls, raw: str) -> PolicyRouter:
        """Parse a JSON document into a router.

        Document shape::

            {
              "default": {"input_compression": true, "jl_dedupe": true},
              "rules": [
                {"model_glob": "claude-opus-*",
                 "endpoint": "v1/messages",
                 "policy": {"caveman_enabled": true, "caveman_level": "lite"}},
                {"model_glob": "claude-haiku-*",
                 "policy": {"input_compression": false}}
              ]
            }

        Raises:
            ValueError: on invalid JSON or malformed structure.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"MIDDLEOUT_POLICIES is not valid JSON ({exc.msg} at line {exc.lineno} "
                f"col {exc.colno})"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"MIDDLEOUT_POLICIES must be a JSON object, got {type(data).__name__}"
            )

        default_policy = _parse_policy(data.get("default"))

        rules_raw = data.get("rules", [])
        if not isinstance(rules_raw, list):
            raise ValueError(
                f"MIDDLEOUT_POLICIES 'rules' must be a list, got {type(rules_raw).__name__}"
            )

        rules: list[PolicyMatch] = []
        for i, rule in enumerate(rules_raw):
            if not isinstance(rule, dict):
                raise ValueError(
                    f"MIDDLEOUT_POLICIES rule #{i} must be a JSON object, "
                    f"got {type(rule).__name__}"
                )
            model_glob = str(rule.get("model_glob", "*"))
            endpoint = str(rule.get("endpoint", "*"))
            policy = _parse_policy(rule.get("policy"))
            rules.append(PolicyMatch(model_glob=model_glob, endpoint=endpoint, policy=policy))

        return cls(rules=rules, default=default_policy)


__all__ = ["CompressionPolicy", "PolicyMatch", "PolicyRouter"]
