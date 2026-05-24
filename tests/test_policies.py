import json

import pytest

from middleout_proxy.policies import CompressionPolicy, PolicyMatch, PolicyRouter


def test_default_policy_used_when_no_rules_match():
    router = PolicyRouter(rules=[], default=CompressionPolicy(input_compression=False))
    resolved = router.resolve(model="claude-sonnet-4", endpoint="v1/messages")
    assert resolved.input_compression is False


def test_default_compression_policy_field_defaults():
    p = CompressionPolicy()
    assert p.input_compression is True
    assert p.jl_dedupe is True
    assert p.caveman_enabled is False
    assert p.caveman_level == "standard"
    assert p.rtk_enabled is False
    assert p.rtk_level == "minimal"
    assert p.output_compression is False
    assert p.max_text_chars is None


def test_first_match_wins():
    opus = CompressionPolicy(caveman_enabled=True, caveman_level="lite")
    haiku = CompressionPolicy(input_compression=False)
    fallback = CompressionPolicy(jl_dedupe=False)
    router = PolicyRouter(
        rules=[
            PolicyMatch(model_glob="claude-opus-*", endpoint="v1/messages", policy=opus),
            PolicyMatch(model_glob="claude-haiku-*", policy=haiku),
            PolicyMatch(model_glob="*", policy=fallback),
        ],
        default=CompressionPolicy(),
    )
    assert router.resolve(model="claude-opus-4", endpoint="v1/messages") is opus
    assert router.resolve(model="claude-haiku-3", endpoint="v1/messages") is haiku
    # rule 1 has endpoint v1/messages, so opus on a different endpoint should fall through
    other = router.resolve(model="claude-opus-4", endpoint="v1/messages/count_tokens")
    assert other is fallback


def test_glob_matching_supports_star():
    star = CompressionPolicy(caveman_enabled=True)
    router = PolicyRouter(rules=[PolicyMatch(model_glob="*", policy=star)])
    assert router.resolve(model="anything-goes", endpoint="v1/messages") is star
    assert router.resolve(model="claude-opus-4", endpoint="v1/anything") is star


def test_glob_matching_prefix_suffix():
    opus_rule = PolicyMatch(model_glob="claude-opus-*", policy=CompressionPolicy(caveman_enabled=True))
    sonnet_rule = PolicyMatch(model_glob="*-sonnet-*", policy=CompressionPolicy(rtk_enabled=True))
    router = PolicyRouter(rules=[opus_rule, sonnet_rule])
    assert router.resolve(model="claude-opus-4-20240229", endpoint="v1/messages").caveman_enabled
    assert router.resolve(model="claude-sonnet-3-5", endpoint="v1/messages").rtk_enabled
    assert not router.resolve(model="claude-haiku-3", endpoint="v1/messages").caveman_enabled


def test_endpoint_literal_match():
    rule = PolicyMatch(
        model_glob="*",
        endpoint="v1/messages/count_tokens",
        policy=CompressionPolicy(input_compression=False),
    )
    router = PolicyRouter(rules=[rule], default=CompressionPolicy(input_compression=True))
    assert router.resolve(model="m", endpoint="v1/messages/count_tokens").input_compression is False
    assert router.resolve(model="m", endpoint="v1/messages").input_compression is True


def test_resolve_with_none_model_only_matches_star_glob():
    rule_specific = PolicyMatch(model_glob="claude-*", policy=CompressionPolicy(caveman_enabled=True))
    rule_star = PolicyMatch(model_glob="*", policy=CompressionPolicy(rtk_enabled=True))
    router = PolicyRouter(rules=[rule_specific, rule_star])
    resolved = router.resolve(model=None, endpoint="v1/messages")
    # The "claude-*" rule should NOT match a missing model name; the "*" rule should.
    assert resolved.rtk_enabled is True
    assert resolved.caveman_enabled is False


def test_from_json_roundtrip():
    raw = json.dumps(
        {
            "default": {"input_compression": True, "jl_dedupe": True},
            "rules": [
                {
                    "model_glob": "claude-opus-*",
                    "endpoint": "v1/messages",
                    "policy": {"caveman_enabled": True, "caveman_level": "lite"},
                },
                {
                    "model_glob": "claude-haiku-*",
                    "policy": {"input_compression": False},
                },
            ],
        }
    )
    router = PolicyRouter.from_json(raw)
    assert len(router.rules) == 2
    opus = router.resolve(model="claude-opus-4", endpoint="v1/messages")
    assert opus.caveman_enabled is True
    assert opus.caveman_level == "lite"
    haiku = router.resolve(model="claude-haiku-3", endpoint="v1/messages")
    assert haiku.input_compression is False
    fallback = router.resolve(model="claude-sonnet-3-5", endpoint="v1/messages")
    assert fallback.input_compression is True
    assert fallback.jl_dedupe is True


def test_from_json_with_max_text_chars():
    raw = json.dumps(
        {
            "rules": [
                {"model_glob": "*", "policy": {"max_text_chars": 4096}},
            ]
        }
    )
    router = PolicyRouter.from_json(raw)
    assert router.resolve(model="anything", endpoint="v1/messages").max_text_chars == 4096


def test_from_json_max_text_chars_null_means_use_default():
    raw = json.dumps({"rules": [{"model_glob": "*", "policy": {"max_text_chars": None}}]})
    router = PolicyRouter.from_json(raw)
    assert router.resolve(model="x", endpoint="v1/messages").max_text_chars is None


def test_from_json_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        PolicyRouter.from_json("{not-json")


def test_from_json_non_object_root_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        PolicyRouter.from_json("[]")


def test_from_json_unknown_policy_field_raises():
    raw = json.dumps({"rules": [{"model_glob": "*", "policy": {"bogus_field": True}}]})
    with pytest.raises(ValueError, match="unknown policy fields"):
        PolicyRouter.from_json(raw)


def test_from_json_rules_must_be_a_list():
    raw = json.dumps({"rules": {"oops": "dict"}})
    with pytest.raises(ValueError, match="'rules' must be a list"):
        PolicyRouter.from_json(raw)


def test_from_env_empty_returns_defaults():
    router = PolicyRouter.from_env(env={})
    assert router.rules == []
    assert router.default == CompressionPolicy()


def test_from_env_reads_middleout_policies_env_var():
    raw = json.dumps(
        {
            "default": {"input_compression": False},
            "rules": [
                {"model_glob": "claude-*", "policy": {"jl_dedupe": False}},
            ],
        }
    )
    router = PolicyRouter.from_env(env={"MIDDLEOUT_POLICIES": raw})
    assert router.default.input_compression is False
    assert router.resolve(model="claude-opus-4", endpoint="v1/messages").jl_dedupe is False


def test_from_env_blank_value_returns_defaults():
    router = PolicyRouter.from_env(env={"MIDDLEOUT_POLICIES": "   "})
    assert router.rules == []
    assert router.default == CompressionPolicy()


def test_compression_policy_is_frozen():
    p = CompressionPolicy()
    with pytest.raises((AttributeError, Exception)):
        p.input_compression = False  # type: ignore[misc]
