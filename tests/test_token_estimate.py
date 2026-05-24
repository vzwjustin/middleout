from middleout_proxy.token_estimate import (
    estimate_tokens,
    estimate_tokens_for_payload,
    summarize_token_stats,
)


def test_empty_string_returns_zero():
    assert estimate_tokens("") == 0


def test_one_word_returns_one():
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("a") == 1


def test_non_empty_text_returns_at_least_one():
    # Any non-empty input should round up to at least 1 token.
    assert estimate_tokens(".") >= 1
    assert estimate_tokens(" ") >= 1


def test_long_english_paragraph_within_thirty_percent_of_len_over_four():
    # Synthetic but lexically diverse "real" English. We don't try to match a
    # specific BPE tokenizer — just stay within 30% of len(text)//4 baseline.
    paragraph = (
        "Once upon a time in a quiet country town there lived a young woman "
        "named Eleanor who spent every weekend reading old novels in the "
        "garden behind her grandmother's house. The garden was overgrown with "
        "wildflowers and the air always carried the faint scent of lavender "
        "and rosemary. Eleanor would sit beneath the apple tree with her "
        "favourite leather-bound book and lose herself for hours at a time. "
    ) * 4
    baseline = len(paragraph) / 4.0
    est = estimate_tokens(paragraph)
    diff = abs(est - baseline) / baseline
    assert diff < 0.30, f"estimate {est} vs baseline {baseline:.0f} differs by {diff:.2%}"


def test_uppercase_heavy_text_falls_back_to_len_over_three_point_five():
    text = "HELLO WORLD ABC DEFG"
    est = estimate_tokens(text)
    expected = round(len(text) / 3.5)
    assert est == expected


def test_symbol_heavy_text_falls_back_to_len_over_three_point_five():
    # ≥ 30% punctuation triggers the symbol-heavy fallback.
    text = "!@#$%^&*()_+-=[]{}|;:,./<>?"
    est = estimate_tokens(text)
    expected = round(len(text) / 3.5)
    assert est == expected


def test_estimate_is_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    a = estimate_tokens(text)
    b = estimate_tokens(text)
    assert a == b


def test_payload_aggregation_walks_system_and_messages():
    payload = {
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well, thank you."},
        ],
    }
    total = estimate_tokens_for_payload(payload)
    system_tokens = estimate_tokens("You are a helpful assistant.")
    user_tokens = estimate_tokens("Hello, how are you?")
    assistant_tokens = estimate_tokens("I am doing well, thank you.")
    assert total == system_tokens + user_tokens + assistant_tokens


def test_payload_aggregation_handles_list_content_blocks():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first chunk of words"},
                    {"type": "text", "text": "second chunk of words"},
                ],
            }
        ]
    }
    total = estimate_tokens_for_payload(payload)
    assert total == estimate_tokens("first chunk of words") + estimate_tokens(
        "second chunk of words"
    )


def test_payload_aggregation_handles_tool_result_blocks():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "raw tool output text"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "wrapped tool output"}],
                    },
                ],
            }
        ]
    }
    total = estimate_tokens_for_payload(payload)
    expected = estimate_tokens("raw tool output text") + estimate_tokens(
        "wrapped tool output"
    )
    assert total == expected


def test_payload_aggregation_safe_on_empty_payload():
    assert estimate_tokens_for_payload({}) == 0
    assert estimate_tokens_for_payload({"messages": []}) == 0


def test_payload_aggregation_safe_on_non_dict():
    assert estimate_tokens_for_payload(None) == 0  # type: ignore[arg-type]
    assert estimate_tokens_for_payload("not a dict") == 0  # type: ignore[arg-type]


def test_summarize_token_stats_shape():
    payload = {
        "system": "sys prompt",
        "messages": [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello back"},
        ],
    }
    summary = summarize_token_stats(payload)
    assert set(summary.keys()) == {"total", "system", "messages"}
    assert summary["system"] == estimate_tokens("sys prompt")
    assert len(summary["messages"]) == 2
    assert summary["messages"][0]["role"] == "user"
    assert summary["messages"][1]["role"] == "assistant"
    assert summary["messages"][0]["tokens"] == estimate_tokens("hi there")
    assert summary["messages"][1]["tokens"] == estimate_tokens("hello back")
    assert summary["total"] == (
        summary["system"] + sum(m["tokens"] for m in summary["messages"])
    )


def test_summarize_token_stats_safe_on_non_dict():
    summary = summarize_token_stats(None)  # type: ignore[arg-type]
    assert summary == {"total": 0, "system": 0, "messages": []}
