from middleout_proxy.compression import PayloadCompressor, middle_out_text
from middleout_proxy.config import Settings
from middleout_proxy.jl import cosine, signed_jl_projection


def test_middle_out_preserves_edges():
    text = "A" * 5000 + "MIDDLE" * 1000 + "Z" * 5000
    compressed = middle_out_text(
        text, max_chars=2000, min_omission_chars=500, head_fraction=0.5
    )
    assert len(compressed) <= 2100
    assert compressed.startswith("A" * 100)
    assert compressed.endswith("Z" * 100)
    assert "middle-out compressed locally" in compressed


def test_payload_compressor_changes_long_user_text():
    settings = Settings(max_text_chars=1000, min_omission_chars=200, jl_dedupe_enabled=False)
    payload = {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "x" * 5000}],
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    assert transformed["messages"][0]["content"] != payload["messages"][0]["content"]
    assert audit.chars_saved > 0
    assert audit.events[0].mode == "middle-out"


def test_jl_sketch_similarity_for_near_duplicate_text():
    a = " ".join(["function calculate_total amount tax discount"] * 1000)
    b = " ".join(["function calculate_total amount tax discount"] * 998 + ["tiny change"])
    c = " ".join(["banana orange guitar river mountain"] * 1000)
    va = signed_jl_projection(a, dims=256)
    vb = signed_jl_projection(b, dims=256)
    vc = signed_jl_projection(c, dims=256)
    assert cosine(va, vb) > 0.95
    assert cosine(va, vc) < 0.5


def test_jl_dedupe_replaces_second_near_duplicate():
    repeated = " ".join(["alpha beta gamma delta epsilon"] * 1000)
    settings = Settings(
        max_text_chars=20_000,
        jl_dedupe_enabled=True,
        jl_min_chars=1000,
        jl_similarity_threshold=0.98,
    )
    payload = {
        "messages": [
            {"role": "user", "content": repeated},
            {"role": "user", "content": repeated + " small trailing edit"},
        ]
    }
    transformed, audit = PayloadCompressor(settings).compress_request_payload(
        payload, endpoint="v1/messages"
    )
    assert "Near-duplicate content omitted" in transformed["messages"][1]["content"]
    assert any(event.mode == "jl-near-duplicate" for event in audit.events)
