from middleout_proxy.config import Settings
from middleout_proxy.metrics import render_prometheus


def _sample_stats() -> dict:
    return {
        "started_at": 1700000000.0,
        "uptime_s": 12.5,
        "requests_total": 17,
        "compressed_requests": 9,
        "chars_saved_in": 1234,
        "chars_saved_out": 56,
        "upstream_errors": 2,
        "cache_hits": 5,
        "cache_misses": 4,
        "protected_blocks": 1,
        "result_cache": {"size": 3, "max_entries": 256, "hits": 5, "misses": 4},
    }


def test_render_starts_with_help_comment():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    assert output.startswith("# HELP "), output[:200]


def test_every_metric_has_type_block():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    lines = output.splitlines()
    help_metrics = {ln.split()[2] for ln in lines if ln.startswith("# HELP ")}
    type_metrics = {ln.split()[2] for ln in lines if ln.startswith("# TYPE ")}
    assert help_metrics, "expected at least one # HELP line"
    assert help_metrics == type_metrics, (
        f"HELP/TYPE mismatch: only in HELP={help_metrics - type_metrics}; "
        f"only in TYPE={type_metrics - help_metrics}"
    )


def test_counter_names_end_in_total():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    counter_names = [
        line.split()[2]
        for line in output.splitlines()
        if line.startswith("# TYPE ") and line.endswith(" counter")
    ]
    assert counter_names, "no counter metrics found"
    for name in counter_names:
        assert name.endswith("_total"), f"counter {name} must end in _total"


def test_gauges_include_uptime_seconds():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    assert "# TYPE middleout_uptime_seconds gauge" in output
    # The value line is non-comment, starts with the metric name and a space.
    value_lines = [
        line for line in output.splitlines() if line.startswith("middleout_uptime_seconds ")
    ]
    assert len(value_lines) == 1
    # parse as float
    parsed = float(value_lines[0].split()[1])
    assert parsed == 12.5


def test_engine_labels_present_for_each_engine():
    settings = Settings(
        caveman_enabled=True,
        rtk_enabled=False,
        jl_dedupe_enabled=True,
        output_compression_enabled=False,
        input_compression_enabled=True,
    )
    output = render_prometheus(_sample_stats(), settings=settings)
    assert 'middleout_engine_enabled{engine="caveman"} 1' in output
    assert 'middleout_engine_enabled{engine="rtk"} 0' in output
    assert 'middleout_engine_enabled{engine="jl_dedupe"} 1' in output
    assert 'middleout_engine_enabled{engine="output"} 0' in output
    assert 'middleout_input_compression_enabled{engine="input"} 1' in output


def test_required_metric_names_present():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    required = {
        "middleout_requests_total",
        "middleout_compressed_requests_total",
        "middleout_upstream_errors_total",
        "middleout_chars_saved_in_total",
        "middleout_chars_saved_out_total",
        "middleout_protected_blocks_total",
        "middleout_cache_hits_total",
        "middleout_cache_misses_total",
        "middleout_uptime_seconds",
        "middleout_cache_size",
        "middleout_cache_max_entries",
        "middleout_input_compression_enabled",
        "middleout_engine_enabled",
        "middleout_jl_similarity_threshold",
    }
    present = {
        line.split()[2] for line in output.splitlines() if line.startswith("# HELP ")
    }
    missing = required - present
    assert not missing, f"missing required metrics: {sorted(missing)}"


def test_counter_values_match_stats():
    settings = Settings()
    stats = _sample_stats()
    output = render_prometheus(stats, settings=settings)
    assert "middleout_requests_total 17" in output
    assert "middleout_compressed_requests_total 9" in output
    assert "middleout_chars_saved_in_total 1234" in output
    assert "middleout_chars_saved_out_total 56" in output
    assert "middleout_upstream_errors_total 2" in output
    assert "middleout_cache_hits_total 5" in output
    assert "middleout_cache_misses_total 4" in output
    assert "middleout_protected_blocks_total 1" in output


def test_cache_size_reads_from_nested_result_cache():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    assert "middleout_cache_size 3" in output
    assert "middleout_cache_max_entries 256" in output


def test_cache_size_reads_flat_keys_when_present():
    # Alternative merge strategy: flat keys at the top level.
    settings = Settings()
    stats = {"uptime_s": 1.0, "size": 7, "max_entries": 64}
    output = render_prometheus(stats, settings=settings)
    assert "middleout_cache_size 7" in output
    assert "middleout_cache_max_entries 64" in output


def test_jl_similarity_threshold_reflects_settings():
    settings = Settings(jl_similarity_threshold=0.9)
    output = render_prometheus(_sample_stats(), settings=settings)
    # We render with 6 decimals.
    assert "middleout_jl_similarity_threshold 0.900000" in output


def test_output_is_newline_terminated():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    assert output.endswith("\n"), "Prometheus exposition must be newline-terminated"


def test_handles_missing_or_none_stats_keys():
    settings = Settings()
    output = render_prometheus({}, settings=settings)
    # All counters present and default to 0
    assert "middleout_requests_total 0" in output
    assert "middleout_cache_hits_total 0" in output
    assert "middleout_cache_size 0" in output


def test_empty_label_values_are_escaped_safely():
    settings = Settings()
    output = render_prometheus(_sample_stats(), settings=settings)
    # Label values are simple identifiers; no unescaped quotes/backslashes.
    for line in output.splitlines():
        if "{" in line and "}" in line:
            inside = line.split("{", 1)[1].split("}", 1)[0]
            # No raw unescaped double-quote between the opening and closing of
            # a label value (i.e. " followed immediately by another ").
            assert '""' not in inside.replace('\\"', "")
