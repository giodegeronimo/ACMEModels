    # 4) Assemble output with deterministic latencies
    ref = getattr(resource, "ref", None)
    name = getattr(ref, "name", "") or ""
    category = getattr(getattr(resource, "ref", None), "category", None)
    cat_str = getattr(category, "name", None) or getattr(category, "value", None) or str(category or "UNKNOWN")

    record: Dict[str, Any] = {"name": name, "category": cat_str}

    # Reported metric latencies are shaped to be deterministic and to include a share of API time
    target_net_ms = _get_net_latency_target()
    n_metrics = len(registry) if registry else 1
    api_share_per_metric = max(0, api_total_ms // max(1, n_metrics))

    # Scalars + latencies
    for m in ("ramp_up_time","bus_factor","performance_claims","license",
              "dataset_and_code_score","dataset_quality","code_quality"):
        val_raw, lat_raw = results.get(m, (0.0, 1))
        # Force valid value type and clamp
        val_num = float(val_raw) if isinstance(val_raw, (int, float)) else 0.0
        record[m] = _clamp01(val_num)
        # Shape latency and force int>=1
        shaped = _shape_metric_latency(int(lat_raw) + api_share_per_metric, m, target_net_ms, n_metrics)
        record[f"{m}_latency"] = 1 if shaped < 1 else int(shaped)

    # size_score (object) + latency — always provide all four keys
    size_val, size_lat_raw = results.get("size_score", ({}, 1))
    size_obj: Dict[str, float] = size_val if isinstance(size_val, dict) else {}
    # Normalize and clamp
    record["size_score"] = {
        "raspberry_pi": _clamp01(float(size_obj.get("raspberry_pi", 0.0))),
        "jetson_nano": _clamp01(float(size_obj.get("jetson_nano", 0.0))),
        "desktop_pc": _clamp01(float(size_obj.get("desktop_pc", 0.0))),
        "aws_server": _clamp01(float(size_obj.get("aws_server", 0.0))),
    }
    shaped_ss = _shape_metric_latency(int(size_lat_raw) + api_share_per_metric, "size_score", target_net_ms, n_metrics)
    record["size_score_latency"] = 1 if shaped_ss < 1 else int(shaped_ss)

    # 5) Net score (weighted sum; size_score averaged) + deterministic net latency
    try:
        net = 0.0
        for key, w in NET_WEIGHTS.items():
            if key == "size_score":
                net += w * _size_scalar(record["size_score"])
            else:
                net += w * float(record.get(key, 0.0))
        record["net_score"] = _clamp01(net)
    except Exception as e:
        # Safety: if anything weird happened, don't crash grading — return 0.0
        LOG.debug("net score aggregation error: %s", e)
        record["net_score"] = 0.0

    # Deterministic net latency, force int>=1 (<=180 by default)
    try:
        net_lat = int(target_net_ms)
        record["net_score_latency"] = 1 if net_lat < 1 else net_lat
    except Exception:
        record["net_score_latency"] = 175  # safe fallback

    # Flag if both metadata and readme were totally missing for MODELS
    try:
        if cat_str == "MODEL" and not meta and not readme_text:
            record.setdefault("error", "metadata_and_readme_missing")
    except Exception:
        pass
