from eval.build_results_tables import MODEL_NAMES, table_label_efficiency


def test_label_efficiency_excludes_random_alias_rows() -> None:
    cells = [
        {
            "model": model,
            "method": "evidence_engine" if model == "halo_compact" else "nearest",
            "regime": regime,
            "label_mode": "coherent",
            "dataset": "example",
            "k": "1",
            "f1_macro": "80.0" if model == "halo_compact" else "40.0",
        }
        for model in MODEL_NAMES
        for regime in ("ordinary", "specialized_novel")
    ] + [
        {
            "model": "halo_compact",
            "method": "nearest",
            "regime": regime,
            "label_mode": "coherent",
            "dataset": "example",
            "k": "1",
            "f1_macro": "70.0",
        }
        for regime in ("ordinary", "specialized_novel")
    ] + [
        {
            "model": "halo_compact",
            "method": "evidence_engine",
            "regime": "ordinary",
            "label_mode": "random_alias",
            "dataset": "example",
            "k": "1",
            "f1_macro": "20.0",
        },
    ]

    table = table_label_efficiency(cells)

    assert "| HALO (ours, native engine) | **80.00** |" in table
    assert "HARNet / 1-NN" in table
    assert "50.00" not in table
