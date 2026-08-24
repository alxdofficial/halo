import pytest

from eval.build_results_tables import MODEL_NAMES, _validate_current_cells, table_label_efficiency


def test_label_efficiency_excludes_random_alias_rows() -> None:
    cells = [
        {
            "model": model,
            "method": method,
            "regime": regime,
            "label_mode": "coherent",
            "dataset": "example",
            "k": "1",
            "f1_macro": "80.0" if method == "evidence_engine" else "40.0",
        }
        for model in MODEL_NAMES
        for regime in ("ordinary", "specialized_novel")
        for method in (("evidence_engine", "nearest", "prototype", "ridge")
                       if model == "halo_compact" else ("nearest", "prototype", "ridge"))
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

    assert "| HALO / retrieve-mix-vote | **80.00** |" in table
    assert "HARNet / 1-NN" in table
    assert "HARNet / prototype" in table
    assert "HARNet / ridge" in table
    assert "linear_head" not in table
    assert "50.00" not in table


def test_current_report_rejects_missing_matched_readout() -> None:
    cells = [
        {
            "model": model,
            "method": method,
            "k": "1",
        }
        for model in MODEL_NAMES
        for method in (("evidence_engine", "nearest", "prototype", "ridge")
                       if model == "halo_compact" else ("nearest", "prototype", "ridge"))
        if not (model == "harnet" and method == "ridge")
    ]

    with pytest.raises(ValueError, match="harnet/ridge"):
        _validate_current_cells(cells)
