import pandas as pd

from src.bank_marketing_analysis import prepare_data, calculate_association


def test_prepare_data_creates_expected_groups():
    sample = pd.DataFrame({
        "age": [19, 35, 70],
        "duration": [30, 120, 700],
        "previous": [0, 1, 2],
        "job": ["student", "admin.", "retired"],
        "pdays": [999, 10, 999],
    })

    result = prepare_data(sample)

    assert result["age_range"].astype(str).tolist() == ["Jovem", "Adulto", "Idoso"]
    assert result["previous_contacts_group"].tolist() == ["Não", "Sim", "Sim"]
    assert result["job_group"].tolist() == ["Outros", "Administrativo", "Outros"]
    assert result["pdays_was_999"].tolist() == [True, False, True]


def test_calculate_association_returns_expected_metrics():
    table = pd.DataFrame([[30, 10], [10, 30]], index=["A", "B"], columns=["no", "yes"])
    result = calculate_association(table, "teste")

    assert result["dof"] == 1
    assert 0 <= result["cramers_v"] <= 1
    assert result["expected_below_5"] == 0


def _model_sample(n=300):
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "campaign": rng.integers(1, 10, n),
        "previous": rng.integers(0, 3, n),
        "job": rng.choice(["admin.", "student", "retired"], n),
        "marital": rng.choice(["married", "single"], n),
        "education": rng.choice(["basic.4y", "university.degree"], n),
        "default": rng.choice(["no", "yes"], n, p=[0.98, 0.02]),
        "housing": rng.choice(["no", "yes"], n),
        "loan": rng.choice(["no", "yes"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["apr", "may", "jun"], n),
        "poutcome": rng.choice(["failure", "nonexistent", "success"], n),
        "y": rng.choice(["no", "yes"], n, p=[0.8, 0.2]),
    })


def test_adherence_summary_natural_order_keeps_first_values():
    from src.bank_marketing_analysis import adherence_summary

    df = pd.DataFrame({
        "campaign": [1, 1, 2, 2, 3, 40],
        "y": ["no", "no", "no", "yes", "no", "yes"],
    })

    by_order = adherence_summary(df, "campaign", sort_by_rate=False)
    by_rate = adherence_summary(df, "campaign")

    assert by_order.index.tolist() == [1, 2, 3, 40]
    assert by_rate.index[0] == 40  # maior taxa, grupo de 1 registro


def test_reference_categories_are_dropped_from_the_model():
    from src.bank_marketing_analysis import (
        fit_logistic_model,
        reference_categories,
    )

    df = _model_sample()
    references = reference_categories(df)
    _, odds = fit_logistic_model(df)

    assert references["poutcome"] == "failure"
    assert references["month"] == "apr"
    assert "poutcome_failure" not in odds["feature"].tolist()
    assert "month_apr" not in odds["feature"].tolist()


def test_odds_table_reports_group_size_and_flags_rare_categories():
    from src.bank_marketing_analysis import fit_logistic_model

    df = _model_sample()
    _, odds = fit_logistic_model(df)

    assert {"n", "aviso"}.issubset(odds.columns)
    assert odds.loc[odds["feature"] == "age", "n"].isna().all()
    rare = odds[odds["n"].notna() & (odds["n"] < 30)]
    assert (rare["aviso"] == "n<30").all()
