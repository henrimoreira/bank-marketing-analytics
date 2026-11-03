from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from scipy.stats.contingency import association
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "y"

# Grupos com menos registros que isso são sinalizados como instáveis.
MIN_GROUP_SIZE = 30

ECONOMIC_INDICATORS = {
    "emp.var.rate": "Taxa de variação do emprego",
    "cons.price.idx": "Índice de preços ao consumidor",
    "cons.conf.idx": "Índice de confiança do consumidor",
    "euribor3m": "Taxa Euribor de 3 meses",
    "nr.employed": "Número de pessoas empregadas",
}

# Variáveis do modelo de regressão logística (duration fica de fora de
# propósito: só é conhecida depois da ligação e causaria vazamento).
NUMERIC_FEATURES = ["age", "campaign", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo '{path}' não encontrado. "
            "Baixe o conjunto de dados no Repositório UCI Machine Learning "
            "(bank-additional.zip), extraia o arquivo "
            "'bank-additional-full.csv' e coloque-o na pasta data/ do projeto."
        )
    return pd.read_csv(path, sep=";")


def prepare_data(df):
    df = df.copy()
    df["duration_range"] = pd.cut(
        df["duration"], [0, 60, 180, 600, np.inf],
        labels=["Curta", "Média", "Longa", "Muito longa"], include_lowest=True
    )
    df["age_range"] = pd.cut(
        df["age"], [0, 20, 61, df["age"].max() + 1],
        right=False, labels=["Jovem", "Adulto", "Idoso"]
    )
    df["previous_contacts_group"] = np.where(df["previous"].eq(0), "Não", "Sim")
    conditions = [
        df["job"].isin(["admin.", "management"]),
        df["job"].isin(["entrepreneur", "technician", "self-employed"]),
        df["job"].isin(["blue-collar", "services", "housemaid"]),
    ]
    labels = ["Administrativo", "Técnico / negócios", "Operacional"]
    df["job_group"] = np.select(conditions, labels, default="Outros")
    # pdays == 999 é um código especial: "não contatado antes".
    df["pdays_was_999"] = df["pdays"].eq(999)
    return df


def calculate_adherence_rate(series):
    """Percentual de 'yes' em uma série da variável-alvo."""
    return (series == "yes").mean() * 100


def adherence_summary(df, group_column, sort_by_rate=True):
    """Total, adesões e taxa por grupo.

    Com ``sort_by_rate=True`` ordena da maior para a menor taxa. Use
    ``sort_by_rate=False`` quando a ordem natural importa (por exemplo,
    número de contatos), para que ``.head()`` devolva os primeiros valores.
    """
    result = df.groupby(group_column, observed=True, dropna=False)[TARGET].agg(
        total="size", adesoes=lambda s: s.eq("yes").sum()
    )
    result["taxa_adesao_pct"] = result["adesoes"] / result["total"] * 100
    if sort_by_rate:
        return result.sort_values("taxa_adesao_pct", ascending=False)
    return result.sort_index()


def calculate_association(table, name, verbose=True):
    """Qui-quadrado de independência e V de Cramér."""
    chi2, p_value, dof, expected = chi2_contingency(table)
    v = association(table, method="cramer")
    result = {
        "analysis": name, "chi2": chi2, "p_value": p_value, "dof": dof,
        "cramers_v": v, "min_expected": expected.min(),
        "expected_below_5": int((expected < 5).sum()),
    }
    if verbose:
        print(f"Análise: {name}")
        print(f"Qui-quadrado: {chi2:.4f}")
        print(f"Valor-p: {p_value:.6g}")
        print(f"Graus de liberdade: {dof}")
        print(f"V de Cramér: {v:.4f}")
        print(f"Menor frequência esperada: {expected.min():.4f}")
        print(f"Células com frequência esperada < 5: {result['expected_below_5']}")
        if result["expected_below_5"]:
            print("Atenção: há frequências esperadas abaixo de 5; "
                  "interprete o teste com cautela.")
        print()
    return result


def save_customer_figures(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        ("job", "adesao_por_profissao.png", "Taxa de adesão por profissão", "Profissão"),
        ("age_range", "adesao_por_faixa_etaria.png", "Taxa de adesão por faixa etária", "Faixa etária"),
        ("previous_contacts_group", "adesao_por_contatos_anteriores.png",
         "Taxa de adesão por contatos anteriores", "Contato anterior"),
        ("duration_range", "adesao_por_duracao.png",
         "Taxa de adesão por duração da ligação", "Faixa de duração"),
    ]
    for col, filename, title, xlabel in figures:
        # Profissões em ordem de taxa; faixas mantêm a ordem natural.
        summary = adherence_summary(df, col, sort_by_rate=(col == "job"))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = summary.index.astype(str)
        ax.bar(labels, summary["taxa_adesao_pct"])
        for position, (n, rate) in enumerate(
            zip(summary["total"], summary["taxa_adesao_pct"])
        ):
            ax.text(position, rate + 0.5, f"n={n:,}", ha="center", fontsize=7)
        ax.set(title=title, ylabel="Taxa de adesão (%)", xlabel=xlabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def analyze_economic_indicators(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in ECONOMIC_INDICATORS:
        grouped = df.groupby(col, observed=True)[TARGET].agg(
            total="size", adesoes=lambda s: s.eq("yes").sum()
        )
        grouped["rate"] = grouped["adesoes"] / grouped["total"] * 100
        rho, p = spearmanr(grouped.index, grouped["rate"])
        rows.append({"indicator": col, "spearman_rho": rho,
                     "p_value_exploratory": p, "distinct_values": len(grouped)})
        fig, ax = plt.subplots(figsize=(8, 4))
        sizes = 12 + 65 * grouped["total"] / grouped["total"].max()
        ax.scatter(grouped.index, grouped["rate"], s=sizes, alpha=.65)
        ax.set(title=f"{col}: taxa de adesão por valor distinto",
               xlabel=col, ylabel="Taxa de adesão (%)")
        fig.tight_layout()
        fig.savefig(output_dir / f"economico_{col.replace('.', '_')}.png", dpi=160)
        plt.close(fig)
    results = pd.DataFrame(rows)
    print("\nSpearman agrupado por valores distintos (exploratório):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("Nota: valores-p não são evidência inferencial definitiva; "
          "indicadores se repetem entre clientes.")
    corr = df[list(ECONOMIC_INDICATORS)].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlação de Pearson entre indicadores econômicos")
    fig.tight_layout()
    fig.savefig(output_dir / "correlacao_indicadores.png", dpi=160)
    plt.close(fig)
    return results, corr


def reference_categories(df):
    """Categoria de referência de cada variável categórica do modelo.

    Com ``drop_first=True``, a referência é a primeira categoria em ordem
    alfabética. Os odds ratios das dummies comparam com ela.
    """
    return {
        col: sorted(df[col].dropna().unique())[0]
        for col in CATEGORICAL_FEATURES
    }


def fit_logistic_model(df):
    """Modelo exploratório ajustado; exclui duration para evitar vazamento.

    Retorna (pipeline, odds). A tabela ``odds`` traz o ``n`` de cada dummy e
    um aviso para categorias com menos de ``MIN_GROUP_SIZE`` registros.
    """
    model_df = df.copy()
    model_df["target_binary"] = model_df[TARGET].map({"no": 0, "yes": 1})

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pd.get_dummies(model_df[features], columns=CATEGORICAL_FEATURES,
                       drop_first=True, dtype=float)
    y = model_df["target_binary"]

    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough"
    )
    pipeline = Pipeline([
        ("scaler", preprocessor),
        ("logistic", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    ])
    pipeline.fit(X, y)
    feature_names = NUMERIC_FEATURES + [
        c for c in X.columns if c not in NUMERIC_FEATURES
    ]
    coefficients = pipeline.named_steps["logistic"].coef_[0]
    odds = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
    })
    # n de cada dummy = quantos registros pertencem à categoria.
    odds["n"] = odds["feature"].map(X.sum())
    odds.loc[odds["feature"].isin(NUMERIC_FEATURES), "n"] = np.nan
    odds["n"] = odds["n"].round().astype("Int64")
    is_rare = (odds["n"] < MIN_GROUP_SIZE).fillna(False).astype(bool)
    odds["aviso"] = np.where(is_rare, "n<30", "")
    odds = odds.sort_values("odds_ratio", ascending=False)
    return pipeline, odds


def run_analysis(data_path, output_dir):
    df = prepare_data(load_data(data_path))
    print(f"Registros: {len(df):,}; colunas: {df.shape[1]}")
    print(f"Taxa geral de adesão: {df[TARGET].eq('yes').mean()*100:.2f}%")
    print("\nTamanho dos grupos e taxa de adesão:")
    for col in ["marital", "education", "job", "age_range", "poutcome",
                "contact", "month", "pdays_was_999"]:
        print(f"\n{col}\n{adherence_summary(df, col).round(2)}")

    # Ordem natural (1, 2, 3, ...): sem isso, head() devolveria os grupos
    # de maior taxa, que são justamente os menores.
    campaign = adherence_summary(df, "campaign", sort_by_rate=False)
    print("\nTaxa por número de contatos na campanha atual (contatos 1 a 15):")
    print(campaign.head(15).round(2))

    print("\nTestes categóricos:")
    tests = [
        ("age_range", "Faixa etária × adesão"),
        ("job", "Profissão × adesão"),
        ("previous_contacts_group", "Contatos anteriores × adesão"),
        ("job_group", "Grupos de profissão × adesão"),
        ("poutcome", "Resultado da campanha anterior × adesão"),
        ("contact", "Canal de contato × adesão"),
        ("month", "Mês × adesão"),
    ]
    for col, label in tests:
        calculate_association(pd.crosstab(df[col], df[TARGET]), label)

    save_customer_figures(df, output_dir)
    economic, corr = analyze_economic_indicators(df, output_dir)

    print("\nModelo de regressão logística exploratório (sem duration):")
    model, odds = fit_logistic_model(df)
    print("\nCategorias de referência (os OR das dummies comparam com elas):")
    for col, reference in reference_categories(df).items():
        print(f"  {col}: {reference}")
    columns = ["feature", "coefficient", "odds_ratio", "n", "aviso"]
    print("\n20 maiores odds ratios:")
    print(odds.head(20)[columns].round(3).to_string(index=False))
    print("\n20 menores odds ratios:")
    print(odds.tail(20).sort_values("odds_ratio")[columns].round(3).to_string(index=False))
    print("\nModelo regularizado do scikit-learn; OR são razões de chances "
          "associadas aos coeficientes.")
    print("Categorias marcadas com 'n<30' são raras: não interprete seus OR.")
    print("Não foram calculados valores-p ou intervalos de confiança para este modelo.")

    print(f"\nGráficos salvos em: {output_dir}")
    return df, economic, corr, model, odds
