"""Post-hoc baseline-covariate sensitivity analysis for the E90 primary outcome."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
RETURN_DIR = (
    ROOT
    / "artifacts"
    / "human_study"
    / "e90"
    / "returns"
    / "return_20260925_real_v1"
)
SCORED = RETURN_DIR / "analysis" / "frozen_script_run" / "e90_scored_participants.csv"
OUT = RETURN_DIR / "validated_analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_result(fit) -> dict:
    interval = fit.conf_int().loc["ensr"]
    return {
        "n": int(fit.nobs),
        "estimate": float(fit.params["ensr"]),
        "se_hc3": float(fit.bse["ensr"]),
        "ci95": [float(interval.iloc[0]), float(interval.iloc[1])],
        "p": float(fit.pvalues["ensr"]),
        "r_squared": float(fit.rsquared),
        "design_rank": int(fit.model.rank),
        "parameter_count": int(len(fit.params)),
        "condition_number": float(fit.condition_number),
    }


def main() -> None:
    workbook = next((RETURN_DIR / "raw").glob("*.xlsx"))
    background = pd.read_excel(workbook, sheet_name=4)
    scored = pd.read_csv(SCORED)
    data = scored.merge(background, on="participant_code", how="left", suffixes=("", "_bg"))
    data["ensr"] = (
        data["internal_condition"].astype(str).str.casefold().eq("ensr").astype(int)
    )

    primary_formula = (
        "narrative_mean ~ ensr + pre_total + pre_form_b + "
        "lantern_familiarity_1_7"
    )
    categorical_formula = (
        primary_formula
        + " + C(age_group) + C(education) + C(gaming_frequency) + "
        "C(genai_frequency)"
    )
    primary = smf.ols(primary_formula, data=data).fit(cov_type="HC3")
    categorical = smf.ols(categorical_formula, data=data).fit(cov_type="HC3")

    ordinal_maps = {
        "age_group": {"18–24": 0, "25–34": 1, "35–44": 2, "45–54": 3},
        "education": {"高中及以下": 0, "大专/本科": 1, "硕士": 2, "博士": 3},
        "gaming_frequency": {
            "从不": 0,
            "每月少于1次": 1,
            "每月1–3次": 2,
            "每周1–3次": 3,
            "每周4次以上": 4,
        },
        "genai_frequency": {
            "从不": 0,
            "每月少于1次": 1,
            "每月1–3次": 2,
            "每周1–3次": 3,
            "每周4次以上": 4,
        },
    }
    for column, mapping in ordinal_maps.items():
        data[f"{column}_ordinal"] = data[column].map(mapping)
        if data[f"{column}_ordinal"].isna().any():
            unknown = sorted(data.loc[data[f"{column}_ordinal"].isna(), column].astype(str).unique())
            raise RuntimeError(f"Unmapped {column} values: {unknown}")
    ordinal_formula = (
        primary_formula
        + " + age_group_ordinal + education_ordinal + gaming_frequency_ordinal "
        "+ genai_frequency_ordinal"
    )
    ordinal = smf.ols(ordinal_formula, data=data).fit(cov_type="HC3")

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_id": "e90_posthoc_baseline_covariate_sensitivity_v1",
        "analysis_role": "exploratory_post_hoc_sensitivity_primary_model_unchanged",
        "outcome": "mean blinded evidence-constrained narrative score, range 0 to 16",
        "estimand": "ENSR minus B4 coefficient",
        "models": {
            "frozen_primary_reproduction": {
                "formula": primary_formula,
                **model_result(primary),
            },
            "expanded_categorical_adjustment": {
                "formula": categorical_formula,
                "added_covariates": [
                    "age group",
                    "education",
                    "gaming frequency",
                    "generative-AI use frequency",
                ],
                **model_result(categorical),
            },
            "expanded_ordinal_check": {
                "formula": ordinal_formula,
                "added_covariates": [
                    "ordered age category",
                    "ordered education category",
                    "ordered gaming frequency",
                    "ordered generative-AI use frequency",
                ],
                **model_result(ordinal),
            },
        },
        "data": {
            "scored_csv": str(SCORED.resolve()),
            "scored_csv_sha256": sha256(SCORED),
            "workbook": str(workbook.resolve()),
            "workbook_sha256": sha256(workbook),
        },
        "interpretation": (
            "The frozen primary model remains the confirmatory analysis. The expanded "
            "models test whether the ENSR coefficient is sensitive to observed baseline "
            "differences and do not redefine the primary estimand."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "e90_baseline_covariate_sensitivity_v1.json"
    md_path = OUT / "e90_baseline_covariate_sensitivity_v1.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cat = report["models"]["expanded_categorical_adjustment"]
    ord_ = report["models"]["expanded_ordinal_check"]
    md_path.write_text(
        "\n".join(
            [
                "# E90 baseline-covariate sensitivity analysis",
                "",
                "The frozen primary model was retained. Two exploratory post-hoc models "
                "added the baseline domains with the largest standardized differences.",
                "",
                "| Model | ENSR minus B4 | 95% CI | P | n |",
                "|---|---:|---:|---:|---:|",
                f"| Frozen primary | {report['models']['frozen_primary_reproduction']['estimate']:.3f} | "
                f"[{report['models']['frozen_primary_reproduction']['ci95'][0]:.3f}, "
                f"{report['models']['frozen_primary_reproduction']['ci95'][1]:.3f}] | "
                f"{report['models']['frozen_primary_reproduction']['p']:.3g} | "
                f"{report['models']['frozen_primary_reproduction']['n']} |",
                f"| Expanded categorical | {cat['estimate']:.3f} | "
                f"[{cat['ci95'][0]:.3f}, {cat['ci95'][1]:.3f}] | {cat['p']:.3g} | {cat['n']} |",
                f"| Expanded ordinal check | {ord_['estimate']:.3f} | "
                f"[{ord_['ci95'][0]:.3f}, {ord_['ci95'][1]:.3f}] | {ord_['p']:.3g} | {ord_['n']} |",
                "",
                "The expanded categorical model is the main sensitivity analysis. The "
                "ordinal model checks that its conclusion is not driven by sparse dummy "
                "categories. Both are exploratory and leave the frozen primary model unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
