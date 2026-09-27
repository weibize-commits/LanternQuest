from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

DATA = "data/human_study/e90_analysis_ready_anonymized.csv"

data = pd.read_csv(DATA)
data["ensr"] = (data["condition"] == "ensr").astype(int)
data = data.rename(columns={"pretest_score": "pre_total", "narrative_score_mean": "narrative_mean"})
model = smf.ols(
    "narrative_mean ~ ensr + pre_total + pretest_form_b + lantern_familiarity_1_7",
    data=data,
).fit(cov_type="HC3")
ci = model.conf_int().loc["ensr"].tolist()
print({"n": int(model.nobs), "ensr_estimate": float(model.params["ensr"]), "ci95": ci, "p": float(model.pvalues["ensr"])})
