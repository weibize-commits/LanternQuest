# Data dictionary

## `human_study/e90_analysis_ready_anonymized.csv`

`participant_id` and `block_id` are release-specific random labels with no published lookup table. `condition` identifies ENSR or B4. `pretest_form_b`, `lantern_familiarity_1_7`, `pretest_score`, `posttest_score`, `day7_score` and `narrative_score_mean` support the main adjusted models. Questionnaire totals and binary behavioural outcomes use the scales described in the manuscript. Blank values denote outcomes unavailable because of withdrawal or non-return.

## `human_study/e90_rater_scores_anonymized.csv`

Two masked raters scored factual consistency, source scope, separation of documented tradition from personal choice, and uncertainty handling from 0 to 4. `total_0_16` is the sum of the four dimensions.

## Aggregate files

Aggregate CSV and JSON files preserve the estimates, confidence intervals, audit rates and method labels reported in the manuscript. They contain no source text, free-text responses or direct identifiers.

## `dynamic_benchmark/case_level_metrics_anonymized.csv`

`public_case_id` is a release-specific case label. `figure_case_number` links the row to the numeric label shown in Figure 10g without releasing the underlying expert-authored case. `perturbation_type`, `method` and the binary or count outcomes support independent recalculation of the reported paired effects. Source locators, reviewed case text, action identifiers, controller rationales and expert comments are excluded.

## `dynamic_benchmark/formal_analysis.json`

Aggregate method summaries, paired case-bootstrap intervals, discordant-case counts and the prespecified mechanism decision for the ten-case frozen test split.
