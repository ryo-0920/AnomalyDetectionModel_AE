# Work Notes: inference-fpr-tpr-selection

## What was reviewed
- `docs/specs/overview.md`
- `docs/specs/requirements.md`
- `docs/specs/design.md`
- `src/gofumi_ae/inference/standard.py`
- `src/gofumi_ae/evaluation/standard.py`

## Clearly observed current behavior
- Inference currently prompts for one scoring target.
- Inference already writes legacy per-file outputs and TF-style run outputs.
- Evaluation already has ON/OFF-oriented summary and threshold-curve code.
- Evaluation currently uses A2-first logic in its ON/OFF threshold analysis.
- Evaluation currently uses interpolation for target-FPR reporting.

## Drafting rule applied
- Do not infer ambiguous product behavior.
- Keep the draft limited to what the user clearly requested.
- Put all material decisions that affect implementation into open issues.

## Resolved decisions now reflected in the draft
- TPR positive population uses the A2-conditioned population, aligned with the current A2-first evaluation style.
- Per-target threshold selection uses only candidates satisfying `FPR <= target`.
- Among those eligible candidates, selection is defined by:
  - `FPR` closest to the target
  - `TPR` maximal
- No extra tie-break rule was added.
- The intended behavior is one integrated flow from inference through FPR/TPR/threshold output.

## Resulting spec posture
- The draft now states only:
  - ON/OFF selection is required
  - FPR/TPR computation is required
  - fixed target FPR values are required
  - per-target saved TPR and threshold are required
- The draft now additionally fixes:
  - A2-conditioned TPR population semantics
  - target-FPR candidate-eligibility semantics
  - target-FPR selection semantics up to the resolved rule
  - integrated-flow semantics
- The draft intentionally still does not choose:
  - score-column semantics
  - output-contract semantics
  - backward-compatibility behavior for the current one-target prompt
