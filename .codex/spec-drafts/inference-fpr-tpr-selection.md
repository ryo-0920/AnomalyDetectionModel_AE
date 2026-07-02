# Draft Spec: inference-fpr-tpr-selection

## Status
- Draft only.
- Do not promote to `docs/specs/**` yet.
- This draft reflects only resolved product decisions plus still-open questions.

## Clearly specified by the request

### Requested behavior
- Inference shall be revised so it can run on datasets selected through the model/artifact threshold flow.
- During inference, the system shall compute FPR and TPR.
- Because ON and OFF datasets are separate, the flow shall allow selecting or specifying both ON and OFF datasets.
- The inference through FPR/TPR/threshold output shall run as one integrated flow.
- The flow shall find threshold candidates for these target FPR values:
  - `1e-4`
  - `1e-3`
  - `1e-2`
  - `1e-1`
  - `1e0`
- After threshold candidates are found, the flow shall output, for each target FPR value:
  - the value when TPR is highest for that target
  - the threshold at that performance
- The flow shall save:
  - the TPR for each target FPR value
  - the threshold corresponding to that saved performance

## Current code context captured for this draft
- `src/gofumi_ae/inference/standard.py`
  - currently prompts for one scoring target
  - currently writes legacy and TF-style inference outputs
- `src/gofumi_ae/evaluation/standard.py`
  - already contains ON/OFF file-level summary logic
  - already contains threshold-curve, target-FPR, and plotting-related logic
  - currently uses ON A2-first logic in its ON/OFF threshold analysis
  - currently uses interpolation for target-FPR reporting rather than selecting a best-threshold row

## Draft scope
- This draft only captures what is clearly requested for the inference/evaluation behavior change.
- This draft does not lock down behavior that the request leaves materially ambiguous.

## Non-goals
- No training-flow change is specified here.
- No change to approved specs in `docs/specs/**` is made here.
- No decision is made here on exact output filenames or directories for the new target-FPR artifacts.

## Requirement candidates that are clearly supported by the request

### RC-001: Separate ON/OFF selection
- The inference-side evaluation flow shall accept both an ON dataset input and an OFF dataset input.

### RC-002: FPR/TPR computation during inference flow
- The inference-side evaluation flow shall compute FPR and TPR using the scored ON/OFF datasets.

### RC-002a: Integrated execution flow
- The inference-side evaluation flow shall execute artifact selection, ON/OFF inference, FPR/TPR computation, threshold-candidate search, and target-FPR result output as one integrated flow.

### RC-003: Fixed target FPR list
- The inference-side evaluation flow shall evaluate target FPR values `1e-4`, `1e-3`, `1e-2`, `1e-1`, and `1e0`.

### RC-004: Threshold-candidate search
- The inference-side evaluation flow shall search threshold candidates from scored results so that each target FPR can be evaluated.

### RC-004a: TPR positive population
- TPR for target-FPR threshold evaluation shall be computed on the A2-conditioned positive population, consistent with the current A2-first evaluation style.

### RC-004b: Per-target threshold selection rule
- For each target FPR value, the eligible threshold candidates shall be limited to rows satisfying `FPR <= target`.
- From those eligible candidates, the flow shall select the candidate:
  - whose `FPR` is closest to the target, and
  - whose `TPR` is maximal
- No additional tie-break rule is specified in this draft.

### RC-005: Saved per-target result
- For each target FPR value, the flow shall save:
  - a TPR result
  - the threshold corresponding to the saved result

## Acceptance-criteria candidates that stay within the clear request
- AC-001: A user can provide or select both an ON dataset target and an OFF dataset target for the inference evaluation flow.
- AC-002: The run produces target-FPR results for exactly `1e-4`, `1e-3`, `1e-2`, `1e-1`, and `1e0`.
- AC-003: For each target FPR value, the saved result includes a TPR value and a threshold value.
- AC-004: The run computes its target-FPR results from ON/OFF-scored data rather than from a single undifferentiated scoring target.
- AC-005: The target-FPR evaluation uses the A2-conditioned positive population.
- AC-006: For each target FPR value, the selected threshold candidate satisfies `FPR <= target`.
- AC-007: Among candidates satisfying `FPR <= target`, the saved result reflects the candidate with FPR closest to the target and TPR maximal, with no extra tie-break behavior specified.

## Explicitly unresolved in this draft
- Which score column is used for threshold search.
- Where the new outputs are saved and what their stable filenames are.
- Whether existing single-target inference remains available unchanged, or only the evaluation-oriented flow changes.

## Promotion readiness
- Not ready for promotion to `docs/specs/**`.
- Not ready for implementation unlock because material product decisions remain open.
