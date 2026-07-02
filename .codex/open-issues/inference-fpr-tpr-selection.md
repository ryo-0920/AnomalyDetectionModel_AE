# Open Issues: inference-fpr-tpr-selection

This file lists only current unresolved product decisions. No default behavior is assumed.

## 1. Score basis for threshold search
- The request does not specify which score should drive the threshold search.
- Current code context is not uniform:
  - `src/gofumi_ae/inference/standard.py` uses file-level `file_prob` for confusion/ROC outputs.
  - `src/gofumi_ae/evaluation/standard.py` defaults threshold analysis to `anomaly_rate`.
- Decision needed:
  - Which file-level score is the spec target for threshold selection?

## 2. Output contract for saved target-FPR results
- The request specifies what values must be saved, but not where or under what filenames.
- Decision needed:
  - Exact output location
  - Exact filename(s)
  - Whether the new artifacts belong only to TF-style outputs, only to legacy outputs, or to both

## 3. Backward-compatibility scope for the current one-target prompt
- Current inference prompts for one scoring target.
- The request requires ON/OFF selection for this evaluation change, but does not say whether the old one-target flow must remain alongside it.
- Decision needed:
  - Replace the one-target evaluation path
  - Or add a separate ON/OFF evaluation mode while keeping the existing single-target path
