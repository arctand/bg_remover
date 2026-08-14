# False REVIEW reduction validation

Branch: `feature/reduce-false-review`
Production baseline: `d1a984cf561ff02058f0cdaf40503893d95f3225` (`v0.2.0-rc1`)

The matting architecture, model revisions, input resolution, PyMatting refinement,
SSDLite model, SAM 2.1 model, GUI, output format, and public READY/REVIEW/FAILED
statuses were not changed.

## Old pilot diagnosis

The immutable 170-image pilot had 20 REVIEW results: 8 useful and 12 false.
The same file can contribute to more than one reason below.

| Reason | Total | Real problem | False REVIEW |
| --- | ---: | ---: | ---: |
| `mask_issue` | 9 | 5 | 4 |
| `low_confidence` | 6 | 4 | 2 |
| `missing_body_part` | 10 | 3 | 7 |
| `human_mask_disagreement` | 10 | 3 | 7 |
| `multiple_people_uncertain` | 10 | 3 | 7 |

The hypothesis was confirmed: large-hole and translucency signals were used both
as SAM triggers and as irreversible final REVIEW reasons. A clean, complete SAM
result could not clear them. False semantic REVIEWs were also caused by small
background people, low-confidence boxes, and a large SSDLite box spanning two
already-supported foreground people. Old incomplete verification affected six
REVIEW rows; four were real problems and two were false REVIEWs, so incomplete
verification was not globally relaxed.

## Policy change

Fast signals now have three explicit classes:

- Hard REVIEW: almost-empty mask, almost-full mask, or too many significant
  disconnected components. SAM cannot clear these.
- Verification trigger: large internal holes, high translucency, or marginal
  coverage of a relevant SSDLite person box. A complete clean SAM result may
  clear only these signals.
- Telemetry: source-frame contact/crop signal and zero person detections. These
  do not independently change status.

SAM errors, unavailable verification, and `checked_people < prompted_boxes`
remain conservative REVIEW for a suspicious result. `FAILED` remains reserved
for technical processing errors.

The existing SAM decision thresholds were not changed:

```text
box coverage              0.45
center coverage           0.55
multiple-person coverage  0.60
minimum person recall     0.82
minimum missing-box ratio 0.02
boundary tolerance ratio  0.006
```

Prompt filtering uses the existing detector prompt confidence of `0.80` plus
two new calibration settings:

```text
sam_prompt_min_relative_area     0.40
sam_prompt_overlap_suppression   0.50
```

A partially missing box is retained conservatively when its relative area is at
least 0.30, coverage at least 0.09, and center coverage below 0.1485. A box
overlapped at least 50% by supported people is filtered only when its own alpha
coverage is below 0.2475. These rules apply only to SAM prompts and never modify
production alpha or RGB.

On the old false REVIEWs, the decision-policy fix cleared the low-confidence
zero-prompt case, severity separation cleared three clean large-hole cases, and
prompt filtering cleared seven background/overlap cases. The one remaining
false REVIEW is `p3m_np_275`, where SAM evaluated 0/1 prompts; it intentionally
keeps the conservative incomplete-verification policy.

Reporting now separately exposes `fast_qc_hard_reasons`,
`verification_triggers`, `telemetry_signals`, `sam_requested`, `sam_ran`,
`sam_result`, and `final_review_reasons`. `review_reason` remains as a compatible
alias for final reasons. `PIPELINE_SCHEMA_VERSION` is 3, so Resume cannot reuse
decisions produced by the previous policy fingerprint.

## Old 170 regression

| Metric | RC1 policy | New policy |
| --- | ---: | ---: |
| READY | 150 | 161 |
| REVIEW | 20 | 9 |
| FAILED | 0 | 0 |
| False REVIEW among REVIEW | 12/20 (60.0%) | 1/9 (11.1%) |
| REVIEW precision | 8/20 (40.0%) | 8/9 (88.9%) |
| SAM invocation | 29/170 (17.1%) | 22/170 (12.9%) |

All six critical missing-person/body cases remain REVIEW:

- `complex_03_unsplash.jpg`
- `complex_04_unsplash.jpg`
- `complex_05_unsplash.jpg`
- `complex_10_unsplash.jpg`
- `complex_14_pexels.jpg`
- `complex_19_pexels.jpg`

The additional useful non-critical REVIEWs (`complex_06_unsplash.jpg` foreground
artifact and `complex_09_unsplash.jpg` non-person/almost-empty result) also remain
REVIEW.

## Independent validation

The validation set contains 125 new files: 120 deterministic P3M-500-NP images
and five deliberately difficult public Wikimedia scenes with crowds, small
people, occlusion, objects, and complex backgrounds. SHA-256 comparison found no
overlap with either the old 20 regression photos or the 170-image pilot, and no
duplicates inside the new set.

| Metric | Result |
| --- | ---: |
| Files | 125 |
| READY | 122 |
| REVIEW | 3 |
| FAILED | 0 |
| Random READY sample | 30 |
| Additional READY inspected | 8 (all remaining SAM-confirmed plus complex scenes) |
| Inspected READY usable | 38/38 (100%) |
| Missing-body failures in inspected READY | 0 |
| Inspected REVIEW | 3/3 |
| False REVIEW | 0/3 (0%) |
| REVIEW precision | 3/3 (100%) |
| SAM invocation | 10/125 (8.0%) |

All REVIEW and all FAILED results were required to be inspected (there were no
FAILED results). READY inspection used original/white/gray/black/contrast
previews; no critical mask failure or unacceptable halo was found in the 38
inspected READY files. The P3M portion is portrait-heavy, so the five difficult
public scenes were retained as a separate challenge layer rather than claiming
that P3M alone covers all production conditions.

## RTX 5070 performance

Combined production-mode timings for the 125 validation images:

| Metric | Result |
| --- | ---: |
| Total processing time | 80.877 s |
| Average | 0.647 s/photo |
| Median | 0.609 s |
| P95 | 0.774 s |
| Peak VRAM | 3.027 GiB |

The final old-170 rerun was 0.647 s average, 0.609 s median, 0.825 s P95, and
3.027 GiB peak VRAM, consistent with the RC1 reference. Always-on models are
loaded once per batch; SAM remains lazy/conditional. Prompt-overlap calculation
uses rectangle geometry rather than an image-sized temporary bitmap.

## Acceptance

- Old critical cases retained in REVIEW: **6/6**.
- Confirmed critical missing-body failures in new READY: **0**.
- New inspected READY precision: **100%** (38/38; sample estimate).
- False REVIEW among new REVIEW: **0%**.
- Old calibrated false REVIEW: **11.1%**, below the 30% target.
- Old calibrated REVIEW precision: **88.9%**, above the 70% target.
- Technical failure rate did not increase.

Recommendation: **SAFE TO MERGE**, subject to the normal separate approval to
merge. The known motion-blur contamination limitation remains unchanged.
