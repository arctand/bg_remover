# BG Remover quality benchmark

## Scope

This research branch is based on production commit `847faa61b59adfeb7d5e0af28c84fba9e1d2ea78`.
It does not modify the production processing pipeline, GUI, QC implementation, or launchers.
All research code lives in `quality_benchmark/`, `quality_benchmark.py`, and
`benchmark_quality.yaml`.

The benchmark used the 20 local files `01_person.jpg` through `20_person.jpg` from
`C:\Users\user\BGCRUSHER_TEST_PHOTOS`. Photos, masks, previews, contact sheets, and
metrics remain local in the gitignored `benchmark_output/` directory.

Hardware and runtime:

- NVIDIA GeForce RTX 5070, 11.94 GiB VRAM;
- Python 3.12.10;
- PyTorch 2.11.0+cu130, CUDA runtime 13.0;
- FP16 BiRefNet inference;
- one heavy model resident in VRAM at a time;
- model download/load time excluded from per-photo latency; image inference and
  foreground-estimation time included.

The sample has no ground-truth alpha mattes. Therefore this run cannot honestly report
SAD, MSE, gradient, or connectivity accuracy. Mask conclusions use a visual audit of
the named regression cases, pairwise mask agreement, existing QC, and agreement with
an independent SAM 2.1 prompted mask. Foreground RGB conclusions use visual composites
plus a reconstruction-error proxy based on the observed color equation and a locally
estimated background. These are decision aids, not a substitute for a labeled matting
test set.

## Implementations actually tested

### Mask generation

| Variant | Exact model / method | Input | Notes |
| --- | --- | ---: | --- |
| Current HR | `ZhengPeng7/BiRefNet_HR-matting@5d6b6f8` | 2048² | Current production model and preprocessing. |
| Portrait | `ZhengPeng7/BiRefNet-portrait@ecdeb624` | 1024² | Official portrait/human weights, trained for P3M/human matting. |
| Dynamic secondary | `ZhengPeng7/BiRefNet_dynamic-matting@074df545` | 2048² | Current configured secondary candidate. |
| Person-guided HR | HR model above, applied to expanded SSDLite person boxes | 2048² per crop | Implements the official box-guided crop/paste approach with an 8% detector-box margin. It is not a distinct person-guided checkpoint. |

Official references: [BiRefNet repository](https://github.com/ZhengPeng7/BiRefNet),
[HR-matting weights](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting),
[portrait weights](https://huggingface.co/ZhengPeng7/BiRefNet-portrait), and the
[official box-guided notebook](https://colab.research.google.com/drive/1B6aKZ3ekcvKMkSBn0N5mCASLUYMp0whK).
BiRefNet source code is MIT licensed. The HR and dynamic Hugging Face cards identify
MIT; the portrait card did not expose a license tag through the Hub API during this run,
so the portrait weight license should be confirmed before commercial release even
though it is published by the official MIT-licensed project.

### Foreground RGB refinement

All refiners received exactly the same `hr_matting` alpha, so they do not change mask
coverage.

| Variant | Implementation | License |
| --- | --- | --- |
| Raw RGB | Original RGB under the predicted alpha | Project |
| Current edge | Existing `bgremover.edge.decontaminate_rgb` | Project |
| Official BiRefNet | GPU `refine_foreground`, two-pass blur-fusion estimator from official `image_proc.py`, radius 90 | MIT |
| PyMatting ML | `PyMatting==1.1.15`, `estimate_foreground_ml` | MIT |

References: [official BiRefNet foreground implementation](https://github.com/ZhengPeng7/BiRefNet/blob/main/image_proc.py)
and [PyMatting](https://github.com/pymatting/pymatting).

### Independent verification

| Variant | Role | License |
| --- | --- | --- |
| `ssdlite320_mobilenet_v3_large` | Fast COCO person detection and current box-coverage QC | TorchVision BSD-style |
| `facebook/sam2.1-hiera-small@ee5bba1d` | Strong prompted person-mask comparison using the SSDLite boxes | Apache-2.0 |

SAM 2.1 is the [official Meta implementation](https://github.com/facebookresearch/sam2)
and [official Small checkpoint](https://huggingface.co/facebook/sam2.1-hiera-small).
It was installed from SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

## RTX 5070 measurements

### Masks

| Variant | Successful | Errors | Avg s/photo | Median | P95 | Peak VRAM GiB | Mean SAM IoU* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HR-matting | 20 | 0 | 1.733 | 1.419 | 1.788 | 9.422 | 0.849 |
| Portrait | 20 | 0 | 0.228 | 0.217 | 0.268 | 2.688 | 0.851 |
| Dynamic-matting | 20 | 0 | 1.252 | 1.107 | 1.298 | 9.421 | 0.841 |
| HR person-guided | 19 | 1 | 2.266 | 1.414 | 6.752 | 9.421 | 0.877 |

`*` SAM agreement exists for 19 cases. It is not ground truth. Guided masks naturally
score better against SAM because both are constrained by the same SSDLite boxes; this
does not reward retention of a motorcycle, basket, architecture, or other desired
non-person foreground.

Portrait and HR have mean pairwise binary IoU `0.945`. The largest differences are
`19_person`, `02_person`, `09_person`, `13_person`, and `06_person`, which is why the
recommendation is based on their visual results rather than the average alone.

### Foreground refinement

| Variant | Avg s/photo | Median | P95 | Peak VRAM GiB | Mean reconstruction MAE (lower is better) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw RGB | 0.000 | 0.000 | 0.000 | 0.000 | 0.05869 |
| Current edge | 2.992 | 2.006 | 6.282 | 0.000 | 0.04136 |
| Official BiRefNet | 0.099 | 0.076 | 0.155 | 0.549 | 0.04315 |
| PyMatting ML | 0.322 | 0.258 | 0.496 | 0.000 | **0.03946** |

For the explicit halo case `12_person`, reconstruction MAE was `0.02931` for current
edge, `0.03265` for official BiRefNet refinement, and **`0.02854` for PyMatting ML**.
The white/gray/black/contrast composites also show PyMatting removing old-background
edge color while the reconstruction proxy remains lowest; the current method is much
slower on these full-resolution files.

### Verifiers

| Verifier | Successful | Errors | Avg s/photo | Median | P95 | Peak VRAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SSDLite detector | 20 | 0 | 0.063 | 0.037 | 0.070 | 0.201 |
| SAM 2.1 Hiera Small | 19 | 1 | 0.088 | 0.078 | 0.127 | 0.611 |

`13_person` contains architecture and tiny people. SSDLite returned no person box, so
strict person-guided BiRefNet and prompted SAM 2.1 were not run for this case. This is a
real limitation: SAM 2.1 is strong after a valid prompt, but it does not replace the
detector or provide a reliable unprompted production decision in this implementation.

## Regression findings

| Case(s) | Finding |
| --- | --- |
| `06_person` | HR and dynamic keep the two people but omit more surrounding chain/chair context. Portrait preserves the complete human grouping and more attached context. Guidance does not repair an already weak or tight detector box. |
| `12_person` | Mask differences are small; the visible failure is foreground RGB contamination. PyMatting ML is the best tested halo remover. |
| `04`, `14`, `16`, `18` | Portrait retains hair detail at least as well as HR in the generated composites. RGB refinement, not a different alpha threshold, is needed for old-background color in translucent hair. |
| `02_person` | Full-frame HR, portrait, and dynamic retain the motorcycle. Strict person guidance catastrophically removes most of it (HR/guided pairwise IoU `0.130`). |
| `10_person` | All full-frame models retain the basket; guidance happens to retain it because it falls inside the person box, but that is detector-box dependent. |
| `13_person` | HR and portrait retain the architecture. Dynamic collapses to foreground ratio `0.0108`; guided and SAM verifier cannot run because SSDLite detects no person. |
| `03`, `11`, `17` | Every model receives the same false `cropped_source` review. This is a QC heuristic regression, not a matting-model problem. |

## Concrete recommendation

For the next production experiment, use exactly this staged design:

```text
SSDLite320 MobileNet V3 person detection / advisory guidance
(never a hard crop; full-frame fallback is mandatory)
→
ZhengPeng7/BiRefNet-portrait @ ecdeb624, 1024×1024
→
PyMatting 1.1.15 estimate_foreground_ml foreground RGB refinement
→
existing fast QC, but cropped_source is telemetry-only unless corroborated
→
facebook/sam2.1-hiera-small @ ee5bba1d only for suspicious cases with valid boxes
```

This recommended always-on path (SSDLite + portrait + PyMatting, excluding file I/O,
model load, and the existing lightweight numerical QC) measured:

- average: **0.613 s/photo**;
- median: **0.510 s/photo**;
- P95: **0.828 s/photo**;
- peak VRAM: **2.688 GiB**, because stages are sequential;
- benchmark stage errors: **0**.

Running SAM 2.1 for every box-valid image would add about `0.088 s/photo`; production
should call it only when fast QC, model disagreement, or low detector/mask coverage is
suspicious.

## Direct decisions

- **Keep `BiRefNet_HR-matting`?** Keep it available as a research/quality fallback, but
  do not keep it as the default for this person-heavy workload. Portrait produced the
  best quality/performance trade-off and used 71% less peak VRAM. A labeled alpha test
  set is still required before deleting HR support.
- **Is portrait better?** Yes for this 20-photo regression set: similar or slightly
  better SAM agreement, cleaner behavior on `06` and `13`, 0 errors, and about 7.6×
  lower inference latency than HR.
- **Is person guidance needed?** Detection is useful as advisory context and for
  verification. Hard crop/paste guidance is not safe as the primary mask: it breaks
  `02`, cannot handle `13`, and becomes slow for multiple people.
- **Best halo refinement?** PyMatting `estimate_foreground_ml`. It has the lowest mean
  reconstruction error and wins `12_person`; it is also much faster than current edge
  decontamination on these full-resolution images. The official BiRefNet refiner is the
  best speed-oriented alternative, but not the best halo result in this run.
- **Is SAM 2 needed?** Yes as an escalated verifier for suspicious cases, not as an
  always-on generator and not without a valid prompt.
- **Keep current SSDLite?** Yes. It is cheap and supplies detection/box guidance, but a
  zero-box result must not reject or erase the image. It remains necessary to prompt
  this SAM 2.1 design.
- **Keep secondary `BiRefNet_dynamic-matting`?** No for production. It uses essentially
  the same peak VRAM as HR, has lower SAM agreement, and catastrophically fails
  `13_person`. Keep only as a research comparison if desired.
- **What to do with `cropped_source`?** Demote it from an automatic REVIEW reason to a
  weak telemetry signal. Escalate only when it is corroborated by low person-mask
  coverage, SAM disagreement, or another semantic missing-body signal. Normal bust,
  waist, and frame-edge portraits (`03`, `11`, `17`) must not be rejected solely for
  touching the frame.

## Reproduction and local artifacts

```powershell
.venv\Scripts\python.exe -m pip install -r benchmark-requirements.txt
.venv\Scripts\python.exe quality_benchmark.py --config benchmark_quality.yaml
```

The default config resolves the local photos as `../BGCRUSHER_TEST_PHOTOS` and writes:

- `benchmark_output/metrics.csv`;
- `benchmark_output/metrics.json`;
- `benchmark_output/contact_sheets/regression_cases.jpg`;
- `benchmark_output/contact_sheets/masks_*.jpg`;
- `benchmark_output/contact_sheets/refinement_*.jpg`;
- `benchmark_output/side_by_side/01_person.jpg` through `20_person.jpg`;
- per-variant masks and previews under `benchmark_output/masks/` and
  `benchmark_output/previews/`.

The entire `benchmark_output/` directory is gitignored and must not be committed.
