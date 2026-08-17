# Artifact detection validation

Branch: `feature/artifact-detection`
Baseline: `762837bf456801143456d854a2cd72321ddf25a7` (`v0.2.0-rc2`)

## Scope and design

The RC2 matting and semantic-safety pipeline is unchanged. The new
`bgremover/artifacts.py` layer reads the original RGB image and final RGBA result,
downscales them to at most 512 pixels on the long side, and reports quality
evidence without modifying RGB or alpha.

For each stable alpha-transition pixel, the detector follows the local alpha
gradient inward to an opaque foreground reference and outward to the original
background. It measures:

- `residual_background_color`: edge colour moving from the inner foreground in
  the direction of the original background. Chromatic and neutral/luminance
  scores are kept separately, so the rule is not red-specific.
- `loose_edge_noise`: robust 90th-percentile transition width along the local
  edge normal, normalized for analysis resolution.
- `overcut_compact_structure`: an outside pixel resembles the inner foreground
  more than the farther original background, with low local texture.
- `weak_hair_edge`: the same omitted-detail evidence in a textured edge zone.
- `local_alpha_instability`: small isolated low-alpha components outside the
  supported core.
- `semantic_edge_anomaly`: agreement between chromatic residual and loose-edge
  scores. A separate local-agreement score requires both signals to occur on the
  same compact edge segment.

White, gray, black, and contrast composites are explicitly available in the
six-panel debug preview. For numeric residual detection the canvas term cancels
when actual and locally expected composites are compared; alpha scales the
remaining foreground-colour error. The detector therefore computes that shared
visibility term once instead of allocating four full composite arrays.

## Severity and QC integration

Artifact evidence has its own `hard`, `weak`, `telemetry`, and `none` severity.

- Hard `edge_artifact` is allowed to add a final REVIEW reason. It requires a
  calibrated, localized combination of background-colour residual, abnormal
  edge width, and compact low-texture spatial agreement. Independently strong
  compact damage plus alpha instability is a second hard path.
- Loose-edge, local-instability, and combined semantic-edge signals are weak
  triggers. They are reported in telemetry but do not force REVIEW and do not
  invoke SAM, because SAM verifies body completeness rather than fine edge RGB.
- Standalone residual, overcut, or hair evidence remains telemetry. This avoids
  treating legitimate coloured/translucent hair as a hard defect.
- If the person detector finds no relevant person, artifact evidence is retained
  but cannot become a hard decision. Existing QC and SAM reasons retain their
  original priority and behavior.

The report now includes per-class scores, total score, flags, severity, hard
details, weak triggers, telemetry, edge sample count, and local agreement. The
pipeline schema/fingerprint was increased from 3 to 4, so Resume cannot reuse an
RC2 report with the older decision schema.

The intentionally bounded hard ranges distinguish a localized anomalous patch
from pervasive, internally consistent soft hair. During calibration, unbounded
scores incorrectly marked 25/120 previously inspected usable P3M portraits as
hard; the released thresholds reduce that independent set to zero hard artifact
decisions while retaining the required localized regression.

## Validation

### Known 20-photo regression

A fresh full pipeline run produced `18 READY / 2 REVIEW / 0 FAILED`.

| Case | Artifact result | Final result |
| --- | --- | --- |
| `04_person` | no flag (telemetry only) | READY |
| `12_person` | hard: residual `0.7208`, loose `0.6667`, local agreement `0.4630` | REVIEW: `edge_artifact` |
| `14_person` | weak neutral residual + loose + semantic anomaly | READY |
| `16_person` | weak residual + loose + semantic anomaly | READY |
| `18_person` | weak neutral residual + loose + semantic anomaly | READY |
| `19_person` | weak artifact evidence | existing semantic REVIEW preserved |

`11_person`, a visually good ordinary portrait, had only a conservative neutral
residual telemetry flag and no weak/hard decision. Other standalone colour/hair
observations also stayed telemetry and did not change status.
The required `12_person` decision is based only on image/alpha measurements; no
filename or case-specific rule exists.

### False-positive and safety sets

The analyzer was run over saved, immutable RC2 outputs:

| Set | Files | Hard artifact | Consequence |
| --- | ---: | ---: | --- |
| Independent P3M validation | 120 | 0 | no new REVIEW |
| Old calibrated pilot | 170 | 0 | no new REVIEW; existing 9 semantic/QC REVIEW decisions remain |
| Independent complex scenes | 5 | 0 | no new REVIEW |

Weak/telemetry signals are deliberately more common and are visible for future
calibration, but they cannot heat up READY/REVIEW. All existing missing-body,
multiple-person, SAM error, Resume, and reporting tests remain green.

The reproducible offline command is:

```powershell
.venv\Scripts\python.exe quality_benchmark\artifact_validation.py ORIGINALS RUN --summary
```

Add `--previews PATH` to create local six-panel previews for hard cases. Generated
images and benchmark outputs remain ignored by git.

## Performance

The same RTX 5070 20-photo run was compared with the saved RC2 run:

| Metric | RC2 | Artifact layer |
| --- | ---: | ---: |
| Average | 1.2027 s | 1.3586 s |
| Median | 0.8942 s | 1.0507 s |
| P95 | 2.1315 s | 1.9665 s |
| Peak VRAM | 3.0273 GiB | 3.0273 GiB |

The CPU-only detector added about 0.156 seconds to average and median processing
on this small benchmark (approximately 13% average). It adds no model and no GPU
memory. P95 did not regress.

## Tests and limitations

`49` tests pass. New controlled tests cover non-red chromatic and neutral residual
colour, loose edge, clean solid edge, hard versus weak batch integration, and
semantic suppression.

Known limitations:

- compact anatomy is inferred from local texture/shape rather than a dedicated
  face-part model;
- neutral spill is intentionally conservative and normally telemetry-only;
- natural coloured hair can produce a high standalone residual score;
- thresholds are calibrated at the 512-pixel analysis scale and should be
  revalidated if that scale or the foreground refiner changes.

## Recommendation

`RECOMMEND ARTIFACT LAYER`

It detects the required edge-quality regression, preserves semantic safety,
creates no hard artifact decisions on the 290-image false-positive calibration,
and adds bounded CPU cost with unchanged VRAM.
