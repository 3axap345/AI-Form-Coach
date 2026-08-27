# Form-analysis intent and calibration status

## Status and ownership

The rule-based feedback in this prototype is an engineering heuristic, not a medical
or clinical assessment. **Owner:** project owner. Any change to medical claims,
coaching-rule meaning, label semantics, or the thresholds below requires explicit
project-owner approval, a matching test update, and an update to this document.

No threshold below has a validated clinical source. Their source is the MVP's current
engineering implementation and synthetic regression fixtures. **Calibration TODO:**
calibrate against labelled target-camera recordings, record the dataset and owner,
then replace the applicable TODO rows with evidence before making stronger claims.

## Form-analysis thresholds

| Setting | Current intent | Source/status | Owner and calibration TODO |
| --- | --- | --- | --- |
| `shallow_depth_knee_angle = 115°` | Flag a shallow bottom position when the visibility-weighted knee angle reaches this relative score. | Engineering heuristic; not clinically validated. | Project owner; calibrate depth labels from target-camera data. |
| `excessive_forward_lean_angle = 28°` | Flag large torso lean at the bottom of a repetition. | Engineering heuristic; not clinically validated. | Project owner; calibrate against reviewed captures. |
| `knee_valgus_ratio_drop = 0.18` | Flag a material reduction of knee width relative to foot width at the bottom. | Engineering heuristic; not clinically validated. | Project owner; review label definition and calibrate with target captures. |
| `heel_instability_threshold = 0.12` | Flag heel movement relative to ankle/foot landmarks. | Engineering heuristic; not clinically validated. | Project owner; calibrate for camera distance and landmark noise. |
| `form_analysis_min_visibility = 0.4` | Exclude low-confidence frames from knee-angle weighting. | Implementation safety threshold. | Project owner; validate false-positive/false-negative trade-off. |

## Live quality thresholds

| Setting | Current intent | Source/status | Owner and calibration TODO |
| --- | --- | --- | --- |
| `min_avg_visibility = 0.6` | Reject a repetition whose mean landmark confidence is too low. | Implementation safety threshold. | Project owner; calibrate with target camera and MediaPipe version. |
| `min_keypoint_visibility = 0.4` | Reject if the per-frame mean visibility of critical hip/knee/ankle points is below this confidence. | Implementation safety threshold. | Project owner; calibrate against rejected capture examples. |
| `max_missing_frame_ratio = 0.15` | Reject repetitions with too much pose-tracking loss. | Implementation safety threshold. | Project owner; calibrate at intended FPS and camera placement. |
| `min_bbox_area_ratio = 0.05` / `max_bbox_area_ratio = 0.85` | Keep subject framing within usable bounds. | Implementation safety threshold. | Project owner; validate with intended capture distances. |

## UI-PRMD label mapping

`collector/canonical.py` maps `C01` to `correct` and `C02` to `incorrect`.
The current source is the local converted-tree convention used during this MVP, not a
verified upstream UI-PRMD label citation. **Owner:** project owner. **TODO:** verify
the class semantics against primary UI-PRMD documentation before publishing training
metrics or treating the mapping as ground truth. The mapping is tested in
`collector/tests/test_pipeline.py` and must not be silently changed.
