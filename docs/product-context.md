# Product context

## Purpose and users

AI Form Coach is a prototype for developers or research operators who collect
labelled squat repetitions from a camera and inspect basic technique feedback. The
person in front of the camera is the exercise participant, not a patient receiving a
medical diagnosis.

The primary supported workflow is: select a collection label, record one squat,
detect a completed repetition, reject poor capture quality, save a canonical
`[60, 12, 4]` sample, and show classification/rule-based feedback. Offline UI-PRMD
tools are for training experiments, not a dependency of the live application.

## Supported and unsupported claims

- Supported: the app can report capture-quality failures and heuristic squat-form
  observations from the configured camera model and thresholds.
- Supported: feedback is an engineering aid for data collection and coaching review.
- Unsupported: medical diagnosis, injury prediction, rehabilitation prescription,
  clinical validation, multi-person selection, or guarantees that a labelled sample
  reflects a person's true technique.

## Terms and UX constraints

- **Landmarks** are the selected 12 MediaPipe body points, ordered by
  `collector/canonical.py`.
- **Repetition** is a completed `STANDING → DESCENDING → BOTTOM → ASCENDING →
  STANDING` state-machine cycle.
- **Quality rejection** means a repetition is not sent to preprocessing, inference,
  or storage.
- **Feedback** is a short, observable message such as “Knees moving inward” or
  “Squat depth appears shallow”; it must not imply diagnosis or prescribe treatment.

The live HUD must keep feedback concise, show capture/rejection state, and remain
usable when no pose, camera, model, or external dataset is available. Do not silently
change landmark order, normalisation, or feedback wording without tests and a README
or product-context update.
