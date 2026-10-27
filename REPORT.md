# NeuroScan AI — Model Report

**Model:** stage 1 tumor detector, version 7 (`best_model_detection_v7.pth`)
**Task:** does this brain MRI slice show a tumor? (`Normal` vs `tumor`)
**Status:** research prototype. **Not for clinical use.**

## Summary

v7 reliably clears healthy brains, including children, and catches large aggressive tumors such as glioblastoma. It misses smaller or well-defined tumors. On 6 fresh patients it was never tuned on, it got **5 of 6 right at the patient level**; the one miss was a neurocytoma.

The headline figure that earlier versions of this project reported (98% accuracy) measured the wrong thing. Most of the work went into finding out why, and fixing the data rather than the model.

## Model

| | |
|---|---|
| Architecture | MobileNetV2, ImageNet-pretrained, all layers fine-tuned |
| Starting point | v1 detector, trained on Br35H only |
| Training data | Br35H (1,949 images, axial T2) + 816 normal and 208 tumor slices hand-collected from Radiopaedia |
| Hand-collected data | plain T1/T2 only (no contrast), sagittal and coronal, only slices where the tumor is visible |
| Class balance | tumor slices repeated ×4 to match normals |
| Augmentation | resolution jitter, random crop, blur, flips, rotation, and strong brightness / contrast / gamma / equalisation changes |
| Training | 15 epochs, SGD, lr 0.001; [finetune_v2.py](finetune_v2.py) |

## Results

### Same-source test (upper bound)

| Br35H held-out split | Correct |
|---|---|
| Normal (225 images) | 98.7% |
| Tumor (225 images) | 97.3% |

These come from the same dataset the model trained on, so they overstate real performance.

### Fresh validation — 6 patients never used for any decision

Different contributors, scanners and ages. Numbers are the **% of each patient's slices flagged as tumor**.

| Patient | Truth | v1 | v6 | **v7** |
|---|---|---|---|---|
| 3-month-old boy | normal | 24% | 7% | **6%** |
| 9-year-old boy | normal | 33% | 4% | **2%** |
| 30-year-old woman | normal | 43% | 0% | **3%** |
| 60-year-old man, gliosarcoma | tumor | 70% | 70% | **70%** |
| Adult, glioblastoma | tumor | 45% | 51% | **47%** |
| 20-year-old man, extraventricular neurocytoma | tumor | 52% | 7% | **2%** |

Any patient-level cutoff between about 7% and 45% classifies 5 of 6 correctly. No cutoff catches the neurocytoma, which is flagged less than the healthy infant.

### Development validation — 6 patients used while building the model

These were used to compare versions, so the scores below are somewhat optimistic. Tumor rows also show **visible-tumor slices caught**, where slices were marked.

| Patient | Truth | v1 | v5 | v6 | **v7** |
|---|---|---|---|---|---|
| Normal patient 1 | normal | 44% | 4% | 1% | **4%** |
| Normal patient 2 | normal | 58% | 2% | 0% | **0%** |
| Normal patient 3 (unfamiliar scanner) | normal | 99% | 48% | 54% | **13%** |
| Tumor patient 1 | tumor | 20/23 | 17/23 | 18/23 | **19/23** |
| Tumor patient 2 (cystic) | tumor | 15/15 | 10/15 | 5/15 | **8/15** |
| Tumor patient 3 (child, round lesion) | tumor | 91% | 24% | 9% | **2%** |

v1's high tumor numbers are not skill: it flags 44–99% of every healthy patient too.

## What changed between versions

Each version fixed a shortcut the previous one had learned. In every case the fix was in the data, not the network.

| Version | Change | What it revealed |
|---|---|---|
| Original | one 10-class model | 98% on its own test set, ~21% on a real outside scan — it learned image sharpness from which dataset an image came from |
| v1 | split into detection + subtyping, each from one source | outside axial scans improved; 0% on sagittal (Br35H is axial only) |
| v2–v4 | added hand-collected multi-plane scans | labelling whole tumor patients as "tumor" taught it that healthy anatomy is a tumor; contrast tumors vs plain normals taught it to detect contrast |
| v5 | per-slice labels, no contrast, balanced classes | first model good on both tumors and normals — but secretly used "sagittal T2 = tumor" (it flagged 49% of healthy T2 scans) |
| v6 | added healthy T2 scans | shortcut removed; more cautious; called an unfamiliar healthy scanner's images tumor (54%) |
| v7 | strong brightness/contrast augmentation | false alarms on that scanner fell from 54% to 13%; visible-tumor detection rose to 27/38 |

A learning curve ([learning_curve.py](learning_curve.py)) also showed that more Br35H data made outside performance **worse**: 10× more images raised Br35H accuracy from 0.81 to 0.99 but cut accuracy on outside normal scans from 0.83 to 0.56. The gap was coverage, not volume.

## Limitations

- **Small test set.** 12 outside patients in total. Results will move as more are added.
- **Misses small or well-defined tumors.** Neurocytoma and a child's round lesion were essentially missed. Detection is reliable mainly for large, aggressive tumors.
- **Labels are not radiologist-reviewed.** Visible-tumor slices were marked from thumbnails with AI assistance.
- **Per-slice, not per-scan.** Slice order isn't preserved in the downloaded images, so the model can't use neighbouring slices; the patient-level answer is a simple percentage cutoff chosen by inspection.
- **Not the default yet.** `predict.py` still loads v1. Use v7 with `--detector best_model_detection_v7.pth`.
- **Weights not in the repository.** `.pth` files are git-ignored; train with `finetune_v2.py`.
- **Stage 2 (tumor type) is separate and weaker:** 66% on outside scans, and it expects contrast images.

## Reproduce

```bash
NEW_MODEL=best_model_detection_v7.pth python finetune_v2.py
python evaluate_validation.py v7=best_model_detection_v7.pth
```

Training and validation scans come from Radiopaedia (CC BY-NC-SA) and are not redistributed here. Fresh-validation cases courtesy of Dalia Ibrahim, Ryan Thibodeau, Morlie L Wang and Hani M. Al Salam, Radiopaedia.org.
