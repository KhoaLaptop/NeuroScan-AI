# NeuroScan AI: Brain Tumor Classification from MRI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

**NeuroScan AI** classifies brain MRI scans as normal or tumorous, and subtypes tumors as meningioma, glioma, or pituitary tumor. It uses transfer learning on MobileNetV2, split across two models rather than one.

This is a research project. The full model report is in [REPORT.md](REPORT.md). See [Known limitations](#known-limitations) before reading anything into the accuracy numbers — the headline scores are measured on same-source test splits and **do not** reflect performance on scans from elsewhere.

## Architecture: two stages, not one

The obvious design is a single 4-way classifier over `normal / meningioma / glioma / pituitary`. That was the original approach here, and it scored 98% while being substantially broken. The reason is worth understanding, because it is easy to reproduce by accident.

Public brain-MRI datasets are usually single-purpose: one dataset provides tumor images, a different one provides healthy images. If you merge them, **the source dataset perfectly predicts the label**. The network then has no reason to learn anatomy — it can separate the classes by whatever incidental artifact distinguishes the two acquisition pipelines. Here the tumor images were uniform 512×512 clinical exports and the normals were heterogeneous web images (modal size 225×225), so the model learned image sharpness.

This was measurable. Resizing an external test scan from 512×512 to 225×225 — changing no anatomy whatsoever — more than doubled its accuracy. Augmentation did not fix it: suppressing the resolution cue just moved the model onto field-of-view instead, and suppressing that moved it somewhere else again. The correlation is in the label structure, so no input transform removes it.

The fix is structural. Each stage draws **all of its classes from a single source**, so source cannot predict label within either task:

| Stage | Task | Classes | Source |
|---|---|---|---|
| 1 | Detection | `Normal`, `tumor` | Br35H (`yes`/`no` from one pipeline) |
| 2 | Subtyping | `meningioma`, `glioma`, `pituitary tumor` | figshare (Cheng et al.) |

At inference, stage 1 runs first; only images it calls tumorous go on to stage 2.

The two stages deliberately look at different kinds of scan:

| Stage | Question | Scan type | Why |
|---|---|---|---|
| 1 | Is there a tumor? | **Plain** T1 / T2 (no contrast) | Healthy people almost never get contrast, so a contrast-trained detector learns "contrast was injected", not "tumor present" |
| 2 | Which type? | **Contrast** T1 | Every input is already a tumor, so contrast can't leak the label — and enhancement pattern is what distinguishes the types |

## Performance

### Same-source scores (upper bounds)

| Stage | Accuracy on its own held-out split |
|---|---|
| 1 — Detection (Br35H) | 98% |
| 2 — Subtyping (figshare) | 95% |

Both collapse on outside scans. Treat them as ceilings, not estimates.

### Stage 1 on outside scans

**Full results, version history and limitations: [REPORT.md](REPORT.md).**

The current detector is **v7**. On 6 fresh patients it was never tuned on (3 healthy, including two children, and 3 tumors), it gets 5 of 6 right at the patient level. It clears every healthy patient and catches the large aggressive tumors, but misses a neurocytoma. v1 flags 24–43% of each healthy patient's slices as tumor; v7 flags at most 6%.

What got there — each was necessary, none sufficient alone:
1. **Multi-plane, multi-sequence data** for both classes (Br35H is axial T2 only).
2. **Per-slice labels.** Labelling every slice of a tumor patient as "tumor" taught the model that healthy-looking anatomy is a tumor.
3. **No contrast on either side.** With contrast tumors against plain normals, v4 learned to detect contrast and caught 9% of tumors on a plain scan.
4. **Every plane and sequence in both classes.** When sagittal T2 appeared only in tumors, v5 learned "sagittal T2 = tumor" and flagged half of healthy T2 scans.
5. **Strong brightness/contrast augmentation.** v6 called an unfamiliar healthy scanner's images tumor 54% of the time; v7 cut that to 13%.

### Does stage 1 need more data, or different data?

A learning curve trained stage 1 on 10% → 100% of Br35H and scored it on outside normal scans ([learning_curve.py](learning_curve.py)):

| Br35H images | Br35H accuracy | Outside normal scans |
|---|---|---|
| 194 | 0.81 | 0.83 |
| 974 | 0.97 | 0.63 |
| 1,949 | 0.99 | 0.56 |

More axial T2 data made the model **worse** on real scans. The gap was coverage — planes and sequences — not volume.

### Stage 2 on outside scans

| Outside images | Correct |
|---|---|
| Meningioma, contrast (57 slices) | 68% — rest called glioma |
| Pituitary mass, sagittal (38) | 50% — rest called meningioma |
| Pituitary mass, coronal (13) | 100% |
| **Overall (108)** | **66%** |

Real signal, but far from the 95% same-source figure. Not yet fine-tuned on outside data.

## Known limitations

- **Small external test set.** Stage 1 has been tested on 12 outside patients and stage 2 on about three. These are encouraging signs, not validated results.
- **`predict.py` still uses v1 by default.** v7 is much better on outside scans; use it with `--detector best_model_detection_v7.pth`. It still misses smaller and well-defined tumors.
- **The two stages need different images.** Stage 1 expects a plain scan; stage 2 expects a contrast scan. `predict.py` currently passes the same image to both.
- **Stage 2 has no reject option.** It always returns meningioma, glioma or pituitary, so any other tumor — lymphoma, medulloblastoma, ependymoma — gets a confidently wrong label.
- **Coverage gaps in stage 1 training.** No sagittal T2 normals and no normal infant scans, while several tumor cases are infants — "small head" could become a tumor cue.
- **Slice labels are not radiologist-reviewed.** Visible-tumor marks were made from thumbnails; subtle lesions may be mislabelled.

## Installation

```bash
git clone https://github.com/KhoaLaptop/NeuroScan-AI.git
cd NeuroScan-AI
pip install -r requirements.txt
```

## Usage

### Inference

A single image:

```bash
python predict.py path/to/scan.jpg
```

A whole folder, searched recursively, with results written to CSV:

```bash
python predict.py path/to/folder --output predictions.csv
```

Output reports the stage 1 confidence for every image, plus a stage 2 confidence for anything routed to subtyping:

```text
Prediction: meningioma
Detection confidence: 0.9871
Subtype confidence: 0.9342
```

### Training

Each stage trains separately:

```bash
python main.py --task detection --epochs 25
python main.py --task subtype --epochs 25
```

Checkpoints are written to `best_model_detection.pth` and `best_model_subtype.pth`. Training auto-detects CUDA, then Apple MPS, then CPU, and saves whenever validation accuracy improves.

To fine-tune the detector on your own multi-plane data and compare it against the original on held-out patients:

```bash
NEW_MODEL=best_model_detection_v6.pth python finetune_v2.py
```

It starts from `best_model_detection.pth`, trains on Br35H plus the hand-collected folders, refuses to run if any test image also appears in training, and prints an old-vs-new table. `learning_curve.py` measures whether more Br35H data helps outside scans.

### Evaluation

```bash
python test.py --task detection
python test.py --task subtype
```

Writes `confusion_matrix_<task>.png` alongside a per-class report.

## Adding data

Dataset locations are defined in `TASK_ROOT_DIRS` in [main.py](main.py). Each root directory should contain one folder per class. Recognised folder names:

| Stage | Folder names |
|---|---|
| Detection | `Normal` or `no`; `tumor` or `yes` |
| Subtyping | `meningioma`; `glioma`; `pituitary tumor` or `pituitary` |

Aliases are defined in `TASKS` in [src/dataset.py](src/dataset.py). Folders not listed in the active task's class map are ignored, so a source containing extra classes can be pointed at directly.

For fine-tuning, hand-collected data lives in `Tumor Dataset/` (not committed — see below). Folders are set in `NEW_NORMAL_DIRS` and `TUMOR_DIRS` in [finetune_v2.py](finetune_v2.py); held-out test patients go in `Tumor Dataset/validation/` and are never trained on.

**The one rule that matters:** whatever you add, do not let any acquisition property line up with a class boundary. If every new normal comes from one scanner and every new tumor from another, or all normals are T2 and all tumors are T1, you have rebuilt the original bug in a new dimension. Each source, plane and sequence should appear across *all* classes of the stage it feeds.

Rules that came out of collecting data for stage 1:

- **Label slices, not patients.** Only save slices where the tumor is actually visible. A tumor patient's scan is mostly tumor-free slices.
- **No contrast for stage 1.** Use plain T1 / T2 on both sides. Contrast belongs to stage 2.
- **One folder per case and stack**, e.g. `tumor/case_125_glioblastoma/coronal_T2/`. Patient identity is what makes patient-level splits possible; hash filenames from browser downloads lose it.
- **Count patients, not images.** 200 slices of one brain is one sample.

Radiopaedia cases are licensed CC BY-NC-SA, so `Tumor Dataset/`, `Axial_T1/` and `Sagittal_T1/` are git-ignored rather than redistributed here.

## Medical disclaimer

**NeuroScan AI is a research tool and is NOT intended for clinical diagnosis.** It should not be used to inform medical decisions, and it is not a substitute for professional evaluation. Always consult a qualified healthcare provider.

## Data sources

- [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) — stage 1
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (figshare / Cheng et al. component) — stage 2
- External validation case courtesy of Frank Gaillard, [Radiopaedia.org rID 37605](https://radiopaedia.org/cases/normal-brain-mri-6)
- Additional fine-tuning and test cases from [Radiopaedia.org](https://radiopaedia.org), including the "1. MRI BRAIN TUMOUR" playlist — CC BY-NC-SA, not redistributed

## License

MIT.
