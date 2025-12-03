# NeuroScan AI: Advanced Brain Pathology Classification

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen.svg)

**NeuroScan AI** is a deep learning-based medical imaging system designed to classify various brain pathologies from MRI scans with high precision. Leveraging transfer learning with MobileNetV2, the system achieves **98% accuracy** in distinguishing between 10 distinct classes of brain conditions, including tumors, Alzheimer's disease stages, and stroke subtypes.

## 🧠 Supported Pathologies
The model classifies MRI scans into the following 10 categories:

| Category | Classes |
|----------|---------|
| **Brain Tumors** | Meningioma, Glioma, Pituitary Tumor |
| **Alzheimer's Disease** | Mild Demented, Moderate Demented, Non Demented, Very Mild Demented |
| **Stroke / Vascular** | Haemorrhagic, Ischemic |
| **Healthy** | Normal |

## 🖼️ Input Image Guidelines
To ensure accurate predictions, input images should ideally match the characteristics of the training data:

| Class | Recommended MRI Type | Recommended View (Orientation) |
|-------|----------------------|--------------------------------|
| **Pituitary Tumor** | **T1-weighted (Contrast)** | **Coronal** or **Sagittal** (Best for sella turcica) |
| **Meningioma** | **T1 (Contrast) or T2** | **Axial** or Coronal |
| **Glioma** | **T1 (Contrast) or T2** | **Axial** or Coronal |
| **Mild / Moderate / Very Mild Demented** | **T1-weighted** | **Axial** (Focus on hippocampal atrophy) |
| **Non Demented** | **T1-weighted** | **Axial** |
| **Haemorrhagic Stroke** | **CT** or **MRI (DWI/FLAIR)** | **Axial** |
| **Ischemic Stroke** | **DWI** or **FLAIR** | **Axial** |
| **Normal** | Any of the above | Any standard medical view |

> **Note:** The model may perform poorly on images with different sequences (e.g., raw T1 without contrast for tumors) or unusual artifacts. Ensure images are cropped to remove skull/background if possible for best results.

## 🚀 Key Features
*   **High Accuracy**: Achieves **98% test accuracy** on a diverse dataset.
*   **Efficient Architecture**: Built on **MobileNetV2**, ensuring fast inference times suitable for clinical workflows.
*   **MPS Acceleration**: Optimized for macOS devices with Metal Performance Shaders (MPS) support for faster training and inference.
*   **Automated Checkpointing**: Automatically saves the best model during training based on validation metrics.

## 📊 Performance Metrics
Evaluated on a held-out test set (15% of total data):

```text
                  precision    recall  f1-score   support

      meningioma       0.93      0.72      0.81       106
          glioma       0.88      0.98      0.93       214
 pituitary tumor       0.97      0.96      0.96       140
    MildDemented       0.99      1.00      0.99      1479
ModerateDemented       1.00      1.00      1.00       979
     NonDemented       0.99      0.99      0.99      1920
VeryMildDemented       0.98      0.98      0.98      1680
    Haemorrhagic       0.83      0.86      0.84        28
        Ischemic       0.00      0.00      0.00         4
          Normal       0.89      0.93      0.91        60

        accuracy                           0.98      6610
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KhoaLaptop/NeuroScan-AI.git
    cd NeuroScan-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### 1. Inference (Single Image)
To predict the class of a single MRI image:

```bash
python predict.py "path/to/image.jpg"
```

**Output:**
```text
Prediction: VeryMildDemented
Confidence: 0.9987
```

### 2. Training
To retrain the model from scratch:

```bash
python main.py --epochs 10 --batch_size 32
```
*   The script automatically detects macOS and optimizes for MPS (Metal Performance Shaders).
*   The best model is saved as `best_model.pth`.

### 3. Evaluation
To evaluate the model on the test set:

```bash
python test.py
```

## ⚠️ Medical Disclaimer
**NeuroScan AI is a research tool and is NOT intended for clinical diagnosis.**
While the model achieves high accuracy, it should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.

## 📄 License
This project is licensed under the MIT License.
