# Deepfake Detection System

CPU-friendly deepfake detection project using transfer learning with EfficientNet-B0. The model crops faces with OpenCV Haar cascades, trains only a lightweight classifier head, evaluates with scikit-learn metrics, and serves predictions through Streamlit.

## Project Structure

```text
DeepLearning Project/
├── dataset/
│   ├── train/
│   │   ├── real/ or real_image/
│   │   └── fake/ or fake_image/
│   ├── val/
│   │   ├── real/ or real_image/
│   │   └── fake/ or fake_image/
│   └── test/
│       ├── real/ or real_image/
│       └── fake/ or fake_image/
├── models/
│   └── best_model.pth
├── src/
│   ├── __init__.py
│   ├── check_data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## What This Project Does

- Loads images with `torchvision.datasets.ImageFolder`.
- Supports class folders named `real`/`fake` or `real_image`/`fake_image`.
- Detects faces using OpenCV Haar cascade.
- Crops the largest detected face before training, validation, testing, and inference.
- Resizes face crops to `224x224`.
- Uses ImageNet normalization.
- Fine-tunes only the classifier head of EfficientNet-B0.
- Handles class imbalance with balanced sampling by default.
- Saves a validation-tuned decision threshold in the checkpoint.
- Saves the best checkpoint to `models/best_model.pth`.
- Evaluates precision, recall, F1-score, and confusion matrix.
- Provides a Streamlit demo with image upload and webcam capture.

## Dataset Layout

The expected labeled layout is:

```text
dataset/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

This project also accepts:

```text
real_image/
fake_image/
```

The test split must contain labeled real/fake folders for `src/evaluate.py`, because classification reports and confusion matrices require ground-truth labels.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first training run may download pretrained EfficientNet-B0 weights through `timm`.

## Training

First check class balance:

```bash
python src/check_data.py
```

Run:

```bash
python src/train.py
```

Useful CPU-friendly options:

```bash
python src/train.py --batch-size 8 --epochs 8 --patience 3 --cpu-threads 2 --num-workers 0 --balance-strategy sampler --unfreeze-last-blocks 1
```

Training details:

- Model: EfficientNet-B0 pretrained on ImageNet.
- Frozen layers: most backbone feature parameters.
- Trainable layers: classifier head and, by default, the last EfficientNet block.
- To train only the classifier head, use `--unfreeze-last-blocks 0`.
- Classifier: `Dropout(0.3)` plus `Linear(in_features, 1)`.
- Loss: `BCEWithLogitsLoss`.
- Optimizer: Adam with learning rate `1e-4`.
- Output: one logit where fake is the positive class.
- Early stopping: validation loss.
- Threshold: selected on validation data using balanced accuracy.
- Best checkpoint: `models/best_model.pth`.

## Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluator uses the validation-tuned threshold saved in `models/best_model.pth`. You can override it:

```bash
python src/evaluate.py --threshold 0.5
```

Outputs:

```text
models/evaluation/classification_report.txt
models/evaluation/confusion_matrix.png
```

The report includes precision, recall, and F1-score for `Real` and `Fake`.

## Inference From Python

```python
from PIL import Image
from src.inference import load_detector, predict_image

model, device = load_detector("models/best_model.pth")
image = Image.open("sample.jpg").convert("RGB")
result = predict_image(image, model, device)
print(result)
```

Example output:

```python
{
    "prediction": "Fake",
    "confidence": 0.9321,
    "fake_probability": 0.9321
}
```

## Streamlit App

Run:

```bash
streamlit run app.py
```

The app supports:

- Image upload.
- Webcam capture with `st.camera_input`.
- Face cropping before inference.
- Prediction label: `Real` or `Fake`.
- Confidence score.
- Cached model loading for faster repeated predictions.

## CPU Optimization Notes

- The model trains only the final classifier head.
- `num_workers` defaults to `0`, which is stable on Windows and low-resource machines.
- Batch size defaults to `8`.
- `torch.no_grad()` is used during validation, evaluation, and inference.
- The Streamlit model is cached with `st.cache_resource`.
- Inference processes one image at a time.
- Face detection falls back to the full image if no face is detected.

## Common Issues

If evaluation fails with a message about real/fake folders, make sure `dataset/test/` contains labeled class folders:

```text
dataset/test/real/
dataset/test/fake/
```

If training fails while downloading pretrained weights, check your internet connection or pre-download/cache the EfficientNet-B0 weights used by `timm`.

If OpenCV cannot load the Haar cascade, reinstall OpenCV:

```bash
pip install --upgrade opencv-python
```
