# 🍎🍅 Apple vs Tomato Image Classification (Custom CNN)

**Repo:** [github.com/khan-emran/Apple-vs-Tomato-Image-Classification-Custom-CNN](https://github.com/khan-emran/Apple-vs-Tomato-Image-Classification-Custom-CNN)

A binary image classification project that distinguishes **apples** from **tomatoes** using a custom Convolutional Neural Network built from scratch with TensorFlow/Keras — no transfer learning.

---

## 📌 Project Overview

Apples and tomatoes are visually similar in shape and color, making this a fun, non-trivial classification problem. This project trains a custom CNN on Google Colab (GPU-accelerated) to classify an input image as either **apple** or **tomato**, and includes single-image inference on unseen samples.

---

## 📂 Dataset

- **Source:** [Apples or Tomatoes - Image Classification (Kaggle)](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)
- **Classes:** `apples`, `tomatoes`
- **Split used:**
  - Train: 294 images
  - Test/Validation: 97 images
- **Input size:** resized to 256×256×3 (RGB)

> ⚠️ The dataset is **not included** in this repository due to size/licensing. Download it from Kaggle and place it under `data/` as shown in the structure below.

```bash
# Using Kaggle API
kaggle datasets download -d samuelcortinhas/apples-or-tomatoes-image-classification
unzip apples-or-tomatoes-image-classification.zip -d data/
```

Expected folder layout after extraction:
```
data/
├── train/
│   ├── apples/
│   └── tomatoes/
└── test/
    ├── apples/
    └── tomatoes/
```

---

## 🏗️ Model Architecture

A custom Sequential CNN with 3 convolutional blocks (Conv2D → BatchNorm → MaxPooling) followed by a fully-connected classifier head:

| # | Layer | Output Shape | Params |
|---|-------|--------------|--------|
| 1 | Conv2D (32 filters, 3×3, ReLU) | (254, 254, 32) | 896 |
| 2 | BatchNormalization | (254, 254, 32) | 128 |
| 3 | MaxPooling2D (2×2, stride 2) | (127, 127, 32) | 0 |
| 4 | Conv2D (64 filters, 3×3, ReLU) | (125, 125, 64) | 18,496 |
| 5 | BatchNormalization | (125, 125, 64) | 256 |
| 6 | MaxPooling2D (2×2, stride 2) | (62, 62, 64) | 0 |
| 7 | Conv2D (128 filters, 3×3, ReLU) | (60, 60, 128) | 73,856 |
| 8 | BatchNormalization | (60, 60, 128) | 512 |
| 9 | MaxPooling2D (2×2, stride 2) | (30, 30, 128) | 0 |
| 10 | Flatten | (115,200) | 0 |
| 11 | Dense (128, ReLU) | (128) | 14,745,728 |
| 12 | Dropout (0.1) | (128) | 0 |
| 13 | Dense (64, ReLU) | (64) | 8,256 |
| 14 | Dropout (0.1) | (64) | 0 |
| 15 | Dense (1, Sigmoid) | (1) | 65 |

**Total params:** 14,848,193 (56.64 MB) — 14,847,745 trainable / 448 non-trainable

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy
- **Epochs:** 30 · **Batch size:** 32 · **Image size:** 256×256

---

## ⚙️ Setup & Usage

This project was developed and run in **Google Colab** with GPU acceleration (Tesla T4).

### Option A — Run in Google Colab (as built)
1. Upload `AppleTomatoImageClassificationCustom_CNN.ipynb` to Colab.
2. Mount Google Drive and place `archive.zip` (the Kaggle dataset) in your Drive.
3. Run all cells top to bottom — the notebook mounts Drive, unzips the dataset, builds/trains the CNN, and runs inference on a sample image.

### Option B — Run locally
```bash
git clone https://github.com/khan-emran/Apple-vs-Tomato-Image-Classification-Custom-CNN.git
cd Apple-vs-Tomato-Image-Classification-Custom-CNN

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook AppleTomatoImageClassificationCustom_CNN.ipynb
```
Update the dataset paths (currently pointed at `/content/data/...` for Colab) to your local `data/` folder before running locally.

### Predicting on a new image
The notebook loads an image with OpenCV, resizes it to 256×256, and passes it through the trained model:
```python
test_img = cv2.imread("data/train/apples/img_p3_121.jpeg")
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
result = model.predict(test_input)
print("Apple" if int(result[0][0]) == 0 else "Tomato")
```

---

## 📊 Results

Final epoch (30/30):

| Metric | Value |
|--------|-------|
| Train Accuracy | 96.6% |
| Train Loss | 0.306 |
| Validation Accuracy | 71.1% |
| Validation Loss | 4.99 |

**Observation:** Training accuracy climbs steadily to ~96–97% while validation accuracy plateaus around 55–71% and validation loss increases after early epochs — a clear sign of **overfitting**, expected given the small dataset (only 294 training images) relative to the model's ~14.8M parameters.

### Ways to reduce overfitting (noted in the notebook, not yet applied)
- Add more training data
- Data augmentation (flip, rotate, zoom, brightness)
- Increase Dropout rate
- Reduce model complexity / parameter count
- Early stopping on validation loss

---

## 🧪 Tech Stack

- Python 3
- TensorFlow / Keras
- OpenCV (`cv2`) — image loading/preprocessing for inference
- Matplotlib — accuracy/loss curve plots
- Google Colab (GPU: Tesla T4)

---

## 📁 Repository Structure

Current (matches this notebook-based implementation):

```
Apple-vs-Tomato-Image-Classification-Custom-CNN/
│
├── AppleTomatoImageClassificationCustom_CNN.ipynb   # main notebook: data loading, model, training, inference
├── data/
│   ├── train/
│   │   ├── apples/
│   │   └── tomatoes/
│   └── test/
│       ├── apples/
│       └── tomatoes/
├── results/
│   ├── accuracy_curve.png
│   └── loss_curve.png
├── models/
│   └── best_model.h5          # saved trained weights (if exported)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Suggested future structure (if refactored into modular scripts)
```
Apple-vs-Tomato-Image-Classification-Custom-CNN/
│
├── data/
│   ├── train/{apples,tomatoes}/
│   └── test/{apples,tomatoes}/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── best_model.h5
├── results/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔮 Future Improvements

- [ ] Apply data augmentation to reduce the train/validation accuracy gap
- [ ] Add early stopping / model checkpointing
- [ ] Refactor notebook into modular `src/` scripts (see structure above)
- [ ] Compare against transfer learning (MobileNetV2, EfficientNet)
- [ ] Add a confusion matrix and classification report on the test set
- [ ] Deploy as a simple Streamlit/Flask demo app

---

## 🤝 Contributing

Contributions and suggestions are welcome — feel free to open an issue or pull request.

---

## 📄 License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dataset by [Samuel Cortinhas on Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)
