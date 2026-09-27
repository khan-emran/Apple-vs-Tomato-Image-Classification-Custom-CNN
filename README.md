# 🍎🍅 Apple vs Tomato Image Classification (Custom CNN)

A deep learning project that classifies images as either **apple** or **tomato** using a custom-built Convolutional Neural Network (CNN) implemented from scratch (no transfer learning).

---

## 📌 Project Overview

Apples and tomatoes often look visually similar in shape and color, making them a fun and non-trivial binary image classification challenge. This project builds, trains, and evaluates a custom CNN architecture to distinguish between the two classes.

**Goal:** Given an input image, predict whether it contains an apple or a tomato.

---

## 📂 Dataset

- **Source:** [Apples or Tomatoes - Image Classification (Kaggle)](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)
- **Classes:** `apple`, `tomato`
- **Format:** RGB images, organized into `train/` and `test/` folders by class

> ⚠️ The dataset is **not included** in this repository due to size and licensing. Download it from Kaggle and place it in the `data/` folder as shown in the structure below.

### Download Instructions
```bash
# Using Kaggle API
kaggle datasets download -d samuelcortinhas/apples-or-tomatoes-image-classification
unzip apples-or-tomatoes-image-classification.zip -d data/
```

---

## 🏗️ Model Architecture

A custom CNN built with the following general design (update to match your final architecture):

| Layer | Type | Details |
|-------|------|---------|
| 1 | Conv2D + ReLU | 32 filters, 3x3 kernel |
| 2 | MaxPooling2D | 2x2 |
| 3 | Conv2D + ReLU | 64 filters, 3x3 kernel |
| 4 | MaxPooling2D | 2x2 |
| 5 | Conv2D + ReLU | 128 filters, 3x3 kernel |
| 6 | MaxPooling2D | 2x2 |
| 7 | Flatten | — |
| 8 | Dense + ReLU | 128 units |
| 9 | Dropout | 0.5 |
| 10 | Dense + Sigmoid | 1 unit (binary output) |

- **Loss function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall, F1-score

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/apple-tomato-classification.git
cd apple-tomato-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model
```bash
python src/train.py --epochs 30 --batch_size 32 --img_size 128
```

### 2. Evaluate the model
```bash
python src/evaluate.py --model_path models/best_model.h5
```

### 3. Predict on a single image
```bash
python src/predict.py --image_path samples/test_apple.jpg --model_path models/best_model.h5
```

### 4. Run the notebook (exploratory)
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Train Accuracy | XX% |
| Validation Accuracy | XX% |
| Test Accuracy | XX% |
| F1-score | XX |

Include a confusion matrix and sample predictions here:

```
results/
├── confusion_matrix.png
├── accuracy_loss_curves.png
└── sample_predictions.png
```

---

## 🧪 Tech Stack

- Python 3.x
- TensorFlow / Keras (or PyTorch — update accordingly)
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV / Pillow (image preprocessing)

---

## 📁 Repository Structure

See [Project Structure](#-standard-project-file-structure) below.

---

## 🔮 Future Improvements

- Add data augmentation (rotation, flip, zoom, color jitter)
- Experiment with transfer learning (MobileNet, EfficientNet) for comparison
- Deploy as a web app (Streamlit/Flask) for live predictions
- Add Grad-CAM visualizations for model interpretability

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Samuel Cortinhas on Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)
