# DeepGuard AI — AI-Generated Image Detection

A CNN-based classifier that distinguishes **real** images from **AI-generated** ones,
with a Streamlit web app that shows the prediction and its confidence score.

**Tech stack:** Python, Keras (TensorFlow), CNN, OpenCV, Scikit-learn, Streamlit

## Project structure

```
deepguard-ai/
├── app.py            # Streamlit web app
├── config.py         # Paths and hyperparameters
├── preprocess.py     # OpenCV preprocessing + data generator
├── model.py          # CNN architecture
├── train.py          # Training script
├── evaluate.py       # Accuracy / F1 / confusion matrix on the test set
├── predict.py        # Command-line prediction
├── requirements.txt
├── data/             # Put the dataset here (see below)
├── models/           # Trained model is saved here
└── outputs/          # Training curves, metrics, confusion matrix
```

## 1. Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate    |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset

Any real-vs-AI image dataset works as long as it follows this layout:

```
data/
├── train/
│   ├── FAKE/   # AI-generated images
│   └── REAL/   # real photos
└── test/
    ├── FAKE/
    └── REAL/
```

A good public option is **CIFAKE** on Kaggle
(`birdy654/cifake-real-and-ai-generated-synthetic-images`), which already uses
`train/FAKE`, `train/REAL`, `test/FAKE`, `test/REAL` folders:

```bash
pip install kaggle
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data
```

A validation set (15%) is split automatically from `data/train`.

## 3. Preprocessing

Handled in `preprocess.py`:
1. Read with OpenCV and convert BGR → RGB
2. Resize to **256 × 256**
3. Normalize pixel values to **[0, 1]**
4. Stack into float32 tensors of shape `(batch, 256, 256, 3)` for the CNN

## 4. Train

```bash
python train.py                 # defaults in config.py
python train.py --epochs 30 --batch-size 64 --lr 0.0005
```

The best model is saved to `models/deepguard_cnn.keras`; training curves go to `outputs/`.

> Training on 256×256 images is heavy on CPU. A GPU (or Google Colab) is recommended.
> On a CPU, you can lower `IMG_SIZE` in `config.py` for quick experiments.

## 5. Evaluate

```bash
python evaluate.py
```

Prints accuracy, macro F1-score, precision, recall, ROC-AUC and a classification report,
and saves `outputs/confusion_matrix.png` and `outputs/test_metrics.json`.

## 6. Predict from the command line

```bash
python predict.py some_image.jpg another.png
```

## 7. Run the web app

```bash
streamlit run app.py
```

Upload an image and the app shows **Real** or **AI-Generated** with a confidence score.

## Model architecture

4 convolutional blocks (32 → 64 → 128 → 256 filters), each with two 3×3 Conv2D +
BatchNorm + ReLU layers, MaxPooling and Dropout, followed by GlobalAveragePooling,
a 256-unit Dense layer, Dropout(0.5) and a sigmoid output giving P(real).
Trained with Adam + binary cross-entropy, with early stopping and learning-rate reduction.
