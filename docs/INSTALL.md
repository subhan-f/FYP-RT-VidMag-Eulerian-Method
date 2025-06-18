# Installation

## Requirements

- Python 3.9+
- `opencv-python-headless`
- `numpy`
- `scipy`
- `scikit-image`
- `scikit-learn`
- `pyrtools`
- `tqdm`
- `joblib`

Install everything with:

```bash
pip install opencv-python-headless numpy scipy scikit-image scikit-learn pyrtools tqdm joblib
```

The code runs on CPU by default. GPU acceleration is not required, though OpenCV can use CUDA if available.

## Colab Quick-Start

```bash
!git clone https://github.com/yourname/motor-fault-classifier.git
%cd motor-fault-classifier
!pip install opencv-python-headless numpy scipy scikit-image scikit-learn pyrtools tqdm joblib
!python train_classifier.py    # run the demo
```

## Troubleshooting

- **`ImportError: cv2`** – ensure `opencv-python-headless` is installed.
- **No video output** – verify that your MP4 uses a codec supported by OpenCV.
- **Model training slow** – reduce `GridSearchCV` parameters or use a machine with more cores.

Refer back to the [Usage guide](USAGE.md) for CLI examples.
