# Usage Guide

This walkthrough mirrors the numbered prompts in the project description.
Each command prints a short message so you can verify the stage works.

## P2 – Frame Extraction
```bash
python extract_frames.py demo.mp4 --fps 30
```
Expected output:
```
Extracted N frames of shape HxW
```

## P3 – Video Stabilisation
```bash
python stabilise_frames.py demo.mp4 --fps 30 --plot
```
Expected output:
```
Stabilised N frames
Saved comparison to stabilise_demo.png
```
Check the PNG for reduced motion.

## P4 – ROI Cropping
```bash
python crop_roi.py
```
Expected output shows original and cropped shapes.

## P5 – Laplacian Pyramid
```bash
python laplacian_utils.py
```
```
PSNR: 128.xx dB
```
A PSNR greater than 40 dB confirms correct reconstruction.

## P6 – Temporal Filtering
```bash
python temporal_filter.py
```
Look for the filtered peak frequency equal to the target.

## P7 – Motion Magnification
```bash
python magnify_motion.py input.mp4 --fps 30 --flow 0.5 --fhigh 3 --alpha 10
```
```
Wrote input_magnified.mp4 with N frames
```

## P8 – Energy Features
```bash
python feat_energy.py
```
Prints a 3‑element energy vector.

## P9 – Optical Flow Histogram
```bash
python feat_flow.py
```
Outputs an 8 or 16 element histogram.

## P10 – FFT Sidebands
```bash
python feat_fft.py
```
Returns amplitudes at the fundamental and second harmonic.

## P11 – Building Feature Vectors
```bash
python build_features.py
```
Shows the raw feature matrix shape and PCA‑reduced shape.

## P12 – Training the Classifier
```bash
python train_classifier.py
```
Displays accuracy, precision, recall and F1.

## P13 – Full Inference
```bash
python infer.py abnormal.mp4 svm_model.joblib --fps 30 --f1 25.0
```
Example output:
```
Prediction: Abnormal (prob=0.87)
```

### Real-Time Demo
To process a live webcam feed, record a short clip using `ffmpeg` or another tool
and pass the saved MP4 to `infer.py`. Streaming directly from the camera requires
modifying the script to read from `cv2.VideoCapture(0)`.
