# Eulerian-Magnification Motor-Fault Classifier

```
      ____
 ___/    \___
|            |
|   MOTOR    |
|____________|
```

A modular pipeline that amplifies subtle vibrations of an induction motor from
regular video and classifies them as **Normal** or **Abnormal**. It relies on
Eulerian Video Magnification (EVM) and classical machine learning features.

## Overview

1. **Frame Extraction** – load a video and resample frames.
2. **Stabilisation** – remove camera shake using ORB + RANSAC.
3. **ROI Cropping** – detect the red sticker on the motor casing.
4. **Motion Magnification** – Laplacian pyramid + temporal band-pass.
5. **Feature Extraction** – energy stats, optical-flow histograms, FFT sidebands.
6. **Classification** – PCA then SVM-RBF.

### Architecture

```
Video -> Extract -> Stabilise -> Crop -> Magnify -> Features -> PCA -> SVM -> Label
```

### Pipeline Flowchart

```
 +-------------+      +----------------+      +-------------+      +-------------+
 | Video Input | -->  | Frame Extract  | -->  | Stabilise   | -->  | Crop ROI    |
 +-------------+      +----------------+      +-------------+      +-------------+
                                                              |
                                                              v
                     +--------------+      +----------------------+      +-------------------+
                     | Laplacian Py | -->  | Temporal Band-pass   | -->  | Motion Magnified  |
                     +--------------+      +----------------------+      +-------------------+
                                                              |
                                                              v
                        +--------------+      +---------+
                        | Features f(x)| -->  |  SVM-RBF|
                        +--------------+      +---------+
```

### Citations

1. Wu, H., Rubinstein, M., Shih, E., Guttag, J., Durand, F., & Freeman, W. (2012).
   *Eulerian Video Magnification for Revealing Subtle Changes in the World*.
   **ACM TOG**.
2. Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W. (2013).
   *Phase-Based Video Motion Processing*. **ACM TOG**.
3. Patel, A. & Yilmaz, A. (2015). *Video-based vibration measurement of rotating
   machinery*. **Mechanical Systems and Signal Processing**.
4. Chen, S. et al. (2019). *High-speed video magnification for vibration
   monitoring*. **IEEE T-IM**.
5. Yang, J. & Zhen, L. (2021). *Machine fault diagnosis using optical flow and
   video magnification*. **Sensors**.

See also the [Installation guide](INSTALL.md) and the detailed [API](API_REFERENCE.md).
