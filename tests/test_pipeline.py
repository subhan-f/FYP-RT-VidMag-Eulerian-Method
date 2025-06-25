
import os
import numpy as np
import cv2

from extract_frames import extract_frames
from stabilise_frames import stabilise_frames
from crop_roi import crop_roi
from laplacian_utils import build_laplacian_pyramid, reconstruct_from_laplacian
from temporal_filter import temporal_bandpass
from magnify_motion import magnify_motion
from feat_energy import energy_features
from feat_flow import flow_hist_features
from feat_fft import fft_sideband_features
from build_features import build_feature_vector, fit_pca, transform_pca
from train_classifier import train_classifier
from infer import infer


def create_dummy_video(path, n_frames=5, size=(64, 64), shift=0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30, size)
    for i in range(n_frames):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(img, (10 + i * shift, 20), (30 + i * shift, 40), (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def test_extract_and_crop(tmp_path):
    vid = tmp_path / 'dummy.mp4'
    create_dummy_video(str(vid))
    frames = extract_frames(str(vid), fps=30)
    assert len(frames) == 5
    h, w = frames[0].shape[:2]
    assert (h, w) == (64, 64)
    cropped = crop_roi(frames)
    ch, cw = cropped[0].shape[:2]
    assert ch <= h and cw <= w


def test_stabilise_frames(tmp_path):
    vid = tmp_path / 'shift.mp4'
    create_dummy_video(str(vid), shift=2)
    frames = extract_frames(str(vid))
    stabilised = stabilise_frames(frames)
    diff_before = np.mean(np.abs(np.array(frames[0], dtype=np.float32) - np.array(frames[-1], dtype=np.float32)))
    diff_after = np.mean(np.abs(np.array(stabilised[0], dtype=np.float32) - np.array(stabilised[-1], dtype=np.float32)))
    assert diff_after <= diff_before


def test_laplacian_utils():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    pyr = build_laplacian_pyramid(img, levels=3)
    recon = reconstruct_from_laplacian(pyr)
    mse = np.mean((img.astype(np.float32) - recon.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / (mse + 1e-8))
    assert psnr > 40


def test_temporal_bandpass():
    fs = 30
    t = np.arange(0, 2, 1/fs)
    freq = 5.0
    signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    stack = signal[:, None, None].astype(np.float32)
    filtered = temporal_bandpass(stack, fs, freq-1, freq+1)
    amp_before = np.abs(np.fft.rfft(signal))[int(freq)]
    amp_after = np.abs(np.fft.rfft(filtered.squeeze()))[int(freq)]
    assert amp_after >= 0.9 * amp_before


def test_magnify_motion(tmp_path):
    vid = tmp_path / 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(vid), fourcc, 30, (64, 64))
    for i in range(30):
        val = 128 + int(5 * np.sin(2*np.pi*i/10))
        frame = np.full((64, 64, 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    frames = extract_frames(str(vid))
    magnified = magnify_motion(frames, 30, 2, 4, alpha=5, levels=3)
    assert len(magnified) == len(frames)


def test_feature_extractors():
    frames = [(np.random.rand(64,64,3)*255).astype(np.uint8) for _ in range(5)]
    energy = energy_features(frames)
    flow = flow_hist_features(frames, bins=8)
    fft = fft_sideband_features(frames, 30, 5.0)
    feat = build_feature_vector(frames, 30, 5.0)
    assert energy.shape == (3,)
    assert flow.shape == (8,)
    assert fft.shape == (3,)
    assert feat.shape[0] == 22


def test_pca_and_classifier(tmp_path):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=50, n_features=22, random_state=0)
    pca = fit_pca(X, var_ratio=0.9)
    X_pca = transform_pca(pca, X)
    model_path = tmp_path / 'model.joblib'
    train_classifier(X_pca, y, str(model_path))
    assert model_path.exists()


def test_infer(tmp_path):
    vid = tmp_path / 'infer.mp4'
    create_dummy_video(str(vid), n_frames=30)
    frames = extract_frames(str(vid))
    X = np.vstack([build_feature_vector(frames, 30, 5.0) for _ in range(10)])
    y = np.array([0]*5 + [1]*5)
    model_path = tmp_path / 'infer_model.joblib'
    train_classifier(X, y, str(model_path))
    label = infer(str(vid), str(model_path), fs=30, f1=5.0)
    assert label in {'Normal', 'Abnormal'}
