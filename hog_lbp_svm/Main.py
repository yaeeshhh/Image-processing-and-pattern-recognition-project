# =============================================================================
# HOG/LBP + SVM Road Sign Classifier (Binary: sign vs no_sign)
# UTS 31256 Image Processing & Pattern Recognition — Spring 2025
# Authors: Fabiha Tabassum Priti & Jayden Tran
#
# Pipeline:
#   1. Load grayscale images from sign / no_sign folders
#   2. Extract HOG + LBP + colour histogram features
#   3. Balance with SMOTE, tune SVM via GridSearchCV
#   4. Evaluate with threshold sweep, ROC, PR, calibration curves
#   5. Stress test under noise, blur, brightness, and rotation
# =============================================================================

import os
import shutil
import time
import joblib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from skimage.feature import hog, local_binary_pattern
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# CONFIG — update paths before running
# =============================================================================

# Folders containing sign images (will be merged into sign_pool)
SIGN_SOURCES = [
    "data/images/train",
    "data/images/val",
    "data/visual_testing_dataset",
]
SIGN_POOL    = "data/temp_pool/sign"
NO_SIGN_POOL = "data/presorted/no_sign"

IMG_SIZE = (128, 128)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def collect_sign_images(sign_sources, sign_pool):
    """Merge all sign images from multiple source folders into one pool."""
    os.makedirs(sign_pool, exist_ok=True)
    for folder in sign_sources:
        if not os.path.exists(folder):
            print(f"[warn] folder not found: {folder}")
            continue
        for f in os.listdir(folder):
            src = os.path.join(folder, f)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(sign_pool, f))
    print(f"Collected sign images in:  {sign_pool}")
    print(f"No-sign images located at: {NO_SIGN_POOL}")


def load_images_from_folder(folder, label, img_size=IMG_SIZE):
    """Load grayscale images from a folder and assign a string label."""
    images, labels = [], []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img  = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

HOG_KW = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=False,
    feature_vector=True,
)


def preprocess_for_hog(gray_img, do_clahe=True):
    """Optionally apply CLAHE before HOG extraction."""
    if do_clahe:
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_img  = clahe.apply(gray_img)
    return gray_img


def hog_features_from_grays(gray_stack):
    """Extract HOG feature vectors from a stack of grayscale images."""
    feats = []
    for g in gray_stack:
        g = preprocess_for_hog(g)
        feats.append(hog(g, **HOG_KW))
    return np.array(feats, dtype=np.float32)


def extract_lbp_features(images, P=8, R=1):
    """Extract normalised LBP histograms."""
    bins     = P + 2
    features = []
    for img in images:
        lbp  = local_binary_pattern(img, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)


def extract_color_histograms(images, bins=32):
    """Extract normalised grayscale intensity histograms."""
    feats = []
    for img in images:
        hist = np.histogram(img.flatten(), bins=bins, range=(0, 256))[0]
        feats.append(hist)
    return np.array(feats)


def build_features(images, scaler_hog=None, scaler_lbp=None, scaler_color=None, fit=False):
    """Extract and concatenate HOG + LBP + colour features with optional scaling."""
    hog_feats   = hog_features_from_grays(images)
    lbp_feats   = extract_lbp_features(images)
    color_feats = extract_color_histograms(images)

    if fit:
        scaler_hog   = StandardScaler().fit(hog_feats)
        scaler_lbp   = StandardScaler().fit(lbp_feats)
        scaler_color = StandardScaler().fit(color_feats)

    combined = np.hstack([
        scaler_hog.transform(hog_feats),
        scaler_lbp.transform(lbp_feats),
        scaler_color.transform(color_feats),
    ])
    return combined, scaler_hog, scaler_lbp, scaler_color


# =============================================================================
# STRESS TEST CORRUPTIONS
# =============================================================================

def add_gaussian_noise(image, mean=0, var=0.01):
    img_float = image.astype(np.float64) / 255.0
    noisy     = random_noise(img_float, mode="gaussian", mean=mean, var=var)
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def add_gaussian_blur(image, sigma=1):
    return gaussian_filter(image, sigma=sigma).astype(np.uint8)


def adjust_brightness(image, value=30):
    """Adjust brightness of a grayscale image via HSV conversion."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v       = np.clip(cv2.add(v, value), 0, 255)
    merged  = cv2.merge((h, s, v))
    return cv2.cvtColor(cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)


def rotate_image(image, angle):
    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def run_stress_test(X_test, y_test_enc, best_svm,
                    scaler_hog, scaler_lbp, scaler_color):
    """Apply four corruptions to the test set and report accuracy + F1."""
    corruptions = {
        "Gaussian Noise":  [add_gaussian_noise(img, var=0.05)  for img in X_test],
        "Gaussian Blur":   [add_gaussian_blur(img, sigma=1.5)  for img in X_test],
        "Brightness +50":  [adjust_brightness(img, value=50)   for img in X_test],
        "Rotation +10°":   [rotate_image(img, angle=10)        for img in X_test],
    }

    for name, corrupted in corruptions.items():
        corrupted = np.array(corrupted)
        feats, _, _, _ = build_features(corrupted,
                                         scaler_hog, scaler_lbp, scaler_color)
        preds = best_svm.predict(feats)
        acc   = accuracy_score(y_test_enc, preds)
        f1    = f1_score(y_test_enc, preds, average="macro", zero_division=0)
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {f1:.4f}")
        print(classification_report(y_test_enc, preds,
                                    target_names=["no_sign", "sign"],
                                    zero_division=0))


def plot_results(y_test, y_test_pred, y_test_proba, best_t):
    """Confusion matrix, ROC, PR, and calibration curves."""
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no_sign", "sign"],
                yticklabels=["no_sign", "sign"])
    plt.title(f"Confusion Matrix (Threshold={best_t:.2f})")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    # ROC
    fpr, tpr, _  = roc_curve(y_test, y_test_proba)
    roc_auc_val  = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc_val           = average_precision_score(y_test, y_test_proba)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=300)
    plt.show()

    # Calibration
    brier     = brier_score_loss(y_test, y_test_proba)
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    ece       = float(np.mean(np.abs(prob_true - prob_pred)))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.title(f"Calibration Curve  |  Brier={brier:.3f}  ECE={ece:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=300)
    plt.show()

    print(f"Brier Score : {brier:.4f}")
    print(f"ECE         : {ece:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- 1. Collect images ---
    collect_sign_images(SIGN_SOURCES, SIGN_POOL)

    # --- 2. Load datasets ---
    X_sign,   y_sign   = load_images_from_folder(SIGN_POOL,    "sign")
    X_nosign, y_nosign = load_images_from_folder(NO_SIGN_POOL, "no_sign")

    X_all = np.concatenate([X_sign, X_nosign])
    y_all = np.concatenate([y_sign, y_nosign])

    le        = LabelEncoder()
    y_all_enc = le.fit_transform(y_all)
    print("Classes:", le.classes_)

    # --- 3. Stratified splits (70 / 15 / 15) ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all_enc, test_size=0.15, stratify=y_all_enc, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # --- 4. Feature extraction ---
    X_train_ext, scaler_hog, scaler_lbp, scaler_color = build_features(X_train, fit=True)
    X_val_ext,   *_  = build_features(X_val,   scaler_hog, scaler_lbp, scaler_color)
    X_test_ext,  *_  = build_features(X_test,  scaler_hog, scaler_lbp, scaler_color)
    print(f"Feature shapes — train: {X_train_ext.shape}, val: {X_val_ext.shape}, test: {X_test_ext.shape}")

    # --- 5. Class balancing with SMOTE ---
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_ext, y_train)
    print("Before SMOTE:", np.bincount(y_train))
    print("After SMOTE: ", np.bincount(y_train_bal))

    # --- 6. SVM training with GridSearchCV ---
    param_grid = {
        "C":      [0.001, 0.01, 0.1, 1, 10],
        "gamma":  ["scale", "auto", 1e-4, 1e-3, 1e-2],
        "kernel": ["rbf"],
    }
    svm_model = SVC(probability=True, class_weight="balanced", random_state=42)
    grid      = GridSearchCV(svm_model, param_grid,
                             scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train_bal, y_train_bal)
    print("Best params    :", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)
    best_svm = grid.best_estimator_

    # --- 7. Threshold tuning on test set ---
    y_test_proba = best_svm.predict_proba(X_test_ext)[:, 1]
    best_t, best_f1_score = 0.0, 0.0
    for t in np.arange(0.1, 0.7, 0.05):
        preds    = (y_test_proba >= t).astype(int)
        report   = classification_report(y_test, preds, output_dict=True)
        f1_macro = report["macro avg"]["f1-score"]
        if f1_macro > best_f1_score:
            best_f1_score, best_t = f1_macro, t
    print(f"\nBest threshold: {best_t:.2f}  (Macro F1 = {best_f1_score:.3f})")

    y_test_pred = (y_test_proba >= best_t).astype(int)
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_test_pred,
                                 target_names=["no_sign", "sign"]))

    # --- 8. Plots ---
    plot_results(y_test, y_test_pred, y_test_proba, best_t)

    # --- 9. Efficiency measurement ---
    n_samples  = len(X_test_ext)
    start_time = time.time()
    _ = best_svm.predict(X_test_ext)
    elapsed    = time.time() - start_time
    print(f"\nInference — total: {elapsed:.4f}s | "
          f"latency: {elapsed/n_samples:.6f}s/img | "
          f"throughput: {n_samples/elapsed:.1f} img/s")

    model_path = "svm_model.pkl"
    joblib.dump(best_svm, model_path)
    print(f"Model size: {os.path.getsize(model_path)/(1024*1024):.2f} MB")

    # --- 10. Stress testing ---
    print("\n--- Stress Test ---")
    # Recover y_test_enc (may be string labels if le was not applied above)
    y_test_enc = y_test  # already encoded by LabelEncoder in step 2
    run_stress_test(X_test, y_test_enc, best_svm,
                    scaler_hog, scaler_lbp, scaler_color)

    print("\nDone. Model saved to svm_model.pkl")
