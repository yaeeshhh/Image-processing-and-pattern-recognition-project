# =============================================================================
# Hybrid CNN-SVM Road Sign Classifier
# UTS 31256 Image Processing & Pattern Recognition — Spring 2025
# Authors: Yaeesh Mahomed & Qian Zhao
#
# Architecture: ResNet50 (feature extractor) + SVM (classifier)
# Dataset: Kaggle Traffic Sign Detection + Roboflow Self-Driving Cars
# =============================================================================

import os
import cv2
import time
import joblib
import warnings
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, top_k_accuracy_score, brier_score_loss
)
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import rotate as scipy_rotate

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIG — update these paths before running
# =============================================================================
FINETUNE_DIR_MAIN = "path/to/Traffic_sign_detection_data"   # Kaggle dataset root
SELFDRIVING_ZIP   = "path/to/Self-Driving_Cars.zip"         # Roboflow zip

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
L2_REG     = 0.01


# =============================================================================
# UTILITIES
# =============================================================================

def extract_zip(zip_path, extract_to):
    import shutil
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def resize_with_padding(img, target_size=(224, 224)):
    """Resize image while preserving aspect ratio, pad with zeros."""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded


# =============================================================================
# DATA LOADING
# =============================================================================

def make_samples_from_yolo(img_dir, lbl_dir):
    """Parse YOLO-format labels and return (img_path, x1, y1, x2, y2, class_id) tuples."""
    samples = []
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        with open(lbl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                class_id, x_c, y_c, w_box, h_box = map(float, line.strip().split())
                x1 = max(int((x_c - w_box / 2) * w_img), 0)
                y1 = max(int((y_c - h_box / 2) * h_img), 0)
                x2 = min(int((x_c + w_box / 2) * w_img), w_img)
                y2 = min(int((y_c + h_box / 2) * h_img), h_img)
                if x2 <= x1 or y2 <= y1:
                    continue
                samples.append((img_path, x1, y1, x2, y2, int(class_id)))
    return samples


def load_main_dataset_for_finetuning(main_dir):
    train_img_dir = os.path.join(main_dir, 'images', 'train')
    val_img_dir   = os.path.join(main_dir, 'images', 'val')
    train_lbl_dir = os.path.join(main_dir, 'labels', 'train')
    val_lbl_dir   = os.path.join(main_dir, 'labels', 'val')
    ft_train = make_samples_from_yolo(train_img_dir, train_lbl_dir)
    ft_val   = make_samples_from_yolo(val_img_dir,   val_lbl_dir)
    print(f"Fine-tuning — Train: {len(ft_train)}, Val: {len(ft_val)}")
    return ft_train, ft_val


def load_selfdriving_dataset_with_splits(base_dir, img_size=IMG_SIZE):
    """Load the Self-Driving Cars (Roboflow) dataset — train / valid / test splits."""
    def load_yolo_split(split_name):
        img_dir = os.path.join(base_dir, split_name, 'images')
        lbl_dir = os.path.join(base_dir, split_name, 'labels')
        if not os.path.exists(img_dir):
            img_dir = os.path.join(base_dir, 'images')
            lbl_dir = os.path.join(base_dir, 'labels')
        images, labels = [], []
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path   = os.path.join(img_dir, img_file)
            label_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')
            if not os.path.exists(label_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    class_id, x_c, y_c, w, h = map(float, line.strip().split())
                    x1 = max(int((x_c - w / 2) * w_img), 0)
                    y1 = max(int((y_c - h / 2) * h_img), 0)
                    x2 = min(int((x_c + w / 2) * w_img), w_img)
                    y2 = min(int((y_c + h / 2) * h_img), h_img)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = resize_with_padding(img[y1:y2, x1:x2], img_size)
                    images.append(crop)
                    labels.append(int(class_id))
        return np.array(images), np.array(labels)

    X_train, y_train = load_yolo_split('train')
    X_val,   y_val   = load_yolo_split('valid')
    X_test,  y_test  = load_yolo_split('test')
    print(f"Self-Driving Cars — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# KERAS SEQUENCE GENERATOR
# =============================================================================

class YoloCropGenerator(Sequence):
    def __init__(self, samples, num_classes, batch_size=32,
                 img_size=(224, 224), shuffle=True, augment_fn=None):
        self.samples     = samples
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.img_size    = img_size
        self.shuffle     = shuffle
        self.augment_fn  = augment_fn
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx     = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_idx]
        X = np.zeros((len(batch_samples), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y = np.zeros((len(batch_samples),), dtype=np.int32)
        for i, (img_path, x1, y1, x2, y2, class_id) in enumerate(batch_samples):
            img = cv2.imread(img_path)
            if img is None:
                crop = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            else:
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                crop = img[y1:y2, x1:x2]
                crop = resize_with_padding(crop, self.img_size)
            if self.augment_fn:
                crop = self.augment_fn(crop)
            X[i] = preprocess_input(crop.astype('float32'))
            y[i] = class_id
        y_cat = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        return X, y_cat


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_model(num_classes, img_size=IMG_SIZE, l2_reg=L2_REG):
    """ResNet50 backbone with custom classification head."""
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(img_size[0], img_size[1], 3))
    # Freeze all but last 5 layers for Stage 1
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    for layer in base_model.layers[-5:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg))(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model


def train_model(model, base_model, train_gen, val_gen):
    """Two-stage training: partial fine-tune → full fine-tune."""
    # Stage 1 — last 5 layers unfrozen
    print("[Stage 1] Training with last 5 layers unfrozen...")
    callbacks1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint('best_cnn_stage1.keras', monitor='val_loss', save_best_only=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=6, callbacks=callbacks1, verbose=1)

    # Stage 2 — full fine-tune with lower LR
    print("[Stage 2] Fine-tuning all layers with lower LR...")
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks2 = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8),
        ModelCheckpoint('best_cnn_stage2.keras', monitor='val_loss', save_best_only=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=6, callbacks=callbacks2, verbose=1)
    return model


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def build_feature_extractor(model):
    return Model(inputs=model.input, outputs=model.layers[-5].output)


def extract_features_batch(images, feature_extractor, batch_size=BATCH_SIZE):
    features = []
    for i in range(0, len(images), batch_size):
        batch = preprocess_input(images[i:i + batch_size].astype('float32'))
        features.append(feature_extractor.predict(batch, verbose=0))
    return np.vstack(features)


# =============================================================================
# TEST-TIME AUGMENTATION (TTA)
# =============================================================================

def apply_tta_transforms(image):
    """Generate 7 augmented views of a single image."""
    transforms = [
        image,
        np.fliplr(image),
        scipy_rotate(image, -5, axes=(0, 1), reshape=False, mode='nearest'),
        scipy_rotate(image,  5, axes=(0, 1), reshape=False, mode='nearest'),
        np.clip(image * 1.1, 0, 255).astype(np.uint8),  # brighter
        np.clip(image * 0.9, 0, 255).astype(np.uint8),  # darker
    ]
    return np.array(transforms)


def predict_with_tta(images, feature_extractor, scaler, svm, use_tta=True):
    """Return averaged class probabilities, optionally using TTA."""
    if not use_tta:
        features = extract_features_batch(images, feature_extractor)
        return svm.predict_proba(scaler.transform(features))
    all_probs = []
    for img in images:
        tta_imgs     = apply_tta_transforms(img)
        tta_features = extract_features_batch(tta_imgs, feature_extractor)
        tta_probs    = svm.predict_proba(scaler.transform(tta_features))
        all_probs.append(np.mean(tta_probs, axis=0))
    return np.array(all_probs)


# =============================================================================
# STRESS TESTING
# =============================================================================

def add_gaussian_noise(images, mean=0, std=10):
    return np.array([np.clip(img + np.random.normal(mean, std, img.shape).astype(np.uint8), 0, 255)
                     for img in images])

def apply_gaussian_blur(images, ksize=(3, 3)):
    return np.array([cv2.GaussianBlur(img, ksize, 0) for img in images])

def adjust_brightness(images, factor=1.2):
    return np.array([np.clip(img * factor, 0, 255).astype(np.uint8) for img in images])

def slight_rotation(images, angle=5):
    rotated = []
    for img in images:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.array(rotated)


CORRUPTION_TYPES = {
    "Gaussian Noise":  lambda x: add_gaussian_noise(x, std=15),
    "Blur":            lambda x: apply_gaussian_blur(x, ksize=(5, 5)),
    "Brightness Up":   lambda x: adjust_brightness(x, factor=1.3),
    "Brightness Down": lambda x: adjust_brightness(x, factor=0.7),
    "Rotation +5°":    lambda x: slight_rotation(x, angle=5),
    "Rotation -5°":    lambda x: slight_rotation(x, angle=-5),
}


def run_stress_tests(X_test, y_test_enc, feature_extractor, scaler, svm, baseline_acc):
    subset_idx    = np.random.choice(len(X_test), min(150, len(X_test)), replace=False)
    stress_images = X_test[subset_idx]
    stress_labels = y_test_enc[subset_idx]
    stress_metrics = {}
    for name, func in CORRUPTION_TYPES.items():
        print(f"\nEvaluating: {name}")
        corrupted = func(stress_images)
        probs     = predict_with_tta(corrupted, feature_extractor, scaler, svm, use_tta=True)
        preds     = np.argmax(probs, axis=1)
        acc = accuracy_score(stress_labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(stress_labels, preds,
                                                       average='macro', zero_division=0)
        stress_metrics[name] = {"accuracy": acc, "precision_macro": p,
                                 "recall_macro": r, "f1_macro": f1}
        print(f"  Accuracy: {acc:.4f}  F1 (macro): {f1:.4f}  Drop: {baseline_acc - acc:.4f}")
    return stress_metrics


# =============================================================================
# EVALUATION & VISUALISATION
# =============================================================================

def evaluate(y_test, y_pred, y_pred_proba, y_test_enc, y_pred_enc,
             label_encoder, feature_extractor, scaler, svm, X_test):
    metrics = {}
    metrics['overall_accuracy'] = accuracy_score(y_test, y_pred)
    metrics['top3_accuracy']    = top_k_accuracy_score(
        y_test_enc, y_pred_proba,
        k=min(3, len(label_encoder.classes_)),
        labels=np.arange(len(label_encoder.classes_))
    )

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, average='macro',  zero_division=0)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test, y_pred, average='micro',  zero_division=0)
    metrics.update({'precision_macro': p_mac, 'recall_macro': r_mac, 'f1_macro': f1_mac,
                    'precision_micro': p_mic, 'recall_micro': r_mic, 'f1_micro': f1_mic})

    present_classes = np.unique(y_test_enc)
    y_test_bin      = label_binarize(y_test_enc, classes=present_classes)

    try:
        metrics['roc_auc'] = roc_auc_score(y_test_bin,
                                            y_pred_proba[:, present_classes],
                                            average='macro', multi_class='ovr')
    except ValueError:
        metrics['roc_auc'] = np.nan

    metrics['pr_auc'] = np.mean([
        average_precision_score(y_test_bin[:, i], y_pred_proba[:, present_classes[i]])
        for i in range(len(present_classes))
    ])
    metrics['brier_score'] = np.mean([
        brier_score_loss(y_test_bin[:, i], y_pred_proba[:, present_classes[i]])
        for i in range(len(present_classes))
    ])

    # ECE
    confidences = np.max(y_pred_proba, axis=1)
    correct     = (np.argmax(y_pred_proba, axis=1) == y_test_enc)
    ece = sum(
        np.abs(np.mean(confidences[mask]) - np.mean(correct[mask])) * np.mean(mask)
        for i in range(10)
        for mask in [(confidences > i / 10) & (confidences <= (i + 1) / 10)]
        if np.sum(mask) > 0
    )
    metrics['ece'] = ece

    # Latency
    start = time.time()
    predict_with_tta(X_test[:20], feature_extractor, scaler, svm, use_tta=True)
    elapsed = time.time() - start
    metrics['latency_ms_with_tta']    = (elapsed / 20) * 1000
    metrics['throughput_fps_with_tta'] = 20 / elapsed

    start = time.time()
    _ = svm.predict(scaler.transform(extract_features_batch(X_test[:100], feature_extractor)))
    elapsed = time.time() - start
    metrics['latency_ms_no_tta']    = (elapsed / 100) * 1000
    metrics['throughput_fps_no_tta'] = 100 / elapsed

    print("\n" + "=" * 60)
    print("FINAL METRICS (with TTA)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    return metrics


def generate_plots(y_test, y_pred, y_pred_proba, label_encoder, y_test_enc):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    if len(label_encoder.classes_) > 20:
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (with TTA)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tta.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Confidence distribution
    correct_confs, incorrect_confs = [], []
    for i, (tl, pl) in enumerate(zip(y_test, y_pred)):
        conf = np.max(y_pred_proba[i])
        (correct_confs if tl == pl else incorrect_confs).append(conf)
    plt.figure(figsize=(8, 5))
    plt.hist([correct_confs, incorrect_confs], bins=30,
             label=['Correct', 'Incorrect'], alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution (Correct vs Incorrect)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- 1. Extract & load datasets ---
    selfdriving_extract = extract_zip(SELFDRIVING_ZIP, 'selfdriving_data')
    ft_train_samples, ft_val_samples = load_main_dataset_for_finetuning(FINETUNE_DIR_MAIN)

    # Remap labels to contiguous range
    all_classes = sorted({s[5] for s in (ft_train_samples + ft_val_samples)})
    label_map   = {c: i for i, c in enumerate(all_classes)}
    def remap(samples):
        return [(p, x1, y1, x2, y2, label_map[c]) for p, x1, y1, x2, y2, c in samples]
    ft_train_samples = remap(ft_train_samples)
    ft_val_samples   = remap(ft_val_samples)
    num_classes_ft   = len(all_classes)

    # --- 2. Build generators ---
    train_gen = YoloCropGenerator(ft_train_samples, num_classes_ft,
                                  batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True)
    val_gen   = YoloCropGenerator(ft_val_samples,   num_classes_ft,
                                  batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False)

    # --- 3. Build & train CNN ---
    model, base_model = build_model(num_classes_ft)
    model = train_model(model, base_model, train_gen, val_gen)
    feature_extractor = build_feature_extractor(model)

    # --- 4. Load Self-Driving Cars dataset ---
    selfdriving_path = os.path.join(
        selfdriving_extract,
        'Self-Driving_Cars.v6-version-4-prescan-416x416.yolov11'
    )
    X_train, X_val, X_test, y_train, y_val, y_test = \
        load_selfdriving_dataset_with_splits(selfdriving_path)

    # --- 5. Extract CNN features ---
    print("Extracting features...")
    X_train_feat = extract_features_batch(X_train, feature_extractor)
    X_val_feat   = extract_features_batch(X_val,   feature_extractor)
    X_test_feat  = extract_features_batch(X_test,  feature_extractor)

    # --- 6. Scale + encode ---
    label_encoder = LabelEncoder()
    y_train_enc   = label_encoder.fit_transform(y_train)
    y_val_enc     = label_encoder.transform(y_val)
    y_test_enc    = label_encoder.transform(y_test)

    scaler          = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train_feat)
    X_val_scaled    = scaler.transform(X_val_feat)
    X_test_scaled   = scaler.transform(X_test_feat)

    # --- 7. Train SVM ---
    print("Training SVM with GridSearch...")
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {'C': [1, 10], 'gamma': ['scale', 0.01]},
        cv=3, n_jobs=-1, verbose=2, scoring='f1_macro'
    )
    svm_grid.fit(X_train_scaled, y_train_enc)
    svm = svm_grid.best_estimator_
    print(f"Best SVM params: {svm_grid.best_params_}")

    # --- 8. Predict (with & without TTA) ---
    print("\nTest evaluation with TTA...")
    y_pred_proba_tta    = predict_with_tta(X_test, feature_extractor, scaler, svm, use_tta=True)
    y_pred_enc          = np.argmax(y_pred_proba_tta, axis=1)
    y_pred              = label_encoder.inverse_transform(y_pred_enc)

    y_pred_proba_no_tta = svm.predict_proba(X_test_scaled)
    y_pred_enc_no_tta   = np.argmax(y_pred_proba_no_tta, axis=1)

    print(f"Accuracy without TTA : {accuracy_score(y_test_enc, y_pred_enc_no_tta):.4f}")
    print(f"Accuracy with TTA    : {accuracy_score(y_test_enc, y_pred_enc):.4f}")

    # --- 9. Full evaluation ---
    baseline_acc = accuracy_score(y_test_enc, y_pred_enc)
    metrics = evaluate(y_test, y_pred, y_pred_proba_tta, y_test_enc, y_pred_enc,
                       label_encoder, feature_extractor, scaler, svm, X_test)

    # --- 10. Stress tests ---
    print("\n--- Stress Testing ---")
    run_stress_tests(X_test, y_test_enc, feature_extractor, scaler, svm, baseline_acc)

    # --- 11. Plots ---
    generate_plots(y_test, y_pred, y_pred_proba_tta, label_encoder, y_test_enc)

    # --- 12. Save artefacts ---
    print("\nSaving models...")
    model.save('final_cnn_model.keras')
    feature_extractor.save('feature_extractor.keras')
    joblib.dump(svm,           'final_svm_model.pkl')
    joblib.dump(scaler,        'feature_scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Done. All artefacts saved.")
