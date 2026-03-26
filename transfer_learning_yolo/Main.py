# =============================================================================
# Transfer Learning Road Sign Classifier — YOLO11n
# UTS 31256 Image Processing & Pattern Recognition — Spring 2025
# Authors: Anush Harutyunyan & Minseok Lee
#
# Pipeline:
#   1. Dataset augmentation (Albumentations, YOLO format)
#   2. Base training on Kaggle Traffic Sign dataset
#   3. Fine-tuning on Roboflow Self-Driving Cars dataset
#   4. Inference with adaptive retry logic
#   5. Comprehensive metrics (mAP, P/R/F1, ECE, Brier, stress test)
# =============================================================================

import os
import json
import random
import time
import yaml
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO
from sklearn.metrics import average_precision_score, roc_auc_score

# =============================================================================
# CONFIG — update dataset paths before running
# =============================================================================
DATASET_ROOT    = "Traffic_sign_detection_data"   # Kaggle dataset root
SIGN_YAML       = "sign.yaml"                      # Roboflow dataset yaml
YOLO_BASE_MODEL = "yolo11n.pt"                     # Pretrained YOLO11n weights


# =============================================================================
# AUGMENTATION PIPELINE
# =============================================================================

YOLO_AUG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.GaussNoise(p=0.15),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                 rotate=(-6, 6), shear=(-4, 4), p=0.7),
        A.CLAHE(clip_limit=2.0, p=0.15),
    ],
    bbox_params=A.BboxParams(
        format="yolo",          # (xc, yc, w, h) normalised to [0, 1]
        label_fields=["class_labels"],
        min_visibility=0.3,     # drop boxes that become too small/hidden
    ),
)


# =============================================================================
# LABEL UTILITIES
# =============================================================================

def read_yolo_labels(label_path: Path):
    """Return (boxes, labels) from a YOLO .txt label file."""
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            cid, xc, yc, w, h = parts
            labels.append(int(float(cid)))
            boxes.append([float(xc), float(yc), float(w), float(h)])
    return boxes, labels


def write_yolo_labels(label_path: Path, boxes, labels):
    """Write YOLO-format labels to file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for (xc, yc, w, h), cid in zip(boxes, labels):
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


_POSSIBLE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def stem_to_image_path(stem: str, img_dir: Path):
    """Find the actual image file for a given stem (filename without ext)."""
    for ext in _POSSIBLE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def scan_train_split(train_img_dir: Path, train_label_dir: Path, nc: int):
    """Return (files_by_cid, stem_to_classes) from a YOLO train split."""
    files_by_cid    = [set() for _ in range(nc)]
    stem_to_classes = defaultdict(set)
    for lbl in train_label_dir.rglob("*.txt"):
        stem = lbl.stem
        boxes, labels = read_yolo_labels(lbl)
        if not labels:
            continue
        for cid in set(labels):
            if 0 <= cid < nc:
                files_by_cid[cid].add(stem)
                stem_to_classes[stem].add(cid)
    files_by_cid = [sorted(s) for s in files_by_cid]
    return files_by_cid, stem_to_classes


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_yolo_dataset(
    dataset_root,
    nc: int,
    target_per_class: int = 1000,
    max_aug_per_source: int = 3,
    seed: int = 41,
    keep_all_boxes_in_aug: bool = True,
    out_subdir=None,
):
    """
    Oversample minority classes in the YOLO train split using Albumentations.

    Args:
        dataset_root:          root containing images/{train,val} and labels/{train,val}
        nc:                    number of classes
        target_per_class:      minimum images per class after augmentation
        max_aug_per_source:    cap augmentations from the same original image
        seed:                  RNG seed
        keep_all_boxes_in_aug: if True, keep all boxes in augmented image;
                               if False, keep only target-class boxes
        out_subdir:            if None, write back into train;
                               otherwise write to e.g. "train_aug"
    """
    rng             = random.Random(seed)
    root            = Path(dataset_root)
    train_img_dir   = root / "images" / "train"
    train_label_dir = root / "labels" / "train"

    if out_subdir:
        out_img_dir = root / "images" / out_subdir
        out_lbl_dir = root / "labels" / out_subdir
    else:
        out_img_dir = train_img_dir
        out_lbl_dir = train_label_dir

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    files_by_cid, _ = scan_train_split(train_img_dir, train_label_dir, nc)
    per_class_counts = [len(v) for v in files_by_cid]
    print(f"[info] current per-class image counts: {per_class_counts}")

    total_added = 0

    for cid in range(nc):
        present = per_class_counts[cid]
        need    = max(0, target_per_class - present)
        if need == 0:
            print(f"[info] class {cid}: already has {present} >= {target_per_class}, skip")
            continue

        candidates = files_by_cid[cid][:]
        if not candidates:
            print(f"[warn] class {cid}: no source images; cannot augment")
            continue

        rng.shuffle(candidates)
        used_times: dict = {}
        augmented_for_cid = 0
        uid_counter       = 0
        i                 = 0

        print(f"[info] augmenting class {cid}: need {need} more")

        while augmented_for_cid < need and candidates:
            stem              = candidates[i % len(candidates)]
            used_times[stem]  = used_times.get(stem, 0)

            if used_times[stem] >= max_aug_per_source:
                i += 1
                if i > len(candidates) * (max_aug_per_source + 2):
                    break
                continue

            img_path = stem_to_image_path(stem, train_img_dir)
            if img_path is None:
                i += 1
                continue

            lbl_path        = train_label_dir / f"{stem}.txt"
            boxes, labels   = read_yolo_labels(lbl_path)
            if not boxes:
                i += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                i += 1
                continue

            class_labels = labels[:]
            aug          = YOLO_AUG(image=img, bboxes=boxes, class_labels=class_labels)
            aug_img, aug_boxes, aug_labels = aug["image"], aug["bboxes"], aug["class_labels"]

            if not aug_boxes:
                i += 1
                continue

            if not keep_all_boxes_in_aug:
                keep = [(b, l) for (b, l) in zip(aug_boxes, aug_labels) if l == cid]
                if not keep:
                    i += 1
                    continue
                aug_boxes, aug_labels = list(zip(*keep))
                aug_boxes, aug_labels = list(aug_boxes), list(aug_labels)

            new_name    = f"{stem}_aug_c{cid}_{uid_counter}"
            uid_counter += 1

            cv2.imwrite(str(out_img_dir / f"{new_name}.jpg"), aug_img)
            write_yolo_labels(out_lbl_dir / f"{new_name}.txt", aug_boxes, aug_labels)

            augmented_for_cid  += 1
            total_added        += 1
            used_times[stem]   += 1
            i                  += 1

        print(f"[info] class {cid}: added {augmented_for_cid} augmented images")

    print(f"[done] augmentation complete. Total new images: {total_added}")


# =============================================================================
# DATA YAML GENERATION
# =============================================================================

def generate_data_yaml(label_path: str, output_path=None):
    """Auto-generate data.yaml by scanning YOLO label files."""
    if os.path.exists("data.yaml"):
        print("[info] data.yaml already exists. Skipping generation")
        return

    ids      = set()
    root     = Path(label_path)
    train_label_dir = root / "labels" / "train"
    val_label_dir   = root / "labels" / "val"
    train_img_dir   = (root / "images" / "train").as_posix()
    val_img_dir     = (root / "images" / "val").as_posix()

    for f in train_label_dir.rglob("*.txt"):
        try:
            with open(f, "r") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ids.add(int(float(s.split()[0])))
        except Exception as e:
            print(f"[Warning] Skipping {f}: {e}")

    if not ids:
        print("[info] No class ids found in labels. Assuming 1 class")
        ids = {0}

    nc    = max(ids) + 1
    names = [f"class_{i}" for i in range(nc)]

    data = {"train": train_img_dir, "val": val_img_dir, "nc": nc, "names": names}
    with open("data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"[info] data.yaml created. nc={nc}, ids={sorted(ids)}")


# =============================================================================
# TRAINING & FINE-TUNING
# =============================================================================

def train_yolo(model_path: str, imgsz: int, epochs: int):
    """Base training on the Kaggle dataset."""
    device  = "0" if torch.cuda.is_available() else "cpu"
    workers = max(2, min(8, os.cpu_count() or 4))
    print(f"[info] device: {device}, workers: {workers}")

    model = YOLO(model_path)
    model.train(data="data.yaml", epochs=epochs, imgsz=imgsz,
                device=device, workers=workers, seed=41, cos_lr=True)

    metrics = model.val(data="data.yaml", imgsz=imgsz, plots=True)
    print(f"[final] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


def finetune_yolo(model_path: str, imgsz: int, epochs: int):
    """
    Two-phase fine-tuning on the Roboflow dataset.
    Phase 1: frozen early layers (freeze=10), low LR → specialise to domain.
    Phase 2: all layers unfrozen, slightly higher LR → refine representations.
    """
    device  = "0" if torch.cuda.is_available() else "cpu"
    workers = max(2, min(8, os.cpu_count() or 4))

    # Phase 1 — frozen
    model = YOLO(YOLO_BASE_MODEL)
    model.load(str(model_path))
    model.train(data=SIGN_YAML, epochs=5, imgsz=imgsz, device=device,
                workers=workers, seed=41, cos_lr=True, resume=False,
                freeze=10, lr0=5e-4, lrf=5e-4)

    # Phase 2 — fully unfrozen
    model = YOLO(YOLO_BASE_MODEL)
    model.load("runs/detect/train2/weights/best.pt")
    model.train(model=YOLO_BASE_MODEL, data=SIGN_YAML, epochs=epochs,
                imgsz=imgsz, device=device, workers=workers, seed=41,
                cos_lr=True, resume=False, freeze=0, lr0=1e-3, lrf=1e-3)

    metrics = model.val(data=SIGN_YAML, imgsz=imgsz, plots=True)
    print(f"[final] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


# =============================================================================
# INFERENCE
# =============================================================================

def pred(model_path: str, input_path: str, conf: float):
    """
    Run inference with adaptive retry:
    - Pass 1: imgsz=640, given conf
    - Retry if 0 detections: lower conf, imgsz=960, TTA, agnostic NMS
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)

    def _run(imgsz, conf_thres, desc):
        res = model.predict(source=input_path, imgsz=imgsz, conf=conf_thres,
                            iou=0.7, device=device, agnostic_nms=False,
                            augment=False, save=True, verbose=False)
        r = res[0]
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[{desc}] imgsz={imgsz} conf={conf_thres:.2f} -> boxes={n}")
        return r

    r = _run(640, conf, "pass1")

    if r.boxes is None or len(r.boxes) == 0:
        conf_retry = max(conf * 0.75, 0.20)
        results    = model.predict(source=input_path, imgsz=960, conf=conf_retry,
                                   iou=0.7, device=device, agnostic_nms=True,
                                   augment=True, save=True, verbose=False)
        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[retry] imgsz=960 conf={conf_retry:.2f} TTA=True agnostic_nms=True -> boxes={n}")

    if r.boxes is not None and len(r.boxes):
        for b in r.boxes[:10]:
            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
            cls_id = int(b.cls[0])
            c      = float(b.conf[0])
            print(f"  {r.names[cls_id]}  conf={c:.2f}  box=({x1},{y1},{x2},{y2})")
    else:
        print("[hint] Still no detections. Try imgsz=1280 or conf ~0.25.")

    return r


# =============================================================================
# METRICS
# =============================================================================

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])


def yolo_to_xyxy(xc, yc, w, h, W, H):
    return [(xc - w / 2) * W, (yc - h / 2) * H,
            (xc + w / 2) * W, (yc + h / 2) * H]


def read_gt_label(lbl_path: Path, W: int, H: int):
    boxes, classes = [], []
    if not lbl_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    with open(lbl_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cid, xc, yc, w, h = s.split()
            boxes.append(yolo_to_xyxy(float(xc), float(yc), float(w), float(h), W, H))
            classes.append(int(float(cid)))
    return np.array(boxes, np.float32), np.array(classes, np.int32)


def iou_matrix(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1  = np.maximum(ax1[:, None], bx1[None, :])
    iy1  = np.maximum(ay1[:, None], by1[None, :])
    ix2  = np.minimum(ax2[:, None], bx2[None, :])
    iy2  = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, thr):
    P, G = len(p_xyxy), len(g_xyxy)
    if P == 0:
        return np.zeros(0, bool), np.zeros(G, bool)
    if G == 0:
        return np.zeros(P, bool), np.zeros(0, bool)
    order    = np.argsort(-p_conf)
    p_xyxy   = p_xyxy[order]
    p_cls    = p_cls[order]
    tp       = np.zeros(P, bool)
    gt_taken = np.zeros(G, bool)
    for i in range(P):
        c = p_cls[i]
        m = (g_cls == c)
        if not np.any(m):
            continue
        ious     = iou_matrix(p_xyxy[i:i + 1], g_xyxy[m])[0]
        cand_idx = np.where(m)[0]
        free     = ~gt_taken[cand_idx]
        if not np.any(free):
            continue
        ious     = ious[free]
        cand_idx = cand_idx[free]
        j        = np.argmax(ious)
        if ious[j] >= thr:
            tp[i]             = True
            gt_taken[cand_idx[j]] = True
    inv       = np.empty_like(order)
    inv[order] = np.arange(P)
    return tp[inv], gt_taken


def brier_score(conf, tp_flags):
    if len(conf) == 0:
        return None
    return float(np.mean((conf - tp_flags.astype(np.float32)) ** 2))


def expected_calibration_error(conf, tp_flags, n_bins=15):
    if len(conf) == 0:
        return None
    conf = np.asarray(conf, np.float32)
    y    = tp_flags.astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    N    = len(conf)
    ece  = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if np.any(m):
            ece += (np.sum(m) / N) * abs(np.mean(y[m]) - np.mean(conf[m]))
    return float(ece)


def prf_at_threshold(per_image_preds, per_image_gts, thr, iou_thr, nc):
    TP = FP = FN = 0
    tp_c = np.zeros(nc, int)
    fp_c = np.zeros(nc, int)
    fn_c = np.zeros(nc, int)
    for pred, gt in zip(per_image_preds, per_image_gts):
        p_xyxy, p_cls, p_conf = pred["xyxy"], pred["cls"], pred["conf"]
        keep   = p_conf >= thr
        p_xyxy, p_cls, p_conf = p_xyxy[keep], p_cls[keep], p_conf[keep]
        tp_flags, gt_taken = greedy_match(p_xyxy, p_conf, p_cls,
                                          gt["xyxy"], gt["cls"], iou_thr)
        TP += int(tp_flags.sum())
        FP += int((~tp_flags).sum())
        FN += int(max(len(gt["cls"]) - gt_taken.sum(), 0))
        for c in range(nc):
            m_pred_c = (p_cls == c)
            if m_pred_c.any():
                tp_c[c] += int(tp_flags[m_pred_c].sum())
                fp_c[c] += int((~tp_flags[m_pred_c]).sum())
            m_gt_c = (gt["cls"] == c)
            if m_gt_c.any():
                fn_c[c] += int((~gt_taken[m_gt_c]).sum())

    P_micro  = TP / (TP + FP) if TP + FP > 0 else 0.0
    R_micro  = TP / (TP + FN) if TP + FN > 0 else 0.0
    F1_micro = (2 * P_micro * R_micro / (P_micro + R_micro)) if (P_micro + R_micro) > 0 else 0.0

    P_macro_list, R_macro_list, F1_macro_list = [], [], []
    for c in range(nc):
        Pc  = tp_c[c] / (tp_c[c] + fp_c[c]) if (tp_c[c] + fp_c[c]) > 0 else 0.0
        Rc  = tp_c[c] / (tp_c[c] + fn_c[c]) if (tp_c[c] + fn_c[c]) > 0 else 0.0
        F1c = (2 * Pc * Rc / (Pc + Rc)) if (Pc + Rc) > 0 else 0.0
        P_macro_list.append(Pc)
        R_macro_list.append(Rc)
        F1_macro_list.append(F1c)

    P_macro  = float(np.mean(P_macro_list))
    R_macro  = float(np.mean(R_macro_list))
    F1_macro = float(np.mean(F1_macro_list))
    return (P_micro, R_micro, F1_micro, TP, FP, FN, P_macro, R_macro, F1_macro)


def sweep_best_f1(per_image_preds, per_image_gts, iou_thr, nc):
    thresholds = np.linspace(0.0, 1.0, 201)
    best    = {"conf": 0.0,
               "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "TP": 0, "FP": 0, "FN": 0},
               "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}
    best_f1 = 0.0
    for t in thresholds:
        Pmi, Rmi, F1mi, TP, FP, FN, Pma, Rma, F1ma = prf_at_threshold(
            per_image_preds, per_image_gts, t, iou_thr, nc)
        if F1mi > best_f1:
            best_f1 = F1mi
            best    = {"conf": float(t),
                       "micro": {"precision": float(Pmi), "recall": float(Rmi),
                                 "f1": float(F1mi), "TP": int(TP), "FP": int(FP), "FN": int(FN)},
                       "macro": {"precision": float(Pma), "recall": float(Rma), "f1": float(F1ma)}}
    return best


def model_metrics(model_path: str, imgsz: int, data_yaml: str = "sign.yaml",
                  iou_thr: float = 0.50, ece_bins: int = 15,
                  stress_max_imgs: int = 150, conf_infer: float = 0.001,
                  conf_eval=None, do_sweep: bool = True) -> dict:
    """
    Comprehensive offline evaluation on the validation split.
    Returns a dict with P/R/F1, PR-AUC, ROC-AUC, ECE, Brier, latency,
    stress-test results, and optional operating-point / sweep metrics.
    """
    data_cfg    = load_yaml(data_yaml)
    val_img_dir = Path(data_cfg.get("val") or data_cfg.get("valid") or data_cfg.get("validation"))
    if "images" in str(val_img_dir):
        val_lbl_dir = Path(str(val_img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    else:
        val_lbl_dir = Path(str(val_img_dir).replace("/images/", "/labels/"))

    nc    = int(data_cfg["nc"])
    names = data_cfg.get("names", [f"class_{i}" for i in range(nc)])

    val_imgs = list_images(val_img_dir)
    if not val_imgs:
        raise RuntimeError(f"No validation images found under: {val_img_dir}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)

    per_image_preds   = []
    per_image_gts     = []
    per_cls_scores    = [[] for _ in range(nc)]
    per_cls_hits      = [[] for _ in range(nc)]
    gt_pos_counts     = np.zeros(nc, int)
    per_img_presence  = []
    lat_ms            = []
    TP = FP = FN      = 0

    for img_path in val_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W  = img.shape[:2]
        lbl   = val_lbl_dir / img_path.with_suffix(".txt").name
        g_xyxy, g_cls = read_gt_label(lbl, W, H)
        for c in g_cls.tolist():
            if 0 <= c < nc:
                gt_pos_counts[c] += 1

        t1   = time.time()
        pred = model.predict(source=str(img_path), imgsz=imgsz, conf=conf_infer,
                             iou=0.70, device=device, verbose=False, agnostic_nms=False)[0]
        lat_ms.append((time.time() - t1) * 1000)

        if pred.boxes is not None and len(pred.boxes):
            p_xyxy = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
            p_cls  = pred.boxes.cls.cpu().numpy().astype(np.int32)
            p_conf = pred.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            p_xyxy = np.zeros((0, 4), np.float32)
            p_cls  = np.zeros((0,), np.int32)
            p_conf = np.zeros((0,), np.float32)

        per_image_preds.append({"xyxy": p_xyxy, "cls": p_cls, "conf": p_conf})
        per_image_gts.append({"xyxy": g_xyxy, "cls": g_cls})

        tp_flags, gt_taken = greedy_match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr)
        for c in range(nc):
            m = (p_cls == c)
            if np.any(m):
                per_cls_scores[c].extend(p_conf[m].tolist())
                per_cls_hits[c].extend(tp_flags[m].astype(int).tolist())
        TP += int(tp_flags.sum())
        FP += int((~tp_flags).sum())
        FN += int(max(len(g_cls) - gt_taken.sum(), 0))

        confmap = defaultdict(float)
        for c in range(nc):
            if np.any(p_cls == c):
                confmap[c] = float(np.max(p_conf[p_cls == c]))
        per_img_presence.append((set(g_cls.tolist()), confmap))

    # Efficiency
    avg_latency_ms  = float(np.mean(lat_ms)) if lat_ms else None
    throughput_fps  = float(1000.0 / avg_latency_ms) if avg_latency_ms else None
    try:
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    except Exception:
        model_size_mb = None

    # Micro P/R/F1
    prec_micro = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec_micro  = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_micro   = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) > 0 else 0.0

    # Macro P/R/F1 and AUCs
    macro_prec, macro_rec, macro_f1 = [], [], []
    pr_auc_values, roc_auc_values   = [], []
    for c in range(nc):
        y_scores = np.array(per_cls_scores[c], np.float32)
        y_true   = np.array(per_cls_hits[c], np.int32)
        P_pos    = int(gt_pos_counts[c])
        tp_c     = int(y_true.sum())
        fp_c     = int(len(y_true) - tp_c)
        fn_c     = int(max(P_pos - tp_c, 0))
        prec_c   = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c    = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c     = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        macro_prec.append(prec_c)
        macro_rec.append(rec_c)
        macro_f1.append(f1_c)

        ap_c = None
        if len(y_scores) > 0 and P_pos > 0 and len(np.unique(y_true)) > 1:
            try:
                ap_c = average_precision_score(y_true, y_scores)
            except Exception:
                pass
        pr_auc_values.append(ap_c)

        roc_c = None
        if len(np.unique(y_true)) > 1:
            try:
                roc_c = roc_auc_score(y_true, y_scores)
            except Exception:
                pass
        roc_auc_values.append(roc_c)

    prec_macro   = float(np.mean(macro_prec))
    rec_macro_v  = float(np.mean(macro_rec))
    f1_macro_v   = float(np.mean(macro_f1))
    pr_auc_macro = float(np.nanmean([v for v in pr_auc_values if v is not None])) \
                   if any(v is not None for v in pr_auc_values) else None
    roc_auc_macro = float(np.nanmean([v for v in roc_auc_values if v is not None])) \
                    if any(v is not None for v in roc_auc_values) else None

    all_scores = np.concatenate([np.array(per_cls_scores[c], np.float32) for c in range(nc)])
    all_hits   = np.concatenate([np.array(per_cls_hits[c], np.int32)    for c in range(nc)])
    ece        = expected_calibration_error(all_scores, all_hits, ece_bins) if len(all_scores) else None
    brier      = brier_score(all_scores, all_hits) if len(all_scores) else None

    # Top-k image-level accuracy
    def top_k_accuracy(per_img, K):
        correct = total = 0
        for present, confmap in per_img:
            if not present:
                continue
            total += len(present)
            topK   = [c for c, _ in sorted(confmap.items(), key=lambda x: -x[1])[:K]]
            correct += sum(1 for c in present if c in topK)
        return correct / total if total > 0 else 0.0

    top1 = top_k_accuracy(per_img_presence, 1)
    top3 = top_k_accuracy(per_img_presence, 3)

    # Stress test
    CORR = {
        "gauss_noise":         A.GaussNoise(p=1.0),
        "motion_blur":         A.MotionBlur(blur_limit=7, p=1.0),
        "brightness_contrast": A.RandomBrightnessContrast(0.3, 0.3, p=1.0),
    }

    def eval_subset(imgs, tfm=None, max_n=stress_max_imgs):
        per_s  = [[] for _ in range(nc)]
        per_h  = [[] for _ in range(nc)]
        gt_cnt = np.zeros(nc, int)
        cnt    = 0
        for ip in imgs:
            if cnt >= max_n:
                break
            im = cv2.imread(str(ip))
            if im is None:
                continue
            H, W  = im.shape[:2]
            lbl   = val_lbl_dir / ip.with_suffix(".txt").name
            g_xyxy, g_cls = read_gt_label(lbl, W, H)
            for c in g_cls.tolist():
                if 0 <= c < nc:
                    gt_cnt[c] += 1
            if tfm is not None:
                im = tfm(image=im)["image"]
            p = model.predict(source=im, imgsz=imgsz, conf=0.001, iou=0.70,
                              device=device, verbose=False, agnostic_nms=False)[0]
            if p.boxes is not None and len(p.boxes):
                px = p.boxes.xyxy.cpu().numpy().astype(np.float32)
                pc = p.boxes.cls.cpu().numpy().astype(np.int32)
                pf = p.boxes.conf.cpu().numpy().astype(np.float32)
            else:
                px = np.zeros((0, 4), np.float32)
                pc = np.zeros((0,), np.int32)
                pf = np.zeros((0,), np.float32)
            tpf, _ = greedy_match(px, pf, pc, g_xyxy, g_cls, iou_thr)
            for c in range(nc):
                m = (pc == c)
                if np.any(m):
                    per_s[c].extend(pf[m].tolist())
                    per_h[c].extend(tpf[m].astype(int).tolist())
            cnt += 1
        aps = []
        for c in range(nc):
            ys = np.array(per_s[c], np.float32)
            yt = np.array(per_h[c], np.int32)
            if len(ys) > 0 and gt_cnt[c] > 0 and len(np.unique(yt)) > 1:
                try:
                    aps.append(average_precision_score(yt, ys))
                except Exception:
                    pass
        return float(np.mean(aps)) if aps else None

    subset    = val_imgs[:min(len(val_imgs), stress_max_imgs)]
    base_map  = eval_subset(subset, None, stress_max_imgs)
    stress    = {}
    for name, aug in CORR.items():
        m = eval_subset(subset, A.Compose([aug]), stress_max_imgs)
        stress[name] = {"mAP50_subset": m,
                        "delta_vs_clean": (m - base_map) if (m is not None and base_map is not None) else None}

    operating_point = None
    if conf_eval is not None:
        Pmi, Rmi, F1mi, TP_e, FP_e, FN_e, Pma, Rma, F1ma = prf_at_threshold(
            per_image_preds, per_image_gts, conf_eval, iou_thr, nc)
        operating_point = {
            "conf_eval": float(conf_eval),
            "micro": {"precision": Pmi, "recall": Rmi, "f1": F1mi,
                      "TP": TP_e, "FP": FP_e, "FN": FN_e},
            "macro": {"precision": Pma, "recall": Rma, "f1": F1ma}
        }

    sweep_best = sweep_best_f1(per_image_preds, per_image_gts, iou_thr, nc) if do_sweep else None

    return {
        "overall": {"top1_accuracy_presence": top1, "top3_accuracy_presence": top3},
        "precision_recall_f1": {
            "micro": {"precision": prec_micro, "recall": rec_micro,  "f1": f1_micro},
            "macro": {"precision": prec_macro, "recall": rec_macro_v, "f1": f1_macro_v}
        },
        "pr_auc":    {"per_class": pr_auc_values,  "macro": pr_auc_macro},
        "roc_auc":   {"per_class": roc_auc_values, "macro": roc_auc_macro,
                      "note": "ROC on detection confidences is non-standard; compare with PR-AUC."},
        "calibration": {"ece": ece, "brier": brier, "bins": ece_bins},
        "efficiency":  {"avg_latency_ms_per_image": avg_latency_ms,
                        "throughput_fps": throughput_fps, "model_size_mb": model_size_mb},
        "stress_test": {"baseline_subset_mAP50": base_map,
                        "corruptions": stress, "subset_size": len(subset)},
        "operating_point": operating_point,
        "sweep_best": sweep_best,
        "meta": {"images_evaluated": len(val_imgs), "iou_thr": iou_thr,
                 "imgsz": imgsz, "nc": nc, "names": names, "data_yaml": str(Path(data_yaml).resolve())}
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_data_yaml(DATASET_ROOT)

    augment_yolo_dataset(
        dataset_root=DATASET_ROOT,
        nc=43,
        target_per_class=200,
        max_aug_per_source=20,
        seed=41,
        keep_all_boxes_in_aug=True,
        out_subdir=None,          # write back into train; or set "train_aug"
    )

    train_yolo(YOLO_BASE_MODEL, imgsz=640, epochs=20)
    finetune_yolo("runs/detect/train/weights/best.pt", imgsz=416, epochs=20)
    pred("runs/detect/train3/weights/best.pt", "road.PNG", conf=0.25)

    metrics_json = model_metrics(
        model_path="runs/detect/train3/weights/best.pt",
        imgsz=416,
        data_yaml=SIGN_YAML,
        conf_eval=0.49
    )

    with open("metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print("\nDone. Metrics saved to metrics.json")
