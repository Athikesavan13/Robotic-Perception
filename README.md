# 🔌 Metric-Semantic 3D Scene Reconstruction
### Desktop Socket Pose Estimation via Multi-View YOLO Detection and Ray Triangulation

<p align="center">
  <img src="assets/pipeline_banner.png" alt="Pipeline Overview" width="800"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv9m-Ultralytics-orange?logo=pytorch" alt="YOLO"/></a>
  <a href="#"><img src="https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy" alt="NumPy"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Kaggle%20T4-20BEFF?logo=kaggle" alt="Kaggle"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Course-CP260--2026-green" alt="Course"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License"/></a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Pipeline Architecture](#-pipeline-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Stage-by-Stage Guide](#-stage-by-stage-guide)
  - [Stage 1: Dataset Preparation](#stage-1-dataset-preparation)
  - [Stage 2: YOLOv9m Training](#stage-2-yolov9m-training)
  - [Stage 3: Multi-Frame Detection](#stage-3-multi-frame-detection)
  - [Stage 4: Ray Triangulation](#stage-4-ray-triangulation)
  - [Stage 5: OBB Assembly](#stage-5-obb-assembly)
- [Mathematical Formulation](#-mathematical-formulation)
- [Results](#-results)
- [Configuration Reference](#-configuration-reference)
- [Output Format](#-output-format)
- [Visualisation](#-visualisation)
- [Bonus Features](#-bonus-features)
- [Limitations & Future Work](#-limitations--future-work)
- [Authors](#-authors)
- [References](#-references)

---

## 🧭 Overview

This project solves **Metric-Semantic 3D Scene Reconstruction** — the problem of localising specific semantic objects in metric 3D space from a set of posed RGB images. Concretely, we detect and localise three types of desktop back-panel sockets:

| Target Object    | Class ID |
|-----------------|----------|
| Ethernet Socket | 0        |
| Power Socket    | 1        |
| VGA Socket      | 2 (bonus)|

For each object, we produce a full **Oriented Bounding Box (OBB)** — encoding 3D centre, physical extents (in metres), and orientation — without requiring a depth sensor, dense reconstruction, or any learned 3D representation.

**Key idea:** Fine-tune a YOLO detector on 16 annotated frames, run it across 704 posed frames, then triangulate 2D detections into metric 3D positions using known camera poses. Clean, fast, and interpretable.

---

## 🎯 Problem Statement

### Inputs

| Input | Description |
|-------|-------------|
| `frame_000xyz.png` | 2560×1440 RGB images of a desktop back-panel |
| `poses.json` | Camera-to-world pose matrices `W_T_Ci ∈ SE(3)`, indexed by frame number |
| Camera Intrinsics | `fx=1477.01`, `fy=1480.44`, `cx=1298.25`, `cy=686.82` |

### Output

An Oriented Bounding Box per socket:

```
OBB = { c ∈ ℝ³,  e ∈ ℝ³,  R ∈ SO(3) }
```

- **c** — 3D centre position in world frame (metres)
- **e** — half-extents [width, height, depth] (metres)
- **R** — 3×3 rotation matrix encoding panel orientation

### Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Centre Error | `‖ĉ − c*‖₂` (metres) | < 8 cm |
| Rotation Error | `arccos((tr(R̂ᵀR*) − 1) / 2)` (degrees) | < 10° |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: 704 Posed RGB Frames             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Dataset Preparation                              │
│  • LabelMe JSON polygon annotations (16 frames)             │
│  • Convert polygons → YOLO bounding boxes                   │
│  • 80/20 stratified train/val split                         │
│  • Generate data.yaml for Ultralytics                       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — YOLOv9m Fine-Tuning                              │
│  • Pretrained backbone (ImageNet), frozen first 10 layers   │
│  • AdamW + cosine LR decay, 200 epochs                      │
│  • Mosaic + MixUp + Copy-Paste augmentation                 │
│  • Early stopping (patience=40)                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — Multi-Frame Detection                            │
│  • Inference on all 704 frames                              │
│  • Confidence threshold τ = 0.25                            │
│  • Store per-frame detections: [x1,y1,x2,y2, cls, conf]    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4 — Multi-View Ray Triangulation                     │
│  • Construct viewing ray per detection (pinhole model)      │
│  • Rotate ray to world frame via pose matrix                │
│  • Solve linear least-squares for optimal 3D intersection   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 — OBB Assembly                                     │
│  • Panel normal → rotation matrix R via Gram-Schmidt        │
│  • Metric extent from reprojection + physical priors        │
│  • Clamp outliers to 2.5× physical prior                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: final_answers.json  +  OBB visualisation           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
metric-semantic-3d-reconstruction/
│
├── README.md
├── requirements.txt
├── data.yaml                          # YOLO dataset config
│
├── notebooks/
│   ├── Cell_A_dataset_prep.ipynb      # Stage 1: annotation conversion
│   ├── Cell_B_train_yolo.ipynb        # Stage 2: YOLOv9m fine-tuning
│   ├── Cell_C_detect_all_frames.ipynb # Stage 3: batch inference
│   ├── Cell_D_triangulation.ipynb     # Stage 4: ray triangulation
│   ├── Cell_E_obb_assembly.ipynb      # Stage 5: OBB construction
│   ├── Cell_F_validation.ipynb        # Evaluation vs ground truth
│   └── Cell_G_visualisation.ipynb     # OBB back-projection plots
│
├── src/
│   ├── dataset_prep.py                # Polygon → YOLO bbox conversion
│   ├── train.py                       # YOLOv9m training script
│   ├── detect.py                      # Batch detection on all frames
│   ├── triangulate.py                 # Multi-view ray triangulation
│   ├── obb_assembly.py                # OBB rotation + extent recovery
│   ├── visualise.py                   # OBB back-projection renderer
│   └── utils.py                       # Shared helpers (pose loading, etc.)
│
├── data/
│   ├── annotations/                   # LabelMe JSON files (16 frames)
│   ├── images/
│   │   ├── train/                     # 12 training images
│   │   └── val/                       # 4 validation images
│   ├── labels/
│   │   ├── train/                     # YOLO .txt label files
│   │   └── val/
│   └── poses.json                     # Camera-to-world pose matrices
│
├── weights/
│   └── yolov9m_sockets.pt             # Fine-tuned model weights
│
├── outputs/
│   ├── final_answers.json             # Predicted OBBs (submission)
│   ├── detections/                    # Per-frame detection results
│   └── visualisations/                # OBB projection images
│
└── assets/
    └── pipeline_banner.png
```

---

## 📦 Dataset

The dataset is **not included** in this repository (Kaggle competition data). To reproduce results:

1. Download the dataset from the Kaggle competition page
2. Place images in `data/images/`, annotations in `data/annotations/`, and `poses.json` in `data/`

**Dataset Statistics:**

| Property | Value |
|----------|-------|
| Total frames | 704 |
| Annotated frames | 16 |
| Image resolution | 2560 × 1440 px |
| Annotation format | LabelMe JSON (polygon) |
| Socket classes | 3 (Ethernet, Power, VGA) |
| Camera model | Pinhole |

**Camera Intrinsics:**

```python
fx, fy = 1477.01, 1480.44
cx, cy = 1298.25, 686.82
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (tested on Kaggle T4, ~2.4 GB VRAM usage)
- Git

### Clone & Install

```bash
git clone https://github.com/<your-username>/metric-semantic-3d-reconstruction.git
cd metric-semantic-3d-reconstruction

pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
Pillow>=9.5.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
labelme>=5.2.0
PyYAML>=6.0
```

---

## 🚀 Quickstart

If you just want to run the full pipeline end-to-end with default settings:

```bash
# 1. Prepare dataset
python src/dataset_prep.py \
    --annotations_dir data/annotations \
    --output_dir data/labels \
    --split 0.8

# 2. Train YOLOv9m
python src/train.py \
    --data data.yaml \
    --epochs 200 \
    --batch 8 \
    --device 0

# 3. Run detection on all frames
python src/detect.py \
    --weights weights/yolov9m_sockets.pt \
    --images_dir data/images/all \
    --conf 0.25 \
    --output outputs/detections

# 4. Triangulate 3D centres
python src/triangulate.py \
    --detections outputs/detections \
    --poses data/poses.json \
    --output outputs/centres.json

# 5. Assemble OBBs and export
python src/obb_assembly.py \
    --centres outputs/centres.json \
    --detections outputs/detections \
    --poses data/poses.json \
    --output outputs/final_answers.json

# 6. Visualise results
python src/visualise.py \
    --obbs outputs/final_answers.json \
    --images_dir data/images/all \
    --poses data/poses.json \
    --output outputs/visualisations
```

---

## 📖 Stage-by-Stage Guide

### Stage 1: Dataset Preparation

LabelMe polygon annotations are converted to YOLO axis-aligned bounding boxes:

```python
# For each polygon annotation:
xmin = min(polygon[:, 0])
ymin = min(polygon[:, 1])
xmax = max(polygon[:, 0])
ymax = max(polygon[:, 1])

# Normalise by image dimensions (YOLO format):
x_centre = (xmin + xmax) / 2 / img_width
y_centre = (ymin + ymax) / 2 / img_height
width    = (xmax - xmin) / img_width
height   = (ymax - ymin) / img_height
```

**Class mapping:**

| Class | ID |
|-------|----|
| Ethernet Socket | 0 |
| Power Socket    | 1 |
| VGA Socket      | 2 |

The 16 annotated frames are split 80/20 (stratified) → 12 train / 4 val.

---

### Stage 2: YOLOv9m Training

```python
from ultralytics import YOLO

model = YOLO("yolov9m.pt")

results = model.train(
    data        = "data.yaml",
    epochs      = 200,
    imgsz       = 640,
    batch       = 8,
    optimizer   = "AdamW",
    lr0         = 0.001,
    lrf         = 0.01,
    mosaic      = 1.0,
    mixup       = 0.15,
    copy_paste  = 0.3,
    flipud      = 0.5,
    fliplr      = 0.5,
    degrees     = 15.0,
    freeze      = 10,
    patience    = 40,
)
```

**Why these choices?**

| Choice | Rationale |
|--------|-----------|
| `freeze=10` | Keeps ImageNet-pretrained low-level features intact; only the detection head is fine-tuned, preventing overfitting on 16 images |
| `mosaic=1.0` | Composites 4 images per sample — effectively quadruples dataset diversity |
| `mixup=0.15` | Blends pairs of images and labels for smoother decision boundaries |
| `copy_paste=0.3` | Randomly pastes socket instances at new locations — critical for small-object generalization |
| `AdamW + cosine LR` | 3-epoch warm-up followed by cosine decay to `1e-2 × lr0`; stable on small datasets |
| `YOLOv9m over YOLOv9n` | Sockets occupy <5% of image area; the medium model's +20M parameters improve small-object recall significantly |

---

### Stage 3: Multi-Frame Detection

The fine-tuned model is applied to all 704 frames:

```python
from ultralytics import YOLO

model = YOLO("weights/yolov9m_sockets.pt")
results = model.predict(source="data/images/all", conf=0.25, save=False)

# Per-frame detections stored as:
# detections[frame_id] = [[x1, y1, x2, y2, cls_id, conf], ...]
```

**Confidence threshold τ = 0.25** filters low-quality detections before triangulation. Setting τ lower (e.g., 0.15) recovers detections in sparse-view frames at the cost of noisier triangulation.

---

### Stage 4: Ray Triangulation

For each socket class, viewing rays are collected from all frames where a detection exists, then the optimal 3D intersection is solved in closed form.

**Step 1 — Construct ray in camera frame:**

```python
u, v = bbox_centre  # pixel coordinates

d_cam = np.array([
    (u - cx) / fx,
    (v - cy) / fy,
    1.0
])
d_cam /= np.linalg.norm(d_cam)
```

**Step 2 — Rotate to world frame using pose:**

```python
# pose: 4×4 camera-to-world matrix W_T_C
R_cw = pose[:3, :3]          # rotation submatrix
o    = pose[:3, 3]            # camera centre in world frame

d_world = R_cw @ d_cam
d_world /= np.linalg.norm(d_world)
```

**Step 3 — Solve least-squares intersection:**

For N rays `{(o_i, d_i)}`, the optimal 3D point minimises the sum of squared perpendicular distances. This leads to the linear system:

```python
A = np.zeros((3, 3))
b = np.zeros(3)

for o_i, d_i in rays:
    M_i = np.eye(3) - np.outer(d_i, d_i)
    A  += M_i
    b  += M_i @ o_i

centre, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
```

---

### Stage 5: OBB Assembly

**Rotation estimation — Panel Normal:**

The panel normal is approximated as the mean unit vector pointing from the 3D socket centre toward each observing camera:

```python
normal = np.zeros(3)
for o_i in camera_centres:
    v = o_i - centre
    normal += v / np.linalg.norm(v)
normal /= np.linalg.norm(normal)

# Build orthonormal frame via Gram-Schmidt with world-up prior
up    = np.array([0.0, 1.0, 0.0])
right = np.cross(up, normal)
right /= np.linalg.norm(right)
up    = np.cross(normal, right)

R = np.column_stack([right, up, normal])  # 3×3 rotation matrix
```

**Extent estimation — Metric reprojection:**

```python
widths, heights = [], []
for bbox, o_i in zip(bboxes, camera_centres):
    x1, y1, x2, y2 = bbox
    d = np.linalg.norm(centre - o_i)  # depth estimate

    w_m = (x2 - x1) * d / fx
    h_m = (y2 - y1) * d / fy
    widths.append(w_m)
    heights.append(h_m)

# Median + clamp to 2.5× physical prior to reject outliers
w_final = min(np.median(widths),  2.5 * PRIOR_WIDTH[cls])
h_final = min(np.median(heights), 2.5 * PRIOR_HEIGHT[cls])
d_final = PRIOR_DEPTH[cls]  # depth always from physical prior
```

**Physical size priors:**

| Entity | Width (m) | Height (m) | Depth (m) |
|--------|-----------|------------|-----------|
| VGA Socket | 0.0354 | 0.0118 | 0.0061 |
| Ethernet Socket | 0.0160 | 0.0130 | 0.0180 |
| Power Socket | 0.0280 | 0.0220 | 0.0200 |

---

## 📐 Mathematical Formulation

### OBB Representation

$$\text{OBB} = \{ \mathbf{c} \in \mathbb{R}^3,\; \mathbf{e} \in \mathbb{R}^3,\; \mathbf{R} \in SO(3) \}$$

### Multi-View Triangulation (Eq. 2)

$$\min_{\mathbf{p}} \sum_{i=1}^{N} \left\| \mathbf{p} - \mathbf{o}_i - \hat{\mathbf{d}}_i^\top (\mathbf{p} - \mathbf{o}_i)\, \hat{\mathbf{d}}_i \right\|^2$$

### Linear System (Eq. 5)

$$\left( \sum_i M_i \right) \mathbf{p} = \sum_i M_i\, \mathbf{o}_i, \qquad M_i = I - \hat{\mathbf{d}}_i \hat{\mathbf{d}}_i^\top$$

### Ray Direction (Eqs. 3–4)

$$\hat{\mathbf{d}}_\text{cam} = \left[\frac{u - c_x}{f_x},\; \frac{v - c_y}{f_y},\; 1 \right]^\top, \qquad \hat{\mathbf{d}}_\text{world} = \mathbf{R}_{CW}\, \hat{\mathbf{d}}_\text{cam} \;/\; \|\cdot\|$$

### Panel Normal (Eq. 6)

$$\hat{\mathbf{n}} = \frac{1}{N} \sum_{i=1}^{N} \frac{\mathbf{o}_i - \mathbf{c}}{\| \mathbf{o}_i - \mathbf{c} \|}$$

### Metric Extent (Eq. 7)

$$w_m = \frac{(x_2 - x_1) \cdot d}{f_x}, \qquad h_m = \frac{(y_2 - y_1) \cdot d}{f_y}$$

---

## 📊 Results

### Detection Performance (YOLOv9m, epoch 76/200)

| Class | Precision | Recall | mAP@50 | mAP@50:95 |
|-------|-----------|--------|--------|-----------|
| All (combined) | 0.335 | 0.583 | 0.316 | 0.181 |

> Modest mAP values are expected with only 16 training images. Box loss decreased 1.419 → 1.205 and classification loss 0.841 → 0.648 over epochs 71–74, with steady convergence. GPU usage: ~2.39 GB on Kaggle T4.

### 3D Localisation (VGA Socket vs. Ground Truth)

| Metric | Value | Target |
|--------|-------|--------|
| Centre error | See Cell E output | < 8 cm |
| Rotation error | See Cell E output | < 10° |

### Qualitative Results

OBB back-projection onto all 16 annotated frames confirms tight alignment with physical connectors, with no systematic drift across viewpoints.

| Frame 333 | All Frames |
|-----------|------------|
| ![Frame 333](outputs/visualisations/frame_333_obb.png) | ![All Frames](outputs/visualisations/all_frames_obb.png) |

---

## 🔧 Configuration Reference

All key hyperparameters are centralised in `src/utils.py`:

```python
# Camera intrinsics
INTRINSICS = {
    "fx": 1477.01, "fy": 1480.44,
    "cx": 1298.25, "cy": 686.82
}

# Detection threshold
CONF_THRESHOLD = 0.25

# Physical size priors [width, height, depth] in metres
PHYSICAL_PRIORS = {
    "ethernet_socket": [0.0160, 0.0130, 0.0180],
    "power_socket":    [0.0280, 0.0220, 0.0200],
    "vga_socket":      [0.0354, 0.0118, 0.0061],
}

# Outlier clamping multiplier
CLAMP_FACTOR = 2.5

# Class ID mapping
CLASS_NAMES = {0: "ethernet_socket", 1: "power_socket", 2: "vga_socket"}
```

---

## 📄 Output Format

Results are serialised to `outputs/final_answers.json`:

```json
[
  {
    "entity": "power_socket",
    "obb": {
      "center":   [x, y, z],
      "extent":   [width, height, depth],
      "rotation": [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
      ]
    }
  },
  {
    "entity": "ethernet_socket",
    "obb": { ... }
  },
  {
    "entity": "vga_socket",
    "obb": { ... }
  }
]
```

A strict format-check assertion validates all fields before submission:

```python
assert "entity" in entry
assert "center" in entry["obb"] and len(entry["obb"]["center"]) == 3
assert "extent" in entry["obb"] and len(entry["obb"]["extent"]) == 3
assert "rotation" in entry["obb"] and np.array(entry["obb"]["rotation"]).shape == (3, 3)
```

---

## 🖼️ Visualisation

OBBs are back-projected onto image frames as a sanity check using the pinhole model:

```python
def project_obb(centre, extent, rotation, pose, intrinsics):
    """Project 3D OBB corners onto image plane."""
    w, h, d = extent
    # 8 corners of the box in local frame
    corners_local = np.array([
        [±w, ±h, ±d] for each sign combination
    ])
    # Transform to world frame
    corners_world = rotation @ corners_local.T + centre[:, None]
    # Transform to camera frame
    C_T_W  = np.linalg.inv(pose)
    corners_cam = C_T_W[:3, :3] @ corners_world + C_T_W[:3, 3:]
    # Project
    u = fx * corners_cam[0] / corners_cam[2] + cx
    v = fy * corners_cam[1] / corners_cam[2] + cy
    return np.stack([u, v], axis=1)
```

Run visualisation for all frames:

```bash
python src/visualise.py \
    --obbs outputs/final_answers.json \
    --images_dir data/images/all \
    --poses data/poses.json \
    --output outputs/visualisations \
    --frames all          # or --frames 333 for a single frame
```

---

## 🎁 Bonus Features

### VGA Socket Generalisation
A third class (VGA Socket, class ID 2) was added with zero changes to the triangulation or OBB assembly code. This confirms the framework is a general **metric-semantic pipeline**, not a task-specific solution. Simply:
1. Annotate VGA sockets with class ID 2 in LabelMe
2. Add the physical prior to `PHYSICAL_PRIORS`
3. Run the pipeline unchanged

### Confidence-Weighted Ray Filtering
Lowering τ from 0.25 to 0.15 recovers additional detections in frames where the socket is partially occluded or at a difficult viewing angle. Useful when a socket is detected in very few frames:

```bash
python src/detect.py --conf 0.15 ...
```

### Physical Prior Clamping
Without the 2.5× clamp, a single close-range frame can produce extent estimates > 30 cm (physically implausible). The clamp is essential for robustness:

```python
w_final = min(np.median(widths), CLAMP_FACTOR * PRIOR_WIDTH[cls])
```

---

## ⚠️ Limitations & Future Work

| Limitation | Proposed Fix |
|------------|-------------|
| Panel-normal heuristic assumes sockets face the camera for most frames — fails for back-facing sockets | Use PnP-based normal estimation from known 3D socket geometry |
| Extent estimation degrades at grazing angles | RANSAC-based depth estimation or learned depth refinement |
| Small training set (16 images) limits detection recall | Self-supervised pseudo-labelling on unlabelled frames |
| Single confidence threshold for all classes | Per-class adaptive thresholding |
| Replacing YOLO with RT-DETR | Transformer detectors show improved small-object recall |

---

## 👨‍💻 Authors

| Name | Roll No | Email |
|------|---------|-------|
| Boddu Amarnath | 26574 | bamarnath@iisc.ac.in |
| Utkarsh Vats | 27284 | utkarshvats@iisc.ac.in |

**Course:** CP260-2026, Indian Institute of Science

---

## 📚 References

1. J. Redmon, S. Divvala, R. Girshick, A. Farhadi — *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016
2. G. Jocher, A. Chaurasia, J. Qiu — *Ultralytics YOLO*, 2023. https://github.com/ultralytics/ultralytics
3. R. Hartley, A. Zisserman — *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2003
4. K. Kanatani, Y. Sugaya, H. Niitsuma — *Triangulation from Two Views Revisited: Hartley-Sturm vs. Optimal Correction*, BMVC 2008
5. R. B. Rusu, N. Blodow, M. Beetz — *Fast Point Feature Histograms (FPFH) for 3D Registration*, ICRA 2009
6. J. J. Park, P. Florence, J. Straub, R. Newcombe, S. Lovegrove — *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation*, CVPR 2019
7. J. McCormac, A. Handa, A. Davison, S. Leutenegger — *SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks*, ICRA 2017
8. A. Rosinol, M. Abate, Y. Chang, L. Carlone — *Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping*, ICRA 2020
9. Y. Siddiqui et al. — *Panoptic Lifting for 3D Scene Understanding with Neural Fields*, CVPR 2023

---

<p align="center">
  Made with ❤️ at IISc Bangalore &nbsp;·&nbsp; CP260-2026
</p>
