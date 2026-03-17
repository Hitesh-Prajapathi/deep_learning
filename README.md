# Deep Learning Labs Repository

This repository contains a set of deep learning lab assignments and experiments (lab1 through lab7), plus YOLO object detection projects (YOLOv5 and YOLOv8). The notebooks and checkpoints are organized to demonstrate training, evaluation, and dataset structure for classification + detection tasks.

## 📁 Root folders

- `lab1/` - Lab 1 notebook and experiments
- `lab2/` - Lab 2 notebook and experiments
- `lab3/` - Lab 3 notebook and experiments
- `lab4/` - Lab 4 dataset, CNN/ensemble experiments
- `lab5/` - Lab 5 notebook and experiments
- `lab6/` - Lab 6 notebook and experiments
- `lab7/` - Lab 7 notebook and experiments
- `yolov5_model/` - YOLOv5 training project, configs, weights, runs
- `yolov8_model/` - YOLOv8 training project, configs, weights, runs
- `LICENSE` - project license

## 🧾 Summary of labs

### lab1
- `2548525_Lab_1_DL.ipynb`
- Likely introduced fundamentals: Python + environment setup, first model, basic data loading, training loop, metric logging.

### lab2
- `2548525_DL_LAB2.ipynb`
- Likely deepened data augmentation, model tuning, evaluation metrics, and train/val pipeline.

### lab3
- `2548525_DL_Lab3.ipynb`
- Likely covered convolutional networks, activation maps, regularization, and advanced architecture components.

### lab4
- `2548525_DL_Lab4.ipynb` (expected on disk)
- `brain_tumor_subset/` dataset structure for brain tumor classification
  - `images/train/`, `images/val/`
  - `labels/train/`, `labels/val/`
- `cnn/` contains:
  - `cnn_from_scratch.ipynb`
  - pre-trained checkpoint: `cnn_cifar10.pth`
  - dataset: `data/cifar-10-batches-py/`
- `ensemble/` contains ensemble modeling coursework and output artifacts

### lab5
- `2548525_DL_Lab5.ipynb`
- Medium complexity experiments, likely cross-validation, architecture comparisons, optimization strategies.

### lab6
- `2548525_DL_Lab_6.ipynb`
- Probably custom model design, metric analysis, or dataset extension.

### lab7
- `2548525_DL_Lab7.ipynb`
- Final lab, summarizing full pipeline, evaluation, maybe deployment or model export.

## 🚀 YOLO workflows

### yolov5_model
- `yolov5_training.ipynb`
- `yolov5_training.yaml`
- `yolov5s.pt`
- Results under `runs/brain_tumor_v5*/`
- Includes `yolov5/` source code from official repository

### yolov8_model
- `yolov8_training.ipynb`
- `yolov8_training.yaml`
- `yolov8s.pt`
- Results under `runs/yolov8/`

## ▶️ Setup and run

1. Install Python packages (PyTorch, torchvision, OpenCV, other notebook dependencies).
2. Start Jupyter in repo root:
   - `jupyter lab` or `jupyter notebook`
3. Open any lab notebook and run cells interactively.

### YOLOv5 quick commands
```bash
cd yolov5_model/yolov5
pip install -r requirements.txt
python train.py --data ../brain_tumor_subset/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 50
python detect.py --weights runs/train/exp/weights/best.pt --source ../brain_tumor_subset/images/val
```

### YOLOv8 quick commands
```bash
cd yolov8_model
# assuming Ultralytics package installed
python -m pip install ultralytics
python train.py --data yolov8_training.yaml --model yolov8s.pt --epochs 50
python val.py --weights runs/yolov8/weights/best.pt --data yolov8_training.yaml
```

## 📌 Notes

- Adjust notebook paths if any dataset path is hardcoded.
- Keep data directories intact for reproducibility.
- Optional: add a root `requirements.txt` to unify dependencies.

## 🔏 Recommended improvements

- add a concise table of contents with direct links to notebooks
- include environment setup script (e.g. `setup.sh`, `requirements.txt`) at repo root
- add execution order guidance (lab1 -> lab7, then YOLO)

---

> Have fun exploring deep learning experiments and training your own models!
