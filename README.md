# connector-detection-transfer-learning
# Connector Detection using Object Detection (TensorFlow vs PyTorch Exploration)

This project was developed in response to an interview assignment that required training a model to detect connectors and deploy it for live webcam inference using TensorFlow.js. Due to practical constraints with Colab GPU availability and deployment limitations, this submission focuses on **deep experimentation and benchmarking** of transfer learning techniques using **TensorFlow** and **PyTorch** object detection models.

---

## 🎯 Assignment Overview

**Objective:**
- Train a model to detect only **"connectors"** from a custom dataset.
- Deploy the model in-browser using TensorFlow.js to perform real-time inference from the webcam.
- Achieve at least **90% mAP (mean Average Precision)**.

**Deliverables (original task):**
- Working web app with live webcam inference
- Hosted link (AWS/Heroku/etc.)
- GitHub repository with model + training code
- Video showcasing the model

---


## 🚧 Limitations

Despite significant experimentation and analysis, a few constraints limited full deployment and results:

1. **No tf.js Deployment Achieved**  
   The original task required running the model via TensorFlow.js on a live webcam in the browser.  
   → Due to time, conversion issues, and limited training output, this step was not completed.

2. **Dataset Was Provided Privately**  
   The custom dataset was part of an interview assignment and cannot be shared publicly.  
   → This affects reproducibility and generalization testing.

3. **No Real-Time Web Demo**  
   No hosted demo (e.g., on AWS, Render, or Netlify) was submitted due to lack of a working tf.js export and compute constraints.

4. **Training Limited by Compute Budget**  
   Training was run on free-tier Google Colab GPU for 3+ hours (10 epochs on ~1500 images).  
   → Insufficient for full convergence or extensive fine-tuning needed to achieve 90%+ mAP.

5. **MaskRCNN Rejected Use Without Mask Labels**  
   Attempt to use segmentation-capable models like `maskrcnn_resnet50_fpn` failed as they required mask labels even when bounding boxes alone were desired.

6. **Model Architecture Compatibility**  
   Some newer SOTA architectures like **DINO DETR** and **EfficientDet** were not used due to:
   - Lack of clear documentation for label remapping
   - Confusion around class prediction layers
   - High compute requirements

7. **Evaluation Only on mAP**  
   Accuracy was measured only using **mean Average Precision (mAP)**. No per-class precision-recall breakdowns were computed.

---
## 🔬 What Was Accomplished

Despite infrastructure limitations, the following key investigations and results were completed:

### ✅ TensorFlow (TF2 Object Detection API):
- Used `hub.KerasLayer`-based pipeline and pre-trained object detection models.
- Limitations encountered:
  - **No way to freeze feature extractor layers via `pipeline.config`**
  - TensorFlow community has left related GitHub issues unanswered since 2020
  - Training without freezing leads to **worse results** than pre-trained baselines

> See: `TensorFlow2ObjectDetectionUsingTFHUB.ipynb` and `tensorflowVsPytorch.pdf`

---

### ✅ PyTorch (Torchvision-based Object Detection):
- Explored multiple models for transfer learning:
  - `fasterrcnn_resnet50_fpn`
  - `maskrcnn_resnet50_fpn`
  - `fcos_resnet50_fpn`
- Issues faced:
  - **FCOS** raised architecture/weight access issues (see GitHub issue raised: #5932)
  - **MaskRCNN** required segmentation mask even when only bounding boxes were needed
  - **FasterRCNN** worked best, but only under **specific label remapping** (`label 77 - Cell Phone`)
- Training vs freezing tests performed with:
  - `requires_grad=False` to freeze feature extractors
  - Custom optimizer setups: **Adam with LambdaLR** vs **SGD with StepLR**

> See:  
> - `PytorchObjectDetectionForCustomDataInferenceAndTraining.ipynb`  
> - `FasterRCNN10epochs*.txt` logs

---

## 📊 Results

| Model | Setup | Final mAP |
|-------|-------|-----------|
| FasterRCNN + Adam + Label 77 | Transfer learning (with label remap) | ✓ Improved mAP |
| FasterRCNN + SGD + No Label | Trained from scratch | ✗ mAP = 0 |
| Pre-trained FCOS | Used as-is | Poor detection |
| MaskRCNN | Failed due to missing masks | N/A |

⏳ **Training Time:** 10 epochs = ~3 hours on Colab GPU for 1512 images  
📉 **Result:** Not enough GPU time or compute budget to achieve full 90%+ mAP within time limits.

---

## 🧠 Key Learnings

- **Model architecture choice + label mapping** is critical in transfer learning.
- TensorFlow's object detection API is limited for custom training unless deeply customized.
- PyTorch offers better transparency with `requires_grad`, pretrained layer access, and faster iteration.
- Cloud-free deployment of larger models is challenging without paid compute or a lightweight substitute like **YOLOv5-nano** or **MobileNet SSD**.
- Latest object detection models (e.g., **FasterRCNN v2**) show substantial mAP improvements (e.g., +9.7).

> See: [FasterRCNN v2 Improvements](https://github.com/pytorch/vision/issues/5307)

---

## 📂 Project Files

