# Cocoa Disease Detection - Documentation

## Overview and Objectives

This solution aims to develop an object detection model to identify multiple diseases in cocoa plant images. Our solution is designed to assist subsistence farmers in Africa by enabling disease detection using entry-level smartphones. The objectives include:

- Accurate multi-class disease detection on cocoa plant images.
- Generalization to unseen diseases not present in the training set.
- Efficient deployment and inference on edge devices.

Expected outcomes include increased early detection of plant diseases, reduced crop losses, and minimized pesticide use.

## Folder structure
  - notebooks/rf-detr-amini-cocoa-challenge.ipynb
  - notebooks/rfdetr-training.ipynb
  - notebooks/rfdetr-inference.ipynb
  - data/train.csv
  - data/test.csv
  - notebooks/wbf.ipynb
  - results/test.csv
  - results/train.csv
  - requirements.txt
  - cocoa_disease_detection_documentation.md

N.B:
- After data preparation, training and inference, it is important to have submission csv file and test.csv coupled with train.csv in the same directory as the `wbf` notebook, the folder is arranged in this way for the purpose of uniformity.
- Running `pip install -r requirements.txt` might not be neccessary as all libraries needed will be installed upon running the notebooks



## ETL Process

### Extract

- **Data Sources**: Image datasets with annotated cocoa plant diseases.
- **Formats**: Images (JPEG, PNG), COCO-style JSON annotations.

### Transform

- **Orientation Fixes**: Badly oriented images were corrected.
- **COCO Conversion**: All images converted to COCO dataset format.

### Load

- **Storage**: Images and annotations loaded into memory and saved in a dataset to be loaded in the `rfdetr-training` notebook

## Data Modeling

- **Model Used**: `RFDETRLarge` from Roboflow's repository.
- **Assumptions**: Diseases can co-occur, images may show multiple labels(very few cases in the provided dataset).
- **Feature Engineering**: Based on raw image inputs and COCO labels.

### Model Training

- **Algorithm**: RF-DETR (DETR-based object detection).
- **Training Time**: 8 hours 24 minutes.
- **Epochs**: 10.
- **Losses**:
  - Train Loss: 6.01317
  - Test Loss: 6.74589
- **Evaluation Metrics**:
  - AP50: 0.78086
  - AP50_90: 0.54817
  - AR50_90: 0.43859
- **EMA Metrics**:
  - AP50: 0.80309
  - AP50_90: 0.57591
  - AR50_90: 0.44415

### Model Validation

Validation was performed using COCO evaluation metrics (AP and AR). Model checkpoints were monitored with Weights & Biases (wandb) to track performance over time.

## Inference 

- **Deployment**: Local inference using trained model weights.
- **Infrastructure**: Local GPU for batch inference.
- **Input**: Test images.
- **Output**: csv file with bounding box coordinates with labels and confidence scores.
- **Postprocessing**: WBF (Weighted Box Fusion) applied for final predictions.
  - For the WBF postprocessing, the image size is added to the initial test dataframe, all these are added in the `rf-detr-amini-cocoa-challenge` notebook, thi is also the data preparation notebook
  - The wbf notebook is also seen as `wbf.ipynb`

## Runtime

- **Training**: 8h 24m
- **Inference**: 15m

## Performance Metrics

- **Wandb-tracked Metrics**:
  - Train/Test Loss
  - AP50, AP50_90, AR50_90 (Base & EMA)
- **Submission Scores**:
  - **Public Leaderboard**: 0.825360226
  - **Private Leaderboard**: 0.825496953
## Resource

- **GPU**:
 - T4 GPU on Kaggle

## Submission

- **File Name**: `rfdetr_9_epochs_full_linear_best_ema_wbf-max.csv`

## General Approach

1. Install necessary packages
2. Load data
3. Fix orientation for bad images
4. Create COCO dataset
5. Train model
6. Evaluate model
7. Run inference on test set
8. Apply Weighted Box Fusion (WBF) on the submission file
