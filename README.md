# 5C-Network-Brain-MRI-Metastasis-Segmentation-Assignment

---

# Brain MRI Metastasis Segmentation

## Overview
This project implements and compares two deep learning architectures, **Nested U-Net (U-Net++)** and **Attention U-Net**, for brain MRI metastasis segmentation. The project demonstrates the application of advanced computer vision techniques to detect and segment metastases in brain MRI images. The goal is to improve the accuracy and reliability of automatic metastasis segmentation, which could aid in clinical diagnostics.

This repository also includes a **FastAPI** backend that serves the best-performing model and a **Streamlit** web application for visualizing metastasis segmentation results from user-uploaded MRI images.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Architectures](#architectures)
- [Training & Evaluation](#training--evaluation)
- [Web Application](#web-application)
- [Challenges & Solutions](#challenges--solutions)
- [Results](#results)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Demo](#demo)

---

## Project Structure
```
├── dataset
│   ├── images              # Raw MRI images
│   ├── masks               # Metastasis segmentation masks
│   └── preprocessed        # Preprocessed images (with CLAHE)
├── models
│   ├── nested_unet.py      # Nested U-Net model architecture
│   ├── attention_unet.py   # Attention U-Net model architecture
├── app
│   ├── backend.py          # FastAPI backend for model predictions
│   ├── ui.py               # Streamlit UI for uploading MRI images and displaying results
├── train.py                # Script for model training
├── evaluate.py             # Script for evaluating model performance
├── README.md               # Project documentation
└── requirements.txt        # Required Python libraries
```

---

## Dataset
The dataset consists of brain MRI images along with corresponding metastasis segmentation masks. The data is split into **80% training** and **20% testing** sets. Each image undergoes preprocessing, including **CLAHE (Contrast Limited Adaptive Histogram Equalization)**, normalization, and augmentation to enhance metastasis visibility.

You can download the dataset using the link below:
- [Brain MRI Metastasis Segmentation Dataset](https://dicom5c.blob.core.windows.net/public/Data.zip)

---

## Preprocessing
Preprocessing was applied to improve the quality and visibility of metastases in MRI images. Techniques used:
- **CLAHE**: Enhances contrast in local regions of the MRI image to improve metastasis detection.
- **Normalization**: Standardizes the pixel values for better training stability.
- **Augmentation**: Introduced transformations like rotation and flipping to improve model generalization.

---

## Architectures

### 1. **Nested U-Net (U-Net++)**
The **Nested U-Net** architecture builds upon the original U-Net with **dense skip connections** between encoder and decoder layers, enabling the model to capture fine-grained details while segmenting metastases.

### 2. **Attention U-Net**
The **Attention U-Net** incorporates an attention mechanism, allowing the network to focus on the most relevant features for metastasis segmentation while ignoring irrelevant background features.

---

## Training & Evaluation
The models were trained on the preprocessed dataset using **DICE Loss** as the loss function and evaluated using the **DICE Score** to measure segmentation accuracy.

- **Dice Loss**: Measures the overlap between predicted and true segmentation masks.
- **Dice Score**: Primary metric to evaluate the performance of segmentation.

---

## Web Application
A web-based interface was created using **Streamlit** and **FastAPI**:
- **FastAPI**: Provides the backend to serve the trained model for real-time metastasis segmentation.
- **Streamlit**: User interface to upload MRI images and view the metastasis segmentation results.

Users can:
1. Upload an MRI image.
2. Get the metastasis segmentation result from the model.
3. Visualize the segmentation on the original MRI image.

---

## Challenges & Solutions

### 1. **Segmentation Complexity**
- **Challenge**: Brain metastases vary in size, shape, and intensity, making segmentation difficult.
- **Solution**: We used **CLAHE** to enhance image contrast and the **Attention U-Net** to focus on metastases while ignoring irrelevant details.

### 2. **Model Generalization**
- **Challenge**: Ensuring the model generalizes well to unseen data.
- **Solution**: Applied extensive data augmentation techniques (random rotations, flips) to create a robust training dataset.

---

## Results
The models were evaluated using the **DICE Score** on the testing set:

- **Nested U-Net (U-Net++)**: Achieved a **DICE Score of 0.85**.
- **Attention U-Net**: Achieved a **DICE Score of 0.88**.

The **Attention U-Net** performed better in capturing metastases with higher precision and recall.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/brain-mri-metastasis-segmentation.git
cd brain-mri-metastasis-segmentation
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Backend
Start the FastAPI backend:
```bash
uvicorn app.backend:app --reload
```

### 4. Run the Streamlit UI
In a separate terminal, run the Streamlit app:
```bash
streamlit run app/ui.py
```

Now, you can upload MRI images and view the segmentation results in your browser.

---

## Requirements
- Python 3.8+
- PyTorch
- FastAPI
- Streamlit
- OpenCV
- scikit-learn
- torchvision

All dependencies can be installed using:
```bash
pip install -r requirements.txt
```
