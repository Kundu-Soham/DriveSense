# Pedestrian Crossing Prediction with Deep Learning 🚗🚸

This repository contains the implementation of a deep learning model that predicts pedestrian crossing intent using multi-frame video sequences from the Joint Attention for Autonomous Driving (JAAD) dataset. 

By combining spatial feature extraction with dynamic temporal modeling and temporal attention, the network accurately classifies whether a pedestrian intends to cross the street.

---

## 📄 Technical Report

For a detailed breakdown of the model architecture, baseline comparisons, quantitative evaluations, and qualitative failure-mode analyses, check out the full project report:

* 📌 **[View Technical Report PDF](./DriveSense%20Full%20Report.pdf)**: Read the 5-page PDF detailing dataset split mechanics, training curves, attention weights, and edge-case evaluations.

---

## 🚀 Quick Start: Google Colab

> **Note:** The complete source code, dataset parsing, training pipelines, evaluation loops, and interactive visualizations are designed to run end-to-end inside **Google Colab**.

* 📌 **[Open Notebook in Google Colab](https://colab.research.google.com/drive/10teHBrmxWfdvdZtnmqPZtvkvrya_MOh6)**: Click to open and run the project interactively with GPU acceleration.
* 📁 **Local Inspection**: You can also browse the `.ipynb` file directly inside the [`notebooks/`](./notebooks) directory of this repository.

---

## ⚙️ Google Drive & Data Setup

> ⚠️ **Directory Path Notice**: The notebook is configured to load raw video sequences and annotations from a specific Google Drive folder structure. 
> 
> If you are running the notebook on your own Google Colab instance:
> 1. Mount your Google Drive in the first cell (`drive.mount('/content/drive')`).
> 2. Update the `BASE_DIR` / path variables in the top configuration cell to match your local Google Drive directory structure.