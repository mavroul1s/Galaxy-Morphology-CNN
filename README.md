# 🌌 Automated Morphological Classification of Galaxies
### A Deep Learning Approach using Custom VGG Architecture

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview
The morphological classification of galaxies is a fundamental task in observational cosmology, providing insights into the formation and evolution of the universe. This project automates this process using a custom-designed **Convolutional Neural Network (CNN)** trained on the **Galaxy Zoo** dataset.

Unlike traditional "black box" solutions, this repository implements a transparent, rigorous **5-phase experimental pipeline** designed to isolate and optimize architectural decisions, achieving high precision with low computational overhead.

### 🎯 Key Objectives
* **Precision:** Predict 37 probabilistic morphological features (e.g., "Smooth", "Spiral", "Bar") with an RMSE < 0.11.
* **Efficiency:** Design a lightweight architecture ($\approx 2.6$M parameters) capable of real-time inference.
* **Interpretability:** Visualize internal feature maps to ensure the model learns physical geometric primitives.

---

## 📂 Repository Structure

```text
├── 📂 assets/               # Visualizations, plots, and architecture diagrams
│   ├── architecture_diagram.jpg
│   ├── mse_rmse.png
│   ├── learned_filters.png
│   └── all_37_classes_labeled.png
│
├── 📂 doc/                  # Formal documentation and scientific reports
│   └── final_report.pdf     # IEEE-formatted scientific paper detailing the methodology
│
└── 📂 src/                  # Source code and implementation
    ├── main.ipynb           # Complete Jupyter Notebook (Data pipeline, Training, Evaluation)
    └── galaxy_zoo_submission.csv  # Final probability predictions
