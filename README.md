# Automated Morphological Classification of Galaxies 🌌

### A Deep Learning Approach using Custom VGG Architecture
**Authors:** Nikolaos Mavros, Kiveli Fotinaki  
**Institution:** University of Thessaly, Dept. of Electrical & Computer Engineering

---

## 📌 Project Overview
The morphological classification of galaxies is a critical task in observational cosmology. This project automates the classification process using a custom-designed **Convolutional Neural Network (CNN)**. Trained on the **Galaxy Zoo** dataset, our model predicts 37 probabilistic morphological features (e.g., "Smooth", "Spiral", "Edge-on") with high precision.

Unlike "black box" solutions, this repository implements a rigorous 5-phase experimental pipeline:
1.  **Architecture Design:** Custom VGG-style CNN (2.6M params).
2.  **Sensitivity Analysis:** Optimization of Learning Rate ($10^{-3}$) and Dropout ($0.2$).
3.  **Scalability Testing:** Verification of sub-linear scaling on large datasets.
4.  **Deployment:** Full-scale training on 61,578 images.
5.  **Evaluation:** Blind testing on 79,975 unlabelled galaxies.

## 📊 Key Results
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Validation RMSE** | **0.104** | Internal evaluation on 20% holdout set. |
| **Test RMSE** | **0.109** | External evaluation on Kaggle blind test set. |
| **Inference Speed** | **57ms** | Average time to process a single image (Real-time). |
| **Throughput** | **720 img/s** | Max processing speed during batch training. |

## 📂 Repository Structure
* `main.ipynb`: The complete, modular Jupyter Notebook containing all 5 experimental phases.
* `final_report.pdf`: The official IEEE-format scientific paper describing the methodology.
* `my_galaxy_model_backup.keras`: The saved weights of the final trained model.
* `galaxy_zoo_submission.csv`: The final probability predictions generated for the Kaggle competition.

## 🧠 Model Architecture
The model is a custom VGG-variant optimized for $64 \times 64$ pixel inputs:
* **Input:** $64 \times 64 \times 3$ (RGB Images)
* **Encoder:** 4 Convolutional Blocks ($32 \to 64 \to 128 \to 256$ filters).
* **Regularization:** Batch Normalization + Max Pooling ($2\times2$) after every block.
* **Head:** Dense layers ($512 \to 256 \to 37$) with optimized Dropout ($0.2$).
* **Output:** 37-way Sigmoid probabilistic regression.

## 🚀 Usage
### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* Pandas, NumPy, Matplotlib

### Running the Code
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Galaxy-Morphology-CNN.git](https://github.com/YOUR_USERNAME/Galaxy-Morphology-CNN.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```
3.  **Run the Notebook:**
    Open `main.ipynb` in Jupyter or Google Colab. The notebook is self-contained and handles data downloading (via Kaggle API), training, and evaluation.

## 📈 Visualizations
The project includes tools to visualize:
* **Learning Curves:** Loss & RMSE over epochs.
* **Learned Filters:** Visualizing the internal weights of the first CNN layer.
* **Predictions:** Comparing model probability outputs against ground-truth images.

---
*Based on the research paper "Automated Morphological Classification of Galaxies using Deep Convolutional Architecture" (2026).*
