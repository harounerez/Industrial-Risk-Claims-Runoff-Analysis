# 💰 Industrial-Risk-Claims-Runoff-Analysis

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Modeling](https://img.shields.io/badge/Methods-Chain%20Ladder%2C%20GAM%2C%20XGBoost-green)](https://scikit-learn.org/)
[![Focus](https://img.shields.io/badge/Focus-Industrial%20Insurance%20Claims-red)](https://github.com/yourusername/yourrepo)

This project implements a **hybrid modeling framework** for technical reserves, combining **traditional actuarial reserving methods** with **modern Machine Learning (ML)** techniques. The goal is to produce a **robust, highly accurate, and interpretable** estimation of the **Run-Off Triangle** for outstanding claims in the industrial insurance segment.

The approach utilizes an extensive claims dataset from the Algerian market to provide a comprehensive forecast of future claims payments, ensuring financial stability and regulatory compliance.

---

## 🎯 Project Objectives

The core objective is to design a high-performance predictive model that achieves the following:

1.  **Granular Reserve Estimation:** Provide a reliable, detailed estimation of the **liquidity triangle (Run-Off Triangle)** for multiple lines of business.
2.  **Hybrid Methodology:** Integrate advanced techniques, including **Generalized Additive Models (GAM)** and **Stochastic Processes**, with state-of-the-art **AI/ML models (XGBoost, LightGBM)**.
3.  **Operational Applicability:** Ensure the final model is both **interpretable** for audit purposes and readily deployable within existing insurance system environments.

---

## ⚙️ Solution Architecture

The project follows a structured data-to-prediction pipeline that integrates both classical and advanced methods at different stages.



**Figure 1: Solution Pipeline Diagram**
> This diagram illustrates the two main parallel approaches: the **Traditional Actuarial Path** (Chain-Ladder and GAM) and the **Supervised Machine Learning Path** (Feature Engineering followed by XGBoost/LightGBM). Both paths feed into the final prediction and reserve calculation.

---

## 📊 Dataset Description

The analysis is based on a comprehensive dataset of industrial risk claims from the Algerian insurance market, covering a 10-year period (2014 to 2023), totaling **90,229 claim observations**.

The dataset focuses on settlements for **Industrial Risks** and includes the following key variables:

| Feature | Description |
| :--- | :--- |
| `Exercice` | Year of claims settlement. |
| `Branche` | The Industrial Risks insurance branch. |
| `Code Produit` / `Désignation Produit` | Specific insurance product codes and names. |
| `Sous-Branche` | Sub-categories within the industrial risks branch. |
| `Date Survenance` | Date the claim occurred (Occurrence Date). |
| `Règlement` | The settlement amount for each claim (**Target Variable**). |

> **Key Insight:** Initial data analysis confirmed that claim development patterns (`sinistralité`) vary significantly depending on the **product type** and **sub-branch**, necessitating a feature-rich or segmented modeling approach.

---

## 🔬 Modeling Approaches

We developed and compared two primary, complementary modeling streams to ensure a comprehensive and robust reserving solution.

### 1. Actuarial and Statistical Reserving (Traditional Approach)

* **Chain-Ladder Method:** Implemented as the foundational model to derive **Link Ratio Factors** and calculate the traditional reserve estimate, serving as a crucial industry benchmark.
* **Generalized Additive Models (GAM):** Used to model development factors or claims payments, offering a flexible, non-linear alternative to standard linear models while maintaining high **interpretability**.

### 2. Machine Learning and Simulation (Predictive Approach)

* **Feature Engineering:** The claims data is unrolled from a triangle structure into a flat dataset suitable for ML training, incorporating features like Occurrence Year, Development Lag, and Product Category.
* **Gradient Boosting Models:** **XGBoost Regressor** and **LightGBM Regressor** were utilized for their superior predictive power in complex, tabular data.
* **Stochastic Simulation:** The final predictions from the ML models can be used as inputs for a **Monte Carlo Simulation** framework to generate a full reserve distribution, thereby quantifying the **Prediction Risk**.

---

## 📈 Results and Visual Analysis

The comparison of methodologies reveals the trade-offs between interpretability and predictive accuracy.



**Figure 2: Comparative Analysis Plots**
> These plots visualize key findings from the Exploratory Data Analysis (EDA) and the model evaluation:
> * **Annual Claim Settlements:** Shows the trend of total `Règlement` (settlement amount) over the different years of settlement (`Exercice`).
> * **Claims Distribution by Product:** Illustrates the volume of claims across various `Code Produit` and `Désignation Produit`, highlighting the most frequent lines of business.
> * **Model Performance Comparison:** Displays a comparison of key metrics (e.g., MAE, RMSE) between the **Traditional Chain-Ladder** approach and the **Machine Learning** models, demonstrating the superior predictive accuracy gained from the latter.

---

## 📂 Repository Contents

| File Name | Description |
| :--- | :--- |
| **`chain ladder + GAM.ipynb`** | **Jupyter Notebook** containing all code for data preprocessing, Run-Off Triangle creation, Chain-Ladder factor calculation, and the implementation of **GAMs** for reserving. |
| **`simulation + ml.ipynb`** | **Jupyter Notebook** containing the code for data transformation, feature generation for supervised learning, training of **XGBoost** and **LightGBM** models, and the structure for post-modeling stochastic simulation. |

---

## 💻 Getting Started

To replicate the project results locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:** Ensure you have Python 3.x and the necessary libraries (e.g., `numpy`, `pandas`, `scikit-learn`, `chainladder`, `pygam`, `xgboost`, `lightgbm`).
3.  **Run the notebooks:** Start by reviewing the technical report, then execute the cells in the **`chain ladder + GAM.ipynb`** and **`simulation + ml.ipynb`** notebooks to follow the modeling workflow and generate the final reserve predictions.
