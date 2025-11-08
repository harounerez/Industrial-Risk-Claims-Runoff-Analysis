# 💰 Advanced Actuarial Claims Reserving and Predictive Modeling

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Modeling](https://img.shields.io/badge/Models-Chain%20Ladder%2C%20GAM%2C%20XGBoost-green)](https://scikit-learn.org/)
[![Industry](https://img.shields.io/badge/Industry-Industrial%20Insurance%20Claims-red)](https://github.com/yourusername/yourrepo)

This project implements a **hybrid modeling framework** for technical reserves, combining **traditional actuarial reserving methods** with **modern Machine Learning (ML)** techniques. The goal is to produce a **robust, highly accurate, and interpretable** estimation of the **Run-Off Triangle** for outstanding claims in the industrial insurance segment.

The approach utilizes an extensive claims dataset from the Algerian market to provide a comprehensive forecast of future claims payments, ensuring financial stability and regulatory compliance.

---

## 🎯 Project Objectives

The primary objective is to design a high-performance predictive model that achieves the following:

1.  **Granular Reserve Estimation:** Provide a reliable, detailed estimation of the **liquidity triangle (Run-Off Triangle)** for multiple lines of business.
2.  **Hybrid Methodology:** Integrate advanced techniques, including **Generalized Additive Models (GAM)** and **Stochastic Processes**, with state-of-the-art **AI/ML models (XGBoost, LightGBM)**.
3.  **Operational Applicability:** Ensure the final model is both **interpretable** for audit purposes and readily deployable within existing insurance system environments.

---

## 📊 Dataset Description

The analysis is based on a comprehensive dataset of industrial risk claims from the Algerian insurance market, providing a deep look into the claims development over a decade (2014 to 2023), totaling **90,229 claim observations**.

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

This stream focuses on established aggregate methods for completing the Run-Off Triangle while enhancing them with statistical flexibility.

* **Chain-Ladder Method:** Implemented as the foundational model to derive **Link Ratio Factors** and calculate the traditional reserve estimate, serving as a crucial industry benchmark.
* **Generalized Additive Models (GAM):** Used to model the development factors or claims payments, offering a flexible, non-linear alternative to standard linear models while maintaining high **interpretability**—essential for regulatory reporting.

### 2. Machine Learning and Simulation (Predictive Approach)

This stream transforms the triangle completion into a supervised regression problem, leveraging advanced non-linear predictors for enhanced accuracy.

* **Feature Engineering:** The claims data is unrolled from a triangle structure into a flat dataset suitable for ML training, incorporating features like Occurrence Year, Development Lag, and Product Category.
* **Gradient Boosting Models:**
    * **XGBoost Regressor:** Utilized for its superior predictive power in complex, tabular data, often yielding the lowest prediction errors.
    * **LightGBM Regressor:** Implemented as an efficient, high-performance alternative, suitable for large-scale datasets.
* **Stochastic Simulation:** The final predictions from the ML models can be used as inputs for a **Monte Carlo Simulation** framework to generate a full reserve distribution, thereby quantifying the **Prediction Risk**.

---

## 📂 Repository Contents

| File Name | Description |
| :--- | :--- |
| **`File Presentation.pdf`** | Document outlining the business context, problem statement, modeling requirements, and a detailed description of the claims data used. |
| **`mathurance_report_team_33.pdf`** | The comprehensive technical report, detailing the data analysis, the implementation of both the traditional (Chain-Ladder + GAM) and ML approaches, and a comparison of the results. |
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
2.  **Install dependencies:** Ensure you have Python 3.x and the necessary libraries (`numpy`, `pandas`, `scikit-learn`, `chainladder`, `pygam`, `xgboost`, `lightgbm`).
3.  **Run the notebooks:** Start by reviewing the technical report, then execute the cells in the **`chain ladder + GAM.ipynb`** and **`simulation + ml.ipynb`** notebooks to follow the modeling workflow and generate the final reserve predictions.
