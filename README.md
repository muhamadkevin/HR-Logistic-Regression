# Employee Attrition Prediction (Optimized for Recall)

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.x-blue?logo=pandas)

This project builds an end-to-end machine learning model to predict employee attrition (turnover) using the IBM HR Analytics dataset. The primary goal is not just to build an *accurate* model, but a *useful* one that can proactively identify employees at high risk of leaving.

---

## 1. The Business Problem

Employee turnover is expensive. It costs companies time and money to recruit, hire, and train new employees. By identifying employees who are likely to leave, HR can proactively intervene with retention strategies (e.g., salary adjustments, workload changes, or new projects) to improve morale and reduce attrition.

## 2. The Challenge: A Highly Imbalanced Dataset

Exploratory Data Analysis (EDA) revealed a significant challenge: the dataset is **highly imbalanced**.
* **No Attrition (Class 0):** ~84% of the dataset
* **Attrition (Class 1):** ~16% of the dataset

A model optimized for pure **Accuracy** will simply learn to "always predict No Attrition" and achieve a high, but misleading, score. The true test of this model is its ability to correctly identify the rare "Attrition" cases.

## 3. Key EDA Insights

Initial analysis confirmed several common-sense drivers of attrition:
* **Overtime:** Employees who work overtime are significantly more likely to leave.
* **Monthly Income:** Lower income levels are strongly correlated with a higher probability of attrition.
* **Job Role:** Certain roles (e.g., Sales Representative) had much higher turnover than others (e.g., Research Director).

## 4. Modeling & Pipeline

A sophisticated preprocessing and modeling pipeline was built using `scikit-learn` to ensure robust and reproducible results.

* **Model:** `LogisticRegression` was chosen because its coefficients are interpretable, allowing us to explain *why* the model makes certain predictions.
* **Preprocessing:** A `ColumnTransformer` was used to apply:
    * `OneHotEncoder` to categorical features.
    * `PolynomialFeatures` to key numeric features (to capture non-linear relationships).
    * `StandardScaler` to all features, which is essential for regularized models.

## 5. Model Iteration: The "Accuracy Trap" vs. The Useful Model

This project demonstrates the critical importance of choosing the correct evaluation metric.

### Iteration 1: The "Accuracy Trap" (Baseline Model)
The initial `GridSearchCV` was run using the default `scoring='accuracy'`.
* **Result:** A high **Accuracy of 87.28%**.
* **The Problem:** The `classification_report` showed this model was useless.
    * **Recall (Class 1): 0.32** — It failed to identify **68%** of employees who were about to leave.

### Iteration 2: Optimizing for What Matters (The Final Model)
To fix this, the model was re-optimized by adding **`class_weight='balanced'`** to the `LogisticRegression` estimator. This forces the model to pay a heavier penalty for misclassifying the rare "Attrition" class. This trade-off accepting a lower overall accuracy to gain a *much* higher recall.

## 6. Final Model Results

The final, optimized model provides a far more actionable tool for HR. The key decision was to optimize for **Recall** on the "Attrition" class, accepting a necessary trade-off in Precision and overall Accuracy to achieve this.

The final results from the optimized model (using `class_weight='balanced'`) on the test data are:

| Metric | Final Score |
| :--- | :--- |
| **Recall (Class 1)** | **70%** |
| **AUC Score** | 0.81 |
| F1-Score (Class 1) | 48% |
| Precision (Class 1) | 37% |
| Accuracy | 76% |

While the overall accuracy dropped, the **Recall for Attrition jumped from 32% to 70%**. This means the model now successfully identifies 70% of at-risk employees, a massive improvement in business value.

### Final Confusion Matrix (Data Tes)
<img width="657" height="561" alt="image" src="https://github.com/user-attachments/assets/cbbc69bd-da9b-4a5a-8bef-5021041abc87" />
``

### Final ROC Curve (Data Tes)
<img width="570" height="565" alt="image" src="https://github.com/user-attachments/assets/ad41d0f7-b9b4-4996-85a7-d33f5eb643e4" />
``
## 7. How to Run

Follow these steps to run the analysis on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/muhamadlevin/HR-Logistic-Regression.git](https://github.com/muhamadlevin/HR-Logistic-Regression.git)
    cd HR-Logistic-Regression
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows
    .\venv\Scripts\activate
    
    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    (This uses the `requirements.txt` file you created.)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch JupyterLab:**
    ```bash
    jupyterlab
    ```
    This will open JupyterLab in your web browser. From there, you can open and run the `EDAmodul5.ipynb` notebook.
