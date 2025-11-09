# Employee Attrition Prediction using Logistic Regression

This project conducts an end-to-end analysis of the IBM HR Analytics Employee Attrition dataset from Kaggle. The primary goal is to understand the key factors driving employee attrition and to build an optimized predictive model using Logistic Regression.

The final model achieves a mean cross-validated accuracy of **87.28%**.

---

## 1. Exploratory Data Analysis (EDA)

The initial analysis focused on identifying the main drivers of employee attrition. The data was visualized to uncover relationships between features and the target variable (`Attrition`).

Key insights from the EDA include:
* **Overtime:** Employees who work overtime have a significantly higher rate of attrition.
* **Monthly Income:** Lower monthly income is strongly correlated with a higher likelihood of attrition.

These two factors were hypothesized to be the most significant predictors.

## 2. Model Development & Preprocessing

To build an interpretable yet powerful predictive model, **Logistic Regression** was chosen. A sophisticated preprocessing pipeline was constructed using Scikit-learn's `Pipeline` and `ColumnTransformer` to prepare the data for the linear model.

This preprocessing workflow includes:

* **Feature Separation:** Data was split into continuous, categorical (text-based), and ordinal (numeric categories) features.
* **One-Hot Encoding (OHE):** Categorical features (like `JobRole`, `MaritalStatus`) were one-hot encoded. `drop='first'` was used to prevent multicollinearity, a requirement for stable Logistic Regression.
* **Polynomial Features:** `PolynomialFeatures` was applied to the continuous numeric features to allow the linear model to capture non-linear relationships and interactions.
* **Scaling:** `StandardScaler` was applied *after* the polynomial transformation to scale all features, which is essential for the performance of a regularized Logistic Regression model.

## 3. Hyperparameter Tuning (GridSearchCV)

`GridSearchCV` was used to systematically test and find the optimal combination of preprocessing steps and model parameters. This ensures the best possible performance and generalizability.

The grid search was configured to tune the following parameters simultaneously:

* **Polynomial Degree:** The `degree` of the `PolynomialFeatures` (testing `[2, 3, 4, 5]`).
* **Regularization Penalty:** The `penalty` for `LogisticRegression` (testing `['l1' (Lasso), 'l2' (Ridge)]`).
* **Regularization Strength:** The `C` parameter for `LogisticRegression` (testing `[0.001, 0.01, 0.1, 1.0, 10.0]`).

## 4. Results & Conclusion

The hyperparameter tuning process (totaling 200 model fits) identified the following optimal parameters:

* **Best `degree`:** `2`
* **Best `penalty`:** `'l2'` (Ridge Regularization)
* **Best `C`:** `0.1`

This combination achieved a final mean cross-validated accuracy of **87.28%**, an improvement over the initial baseline of 86.94%.

This result indicates that a simpler `degree=2` model (parabolic) with strong L2 regularization (`C=0.1`) provided the best balance, preventing the overfitting that occurred with more complex degrees.

Finally, the coefficients of this best-fit model were extracted to validate the initial EDA hypotheses. The results confirmed:
1.  **`OverTime`** had the largest positive coefficient, proving it is the strongest predictor for attrition.
2.  **`MonthlyIncome`** had a strong negative coefficient, confirming that lower pay is a significant driver of attrition.

## 5. Key Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn` (Pipeline, LogisticRegression, GridSearchCV, ColumnTransformer, PolynomialFeatures, StandardScaler)

## 6. Repository Contents

* `EDAmodul5.ipynb`: The main Jupyter Notebook containing all analysis, preprocessing, modeling, and evaluation code.
* `WA_Fn-UseC_-HR-Employee-Attrition.csv`: The raw dataset used for this project.
