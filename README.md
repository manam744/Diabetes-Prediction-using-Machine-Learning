# Diabetes-Prediction-using-Machine-Learning
This project aims to build and optimize machine learning models to predict the presence of diabetes in patients based on medical diagnostic features. In addition to standard model development, we also explored misclassified samples (False Positives and False Negatives) to extract hidden patterns and improve the overall performance of the final model.

## Dataset

We used the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains medical data for female patients of at least 21 years old.

**Features include:**

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0: Non-Diabetic, 1: Diabetic)

## Data Preprocessing

- Replaced invalid zero values in features like `Glucose`, `BloodPressure`, and `BMI` with appropriate statistics (e.g., median).
- Applied standard scaling to numerical features to ensure fair model training.
- Created new interaction features (e.g., `Glucose x Age`, `Pregnancy / Age`, `BMI / SkinThickness`, etc.) to enhance model understanding of feature relationships.
- Split the dataset into training and testing sets using an 80/20 ratio.

## Exploratory Data Analysis (EDA)

- Analyzed feature distributions using histograms and box plots to detect skewness, outliers, and potential correlations.
- Visualized relationships between features and the target variable (`Outcome`) using:
  - Correlation heatmap
  - Pair plots
  - Distribution comparison (diabetic vs non-diabetic)
- Noticed that `Glucose` and `BMI` had the strongest relationship with diabetes presence.
- Identified skewed features and potential interaction patterns to inform feature engineering.

## Baseline Models

We trained and evaluated several baseline models using the original features only:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- CatBoost

Metrics such as accuracy, precision, recall, F1-score, and ROC AUC were calculated.

**Best baseline result:**  
CatBoost achieved the best performance initially, especially in ROC AUC and generalization on test data.

## Feature Engineering

To improve model performance, we created several new features:

- `Glucose x Age` (Glucose_Age_Interaction)
- `Pregnancies / Age` (Pregnancy_Age_Ratio)
- `BMI / SkinThickness` (BMI_Skin_Index)
- Tree-based pattern feature from FP/FN analysis (Error_Pattern_Tree)

These engineered features were added to the dataset and re-evaluated using the top-performing models.

## Error Analysis: False Positives & False Negatives

- After evaluating model predictions, we analyzed misclassified samples:
  - False Positives (Predicted 1, True 0)
  - False Negatives (Predicted 0, True 1)
- Built a decision tree classifier on these samples to identify hidden patterns.
- Extracted interpretable rules and encoded them into a new binary feature: `Error_Pattern_Tree`
- This feature helped the final model to reduce misclassification by learning from previous mistakes.

## Final Model and Hyperparameter Tuning

After evaluating multiple models, **CatBoost** was selected as the final model due to its strong performance.

We then performed hyperparameter tuning using GridSearchCV and Bayesian Optimization to find optimal values for:

- Depth
- Learning rate
- Number of estimators
- L2 regularization

Additionally, we optimized **class weights** to better handle class imbalance and reduce false negatives and false positives.

**Final Model Performance:**

- Accuracy: 0.78  
- F1-score: 0.78  
- ROC AUC: 0.82  
- Confusion Matrix:
    - True Negatives: 78
    - False Positives: 21
    - False Negatives: 13
    - True Positives: 42

## Cross-Validation

To ensure the robustness of our final model, we used **Stratified K-Fold Cross-Validation** (K=5).

This helped validate that the model generalizes well across different data splits and does not overfit.

Cross-validation metrics remained consistent with the test set, confirming the stability of the model.

## Conclusion

This project demonstrates a complete pipeline of classification on medical data:

- Clean preprocessing
- Insightful EDA and feature engineering
- Robust model selection and evaluation
- Learning from model errors to improve predictions

The final CatBoost model achieved a reliable performance and identified interpretable patterns helpful for medical diagnosis support.

✅ All code, analysis, and results are reproducible.

## How to Run

To run the code:

1. Clone the repository:
2. Install dependencies:
3. Run the notebook `diabetes_prediction.ipynb` step by step.

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- matplotlib
- seaborn

- ### Feature Importance
![Feature Importance](feature_importance.png)

### Confusion Matrix
![Confusion Matrix](image/confusion_matrix.png)

### ROC Curve
![ROC Curve](image/roc_curve.png)

