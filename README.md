# Pima Indians Diabetes Prediction Project

This project aims to predict the likelihood of type 2 diabetes in individuals from the Pima Indian population in Arizona using various machine learning models. The ultimate goal is to aid early detection and improve community health outcomes through data-driven insights.

---

## 📊 Dataset

- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Rows**: 768
- **Columns**: 9
- **Target**: `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## 🧪 Features

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## 🔍 Exploratory Data Analysis (EDA)

The dataset was explored and visualized using:

- **Histograms**: To understand feature distributions
- **Boxplots**: To detect outliers
- **Correlation Heatmap**: To identify relationships between features
- **Pairplot (Seaborn)**: To visually compare feature combinations across the outcome classes
- **Missing Value Checks**: Zero values in health metrics like glucose, insulin, etc., were flagged and handled
- **Standardization**: Features were scaled using `StandardScaler`

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost

---

## 🎯 Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- ROC Curve
- AUC (Area Under Curve)

---

## 🏆 Best Model & Results

The ROC curves compared model performance on test data:

| Model                  | AUC  |
|------------------------|------|
| **Naive Bayes**        | **0.83** |
| Random Forest          | 0.82 |
| Logistic Regression    | 0.81 |
| Support Vector Machine | 0.81 |
| Decision Tree          | 0.72 |

🔹 **Naive Bayes** achieved the highest AUC, making it the best-performing model in this case.

---

## 📁 Repository Contents

- `pima_diabetes.ipynb` – Main Jupyter notebook with full code, plots, and analysis
- `pima_diabetes.pptx` – Final project presentation
- `README.md` – Project overview

---

## 🧠 Key Insights

- Glucose, BMI, and Age were strong predictors of diabetes.
- EDA helped identify key trends, missing values, and model inputs.
- Even simple models like **Naive Bayes** can outperform complex models with the right preprocessing.

---

## ✍️ Author

- [Rini Chhabra]  
- Data Analyst