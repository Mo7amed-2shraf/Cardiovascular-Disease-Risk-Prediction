# Cardiovascular Disease Risk Prediction

A project that predicts cardiovascular disease risk from patient clinical data, combining classical machine learning, deep learning, and unsupervised learning techniques — with an interactive Streamlit dashboard for exploration.

##  Overview

This project analyzes a dataset of 70,000 patient records to predict the presence of cardiovascular disease based on clinical and lifestyle features (age, blood pressure, cholesterol, glucose, BMI, smoking, alcohol use, physical activity, etc.).

The pipeline covers:
- **Exploratory Data Analysis (EDA)** — distribution analysis, outlier detection, correlation analysis
- **Feature Engineering** — BMI categories, BMI risk scores, blood pressure categories, interaction features
- **Supervised Learning** — Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning (GridSearchCV)
- **Deep Learning** — a custom DNN (5 hidden layers), a Wide & Deep architecture combining raw features with learned embeddings, and an Autoencoder for unsupervised feature learning
- **Statistical Model Comparison** — paired t-tests comparing deep learning vs. ensemble methods on cross-validated AUC scores
- **Unsupervised Learning** — KMeans clustering, DBSCAN outlier detection, Hierarchical clustering, PCA & t-SNE visualization
- **Deployment** — an interactive Streamlit dashboard for risk exploration and reporting

## Project Structure

```
.
├── Project_Graduation.ipynb   # Main analysis notebook (EDA → ML → DL → Clustering)
├── data/
│   └── cardio_train.csv       # Dataset (not included — see Dataset section below)
├── README.md
└── requirements.txt
```

## Dataset

The dataset used is the **Cardiovascular Disease dataset** — `cardio_train.csv`, a semicolon-separated (`;`) file with **70,000 patient records** and **13 columns**, no missing values.

| Column | Description |
|---|---|
| `id` | Patient identifier (dropped during preprocessing) |
| `age` | Age in days (converted to years in the notebook) — range ≈ 30–65 years |
| `gender` | 1 = female, 2 = male |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | Glucose level — 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | Smoking status (binary) |
| `alco` | Alcohol intake (binary) |
| `active` | Physical activity (binary) |
| `cardio` | **Target** — presence of cardiovascular disease (binary, ~50/50 balanced) |

> Place the dataset file at `data/cardio_train.csv` before running the notebook, or update the path in the data-loading cell.

##  Models

| Approach | Models |
|---|---|
| Supervised (Classical ML) | Logistic Regression, Random Forest, XGBoost |
| Deep Learning | DNN (256→128→64→32→16) with BatchNorm + Dropout, Wide & Deep with categorical embeddings |
| Unsupervised | Autoencoder (12→8→4→8→12), KMeans, DBSCAN, Hierarchical Clustering |
| Dimensionality Reduction | PCA, t-SNE |

Model performance is compared using AUC-ROC, and the deep learning models are statistically compared against the ensemble methods using paired t-tests on cross-validated scores.

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Project_Graduation.ipynb` in Jupyter or Google Colab (a GPU runtime is recommended for the deep learning sections).

### 4. Run the Streamlit dashboard
The dashboard code lives in the final section of the notebook. To run it as a standalone app, copy it into a separate `app.py` file, then:
```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn** (preprocessing, classical ML, clustering, metrics)
- **XGBoost**
- **TensorFlow / Keras** (DNN, Wide & Deep, Autoencoder)
- **SciPy** (statistical testing)
- **Matplotlib**, **Seaborn** (visualization)
- **Streamlit** (interactive dashboard)

##  Author

**Mohamed Ashraf**

 AI Specialization


##  License

This project is open for educational and portfolio purposes.
