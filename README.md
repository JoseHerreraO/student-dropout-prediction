# Predicting Student Dropout and Academic Success

Final project for the **Machine Learning** course in the **B.Sc. in Mathematical Engineering and Artificial Intelligence** at **Universidad Pontificia Comillas (ICAI)**.

This repository studies student dropout, persistence, and academic success in higher education using demographic, socioeconomic, administrative, and academic variables. The project is organized around three complementary tasks:

1. **Multiclass classification** of the final student status (`Dropout`, `Enrolled`, `Graduate`)
2. **Regression** of the second-semester average grade
3. **Unsupervised learning** through PCA and clustering

The full write-up is available in the project report.

---

## Project structure

```text
project-root/
├── src/
│   ├── 01_analysis.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_unsupervised_learning.ipynb
│   └── utils_project.py
├── data/
│   └── rendimiento_estudiantes.csv
├── Report.pdf
├── requirements.txt
└── README.md
```

> The notebooks expect the dataset at `../data/rendimiento_estudiantes.csv`.
> If your folder structure is different, update the path accordingly.

---

## Dataset

The dataset contains information about students in higher education, including:

- demographic and personal variables
- access and previous academic background
- socioeconomic and administrative variables
- macroeconomic context variables
- first-semester academic performance
- second-semester academic performance

The main classification target is:

- `objetivo` -> final academic status (`abandono`, `matriculado`, `graduado`)

The regression target is:

- `nota_media_2sem` -> second-semester average grade

**Source:** Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021).
*Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository.

---

## Methodology

### 1. Exploratory data analysis

The analysis notebook inspects:

- dataset shape and variable types
- missing values and duplicates
- class balance
- continuous, discrete, and categorical variables
- distributions and boxplots
- potential outliers using the IQR rule

### 2. Classification

Two prediction scenarios are considered:

- **Early scenario**: only pre-university and administrative/context variables
- **First-semester scenario**: early variables plus first-semester academic performance

Models compared:

- **Multinomial Logistic Regression**
- **Random Forest Classifier**

Main evaluation metric:

- **Macro-F1**, complemented with accuracy and balanced accuracy

### 3. Regression

Target:

- `nota_media_2sem`

Models compared:

- **Ridge Regression**
- **Random Forest Regressor**

Evaluation metrics:

- **MAE**
- **RMSE**
- **R²**

To avoid **data leakage**, the regression task excludes second-semester variables that are contemporaneous with or too close to the target.

### 4. Unsupervised learning

A smaller interpretable subset of numerical variables is standardized and analyzed with:

- **PCA**
- **K-Means clustering**
- **Hierarchical clustering (Ward linkage)**

Cluster quality is explored using:

- **elbow method**
- **silhouette score**

---

## Results at a glance

| Task | Best model | Key metric |
|---|---|---|
| Classification (first-semester scenario) | Logistic Regression | Test macro-F1 = 0.662 |
| Regression (`nota_media_2sem`) | Ridge | Test R² = 0.742, RMSE = 2.644 |
| Clustering (PCA + K-Means, k=3) | K-Means | Silhouette = 0.301 |

Full metrics, confusion matrices and diagnostic plots are available in the
corresponding notebooks and in `Report.pdf`.

## Main findings

- First-semester academic performance is the strongest signal across all tasks.
- Classification improves substantially once first-semester variables are included.
- In regression, prior academic performance explains a large share of second-semester grades.
- In unsupervised learning, clusters are meaningfully associated with different academic trajectories, especially dropout risk.

---

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the notebooks locally:

```bash
jupyter notebook
```

---

## Reproducibility notes

- A fixed random seed (`42`) is used throughout the project whenever applicable.
- Shared preprocessing utilities are implemented in `utils_project.py` and imported directly in every notebook (including the outlier-detection helper used in the EDA notebook), so there is a single source of truth for these functions.
- The code uses `Pipeline` and `ColumnTransformer` to keep preprocessing and modeling together and reduce leakage risk.

---

## Author

**José Herrera Ortiz**

Machine Learning - Universidad Pontificia Comillas (ICAI)