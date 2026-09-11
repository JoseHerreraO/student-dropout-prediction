# José Herrera Ortiz - Universidad Pontificia Comillas ICAI

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

# 1. DATA LOADING
def load_data(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path, sep=";")
    return df


# 2. PROJECT VARIABLES
def keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Filter a list of column names, keeping only those present in df."""
    return [col for col in columns if col in df.columns]


def get_project_variables(df: pd.DataFrame) -> dict:
    """
    Define and group every feature used across the project's three tasks.

    Groups features into demographic, access, socioeconomic, macroeconomic,
    first-semester and second-semester blocks, then derives the specific
    feature sets required by the classification, regression and
    unsupervised-learning tasks (see README.md for a description of each
    scenario). Every list is filtered through `keep_existing_columns` so the
    function degrades gracefully if a column is missing from `df`.

    Returns:
        A dict mapping each variable-group name (e.g. "classification_features_early",
        "regression_features", "unsupervised_features") to the corresponding
        list of column names, plus the two target column names.
    """
    classification_target = "objetivo"
    regression_target = "nota_media_2sem"

    demographic_features = [
        "estado_civil",
        "nacionalidad",
        "genero",
        "edad_al_matricularse",
        "desplazado",
        "internacional",
        "necesidades_educativas_especiales",
    ]

    access_features = [
        "modo_solicitud",
        "orden_solicitud",
        "curso",
        "asistencia_diurna_vespertina",
        "cualificacion_previa",
        "nota_cualificacion_previa",
        "nota_admision",
        "cualificacion_madre",
        "cualificacion_padre",
        "ocupacion_madre",
        "ocupacion_padre",
    ]

    socioeconomic_features = [
        "deudor",
        "matricula_al_dia",
        "becado",
    ]

    macro_features = [
        "tasa_desempleo",
        "tasa_inflacion",
        "pib",
    ]

    first_sem_features = [
        "asignaturas_1sem_convalidadas",
        "asignaturas_1sem_matriculadas",
        "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",
        "nota_media_1sem",
        "asignaturas_1sem_sin_evaluacion",
    ]

    second_sem_features = [
        "asignaturas_2sem_convalidadas",
        "asignaturas_2sem_matriculadas",
        "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",
        "nota_media_2sem",
        "asignaturas_2sem_sin_evaluacion",
    ]

    continuous_features = [
        "nota_cualificacion_previa",
        "nota_admision",
        "nota_media_1sem",
        "nota_media_2sem",
        "tasa_desempleo",
        "tasa_inflacion",
        "pib",
    ]

    discrete_features = [
        "orden_solicitud",
        "edad_al_matricularse",
        "asignaturas_1sem_convalidadas",
        "asignaturas_1sem_matriculadas",
        "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",
        "asignaturas_1sem_sin_evaluacion",
        "asignaturas_2sem_convalidadas",
        "asignaturas_2sem_matriculadas",
        "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",
        "asignaturas_2sem_sin_evaluacion",
    ]

    initial_features = (
        demographic_features
        + access_features
        + socioeconomic_features
        + macro_features
    )

    classification_features_early = initial_features.copy()
    classification_features_first_sem = initial_features + first_sem_features
    regression_features = initial_features + first_sem_features

    regression_excluded_features = [
        "asignaturas_2sem_convalidadas",
        "asignaturas_2sem_matriculadas",
        "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",
        "asignaturas_2sem_sin_evaluacion",
    ]

    unsupervised_features = [
        "edad_al_matricularse",
        "nota_cualificacion_previa",
        "nota_admision",
        "asignaturas_1sem_matriculadas",
        "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",
        "nota_media_1sem",
        "asignaturas_1sem_sin_evaluacion",
        "tasa_desempleo",
        "tasa_inflacion",
        "pib",
    ]

    all_numeric_features = continuous_features + discrete_features
    categorical_features = [
        col for col in df.columns
        if col not in all_numeric_features + [classification_target]
    ]

    variables = {
        "classification_target": classification_target,
        "regression_target": regression_target,
        "demographic_features": keep_existing_columns(df, demographic_features),
        "access_features": keep_existing_columns(df, access_features),
        "socioeconomic_features": keep_existing_columns(df, socioeconomic_features),
        "macro_features": keep_existing_columns(df, macro_features),
        "first_sem_features": keep_existing_columns(df, first_sem_features),
        "second_sem_features": keep_existing_columns(df, second_sem_features),
        "initial_features": keep_existing_columns(df, initial_features),
        "continuous_features": keep_existing_columns(df, continuous_features),
        "discrete_features": keep_existing_columns(df, discrete_features),
        "categorical_features": keep_existing_columns(df, categorical_features),
        "classification_features_early": keep_existing_columns(df, classification_features_early),
        "classification_features_first_sem": keep_existing_columns(df, classification_features_first_sem),
        "regression_features": keep_existing_columns(df, regression_features),
        "regression_excluded_features": keep_existing_columns(df, regression_excluded_features),
        "unsupervised_features": keep_existing_columns(df, unsupervised_features),
    }

    return variables


def check_defined_variables(df: pd.DataFrame, variables_dict: dict) -> dict:
    """
    Returns a dictionary with missing columns for each variable list.
    Useful for debugging if any column name does not match.
    """
    missing = {}
    for key, value in variables_dict.items():
        if isinstance(value, list):
            missing[key] = [col for col in value if col not in df.columns]
    return missing


# 3. EDA
def detect_iqr_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    summary = []

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        pct_outliers = 100 * n_outliers / len(df)

        summary.append({
            "variable": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": int(n_outliers),
            "pct_outliers": round(pct_outliers, 2),
        })

    return pd.DataFrame(summary).sort_values("pct_outliers", ascending=False)


# 4. PREPROCESSING
def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a feature matrix's columns into (numeric, categorical) lists by dtype."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    return numeric_features, categorical_features


def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes and (optionally) scales numeric
    features, and imputes and one-hot encodes categorical features.

    Numeric features are median-imputed (robust to outliers) and, when
    `scale_numeric` is True, standardized. Categorical features are imputed
    with the most frequent category and one-hot encoded, ignoring unseen
    categories at inference time.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str]
) -> list[str]:
    """
    Recover the expanded feature names produced by a fitted preprocessor.

    Numeric feature names are unchanged by the pipeline, while categorical
    features are expanded into their one-hot encoded names. Useful for
    interpreting model coefficients or feature importances after fitting.
    """
    feature_names = []
    feature_names.extend(numeric_features)

    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_names = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(encoded_names)

    return feature_names


# 5. BUILD X / y MATRICES
def build_classification_data(
    df: pd.DataFrame,
    variables: dict,
    scenario: str = "early"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the (X, y) matrices for the classification task.

    Args:
        scenario: "early" uses only pre-university/administrative variables;
            "first_sem" additionally includes first-semester academic
            performance. See README.md for details on both scenarios.
    """
    if scenario == "early":
        X = df[variables["classification_features_early"]].copy()
    elif scenario == "first_sem":
        X = df[variables["classification_features_first_sem"]].copy()
    else:
        raise ValueError("scenario must be 'early' or 'first_sem'")

    y = df[variables["classification_target"]].copy()
    return X, y


def build_regression_data(
    df: pd.DataFrame,
    variables: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the (X, y) matrices for the regression task.

    Uses only `variables["regression_features"]`, which already excludes
    second-semester variables that would leak information about the target
    (`nota_media_2sem`).
    """
    X = df[variables["regression_features"]].copy()
    y = df[variables["regression_target"]].copy()
    return X, y


def build_unsupervised_data(
    df: pd.DataFrame,
    variables: dict
) -> pd.DataFrame:
    """Select the reduced, interpretable feature subset used for PCA and clustering."""
    return df[variables["unsupervised_features"]].copy()