# data_loader.py
"""Data loading utilities.

For the career advisor app the main entry point is generate_career_data()
from data_synthetic.  The HR-dataset loader is retained for experimental use.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


# ---------------------------------------------------------------------------
# IBM HR attrition dataset (legacy / experimental)
# ---------------------------------------------------------------------------

def load_hr_data() -> pd.DataFrame:
    try:
        import kagglehub
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        file_path = os.path.join(path, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        df = pd.read_csv(file_path)
        print("HR dataset loaded via kagglehub.")
    except Exception:
        url = ("https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
               "/master/data/emp_attrition.csv")
        df = pd.read_csv(url)
        print("HR dataset loaded from GitHub.")
    return df


def preprocess_hr_data(df: pd.DataFrame):
    df = df.copy()
    df['T'] = (df['YearsSinceLastPromotion'] <= 1).astype(int)
    Y = df['MonthlyIncome'].values.astype(float)

    X_cols = ['Age', 'Education', 'JobSatisfaction', 'StockOptionLevel',
              'YearsAtCompany', 'YearsInCurrentRole']
    # Оставляем только колонки, которые реально присутствуют в датасете
    # (kagglehub и GitHub-версии CSV могут отличаться именованием)
    X_cols = [c for c in X_cols if c in df.columns]
    if not X_cols:
        raise ValueError(
            "HR-датасет не содержит ожидаемых признаков. "
            "Проверьте источник данных и именование колонок."
        )
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df = pd.get_dummies(df, columns=['Department', 'JobRole'], drop_first=True)

    W_ohe_cols = [c for c in df.columns if c.startswith('Department_') or c.startswith('JobRole_')]
    W_cols_final = ['PerformanceRating', 'WorkLifeBalance', 'OverTime'] + W_ohe_cols

    X = df[X_cols].values.astype(float)
    W = df[W_cols_final].values.astype(float)
    T = df['T'].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_W = StandardScaler()
    W_scaled = scaler_W.fit_transform(W)

    return X_scaled, W_scaled, T, Y, X_cols, W_cols_final


# ---------------------------------------------------------------------------
# Main entry point used by the career advisor app
# ---------------------------------------------------------------------------

def load_and_prepare_data(data_source: str = 'synthetic', n_synthetic: int = 5000):
    """Load and return a prepared dataset.

    Parameters
    ----------
    data_source : 'synthetic' (default) or 'hr'
    n_synthetic : number of rows to generate when data_source='synthetic'

    Returns
    -------
    'synthetic' : tuple[pd.DataFrame, list[str], list[str]]
        (df, feature_cols, outcome_cols)
        feature_cols — признаки без утечки (leaky cols исключены)
        outcome_cols — все горизонты: 6m, 1y, 2y для salary/satisfaction/promoted/wlb
    'hr' : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, None, list[str], list[str]]
        (X_scaled, W_scaled, T, Y, None, feature_names_X, feature_names_W)
    """
    if data_source == 'synthetic':
        # BUG FIX: was calling non-existent generate_synthetic_data(); now uses generate_career_data()
        from data_synthetic import generate_career_data
        from prediction_service import get_all_outcome_columns
        df = generate_career_data(n=n_synthetic)
        # current_salary / job_satisfaction / work_life_balance исключены:
        # они внесены в _LEAKY_COLS в causal_model.py и молча выбрасываются при обучении.
        feature_cols = [
            'age', 'gender', 'region', 'education_years', 'has_master', 'has_phd',
            'has_certificate', 'total_experience', 'industry_experience',
            'current_job_tenure', 'num_previous_jobs', 'current_industry',
            'job_level', 'skills_count', 'experience_gap', 'job_stability',
            'age_x_skills', 'exp_x_edu',
        ]
        # Все горизонты 6m/1y/2y — иначе модель обучается только на 2y
        outcome_cols = get_all_outcome_columns()
        return df, feature_cols, outcome_cols

    elif data_source == 'hr':
        df = load_hr_data()
        X, W, T, Y, feature_names_X, feature_names_W = preprocess_hr_data(df)
        return X, W, T, Y, None, feature_names_X, feature_names_W

    else:
        raise ValueError("data_source must be 'synthetic' or 'hr'")