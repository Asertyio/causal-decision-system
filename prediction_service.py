# services/prediction_service.py
"""
Сервисный слой для прогнозов и SHAP-интерпретаций.
"""
import pandas as pd
import numpy as np
import shap

# Маппинг горизонтов
HORIZON_MAP = {
    "6 месяцев": "6m",
    "1 год": "1y",
    "2 года": "2y"
}

# Исходы
METRICS = ['salary', 'satisfaction', 'promoted', 'wlb']


def get_all_outcome_columns():
    """Все возможные колонки исходов для обучения."""
    cols = []
    for m in METRICS:
        for h in ['6m', '1y', '2y']:
            cols.append(f'{m}_{h}')
    return cols


def resolve_outcome_columns(available_outcomes: list, horizon_key: str) -> dict:
    """Возвращает словарь col_name для выбранного горизонта с fallback на 2y.

    Raises
    ------
    ValueError если ни целевая колонка, ни fallback не найдены для какого-либо метрика.
    """
    oc = {}
    for metric in METRICS:
        col = f'{metric}_{horizon_key}'
        if col in available_outcomes:
            oc[metric] = col
        else:
            fallback = f'{metric}_2y'
            if fallback in available_outcomes:
                oc[metric] = fallback
            else:
                raise ValueError(
                    f"Ни '{col}', ни fallback '{fallback}' не найдены в available_outcomes. "
                    f"Доступные исходы: {available_outcomes}"
                )
    return oc


def prepare_user_features(age, gender, region, education_years, has_master, has_phd,
                          has_certificate, total_exp, industry_exp, current_tenure,
                          prev_jobs, industry, job_level, skills):
    """Формирует DataFrame с признаками пользователя."""
    return pd.DataFrame([{
        'age': age,
        'gender': gender,
        'region': region,
        'education_years': education_years,
        'has_master': has_master,
        'has_phd': has_phd,
        'has_certificate': int(has_certificate),
        'total_experience': total_exp,
        'industry_experience': industry_exp,
        'current_job_tenure': current_tenure,
        'num_previous_jobs': prev_jobs,
        'current_industry': industry,
        'job_level': job_level,
        'skills_count': skills,
        'experience_gap': total_exp - industry_exp,
        'job_stability': current_tenure / (total_exp + 1),
        'age_x_skills': age * skills / 100.0,
        'exp_x_edu': total_exp * education_years / 10.0,
    }])


def get_shap_explanation(trainer, user_X_proc, action, outcome='salary_2y',
                         background_X: np.ndarray = None):
    """Возвращает объект SHAP-значений для одного пользователя и действия.

    Parameters
    ----------
    trainer       : обученный CausalModelTrainer.
    user_X_proc   : pd.DataFrame — признаки пользователя после OHE (1 строка).
    action        : str — название действия.
    outcome       : str — название исхода, например 'salary_2y'.
    background_X  : np.ndarray — фоновая выборка для KernelExplainer.
                    Должна содержать репрезентативную выборку из обучающих данных
                    (рекомендуется 50–200 строк). Если None — используется
                    среднее значение признаков пользователя как заглушка
                    (низкое качество объяснений).

    Returns
    -------
    shap.Explanation или None если действие/исход недоступны.
    """
    if action not in trainer.models or outcome not in trainer.models[action]:
        return None
    model = trainer.models[action][outcome]
    feature_names = trainer.feature_names_after_ohe

    # atleast_1d критичен: ravel() на одной строке даёт скаляр,
    # что ломает KernelExplainer
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return np.atleast_1d(model.const_marginal_effect(X).ravel())

    # Используем переданный фон или среднее пользователя как заглушку
    if background_X is not None:
        bg = background_X
    else:
        bg = user_X_proc.values  # заглушка низкого качества — передайте background_X

    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(user_X_proc.values)
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=user_X_proc.values,
        feature_names=feature_names,
    )