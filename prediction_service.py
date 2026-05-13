# services/prediction_service.py
"""
Сервисный слой для прогнозов и SHAP-интерпретаций.
"""
import pandas as pd
import numpy as np
from interpreter import ModelInterpreter
from causal_model import CausalModelTrainer, BASELINE_ACTION

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


def resolve_outcome_columns(available_outcomes, horizon_key):
    """Возвращает словарь col_name для выбранного горизонта с fallback на 2y."""
    oc = {}
    for metric in METRICS:
        col = f'{metric}_{horizon_key}'
        if col in available_outcomes:
            oc[metric] = col
        else:
            fallback = f'{metric}_2y'
            oc[metric] = fallback if fallback in available_outcomes else None
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


def get_shap_explanation(trainer, user_X_proc, action, outcome='salary_2y'):
    """Возвращает объект SHAP-значений для одного пользователя и действия."""
    if action not in trainer.models or outcome not in trainer.models[action]:
        return None
    model = trainer.models[action][outcome]
    feature_names = trainer.feature_names_after_ohe
    # Используем KernelExplainer через функцию const_marginal_effect
    # (возвращает 1D массив для бинарного лечения)
    def predict_fn(X):
        return model.const_marginal_effect(X).ravel()
    # background: берём средний профиль (или несколько строк) из обученных данных
    # можно использовать среднее значение X (но в виде массива)
    bg = user_X_proc.values[:1]  # простая заглушка, лучше сохранённый образец
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(user_X_proc.values)
    return shap.Explanation(values=shap_values, base_values=explainer.expected_value,
                            data=user_X_proc.values, feature_names=feature_names)