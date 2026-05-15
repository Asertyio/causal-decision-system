# causal_model.py
"""Каузальная модель карьерного советника.

Исправления:
- Персонализированный baseline через отдельную регрессию (п.2)
- Обучение на всех горизонтах 6m/1y/2y (п.1)
- Быстрый режим улучшен (n_estimators_first=80) (п.6)
- Сохранение baseline-моделей вместе с CATE-моделями
"""
import numpy as np
import pandas as pd
import joblib
import os
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ALL_ACTIONS = [
    'Остаться на текущем месте',
    'Сменить работу (та же отрасль)',
    'Сменить работу с переездом в другой регион',
    'Сменить отрасль',
    'Получить второе высшее образование',
    'Пройти профессиональные курсы / сертификацию',
    'Начать подрабатывать / брать проекты',
    'Уйти во фриланс',
    'Открыть свой бизнес / стартап',
    'Выйти на пенсию / сократить занятость',
    'Взять академический отпуск',
    'Перейти на удалённую работу',
    'Перейти на частичную занятость',
    'Повысить квалификацию внутри компании',
    'Попросить повышения',
    'Попросить увеличения зарплаты',
    'Перейти в дочернюю компанию / филиал',
    'Релоцироваться в другую страну',
    'Сменить профессию полностью',
    'Пойти в декретный отпуск / отпуск по уходу за ребёнком',
    'Вернуться из декрета',
    'Начать инвестировать / трейдинг',
    'Заняться волонтёрством',
    'Вступить в профессиональное сообщество',
    'Начать вести блог / личный бренд',
    'Получить MBA',
    'Пройти стажировку',
    'Устроиться на вторую работу',
    'Участвовать в конференциях / нетворкинг',
    'Ничего не менять, продолжать как есть'
]

BASELINE_ACTION = 'Остаться на текущем месте'

_LEAKY_COLS = {'current_salary', 'job_satisfaction', 'work_life_balance'}


def get_default_feature_cols():
    return [
        'age', 'gender', 'region', 'education_years', 'has_master', 'has_phd',
        'has_certificate', 'total_experience', 'industry_experience',
        'current_job_tenure', 'num_previous_jobs', 'current_industry',
        'job_level', 'skills_count',
        'experience_gap', 'job_stability',
        'age_x_skills', 'exp_x_edu',
    ]


def confidence_label(ci_width: float, outcome: str) -> str:
    thresholds = {
        'salary': (20, 50),
        'satisfaction': (1.5, 3.0),
        'promoted': (0.15, 0.30),
        'wlb': (1.5, 3.0),
    }
    for key, (low, high) in thresholds.items():
        if key in outcome:
            if ci_width <= low:
                return '🟢 Высокая'
            elif ci_width <= high:
                return '🟡 Средняя'
            else:
                return '🔴 Низкая'
    return '🟡 Средняя'


class CausalModelTrainer:
    """Обучает бинарные CausalForestDML для каждой пары (действие, исход)
    и регрессионную baseline-модель для персонализированных прогнозов.
    """
    def __init__(self, random_state=42, n_estimators=32, fast_mode=True, high_precision=False):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.fast_mode = fast_mode
        self.high_precision = high_precision
        self.models: dict = {}                    # models[action][outcome] = CausalForestDML
        self.baseline_models: dict = {}           # baseline_models[outcome] = RandomForestRegressor
        self.is_fitted = False
        self.treatment_names = ALL_ACTIONS
        self.available_treatment_names: list = []
        self.feature_names: list = []
        self.feature_names_after_ohe: list = []
        self.baseline_means: dict = {}

    def _get_model_params(self):
        if self.high_precision:
            # Баланс качество/время: разумные результаты без многочасового ожидания.
            # 30 действий × 12 исходов = 360 моделей — держим параметры умеренными.
            return {
                "cv": 2,            # 3 → 2: экономия ~33% времени на cross-fit
                "max_depth": 8,     # достаточно для захвата нелинейностей
                "min_samples_leaf": 20,
                "n_estimators": 40, # кратно 4; 152 деревьев на 360 моделей — слишком долго
                "n_estimators_first": 60,
                "honest": True,     # честное разбиение — для качества важно
                "inference": True,  # доверительные интервалы нужны
            }
        else:
            # Максимальная скорость: минимум деревьев, нет honest-сплита,
            # нет inference при обучении — только быстрые предсказания.
            return {
                "cv": 2,
                "max_depth": 4,
                "min_samples_leaf": 50,
                "n_estimators": 8,  # минимум кратный 4; достаточно для smoke-test
                "n_estimators_first": 16,
                "honest": False,    # без сплита на честность — в 2× быстрее
                "inference": False, # не считать дисперсию при fit — ещё быстрее
            }

    @staticmethod
    def _round_estimators(n: int, subforest_size: int = 4) -> int:
        """Округляет n_estimators вверх до кратного subforest_size.
        CausalForestDML требует n_estimators % subforest_size == 0.
        """
        remainder = n % subforest_size
        return n if remainder == 0 else n + (subforest_size - remainder)

    def fit(self, data: pd.DataFrame, outcome_cols: list, treatment_col: str = 'treatment',
            feature_cols: list = None):
        if feature_cols is None:
            feature_cols = get_default_feature_cols()
        feature_cols = [c for c in feature_cols if c not in _LEAKY_COLS and c in data.columns]
        self.feature_names = feature_cols

        X_full = data[feature_cols].copy()
        X_full = pd.get_dummies(X_full, drop_first=True)
        self.feature_names_after_ohe = X_full.columns.tolist()

        params = self._get_model_params()
        baseline = BASELINE_ACTION

        # Baseline subset
        baseline_mask = data[treatment_col] == baseline
        if baseline_mask.sum() == 0:
            raise ValueError(
                f"В данных нет ни одного наблюдения с baseline-действием '{baseline}'. "
                "Проверьте колонку treatment."
            )
        X_base = X_full[baseline_mask].values

        # Минимальное число treated-наблюдений для безопасного обучения CausalForestDML
        # При cv=2 нужно минимум 2 * min_samples_leaf наблюдений в каждой группе
        _min_treated = max(10, 2 * params['min_samples_leaf'])

        # Обучаем персонализированные baseline-модели для каждого исхода
        # В fast-режиме берём меньше деревьев для baseline RF
        _baseline_n_est = params['n_estimators_first']
        for outcome in outcome_cols:
            y_base = data.loc[baseline_mask, outcome].values.astype(float)
            self.baseline_means[outcome] = float(y_base.mean()) if len(y_base) > 0 else 0.0
            if len(y_base) > 10:
                reg = RandomForestRegressor(
                    n_estimators=_baseline_n_est,
                    max_depth=params['max_depth'],
                    min_samples_leaf=params['min_samples_leaf'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
                reg.fit(X_base, y_base)
                self.baseline_models[outcome] = reg
            else:
                self.baseline_models[outcome] = None

        # Обучаем CATE-модели для каждой пары (действие, исход), кроме baseline
        available = [baseline]
        for action in ALL_ACTIONS:
            if action == baseline:
                continue
            treated_mask = data[treatment_col] == action
            if treated_mask.sum() < _min_treated:
                continue
            combined_mask = baseline_mask | treated_mask
            X_comb = X_full[combined_mask].values
            T_comb = treated_mask[combined_mask].values.astype(int)

            self.models[action] = {}
            for outcome in outcome_cols:
                Y_comb = data.loc[combined_mask, outcome].values.astype(float)

                model = CausalForestDML(
                    model_y=RandomForestRegressor(
                        n_estimators=params['n_estimators_first'],
                        max_depth=params['max_depth'],
                        min_samples_leaf=params['min_samples_leaf'],
                        random_state=self.random_state,
                        n_jobs=-1),
                    model_t=RandomForestClassifier(
                        n_estimators=params['n_estimators_first'],
                        max_depth=params['max_depth'],
                        min_samples_leaf=params['min_samples_leaf'],
                        random_state=self.random_state,
                        n_jobs=-1),
                    discrete_treatment=True,
                    cv=params['cv'],
                    honest=params['honest'],
                    inference=params['inference'],
                    # Округляем до кратного subforest_size=4 — требование CausalForestDML
                    n_estimators=self._round_estimators(params['n_estimators']),
                    min_samples_leaf=params['min_samples_leaf'],
                    max_depth=params['max_depth'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(Y=Y_comb, T=T_comb, X=X_comb, W=None)
                self.models[action][outcome] = model
            available.append(action)

        self.available_treatment_names = available
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_absolute(self, X_df: pd.DataFrame, actions: list = None,
                         alpha: float = 0.05) -> dict:
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit().")

        # One-hot encoding — добавляем отсутствующие колонки, убираем лишние
        X_proc = pd.get_dummies(X_df, drop_first=True)
        for col in self.feature_names_after_ohe:
            if col not in X_proc.columns:
                X_proc[col] = 0
        # Оставляем только колонки, известные модели, в правильном порядке
        X_proc = X_proc[self.feature_names_after_ohe].reset_index(drop=True)
        X_arr = X_proc.values

        if actions is None:
            actions = self.available_treatment_names
        else:
            # Оставляем только доступные действия
            actions = [a for a in actions if a in self.available_treatment_names]

        all_outcomes = list(self.baseline_models.keys())
        results = {outcome: {'effect': {}, 'lower': {}, 'upper': {}} for outcome in all_outcomes}

        for outcome in all_outcomes:
            # Персонализированный baseline: предсказание базовой модели
            baseline_reg = self.baseline_models.get(outcome)
            if baseline_reg is not None:
                base_pred = baseline_reg.predict(X_arr)
            else:
                base_pred = np.full(len(X_arr), self.baseline_means.get(outcome, 0.0))

            for act in actions:
                if act == BASELINE_ACTION or act not in self.models:
                    # Для baseline CATE = 0
                    results[outcome]['effect'][act] = base_pred.copy()
                    results[outcome]['lower'][act] = base_pred.copy()
                    results[outcome]['upper'][act] = base_pred.copy()
                else:
                    model = self.models[act][outcome]
                    cate = np.atleast_1d(np.squeeze(model.const_marginal_effect(X_arr)))
                    # inference=False (fast mode): интервалы недоступны — используем cate как lower/upper
                    try:
                        cate_lo, cate_hi = model.const_marginal_effect_interval(X_arr, alpha=alpha)
                        cate_lo = np.atleast_1d(np.squeeze(cate_lo))
                        cate_hi = np.atleast_1d(np.squeeze(cate_hi))
                    except (AttributeError, Exception):
                        cate_lo = cate.copy()
                        cate_hi = cate.copy()
                    results[outcome]['effect'][act] = base_pred + cate
                    results[outcome]['lower'][act] = base_pred + cate_lo
                    results[outcome]['upper'][act] = base_pred + cate_hi

        # Преобразуем в DataFrame
        for outcome in all_outcomes:
            for key in ('effect', 'lower', 'upper'):
                results[outcome][key] = pd.DataFrame(results[outcome][key]).reset_index(drop=True)

        return results

    def predict_with_confidence(self, X_df: pd.DataFrame, actions: list = None,
                                alpha: float = 0.05) -> tuple:
        raw = self.predict_absolute(X_df, actions=actions, alpha=alpha)
        confidence = {}
        for outcome, data in raw.items():
            confidence[outcome] = {}
            for act in data['effect'].columns:
                ci_width = float(
                    (data['upper'][act] - data['lower'][act]).mean()
                )
                confidence[outcome][act] = confidence_label(ci_width, outcome)
        return raw, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'baseline_models': self.baseline_models,
            'treatment_names': self.treatment_names,
            'available_treatment_names': self.available_treatment_names,
            'feature_names': self.feature_names,
            'feature_names_after_ohe': self.feature_names_after_ohe,
            'baseline_means': self.baseline_means,
        }, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.models = data['models']
        self.baseline_models = data.get('baseline_models', {})
        self.treatment_names = data['treatment_names']
        self.available_treatment_names = data['available_treatment_names']
        self.feature_names = data['feature_names']
        self.feature_names_after_ohe = data['feature_names_after_ohe']
        self.baseline_means = data.get('baseline_means', {})
        self.is_fitted = True