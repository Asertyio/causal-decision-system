# causal_model.py
"""Каузальная модель карьерного советника.

Улучшения v2:
- Отдельная бинарная CausalForestDML на каждое действие vs baseline (п. 2)
- Исключение current_salary / job_satisfaction / work_life_balance из ковариат (п. 3)
- Добавлены interaction features age_x_skills, exp_x_edu (п. 6)
- Категория уверенности (высокая/средняя/низкая) по ширине ДИ (п. 5)
- Кэширование модели через joblib (п. 10)
"""
import numpy as np
import pandas as pd
import joblib
import os
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Full action list — must match data_synthetic.py and app.py exactly
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

# ── УЛУЧШЕНИЕ 3: Baseline-значения исключены из признаков модели ──
# Эти колонки используются только в outcome generation, не как ковариаты.
_LEAKY_COLS = {'current_salary', 'job_satisfaction', 'work_life_balance'}


def get_default_feature_cols():
    """Ковариаты модели — без утечки данных."""
    return [
        'age', 'gender', 'region', 'education_years', 'has_master', 'has_phd',
        'has_certificate', 'total_experience', 'industry_experience',
        'current_job_tenure', 'num_previous_jobs', 'current_industry',
        'job_level', 'skills_count',
        'experience_gap', 'job_stability',
        # Interaction features (п. 6)
        'age_x_skills', 'exp_x_edu',
    ]


def confidence_label(ci_width: float, outcome: str) -> str:
    """Категория уверенности прогноза по ширине ДИ (п. 5).

    Пороги подобраны отдельно для каждого типа исхода.
    """
    thresholds = {
        'salary': (20, 50),       # тыс. руб.
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
    # fallback
    return '🟡 Средняя'


class CausalModelTrainer:
    """Обучает отдельный CausalForestDML для каждого действия (бинарный T).

    Улучшение п. 2: вместо одного многоклассового леса — по одному бинарному
    лесу на каждое действие vs baseline.  Это даёт более чистые CATE и
    позволяет параллельно обучать независимые модели.
    """

    def __init__(self, random_state=42, n_estimators=32, fast_mode=True, high_precision=False):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.fast_mode = fast_mode
        self.high_precision = high_precision
        # models[action][outcome] = fitted CausalForestDML
        self.models: dict = {}
        self.is_fitted = False
        self.treatment_names = ALL_ACTIONS
        self.available_treatment_names: list = []
        self.feature_names: list = []
        self.feature_names_after_ohe: list = []
        self.baseline_means: dict = {}

    def _get_model_params(self):
        if self.high_precision:
            return {
                "cv": 3,
                "max_depth": 15,
                "min_samples_leaf": 10,
                "n_estimators": 150,
                "n_estimators_first": 100,
            }
        else:
            return {
                "cv": 2,
                "max_depth": 4,
                "min_samples_leaf": 30,
                "n_estimators": 20,
                "n_estimators_first": 20,
            }

    def fit(self, data: pd.DataFrame, outcome_cols: list, treatment_col: str = 'treatment',
            feature_cols: list = None):
        """Train one binary CausalForestDML per (action, outcome) pair.

        Для каждого действия выбираются строки: treated (это действие) + control
        (baseline).  Обучается бинарная DML T∈{0,1}.
        """
        if feature_cols is None:
            feature_cols = get_default_feature_cols()
        # Filter out any leaky columns that may have slipped in
        feature_cols = [c for c in feature_cols if c not in _LEAKY_COLS and c in data.columns]
        self.feature_names = feature_cols

        X_full = data[feature_cols].copy()
        X_full = pd.get_dummies(X_full, drop_first=True)
        self.feature_names_after_ohe = X_full.columns.tolist()

        params = self._get_model_params()
        baseline = BASELINE_ACTION

        # Baseline subset
        baseline_mask = data[treatment_col] == baseline
        X_base = X_full[baseline_mask].values

        for outcome in outcome_cols:
            Y_base = data.loc[baseline_mask, outcome].values.astype(float)
            self.baseline_means[outcome] = float(Y_base.mean()) if len(Y_base) > 0 else 0.0

        available = []
        for action in ALL_ACTIONS:
            if action == baseline:
                # Baseline stored separately; still counts as available
                available.append(action)
                continue
            treated_mask = data[treatment_col] == action
            if treated_mask.sum() < 5:
                continue  # too few samples — skip

            combined_mask = baseline_mask | treated_mask
            X_comb = X_full[combined_mask].values
            T_comb = treated_mask[combined_mask].values.astype(int)

            self.models[action] = {}
            for outcome in outcome_cols:
                Y_comb = data.loc[combined_mask, outcome].values.astype(float)

                model = CausalForestDML(
                    model_y=RandomForestRegressor(
                        n_estimators=params['n_estimators_first'],
                        random_state=self.random_state,
                        min_samples_leaf=params['min_samples_leaf'],
                        n_jobs=-1),
                    model_t=RandomForestClassifier(
                        n_estimators=params['n_estimators_first'],
                        random_state=self.random_state,
                        min_samples_leaf=params['min_samples_leaf'],
                        n_jobs=-1),
                    discrete_treatment=True,
                    cv=params['cv'],
                    honest=True,
                    n_estimators=params['n_estimators'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_depth=params['max_depth'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(Y=Y_comb, T=T_comb, X=X_comb, W=None)
                self.models[action][outcome] = model

            available.append(action)

        self.available_treatment_names = [a for a in ALL_ACTIONS if a in available]
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_absolute(self, X_df: pd.DataFrame, actions: list = None,
                         alpha: float = 0.05) -> dict:
        """Return predicted absolute outcome values for each action.

        Returns
        -------
        dict keyed by outcome name, each value a dict with
        'effect', 'lower', 'upper' — DataFrames (n_rows × n_actions).
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_proc = pd.get_dummies(X_df, drop_first=True)
        for col in self.feature_names_after_ohe:
            if col not in X_proc.columns:
                X_proc[col] = 0
        X_proc = X_proc[self.feature_names_after_ohe].reset_index(drop=True)
        X_arr = X_proc.values

        if actions is None:
            actions = self.available_treatment_names

        unknown = [a for a in actions if a not in self.available_treatment_names]
        if unknown:
            raise ValueError(
                f"Actions not in trained model: {unknown}\n"
                f"Available: {self.available_treatment_names}"
            )

        outcome_cols = list(next(iter(self.models.values())).keys()) if self.models else []
        # Also collect outcomes from baseline_means
        all_outcomes = list(self.baseline_means.keys())

        results = {outcome: {'effect': {}, 'lower': {}, 'upper': {}} for outcome in all_outcomes}

        for outcome in all_outcomes:
            baseline_mean = self.baseline_means[outcome]
            for act in actions:
                n = len(X_arr)
                if act == BASELINE_ACTION or act not in self.models:
                    # Baseline — CATE = 0 by definition
                    results[outcome]['effect'][act] = np.full(n, baseline_mean)
                    results[outcome]['lower'][act] = np.full(n, baseline_mean)
                    results[outcome]['upper'][act] = np.full(n, baseline_mean)
                else:
                    model = self.models[act][outcome]
                    # Binary DML: T=1 is the action; CATE shape (n,) or (n,1)
                    cate = model.const_marginal_effect(X_arr)
                    cate_lo, cate_hi = model.const_marginal_effect_interval(X_arr, alpha=alpha)
                    cate = np.squeeze(cate)
                    cate_lo = np.squeeze(cate_lo)
                    cate_hi = np.squeeze(cate_hi)
                    results[outcome]['effect'][act] = baseline_mean + cate
                    results[outcome]['lower'][act] = baseline_mean + cate_lo
                    results[outcome]['upper'][act] = baseline_mean + cate_hi

        # Convert inner dicts to DataFrames
        for outcome in all_outcomes:
            for key in ('effect', 'lower', 'upper'):
                results[outcome][key] = pd.DataFrame(results[outcome][key]).reset_index(drop=True)

        return results

    def predict_with_confidence(self, X_df: pd.DataFrame, actions: list = None,
                                alpha: float = 0.05) -> dict:
        """predict_absolute + confidence labels per (action, outcome) pair."""
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
    # Persistence (п. 10)
    # ------------------------------------------------------------------

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'treatment_names': self.treatment_names,
            'available_treatment_names': self.available_treatment_names,
            'feature_names': self.feature_names,
            'feature_names_after_ohe': self.feature_names_after_ohe,
            'baseline_means': self.baseline_means,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.models = data['models']
        self.treatment_names = data['treatment_names']
        self.available_treatment_names = data['available_treatment_names']
        self.feature_names = data['feature_names']
        self.feature_names_after_ohe = data['feature_names_after_ohe']
        self.baseline_means = data.get('baseline_means', {})
        self.is_fitted = True
        print(f"Model loaded from {filepath}")