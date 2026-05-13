# validator.py
"""Валидация каузальной модели карьерного советника.

Содержит:
- compute_metrics       — MSE / R² / Pearson для любых y_true / y_pred
- placebo_test          — перемешивание treatment, проверка что CATE ≈ 0
- calibration_test      — калибровка CATE по бинам, усреднённая по нескольким действиям
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from data_synthetic import generate_career_data
from causal_model import CausalModelTrainer, get_default_feature_cols, BASELINE_ACTION


# ──────────────────────────────────────────────────────────────────────
# Базовые метрики
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MSE, R² и корреляция Пирсона между истинными и предсказанными значениями.

    Parameters
    ----------
    y_true, y_pred : np.ndarray — одномерные или двумерные массивы одинакового размера.

    Returns
    -------
    dict с ключами 'MSE', 'R2', 'PearsonR'.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true.ravel(), y_pred.ravel())
    return {'MSE': float(mse), 'R2': float(r2), 'PearsonR': float(corr)}


# ──────────────────────────────────────────────────────────────────────
# Placebo-тест
# ──────────────────────────────────────────────────────────────────────

def placebo_test(
    data: pd.DataFrame,
    outcome_cols: list,
    feature_cols: list,
    treatment_col: str = 'treatment',
    n_permutations: int = 20,
    random_state: int = 0,
) -> dict:
    """Перемешивает treatment и проверяет, что средний CATE ≈ 0.

    Если модель корректна, перемешанные лейблы не должны давать
    систематического эффекта. Высокое среднее ATE после перемешивания
    указывает на утечку данных или переобучение.

    Parameters
    ----------
    data          : pd.DataFrame — обучающие данные.
    outcome_cols  : list — список исходов; тест использует первый.
    feature_cols  : list — ковариаты модели.
    treatment_col : str — колонка с treatment.
    n_permutations: int — число перестановок.
    random_state  : int — seed для воспроизводимости.

    Returns
    -------
    dict с ключами:
        'mean_ate'  — среднее ATE по перестановкам,
        'std_ate'   — стандартное отклонение ATE,
        'p_value'   — двусторонний p-value (доля случаев |ATE| ≥ наблюдаемого),
        'ate_list'  — список ATE по каждой перестановке.
    """
    rng = np.random.default_rng(random_state)
    ate_list = []
    first_outcome = outcome_cols[0]

    for perm_idx in range(n_permutations):
        perm_seed = int(rng.integers(0, 100_000))
        T_perm = data[treatment_col].sample(frac=1, random_state=perm_seed).values
        data_perm = data.copy()
        data_perm[treatment_col] = T_perm

        temp_trainer = CausalModelTrainer(random_state=perm_seed, fast_mode=True)
        temp_trainer.fit(
            data_perm,
            [first_outcome],
            treatment_col=treatment_col,
            feature_cols=feature_cols,
        )

        # Предсказываем на небольшой подвыборке для скорости
        X_sample = data_perm[feature_cols].head(200)
        preds = temp_trainer.predict_absolute(X_sample)
        ate = float(preds[first_outcome]['effect'].values.mean())
        ate_list.append(ate)
        print(f"  Placebo [{perm_idx + 1}/{n_permutations}]: ATE={ate:.3f}")

    ate_arr = np.array(ate_list)
    p_value = float(2 * min(np.mean(ate_arr >= 0), np.mean(ate_arr <= 0)))

    return {
        'mean_ate': float(np.mean(ate_arr)),
        'std_ate': float(np.std(ate_arr)),
        'p_value': p_value,
        'ate_list': ate_list,
    }


# ──────────────────────────────────────────────────────────────────────
# Тест калибровки
# ──────────────────────────────────────────────────────────────────────

def calibration_test(
    trainer: CausalModelTrainer,
    data_test: pd.DataFrame,
    outcome: str = 'salary_2y',
    treatment_col: str = 'treatment',
    n_bins: int = 5,
    n_actions: int = 5,
) -> dict:
    """Оценивает калибровку CATE: сравнивает предсказанный и эмпирический эффект.

    Алгоритм для каждого действия:
      1. Разбиваем наблюдения на n_bins бинов по предсказанному CATE.
      2. В каждом бине считаем эмпирический эффект как
         mean(Y | T=action) − mean(Y | T=baseline).
      3. Вычисляем корреляцию Пирсона между предсказанными и эмпирическими эффектами.
    Итоговая корреляция — среднее по всем действиям, для которых в бинах
    оказалось достаточно наблюдений обоих типов.

    Parameters
    ----------
    trainer       : CausalModelTrainer — обученная модель.
    data_test     : pd.DataFrame — тестовая выборка.
    outcome       : str — целевой исход.
    treatment_col : str — колонка с treatment.
    n_bins        : int — число бинов квантилей.
    n_actions     : int — сколько не-baseline действий использовать.

    Returns
    -------
    dict с ключами:
        'per_action'     — dict{action: {'predicted', 'actual', 'correlation'}},
        'mean_correlation' — средняя корреляция по действиям (основная метрика),
        'n_actions_used'   — сколько действий дали валидный результат.
    """
    feature_cols = trainer.feature_names
    available_fc = [c for c in feature_cols if c in data_test.columns]
    X_test = data_test[available_fc]
    Y_test = data_test[outcome].values
    T_test = data_test[treatment_col].values

    non_baseline = [
        a for a in trainer.available_treatment_names
        if a != BASELINE_ACTION
    ][:n_actions]

    if not non_baseline:
        return {
            'per_action': {},
            'mean_correlation': np.nan,
            'n_actions_used': 0,
        }

    per_action = {}
    correlations = []

    for action in non_baseline:
        preds = trainer.predict_absolute(X_test, actions=[BASELINE_ACTION, action])
        if outcome not in preds:
            continue

        cate_pred = (
            preds[outcome]['effect'][action].values
            - preds[outcome]['effect'][BASELINE_ACTION].values
        )

        # Разбиваем на бины по квантилям предсказанного CATE
        bin_edges = np.percentile(cate_pred, np.linspace(0, 100, n_bins + 1))
        # digitize: бины 1..n_bins
        bin_idx = np.digitize(cate_pred, bin_edges[:-1])

        pred_effects, actual_effects = [], []
        for b in range(1, n_bins + 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue

            pred_effects.append(float(np.mean(cate_pred[mask])))

            y_treated = Y_test[mask][T_test[mask] == action]
            y_control = Y_test[mask][T_test[mask] == BASELINE_ACTION]

            if len(y_treated) > 0 and len(y_control) > 0:
                actual_effects.append(float(np.mean(y_treated) - np.mean(y_control)))
            else:
                actual_effects.append(np.nan)

        pred_arr = np.array(pred_effects)
        actual_arr = np.array(actual_effects)
        valid = ~np.isnan(actual_arr)

        if valid.sum() > 1:
            corr = float(np.corrcoef(pred_arr[valid], actual_arr[valid])[0, 1])
        else:
            corr = np.nan

        per_action[action] = {
            'predicted': pred_arr,
            'actual': actual_arr,
            'correlation': corr,
        }

        if not np.isnan(corr):
            correlations.append(corr)

    mean_corr = float(np.mean(correlations)) if correlations else np.nan

    return {
        'per_action': per_action,
        'mean_correlation': mean_corr,
        'n_actions_used': len(correlations),
    }