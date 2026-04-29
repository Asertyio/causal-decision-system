# validator.py
"""Validation utilities for the causal career model."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from data_synthetic import generate_career_data
from causal_model import CausalModelTrainer, get_default_feature_cols, BASELINE_ACTION


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MSE, R², and Pearson correlation between true and predicted values."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true.ravel(), y_pred.ravel())
    return {'MSE': mse, 'R2': r2, 'PearsonR': corr}


def placebo_test(
    data: pd.DataFrame,
    outcome_cols: list,
    feature_cols: list,
    treatment_col: str = 'treatment',
    n_permutations: int = 20,
    random_state: int = 0,
) -> dict:
    """Permute treatment labels and check that average CATE ≈ 0."""
    rng = np.random.default_rng(random_state)
    ate_placebo = []

    for perm_idx in range(n_permutations):
        perm_seed = int(rng.integers(0, 100_000))
        T_perm = data[treatment_col].sample(frac=1, random_state=perm_seed).values
        data_perm = data.copy()
        data_perm[treatment_col] = T_perm

        temp_trainer = CausalModelTrainer(random_state=perm_seed, fast_mode=True)
        temp_trainer.fit(data_perm, outcome_cols[:1], treatment_col=treatment_col,
                         feature_cols=feature_cols)

        preds = temp_trainer.predict_absolute(data_perm[feature_cols].head(200))
        first_outcome = outcome_cols[0]
        ate = preds[first_outcome]['effect'].values.mean()
        ate_placebo.append(ate)
        print(f"  Placebo [{perm_idx+1}/{n_permutations}]: ATE={ate:.3f}")

    ate_placebo = np.array(ate_placebo)
    p_value = 2 * min(np.mean(ate_placebo >= 0), np.mean(ate_placebo <= 0))
    return {
        'mean_ate': float(np.mean(ate_placebo)),
        'std_ate': float(np.std(ate_placebo)),
        'p_value': float(p_value),
    }


def calibration_test(
    trainer: CausalModelTrainer,
    data_test: pd.DataFrame,
    outcome: str = 'salary_2y',
    treatment_col: str = 'treatment',
    n_bins: int = 5,
) -> dict:
    """Bin observations by predicted CATE and compare to empirical effect."""
    feature_cols = trainer.feature_names
    available_fc = [c for c in feature_cols if c in data_test.columns]
    X_test = data_test[available_fc]
    Y_test = data_test[outcome].values
    T_test = data_test[treatment_col].values

    baseline = BASELINE_ACTION
    # Pick first available non-baseline action
    treated_name = next(
        (a for a in trainer.available_treatment_names if a != baseline),
        baseline
    )

    preds = trainer.predict_absolute(X_test, actions=[baseline, treated_name])
    if outcome not in preds:
        return {'predicted': np.array([]), 'actual': np.array([]), 'correlation': np.nan}

    cate_pred = (preds[outcome]['effect'][treated_name].values
                 - preds[outcome]['effect'][baseline].values)

    bins = np.percentile(cate_pred, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(cate_pred, bins[:-1])

    pred_effects, actual_effects = [], []
    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        pred_effects.append(float(np.mean(cate_pred[mask])))
        y1 = Y_test[mask][T_test[mask] == treated_name]
        y0 = Y_test[mask][T_test[mask] == baseline]
        if len(y1) > 0 and len(y0) > 0:
            actual_effects.append(float(np.mean(y1) - np.mean(y0)))
        else:
            actual_effects.append(np.nan)

    pred_effects  = np.array(pred_effects)
    actual_effects = np.array(actual_effects)
    valid = ~np.isnan(actual_effects)
    corr = float(np.corrcoef(pred_effects[valid], actual_effects[valid])[0, 1]) if valid.sum() > 1 else np.nan

    return {
        'predicted': pred_effects,
        'actual': actual_effects,
        'correlation': corr,
    }