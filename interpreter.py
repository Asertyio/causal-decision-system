# interpreter.py
"""SHAP-based explainability for the causal career model.

CausalForestDML's const_marginal_effect() returns a (n, K) matrix where
K = number of non-baseline treatment categories.  We expose per-action
waterfall plots by selecting the appropriate column of the SHAP output.
"""
import shap
import numpy as np
import matplotlib.pyplot as plt


class ModelInterpreter:
    """Wraps a trained CausalModelTrainer and computes SHAP explanations.

    Parameters
    ----------
    trainer : CausalModelTrainer  – already fitted trainer.
    X_background : np.ndarray     – background sample for the SHAP kernel (n_bg, n_features).
    feature_names : list[str]     – feature names after one-hot encoding.
    outcome : str                 – which outcome model to explain (default 'salary_2y').
    """

    def __init__(self, trainer, X_background: np.ndarray, feature_names: list,
                 outcome: str = 'salary_2y'):
        self.trainer = trainer
        self.feature_names = feature_names
        self.outcome = outcome

        if outcome not in trainer.models:
            raise ValueError(f"Outcome '{outcome}' not found in trainer.models. "
                             f"Available: {list(trainer.models.keys())}")
        self.model = trainer.models[outcome]
        self.available_treatment_names = trainer.available_treatment_names
        self.treatment_name_to_idx = trainer.treatment_name_to_idx

        # BUG FIX: const_marginal_effect returns shape (n, K) — a multi-output function.
        # We wrap it to return a 1-D array (one CATE value per sample) for a specific
        # action column, so SHAP receives a scalar output per sample.
        # The action is set via self._explain_action_idx before each explain() call.
        self._explain_action_idx = 0

        def _predict_single_action(X: np.ndarray) -> np.ndarray:
            cate = self.model.const_marginal_effect(X)  # (n, K)
            if self._explain_action_idx < cate.shape[1]:
                return cate[:, self._explain_action_idx]
            # Reference category → CATE is identically 0
            return np.zeros(len(X))

        # Use a sample of the background to keep SHAP fast
        bg = X_background[:min(100, len(X_background))]
        self.explainer = shap.Explainer(
            _predict_single_action,
            bg,
            feature_names=feature_names,
        )

    def _set_action(self, action: str):
        """Select which treatment column to explain."""
        if action not in self.treatment_name_to_idx:
            raise ValueError(f"Action '{action}' not available. "
                             f"Available: {self.available_treatment_names}")
        self._explain_action_idx = self.treatment_name_to_idx[action]

    def explain(self, X: np.ndarray, action: str = None):
        """Return SHAP values for X.

        Parameters
        ----------
        X      : np.ndarray – preprocessed feature matrix (n, n_features).
        action : str        – which action to explain; defaults to first available.
        """
        if action is None:
            action = self.available_treatment_names[0]
        self._set_action(action)
        return self.explainer(X)

    def plot_waterfall(self, X: np.ndarray, action: str = None,
                       idx: int = 0, max_display: int = 10):
        """Waterfall plot for sample at position idx.

        Parameters
        ----------
        X          : np.ndarray – preprocessed features.
        action     : str        – treatment action to explain.
        idx        : int        – row index to visualise.
        max_display: int        – max features to show.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if action is None:
            action = self.available_treatment_names[0]
        shap_values = self.explain(X, action=action)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.waterfall_plot(shap_values[idx], max_display=max_display, show=False)
        ax.set_title(f"SHAP contribution to CATE — action: {action}", fontsize=10)
        plt.tight_layout()
        return fig

    def plot_summary(self, X: np.ndarray, action: str = None):
        """Beeswarm summary plot across all samples.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if action is None:
            action = self.available_treatment_names[0]
        shap_values = self.explain(X, action=action)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        ax.set_title(f"SHAP summary — action: {action}", fontsize=10)
        plt.tight_layout()
        return fig