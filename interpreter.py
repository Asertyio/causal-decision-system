# interpreter.py
"""SHAP-интерпретатор для бинарных CausalForestDML моделей карьерного советника.

Совместим с архитектурой causal_model1.py:
  trainer.models[action][outcome] = CausalForestDML (бинарный T∈{0,1})
  trainer.baseline_models[outcome] = RandomForestRegressor

Каждый экземпляр ModelInterpreter привязан к конкретной паре (action, outcome).
Для объяснения нескольких действий создавайте отдельные экземпляры или
используйте фабричную функцию make_interpreter().
"""
import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional


class ModelInterpreter:
    """Объясняет CATE конкретной пары (action, outcome) с помощью SHAP.

    Архитектура causal_model1.py использует отдельный бинарный CausalForestDML
    на каждую пару (действие, исход), поэтому const_marginal_effect() возвращает
    shape (n,) или (n, 1) — скалярный CATE без мультиклассовой индексации.

    Parameters
    ----------
    model        : обученный CausalForestDML для данной пары (action, outcome).
    background_X : np.ndarray shape (n_bg, n_features) — фоновая выборка для SHAP.
    feature_names: list[str] — имена признаков после OHE.
    action_name  : str — человекочитаемое название действия (для заголовков).
    outcome_name : str — человекочитаемое название исхода (для заголовков).
    n_background : int — сколько строк из background_X использовать (для скорости).
    """

    def __init__(
        self,
        model,
        background_X: np.ndarray,
        feature_names: list,
        action_name: str = "",
        outcome_name: str = "",
        n_background: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.action_name = action_name
        self.outcome_name = outcome_name

        # Обёртка: бинарный DML возвращает (n,) или (n,1) → всегда (n,)
        def _predict_fn(X: np.ndarray) -> np.ndarray:
            cate = model.const_marginal_effect(X)
            return np.squeeze(cate)  # (n,)

        bg = background_X[:min(n_background, len(background_X))]

        # shap.Explainer автоматически выбирает TreeExplainer для RF-based DML,
        # что на порядок быстрее KernelExplainer.
        self.explainer = shap.Explainer(
            _predict_fn,
            bg,
            feature_names=feature_names,
        )
        # Базовое значение (среднее CATE по фоновой выборке)
        self.base_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """Вычисляет SHAP-значения для матрицы X.

        Parameters
        ----------
        X : np.ndarray shape (n, n_features) — предобработанные признаки.

        Returns
        -------
        shap.Explanation — объект с .values, .base_values, .data.
        """
        exp = self.explainer(X)
        # Кэшируем базовое значение для внешнего использования
        if self.base_value is None and hasattr(exp, 'base_values'):
            bv = exp.base_values
            self.base_value = float(np.mean(bv)) if bv is not None else None
        return exp

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        X: np.ndarray,
        idx: int = 0,
        max_display: int = 10,
    ) -> plt.Figure:
        """Waterfall-график вклада признаков для одного наблюдения.

        Parameters
        ----------
        X           : np.ndarray — предобработанные признаки.
        idx         : int — индекс строки в X.
        max_display : int — максимальное число признаков на графике.

        Returns
        -------
        matplotlib.figure.Figure
        """
        exp = self.explain(X)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.waterfall_plot(exp[idx], max_display=max_display, show=False)
        title = f"SHAP — вклад признаков в CATE"
        if self.action_name:
            title += f"\nДействие: {self.action_name}"
        if self.outcome_name:
            title += f"  |  Исход: {self.outcome_name}"
        ax.set_title(title, fontsize=10, pad=8)
        plt.tight_layout()
        return fig

    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 15,
    ) -> plt.Figure:
        """Beeswarm-график распределения SHAP-значений по всей выборке.

        Parameters
        ----------
        X           : np.ndarray — предобработанные признаки.
        max_display : int — максимальное число признаков.

        Returns
        -------
        matplotlib.figure.Figure
        """
        exp = self.explain(X)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)
        shap.summary_plot(
            exp.values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        title = "SHAP summary"
        if self.action_name:
            title += f" — {self.action_name}"
        if self.outcome_name:
            title += f" / {self.outcome_name}"
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        return fig

    def plot_bar(
        self,
        X: np.ndarray,
        max_display: int = 15,
    ) -> plt.Figure:
        """Bar-chart средних |SHAP|-значений (глобальная важность признаков).

        Parameters
        ----------
        X           : np.ndarray — предобработанные признаки.
        max_display : int — максимальное число признаков.

        Returns
        -------
        matplotlib.figure.Figure
        """
        exp = self.explain(X)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.plots.bar(exp, max_display=max_display, show=False)
        title = "SHAP — глобальная важность признаков"
        if self.action_name:
            title += f"\n{self.action_name}"
        if self.outcome_name:
            title += f" / {self.outcome_name}"
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────
# Фабричная функция
# ──────────────────────────────────────────────────────────────────────

def make_interpreter(
    trainer,
    action: str,
    outcome: str,
    background_X: np.ndarray,
    n_background: int = 100,
) -> ModelInterpreter:
    """Создаёт ModelInterpreter для пары (action, outcome) из обученного trainer.

    Parameters
    ----------
    trainer      : CausalModelTrainer из causal_model1.py (уже обученный).
    action       : str — название действия (должно быть в trainer.models).
    outcome      : str — название исхода (например 'salary_2y').
    background_X : np.ndarray — фоновая выборка после OHE (из X_full.values).
    n_background : int — размер фоновой выборки для SHAP.

    Returns
    -------
    ModelInterpreter

    Raises
    ------
    ValueError если действие или исход не найдены в trainer.
    """
    if not trainer.is_fitted:
        raise ValueError("Trainer не обучен. Вызовите trainer.fit() перед созданием интерпретатора.")

    if action not in trainer.models:
        available = list(trainer.models.keys())
        raise ValueError(
            f"Действие '{action}' не найдено в trainer.models.\n"
            f"Доступные действия: {available}"
        )

    if outcome not in trainer.models[action]:
        available = list(trainer.models[action].keys())
        raise ValueError(
            f"Исход '{outcome}' не найден для действия '{action}'.\n"
            f"Доступные исходы: {available}"
        )

    model = trainer.models[action][outcome]
    feature_names = trainer.feature_names_after_ohe

    return ModelInterpreter(
        model=model,
        background_X=background_X,
        feature_names=feature_names,
        action_name=action,
        outcome_name=outcome,
        n_background=n_background,
    )