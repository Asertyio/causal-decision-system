# test_model.py
"""Quick smoke-test for the causal career model (v2 API)."""
from sklearn.model_selection import train_test_split
import numpy as np

from data_synthetic import generate_career_data
from causal_model import CausalModelTrainer, get_default_feature_cols
from validator import compute_metrics

print("Generating synthetic career data (n=2000)...")
df = generate_career_data(n=2000, random_state=42)

outcome_cols = ['salary_2y', 'satisfaction_2y', 'promoted_2y', 'wlb_2y']
feature_cols = get_default_feature_cols()
# Keep only cols present in df
feature_cols = [c for c in feature_cols if c in df.columns]

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

print("Training CausalForestDML (fast mode, binary per action)...")
trainer = CausalModelTrainer(n_estimators=48, fast_mode=True, random_state=42)
trainer.fit(df_train, outcome_cols, treatment_col='treatment', feature_cols=feature_cols)
print(f"Available actions in model: {len(trainer.available_treatment_names)}")

print("\nEvaluating on test set (salary_2y)...")
preds = trainer.predict_absolute(df_test[feature_cols])

actual_treatments = df_test['treatment'].values
salary_true = df_test['salary_2y'].values

salary_pred = np.array([
    preds['salary_2y']['effect'].iloc[i][actual_treatments[i]]
    if actual_treatments[i] in preds['salary_2y']['effect'].columns
    else np.nan
    for i in range(len(df_test))
])

valid = ~np.isnan(salary_pred)
metrics = compute_metrics(salary_true[valid], salary_pred[valid])
print(f"  MSE:       {metrics['MSE']:.2f}")
print(f"  R²:        {metrics['R2']:.4f}")
print(f"  Pearson R: {metrics['PearsonR']:.4f}")

print("\nSanity checks on predicted effects (median person):")
median_row = df_test[feature_cols].median().to_frame().T
for col in ['gender', 'region', 'current_industry', 'job_level']:
    if col in median_row.columns:
        median_row[col] = df_test[col].mode()[0]

effects = trainer.predict_absolute(median_row)
salary_effects = effects['salary_2y']['effect'].iloc[0]

print(f"  Relocation abroad:  {salary_effects.get('Релоцироваться в другую страну', float('nan')):.1f} тыс. руб")
print(f"  Stay in place:      {salary_effects.get('Остаться на текущем месте', float('nan')):.1f} тыс. руб")
print(f"  Get MBA:            {salary_effects.get('Получить MBA', float('nan')):.1f} тыс. руб")
print(f"  Reduce employment:  {salary_effects.get('Выйти на пенсию / сократить занятость', float('nan')):.1f} тыс. руб")

# Test confidence labels
_, confidence = trainer.predict_with_confidence(median_row)
print("\nConfidence labels (salary_2y):")
for act, lbl in list(confidence['salary_2y'].items())[:5]:
    print(f"  {act[:50]:50s}: {lbl}")

print("\nAll tests passed ✓")