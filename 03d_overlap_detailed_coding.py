import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = 'matched_sheets\\matched_master_sheet_full_6_gpt-5-mini_bs-1.xlsx'
df = pd.read_excel(file_path)

CATEGORY = "meta_analysis"

df[CATEGORY] = pd.to_numeric(df[CATEGORY], errors='coerce')
df['decision_LLM_2'] = pd.to_numeric(df['decision_LLM_2'], errors='coerce')
df_eval = df.dropna(subset=[CATEGORY, 'decision_LLM_2'])

y_true = df_eval[CATEGORY].astype(int)
y_pred = df_eval['decision_LLM_2'].astype(int)

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_true, y_pred, digits=4)
print('Classification Report:')
print(report)