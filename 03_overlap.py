import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = 'matched_master_sheet_2_gpt-5-mini.xlsx'
df = pd.read_excel(file_path)

df['final-decision_include'] = pd.to_numeric(df['final-decision_include'], errors='coerce')
df['decision_LLM_2'] = pd.to_numeric(df['decision_LLM_2'], errors='coerce')
df_eval = df.dropna(subset=['final-decision_include', 'decision_LLM_2'])

y_true = df_eval['final-decision_include'].astype(int)
y_pred = df_eval['decision_LLM_2'].astype(int)

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_true, y_pred, digits=4)
print('Classification Report:')
print(report)