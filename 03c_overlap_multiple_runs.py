import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score

FILE_1_PATH = 'matched_sheets\\matched_master_sheet_2-temp1_gpt-4.1-mini_bs-1.xlsx'
FILE_2_PATH = 'matched_sheets\\matched_master_sheet_2b-temp1_gpt-4.1-mini_bs-1.xlsx'
FILE_3_PATH = 'matched_sheets\\matched_master_sheet_2c-temp1_gpt-4.1-mini_bs-1.xlsx'

files = [FILE_1_PATH, FILE_2_PATH, FILE_3_PATH]
decision_cols = ['decision_LLM_1', 'decision_LLM_2', 'decision_LLM_3']

df_merged = pd.read_excel(files[0])
df_merged['MesH_ID'] = df_merged['MesH_ID'].astype(str).str.strip()
df_merged.rename(columns={'decision_LLM_2': 'decision_LLM_1'}, inplace=True)

for file, col_name in zip(files[1:], ['decision_LLM_2', 'decision_LLM_3']):
    df = pd.read_excel(file)
    df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()
    df.rename(columns={'decision_LLM_2': col_name}, inplace=True)
    df_merged = df_merged.merge(df[['MesH_ID', col_name]], on='MesH_ID', how='left')

df_merged = df_merged.dropna(subset=decision_cols)
df_merged['final_decision_all_1'] = df_merged[decision_cols].all(axis=1).astype(int)
df_merged['final_decision_any_1'] = df_merged[decision_cols].any(axis=1).astype(int)
df_merged['final_decision_majority_1'] = (df_merged[decision_cols].sum(axis=1) >= 2).astype(int)

df_merged['final-decision_include'] = pd.to_numeric(df_merged['final-decision_include'], errors='coerce')
df_merged = df_merged.dropna(subset=['final-decision_include', 'final_decision_all_1', 'final_decision_any_1', 'final_decision_majority_1'])

y_true = df_merged['final-decision_include'].astype(int)
y_pred_all = df_merged['final_decision_all_1'].astype(int)
y_pred_any = df_merged['final_decision_any_1'].astype(int)
y_pred_majority = df_merged['final_decision_majority_1'].astype(int)

accuracy_all = accuracy_score(y_true, y_pred_all)
accuracy_any = accuracy_score(y_true, y_pred_any)
accuracy_majority = accuracy_score(y_true, y_pred_majority)
print(f'Accuracy (all 1): {accuracy_all:.4f}')
print(f'Accuracy (any 1): {accuracy_any:.4f}')
print(f'Accuracy (majority 1): {accuracy_majority:.4f}')

cm_all = confusion_matrix(y_true, y_pred_all)
cm_any = confusion_matrix(y_true, y_pred_any)
cm_majority = confusion_matrix(y_true, y_pred_majority)
print('Confusion Matrix (all 1):')
print(cm_all)
print('Confusion Matrix (any 1):')
print(cm_any)
print('Confusion Matrix (majority 1):')
print(cm_majority)

report_all = classification_report(y_true, y_pred_all, digits=4)
report_any = classification_report(y_true, y_pred_any, digits=4)
report_majority = classification_report(y_true, y_pred_majority, digits=4)
print('Classification Report (all 1):')
print(report_all)
print('Classification Report (any 1):')
print(report_any)
print('Classification Report (majority 1):')
print(report_majority)

kappa_all = cohen_kappa_score(y_true, y_pred_all)
kappa_any = cohen_kappa_score(y_true, y_pred_any)
kappa_majority = cohen_kappa_score(y_true, y_pred_majority)
print(f"Cohen's Kappa (all 1): {kappa_all:.4f}")
print(f"Cohen's Kappa (any 1): {kappa_any:.4f}")
print(f"Cohen's Kappa (majority 1): {kappa_majority:.4f}")