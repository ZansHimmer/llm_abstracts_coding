import pandas as pd
import openpyxl as px

# version 04/03/2026

file_path = 'master-sheet.xlsx'
df = pd.read_excel(file_path)

relevant_columns = ['MesH_ID', 'title', 'abstract', 'final-decision_include']

df = df[relevant_columns]

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
output_path_df = "shuffled_master_sheet.xlsx"
df_shuffled.to_excel(output_path_df, index=False)

print(df_shuffled.head())

output_path = "papers_original.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_shuffled.iterrows():
        mesh_id = row["MesH_ID"]
        title = row["title"]
        abstract = row["abstract"]

        f.write(f"<start id {mesh_id}>\n")
        f.write(f"id: {mesh_id}\n")
        f.write(f"title: {title}\n")
        f.write(f"abstract: {abstract}\n")
        f.write(f"<end id {mesh_id}>\n\n")


