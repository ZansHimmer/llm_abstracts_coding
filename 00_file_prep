import pandas as pd
import openpyxl as px

file_path = 'master-sheet.xlsx'
df = pd.read_excel(file_path)

relevant_columns = ['MesH_ID', 'title', 'abstract', 'final-decision_include']

df = df[relevant_columns]

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

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


