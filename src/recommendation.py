import pandas as pd

def load_pesticide_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower().replace("\n", " ").replace("  ", " ") for col in df.columns]
    df['disease'] = df['disease'].str.strip().str.lower()
    return df

def normalize(text):
    return text.strip().lower().replace('_', ' ')

def recommend_pesticide(df, disease_name):
    disease_name = normalize(disease_name)
    df['normalized_disease'] = df['disease'].apply(normalize)
    match = df[df['normalized_disease'] == disease_name]
    if not match.empty:
        row = match.iloc[0]
        return {
            "Description": row.get('description', 'N/A'),
            "Pesticide (Small Region)": row.get('pesticide (small region)', 'N/A'),
            "Dosage (Small Region)": row.get('dosage (small region)', 'N/A'),
            "Pesticide (Large Region)": row.get('pesticide (large region)', 'N/A'),
            "Dosage (Large Region)": row.get('dosage (large region)', 'N/A'),
            "Organic Method": row.get('organic method', 'N/A')
        }
    else:
        return {"Note": "No pesticide data found for this disease."}
