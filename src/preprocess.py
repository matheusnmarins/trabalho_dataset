import pandas as pd

def preprocess_temperature(df, temp_col, unit_col):
    """
    Converte temperaturas para Celsius baseado na coluna de unidade.
    """
    def convert(row):
        if row[unit_col].lower() in ['f', 'fahrenheit']:
            return (row[temp_col] - 32) * 5/9
        return row[temp_col]

    df[temp_col] = df.apply(convert, axis=1)
    return df

def load_dataset(path, conversions=None):
    df = pd.read_csv(path, low_memory=False)  # evita warning de mixed types
    if conversions:
        for temp_col, unit_col in conversions:
            if unit_col is not None and unit_col in df.columns:
                df = preprocess_temperature(df, temp_col, unit_col)
    return df
