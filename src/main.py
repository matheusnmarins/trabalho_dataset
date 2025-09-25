import os
from preprocess import load_dataset
from utils import train_and_evaluate

DATA_PATH = os.path.join("data")
RESULTS_PATH = os.path.join("results", "graficos")

os.makedirs(RESULTS_PATH, exist_ok=True)

# ==============================
# Dataset 1: Umidade vs Temperatura
# ==============================
df1 = load_dataset(
    os.path.join(DATA_PATH, "ds_hum_vs_temp_dirty.csv"),
    conversions=[("temperatura", "unidade_temperatura")]
)
metrics1 = train_and_evaluate(
    df1,
    feature_col="temperatura",
    target_col="humidade",
    dataset_name="Umidade vs Temperatura",
    save_path=os.path.join(RESULTS_PATH, "umidade_vs_temp.png")
)

# ==============================
# Dataset 2: MinTemp vs MaxTemp
# ==============================
df2 = load_dataset(
    os.path.join(DATA_PATH, "ds_min_temp_vs_max_temp_raw.csv"),
    conversions=[("MinTemp", None), ("MaxTemp", None)]
)
metrics2 = train_and_evaluate(
    df2,
    feature_col="MinTemp",
    target_col="MaxTemp",
    dataset_name="MinTemp vs MaxTemp",
    save_path=os.path.join(RESULTS_PATH, "min_vs_max.png")
)

# ==============================
# Dataset 3: Salinity vs Temp
# ==============================
df3 = load_dataset(
    os.path.join(DATA_PATH, "ds_salinity_vs_temp_raw.csv"),
    conversions=[("T_degC", None)]
)
metrics3 = train_and_evaluate(
    df3,
    feature_col="Salnty",      
    target_col="T_degC",       
    dataset_name="Salinity vs Temperatura"
)

# ==============================
# Salvar métricas
# ==============================
with open(os.path.join("results", "metrics.txt"), "w") as f:
    f.write("Resultados dos Modelos:\n")
    f.write(f"Dataset 1: {metrics1}\n")
    f.write(f"Dataset 2: {metrics2}\n")
    f.write(f"Dataset 3: {metrics3}\n")
