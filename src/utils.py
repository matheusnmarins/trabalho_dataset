import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(df, feature_col, target_col, dataset_name, save_path=None):
    """
    Treina e avalia um modelo de regressão linear.
    Salva gráfico e retorna métricas.
    """
    df = df.dropna(subset=[feature_col, target_col])

    X = df[[feature_col]].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== Resultados para {dataset_name} ===")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R² : {r2:.3f}")

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    plt.title(f"Reais vs Previstos - {dataset_name}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return {"MAE": mae, "MSE": mse, "R2": r2}
