import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings('ignore')


# 1. CARGAR DATASET


df = pd.read_csv('KAG_energydata_complete.csv')


# 2. VARIABLES TEMPORALES


df['date'] = pd.to_datetime(df['date'])

df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['minute'] = df['date'].dt.minute
df['weekofyear'] = (
    df['date']
    .dt.isocalendar()
    .week
    .astype(int)
)


# 3. VARIABLES REZAGADAS


df['Appliances_prev_1'] = (
    df['Appliances']
    .shift(1)
)

df['Appliances_prev_2'] = (
    df['Appliances']
    .shift(2)
)

# 4. PROMEDIO MÓVIL


df['rolling_mean_3'] = (
    df['Appliances']
    .shift(1)
    .rolling(window=3)
    .mean()
)

# 5. REDUCCIÓN DE RUIDO


# Promedio de temperatura interna
cols_temperatura = [
    'T1', 'T2', 'T3',
    'T4', 'T5', 'T6',
    'T7', 'T8', 'T9'
]

df['T_house_mean'] = (
    df[cols_temperatura]
    .mean(axis=1)
)

# Promedio de humedad interna
cols_humedad = [
    'RH_1', 'RH_2', 'RH_3',
    'RH_4', 'RH_5', 'RH_6',
    'RH_7', 'RH_8', 'RH_9'
]

df['RH_house_mean'] = (
    df[cols_humedad]
    .mean(axis=1)
)


# 6. LIMPIEZA


df.dropna(inplace=True)

print(f"\nFilas del dataset: {len(df):,}")
print(f"Columnas del dataset: {len(df.columns)}")

# 7. VARIABLES


target = 'Appliances'

cols_excluir = [

    # Excluir fecha y target
    'date',
    'Appliances',

    # Variables aleatorias
    'rv1',
    'rv2',

    # Excluir temperaturas individuales
    'T1', 'T2', 'T3',
    'T4', 'T5', 'T6',
    'T7', 'T8', 'T9',

    # Excluir humedades individuales
    'RH_1', 'RH_2', 'RH_3',
    'RH_4', 'RH_5', 'RH_6',
    'RH_7', 'RH_8', 'RH_9'
]

features = [
    c for c in df.columns
    if c not in cols_excluir
]

X = df[features]
y = df[target]

print("\nVariables utilizadas:\n")

for var in features:
    print(var)


# 8. TRAIN / TEST


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTrain: {len(X_train):,}")
print(f"Test : {len(X_test):,}")


# 9. MODELO RANDOM FOREST


print("\nEntrenando Random Forest...")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# 10. PREDICCIONES


y_pred = rf.predict(X_test)


# 11. MÉTRICAS

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print(f"\n{'='*40}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print(f"{'='*40}")


# 12. OVERFITTING

train_r2 = rf.score(X_train, y_train)
test_r2 = rf.score(X_test, y_test)

print("\nComparación Train/Test")

print(f"R² entrenamiento : {train_r2:.4f}")
print(f"R² prueba        : {test_r2:.4f}")


# 13. IMPORTANCIA DE VARIABLES

imp = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(10,8))

plt.barh(
    imp.index,
    imp.values
)

plt.xlabel('Importancia relativa')

plt.title(
    'Importancia de Variables - Random Forest'
)

plt.tight_layout()

plt.savefig(
    'grafica_importancia.png',
    dpi=200
)

plt.show()

print("\nGuardada: grafica_importancia.png")


# 14. REAL VS PREDICHO


plt.figure(figsize=(12,4))

n = 200

plt.plot(
    y_test.values[:n],
    label='Real'
)

plt.plot(
    y_pred[:n],
    '--',
    label='Predicho'
)

plt.title(
    'Consumo real vs predicho'
)

plt.xlabel('Muestras')

plt.ylabel('Consumo energético')

plt.legend()

plt.tight_layout()

plt.savefig(
    'grafica_real_vs_pred.png',
    dpi=200
)

plt.show()

print("Guardada: grafica_real_vs_pred.png")


# 15. DISPERSIÓN


plt.figure(figsize=(6,6))

plt.scatter(
    y_test[:500],
    y_pred[:500],
    alpha=0.4
)

lims = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

plt.plot(
    lims,
    lims,
    'r--',
    label='Predicción perfecta'
)

plt.xlabel('Valor real')

plt.ylabel('Valor predicho')

plt.title(
    f'Dispersión — R² = {r2:.4f}'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'grafica_dispersion.png',
    dpi=200
)

plt.show()

print("Guardada: grafica_dispersion.png")