import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# #### Caricamento del dataset
df = pd.read_csv("19_agosto\esercizio_2\AirQualityUCI.csv", sep=';')

# vediamo seci sono valori nulli
print("Valori nulli per colonna:")
print(df.isnull().sum())

# analizziamo le colonne del dataset
print("Colonne del dataset:")
print(df.columns)

# analizziamo le statistiche del dataset
print("Statistiche del dataset:")
print(df.describe())

# lasciamo solo le colonne  'Date', 'Time', 'CO(GT)'
df = df[['Date', 'Time', 'CO(GT)']]

df = df.dropna()

# vediamo seci sono valori nulli
print("Valori nulli dopo la pulizia:")
print(df.isnull().sum())

# Converte tutte le colonne tranne Date e Time in float
df["CO(GT)"] = pd.to_numeric(df["CO(GT)"].astype(str).str.replace(',', '.'), errors='coerce')

# identifichiamo come buona qualita dell aria o scarsa qualita dell'aria in base alla media globale
df['AirQuality'] = (df['CO(GT)'] < df['CO(GT)'].mean()).astype(int)


# come input valore inquinante CO e la data il resto cancelliamo
X = df[['CO(GT)', 'Date', 'Time']]

y = df['AirQuality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)







