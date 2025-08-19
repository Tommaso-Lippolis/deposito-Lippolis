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

print(df.columns)

df = df.dropna()

# vediamo seci sono valori nulli
print("Valori nulli dopo la pulizia:")
print(df.isnull().sum())

# Converte tutte le colonne tranne Date e Time in float
df["CO(GT)"] = pd.to_numeric(df["CO(GT)"].astype(str).str.replace(',', '.'), errors='coerce')

# identifichiamo come buona qualita dell aria o scarsa qualita dell'aria in base alla media globale
df['AirQuality'] = (df['CO(GT)'] < df['CO(GT)'].mean()).astype(int)

print("Distribuzione della qualità dell'aria:")
print(df['AirQuality'].value_counts())


# come input valore inquinante CO e l ora il resto droppiamo
df['Hour'] = pd.to_datetime(df['Time'], format='%H.%M.%S', errors='coerce').dt.hour
X = df[['Hour']]

y = df['AirQuality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression Classifier Report:")
print(classification_report(y_test, y_pred_log))









