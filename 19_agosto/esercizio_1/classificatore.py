# #### Setup iniziale
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# #### Caricamento del dataset
# Kaggle "AEP_hourly": columns => ["Datetime", "AEP_MW"]
df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"])

# #### Preprocessing
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month

# Creazione della variabile target: 1 se AEP_MW è maggiore della media, altrimenti 0
df["Target"] = (df["AEP_MW"] > df["AEP_MW"].mean()).astype(int)





