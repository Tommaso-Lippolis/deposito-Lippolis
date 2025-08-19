# #### Setup iniziale
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# #### Caricamento del dataset
# Kaggle "AEP_hourly": columns => ["Datetime", "AEP_MW"]
df = pd.read_csv("19_agosto\esercizio_1\AEP_hourly.csv", parse_dates=["Datetime"])

# #### Preprocessing
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month

# Creazione della variabile target: 1 se AEP_MW è maggiore della media, altrimenti 0
df["Target"] = (df["AEP_MW"] > df["AEP_MW"].mean()).astype(int)

X = df[["Hour", "Day", "Month"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)
print("Decision Tree Classifier Report:")
print(classification_report(y_test, y_pred_dt))

# Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_classifier = MLPClassifier(random_state=42, max_iter=1000)
nn_classifier.fit(X_train_scaled, y_train)
y_pred_nn = nn_classifier.predict(X_test_scaled)
print("Neural Network Classifier Report:")
print(classification_report(y_test, y_pred_nn))
