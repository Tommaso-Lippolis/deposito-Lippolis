import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("19_agosto\esercizio_1\AEP_hourly.csv", parse_dates=["Datetime"])

# #### Preprocessing
df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month

# Etichetta: 1 se consumo > mediana, altrimenti 0
df["target"] = (df["AEP_MW"] > df["AEP_MW"].median()).astype(int)

# # Feature: ora, giorno della settimana, mese
# X = df[["hour", "dayofweek", "month"]]
# y = df["target"]

# # Split
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

# print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# # plotta le shape dei set
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.bar(['Train', 'Validation', 'Test'], [len(X_train), len(X_val), len(X_test)], color=['blue', 'orange', 'green'])
# # creiamo una linea tratteggiata al livello del test_size
# plt.axhline(y=len(X_test), color='red', linestyle='--', label='Test Size')
# plt.legend()
# plt.title('Dimensioni dei set di dati')
# plt.xlabel('Set di dati')
# plt.ylabel('Numero di campioni')
# plt.show()


from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Feature e target (come prima)
X = df[["hour", "dayofweek", "month"]]
y = df["target"]

# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
auc_tree = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")

# Neural Network con scaling
mlp_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        ),
    ]
)
auc_mlp = cross_val_score(mlp_pipeline, X, y, cv=skf, scoring="roc_auc")

print(f"Decision Tree AUC: {auc_tree.mean():.3f} ± {auc_tree.std():.3f}")
print(f"Neural Network AUC: {auc_mlp.mean():.3f} ± {auc_mlp.std():.3f}")
