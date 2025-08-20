from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Lezione_3/Mall Customer dataset/Mall_Customers.csv')

data = data.drop(['CustomerID', 'Genre', 'Age'], axis=1)

print(" first 5 rows of the dataset:")
print(data.head())

print("Descriptive statistics of the dataset:")
print(data.describe())

print("Data types of the dataset:")
print(data.info())

print("Checking for missing values in the dataset:")
print(data.isnull().sum())

# cambia il nome delle colonne per comodità
data.columns = ['Annual_Income', 'Spending_Score']

print("First 5 rows after renaming columns:")
print(data.head())



# Standardize the data
scaler = StandardScaler()
