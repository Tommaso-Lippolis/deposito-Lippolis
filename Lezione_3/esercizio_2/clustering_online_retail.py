import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


df = pd.read_csv('Lezione_3\esercizio_2\Online Retail dataset\Online_Retail.csv', encoding='ISO-8859-1')

print('Dimensioni del dataset:', df.shape)

# Visualizziamo le prime righe del dataset
print(df.head())
# Informazioni generali sul dataset
print(df.info())

# Statistiche descrittive del dataset
print('Statistiche descrittive del dataset:')
print(df.describe())

# Controllo dei valori nulli
print('Controllo dei valori nulli:')
print(df.isnull().sum())

# rimuoviamo colonna descrizione
df = df.drop(columns=['Description'])

# controlliamo quanti valori diverti ci sono nella colonna 'Country'
print('Valori unici nella colonna "Country":')
print(df['Country'].nunique())

print(df['InvoiceNo'].nunique())

# identifichiamo le righe di 'InvoiceNo' che iniziano con 'C' (cancellazioni)
print('Righe di "InvoiceNo" che iniziano con "C":')
print(df[df['InvoiceNo'].str.startswith('C')])

# rimuoviamo le righe di InvoiceNo che iniziano con 'C' (cancellazioni)
df = df[~df['InvoiceNo'].str.startswith('C')]

print('Dopo la rimozione delle cancellazioni, il numero di righe è:', len(df))


# feature engineering

print('Feature Engineering: Creazione di nuove colonne')

# creiamo Customer Lifetime Value come totale speso da ogni cliente come moltplicazione di UnitPrice e Quantity
df['Customer Lifetime Value'] = df.groupby('CustomerID')['UnitPrice'].transform(lambda x: x * df['Quantity'])

# Calcola il valore totale per ogni transazione (Quantity * UnitPrice)
df['TotalValue'] = df['Quantity'] * df['UnitPrice']
 
# Analisi completa per ogni cliente
customer_analysis = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',         # Numero totale di acquisti (fatture uniche)
    'TotalValue': ['sum', 'mean'],  # Totale speso e spesa media per transazione
    'InvoiceDate': 'nunique',       # Frequenza di acquisto (giorni unici di acquisto)
    'Country': 'first'              # Paese di appartenenza
}).reset_index()
 
# Semplifica i nomi delle colonne
customer_analysis.columns = ['CustomerID', 'Numero_Acquisti', 'CLV', 'Spesa_Media_per_transazione', 'Frequenza_Acquisto', 'Paese']
 
# Calcola la spesa media per acquisto (non per transazione)
customer_analysis['Spesa_Media_per_Acquisto'] = customer_analysis['CLV'] / customer_analysis['Numero_Acquisti']
 
# Mostra i risultati
print(customer_analysis.head(10))

# label encoding per la colonna 'Country'

le = LabelEncoder()
customer_analysis['Paese'] = le.fit_transform(customer_analysis['Paese'])

print('Colonna "Paese" dopo Label Encoding:')
print(customer_analysis.head())


# standardizzazione delle feature numeriche
scaler = StandardScaler()
numerical_features = ['Numero_Acquisti', 'CLV', 'Spesa_Media_per_transazione', 'Frequenza_Acquisto', 'Spesa_Media_per_Acquisto']

customer_analysis[numerical_features] = scaler.fit_transform(customer_analysis[numerical_features])



   