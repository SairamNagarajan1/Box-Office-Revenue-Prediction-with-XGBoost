# box_office_revenue_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# Step 2: Load the dataset
df = pd.read_csv('boxoffice.csv', encoding='latin-1')

# Step 2.1: Dataset shape
print("Dataset shape:", df.shape)

# Step 2.2: Data types
print(df.info())

# Step 3: Statistical summary
print(df.describe().T)

# Drop irrelevant revenue columns
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)

# Step 3.1: Check missing values
print("Missing values (in %):\n", df.isnull().sum() * 100 / df.shape[0])

# Step 4: Handle missing values
df.drop('budget', axis=1, inplace=True)
for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])
df.dropna(inplace=True)

# Step 4.1: Clean numeric columns

# Remove '$' and ',' and convert to numeric
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]
for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 5: MPAA Rating Distribution
plt.figure(figsize=(10, 5))
sb.countplot(df['MPAA'])
plt.title('MPAA Rating Distribution')
plt.show()

# Step 5.1: Average Domestic Revenue by MPAA
print(df.groupby('MPAA')['domestic_revenue'].mean())

# Step 6: Distribution of numeric features
plt.subplots(figsize=(15, 5))
features = ['domestic_revenue', 'opening_theaters', 'release_days']
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Step 7: Detecting outliers using boxplots
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Step 8: Log transformation
for col in features:
    df[col] = df[col].apply(lambda x: np.log10(x) if x > 0 else 0)

# Step 8.1: Post-log distribution
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Step 9: Genre transformation
vectorizer = CountVectorizer()
features_matrix = vectorizer.fit_transform(df['genres']).toarray()
genres = vectorizer.get_feature_names_out()
for i, genre in enumerate(genres):
    df[genre] = features_matrix[:, i]
df.drop('genres', axis=1, inplace=True)

# Step 9.1: Remove rare genres
if 'action' in df.columns and 'western' in df.columns:
    for col in df.loc[:, 'action':'western'].columns:
        if (df[col] == 0).mean() > 0.95:
            df.drop(col, axis=1, inplace=True)

# Step 10: Label Encoding
for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Step 11: Correlation matrix
plt.figure(figsize=(8, 8))
sb.heatmap(df.select_dtypes(include=np.number).corr() > 0.8, annot=True, cbar=False)
plt.title('Highly Correlated Features (r > 0.8)')
plt.show()

# Step 12: Prepare train/test data
X = df.drop(['title', 'domestic_revenue'], axis=1)
y = df['domestic_revenue'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=22)

# Step 12.1: Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 13: XGBoost Model
model = XGBRegressor()
model.fit(X_train, y_train)

# Step 14: Evaluation
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

print("Training MAE:", mae(y_train, train_preds))
print("Validation MAE:", mae(y_val, val_preds))
