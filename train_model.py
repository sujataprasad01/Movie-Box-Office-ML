import ast
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def safe_literal_eval(x):
    try:
        return len(ast.literal_eval(x)) if isinstance(x, str) else 0
    except (ValueError, SyntaxError):
        return 0

def extract_genre_names(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        genre_names = [genre['name'] for genre in genres_list if 'name' in genre]
        return ', '.join(genre_names)
    except (ValueError, SyntaxError, TypeError):
        return 'Unknown'

train = pd.read_csv('train.csv')

train['budget'] = train['budget'].fillna(train['budget'].median())
train['runtime'] = train['runtime'].fillna(train['runtime'].median())
train['popularity'] = train['popularity'].fillna(train['popularity'].median())
train['genres'] = train['genres'].fillna('Unknown')
train['production_companies'] = train['production_companies'].fillna('Unknown')

train['release_date'] = pd.to_datetime(train['release_date'], errors='coerce')
train['release_year'] = train['release_date'].dt.year
train['release_month'] = train['release_date'].dt.month
train['release_day'] = train['release_date'].dt.day

train['genres'] = train['genres'].apply(extract_genre_names)
train['production_companies_count'] = train['production_companies'].apply(safe_literal_eval)

features = ['budget', 'popularity', 'runtime', 'release_year', 'release_month', 'release_day', 'production_companies_count']
X = train[features]
y = train['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('box_office_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved.")
