import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar datos procesados
train = pd.read_csv('../data/processed/train_features.csv')

# Definir features y target
X = train.drop(['PassengerId', 'Survived'], axis=1)
y = train['Survived']

# Split train-val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Guardar modelo
joblib.dump(model, '../models/titanic_model.pkl')