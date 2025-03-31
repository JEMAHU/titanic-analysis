import pytest
import pandas as pd
from src.model import train_model

def test_model_accuracy():
    # Datos de prueba
    X = pd.DataFrame({
        'Pclass': [1, 3, 2],
        'Age': [30, 25, 40],
        'Fare': [100, 20, 50],
        'Sex_male': [0, 1, 1],
        'Embarked_Q': [0, 0, 1]
    })
    y = pd.Series([1, 0, 1])
    
    model = train_model(X, y)
    pred = model.predict([[1, 30, 100, 0, 0]])
    assert pred[0] == 1