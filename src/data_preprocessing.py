import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # 1. Manejo de valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # 2. Eliminar columnas irrelevantes
    df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    
    # 3. Codificación one-hot
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df

if __name__ == "__main__":
    train = pd.read_csv('../data/train_original.csv')
    test = pd.read_csv('../data/test_original.csv')
    
    train_processed = preprocess_data(train)
    test_processed = preprocess_data(test)
    
    train_processed.to_csv('../data/processed/train_processed.csv', index=False)
    test_processed.to_csv('../data/processed/test_processed.csv', index=False)