import pandas as pd

def create_features(df):
    # 1. Crear feature "FamilySize"
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 2. Crear feature "IsAlone"
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. Discretizar "Fare"
    df['FareCategory'] = pd.cut(df['Fare'], bins=[0, 10, 50, 100, 600], labels=[1, 2, 3, 4])
    
    return df

if __name__ == "__main__":
    train = pd.read_csv('../data/processed/train_processed.csv')
    test = pd.read_csv('../data/processed/test_processed.csv')
    
    train = create_features(train)
    test = create_features(test)
    
    train.to_csv('../data/processed/train_features.csv', index=False)
    test.to_csv('../data/processed/test_features.csv', index=False)