import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pclass_1", "Pclass_2", "Pclass_3", "Age", "FamAboard", "Sex_male",
            "Embarked_Q", "Embarked_S", "Embarked_C"]

def load_and_prepare(csv_path: str):
    """
    Colunas: Survived, Pclass, Age, SibSp, Parch, Sex, Embarked.
    """
    df = pd.read_csv(csv_path)

    df["Age"]       = df["Age"].fillna(df["Age"].median())
    df["Embarked"]  = df["Embarked"].fillna("S")
    df["FamAboard"] = df["SibSp"] + df["Parch"]
    df["Sex_male"]  = (df["Sex"] == "male").astype(int)

    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked")
    for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0
            
    df = pd.get_dummies(df, columns=["Pclass"], prefix="Pclass")
    for col in ["Pclass_1", "Pclass_2", "Pclass_3"]:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURES].fillna(0).values.astype(float)
    y = df["Survived"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler

def build_passenger(pclass, age, fam, sex_male, embarked):
    import numpy as np
    
    pc_1 = 1 if pclass == 1 else 0
    pc_2 = 1 if pclass == 2 else 0
    pc_3 = 1 if pclass == 3 else 0
    
    emb_q = 1 if embarked == "Q" else 0
    emb_s = 1 if embarked == "S" else 0
    emb_c = 1 if embarked == "C" else 0
    
    return np.array([[pc_1, pc_2, pc_3, age, fam, sex_male, emb_q, emb_s, emb_c]], dtype=float)