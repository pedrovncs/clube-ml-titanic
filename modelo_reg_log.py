import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


FEATURE_LABELS = ["Pclass_1", "Pclass_2", "Pclass_3", "Age", "FamAboard", "Sex_male",
                  "Emb_Q", "Emb_S", "Emb_C"]


def train_and_predict(X_scaled, y, passenger_scaled, C, max_iter, solver, random_state):
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=random_state)
    model.fit(X_scaled, y)
    pred  = int(model.predict(passenger_scaled)[0])
    proba = float(model.predict_proba(passenger_scaled)[0][1])
    return pred, proba, model


def plot(model, passenger_scaled, proba):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Regressão Logística", fontsize=12)

    coefs  = model.coef_[0]
    colors = ["#4ECBA0" if c > 0 else "#E05C5C" for c in coefs]
    
    ax.barh(FEATURE_LABELS, coefs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes das Variáveis")
    ax.set_xlabel("Valor do Coeficiente")

    plt.tight_layout()
    plt.show(block=False)