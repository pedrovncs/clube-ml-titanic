import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


FEATURE_LABELS = ["Pclass", "Age", "FamAboard", "Sex_male",
                  "Emb_Q", "Emb_S", "Emb_C"]


def train_and_predict(X_scaled, y, passenger_scaled, C, max_iter, solver, random_state):
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=random_state)
    model.fit(X_scaled, y)
    pred  = int(model.predict(passenger_scaled)[0])
    proba = float(model.predict_proba(passenger_scaled)[0][1])
    return pred, proba, model


def plot(model, passenger_scaled, proba):
    log_odds_range = np.linspace(-6, 6, 300)
    sigmoid = 1 / (1 + np.exp(-log_odds_range))

    passenger_log_odds = (np.dot(model.coef_, passenger_scaled.T) + model.intercept_).item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Logistic Regression", fontsize=12)

    ax1.plot(log_odds_range, sigmoid, color="#4A9FC8", linewidth=2)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax1.axvline(passenger_log_odds, color="gray", linestyle="--", linewidth=1)
    dot_color = "#4ECBA0" if proba >= 0.5 else "#E05C5C"
    ax1.scatter([passenger_log_odds], [proba], s=100, color=dot_color, zorder=5,
                label=f"Passenger: {proba*100:.1f}%")
    ax1.set_xlabel("Log-odds")
    ax1.set_ylabel("P(Survived)")
    ax1.set_title("Sigmoid curve")
    ax1.legend()

    coefs  = model.coef_[0]
    colors = ["#4ECBA0" if c > 0 else "#E05C5C" for c in coefs]
    ax2.barh(FEATURE_LABELS, coefs, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("Feature coefficients")
    ax2.set_xlabel("Coefficient value")

    plt.tight_layout()
    plt.show(block=False)