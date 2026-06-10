import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


FEATURE_LABELS = ["Pclass_1", "Pclass_2", "Pclass_3", "Age", "FamAboard", "Sex_male",
                  "Emb_Q", "Emb_S", "Emb_C"]


def train_and_predict(X, y, passenger, max_depth, criterion, min_samples_leaf, random_state):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X, y)
    pred  = int(model.predict(passenger)[0])
    proba = float(model.predict_proba(passenger)[0][1])
    return pred, proba, model


def plot(model):
    depth  = model.get_depth()
    height = max(5, depth * 2)

    fig, ax = plt.subplots(figsize=(max(12, depth * 4), height))
    fig.suptitle("Árvore de Decisão", fontsize=12)

    plot_tree(
        model,
        ax=ax,
        feature_names=FEATURE_LABELS,
        class_names=["Morreu", "Sobreviveu"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=8,
    )

    plt.tight_layout()
    plt.show(block=False)