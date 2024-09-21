from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.helper.log import log
def plot_knn(X_train, X_test, y_train, y_test, y_pred):
    # Reduzir para 2 dimensões para visualização
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    plt.figure(figsize=(10, 6))

    # Plotar pontos de treino
    plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="viridis",
        label="Treinamento",
        marker="o",
        edgecolor="k",
    )

    # Plotar pontos de teste com as predições
    plt.scatter(
        X_test_2d[:, 0],
        X_test_2d[:, 1],
        c=y_pred,
        cmap="coolwarm",
        label="Teste (Predição)",
        marker="x",
        edgecolor="k",
    )

    plt.title("Classificação KNN (Grupos)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotar_resultado(dados):
    for chave in dados.keys():
        if dados[chave] is None:
            log(f"{chave} esta vazio no json, execute o knn para essa caracteristica")
            raise Exception()
    resultados = {
        'tipo': dados.keys(),
        'acuracia': dados.values()
    }
    plt.figure(figsize=(10, 6))
    plt.bar(resultados["tipo"], resultados["acuracia"], color="skyblue")

    plt.title("Acurácia por Tipo de Característica", fontsize=14)
    plt.xlabel("Tipo de Característica", fontsize=12)
    plt.ylabel("Acurácia (%)", fontsize=12)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()