import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.app.FeatureExtractor import FeatureExtractor


def extrair_caracteristicas(file_path):
    # hist_vermelho, hist_verde, hist_azul = FeatureExtractor.histograma_rgb(file_path)

    # Momentos simples
    (
        media,
        variancia,
        desvio_padrao,
        skewness,
        curtose,
    ) = FeatureExtractor.momento_simples(file_path)
    momentos_simples = np.array(
        [media, variancia, desvio_padrao, skewness, curtose]
    )  # Garantindo que seja um array

    # Momentos geométricos
    # momentos_geometricos, hu_momentos = FeatureExtractor.momento_geometrico(file_path)
    # hu_momentos = np.asarray(hu_momentos).flatten()
    # Local Binary Pattern (LBP)
    # _, hist_lbp = FeatureExtractor.lbp(file_path)

    # Vetor de bordas com Canny
    # canny = FeatureExtractor.canny(file_path)

    return momentos_simples


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


def listar_arquivos(diretorio, limite=None):
    arquivos = []
    for nome_arquivo in os.listdir(diretorio):
        caminho_completo = os.path.join(diretorio, nome_arquivo)
        if os.path.isfile(caminho_completo):
            arquivos.append(caminho_completo)
            if limite and len(arquivos) >= limite:
                break
    return arquivos


def gerar_lista_alternada(caminhos_gato, caminhos_cachorro):
    lista_final = []
    labels = []
    for i in range(len(caminhos_cachorro)):
        lista_final.append(caminhos_gato[i])
        labels.append(1)
        lista_final.append(caminhos_cachorro[i])
        labels.append(0)
    return lista_final, labels


def main():
    diretorio_gato = "src/images/cat"
    diretorio_cachorro = "src/images/dog"

    print("Capturando imagens...")
    caminhos_gato = listar_arquivos(diretorio_gato, 500)
    caminhos_cachorro = listar_arquivos(diretorio_cachorro, 500)

    lista_caminhos, labels = gerar_lista_alternada(caminhos_gato, caminhos_cachorro)
    print(len(lista_caminhos))
    print("Imagens capturadas començando extração de caracteristicas...")
    dados = np.array(
        [extrair_caracteristicas(file_path) for file_path in lista_caminhos]
    )
    print("Extração de caracteristicas concluida...")
    print(dados)

    X_train, X_test, y_train, y_test = train_test_split(dados, labels, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    melhor_acuracia = float("-inf")
    melhor_n_neighbors = 0
    print("Executando KNN para achar melhor vizinho...")
    for i in range(1, 41):
        print(f"{i} de 40")
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn, dados, labels, cv=5)
        acuracia = scores.mean() * 100
        print(f"Acurácia: {acuracia:.2f}%")
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_n_neighbors = i

    print(f"Número de vizinhos ótimo: {melhor_n_neighbors}")
    print(f"Acurácia: {melhor_acuracia:.2f}%")
    print("Iniciando plot dos resultados...")

    knn = KNeighborsClassifier(n_neighbors=melhor_n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Plotando...")
    # Plotar os resultados
    plot_knn(X_train, X_test, y_train, y_test, y_pred)
    print("Algoritmo concluído")


if __name__ == "__main__":
    # main()
    resultados = {
        "tipo": [
            "m_simples",
            "m_geometrico",
            "hu_momento",
            "lbp",
            "canny",
            "hist_vermelho",
            "hist_verde",
            "hist_azul",
        ],
        "acuracia": [55.70, 62.50, 55.70, 71.00, 56.10, 61.70, 62.80, 66.80],
    }

    plt.figure(figsize=(10, 6))
    plt.bar(resultados["tipo"], resultados["acuracia"], color="skyblue")

    plt.title("Acurácia por Tipo de Característica", fontsize=14)
    plt.xlabel("Tipo de Característica", fontsize=12)
    plt.ylabel("Acurácia (%)", fontsize=12)

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
