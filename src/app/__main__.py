import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.app.FeatureExtractor import FeatureExtractor


# Função para combinar as características em um único vetor
def extrair_caracteristicas(file_path):
    # hist_vermelho, hist_verde, hist_azul = FeatureExtractor.histograma_rgb(file_path)

    # Momentos simples
    """(
        media,
        variancia,
        desvio_padrao,
        skewness,
        curtose,
    ) = FeatureExtractor.momento_simples(file_path)
    momentos_simples = np.array(
        [media, variancia, desvio_padrao, skewness, curtose]
    )"""  # Garantindo que seja um array

    # Momentos geométricos
    # momentos_geometricos, hu_momentos = FeatureExtractor.momento_geometrico(file_path)

    # Local Binary Pattern (LBP)
    _, hist_lbp = FeatureExtractor.lbp(file_path)

    # Vetor de bordas com Canny
    # canny = FeatureExtractor.canny(file_path)

    return [hist[0] for hist in hist_lbp]


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
    """
    Lista os arquivos de um diretório.

    :param diretorio: Caminho do diretório para listar os arquivos
    :param limite: Limite opcional para a quantidade de arquivos a serem listados
    :return: Lista com os caminhos completos dos arquivos
    """
    arquivos = []
    for nome_arquivo in os.listdir(diretorio):
        caminho_completo = os.path.join(diretorio, nome_arquivo)
        if os.path.isfile(caminho_completo):
            arquivos.append(caminho_completo)
            if limite and len(arquivos) >= limite:
                break
    return arquivos


def gerar_lista_alternada(caminhos_gato, caminhos_cachorro, limite=25):
    """
    Gera uma lista alternada com caminhos de imagens de gatos e cachorros.

    :param caminhos_gato: Lista de caminhos de imagens de gatos
    :param caminhos_cachorro: Lista de caminhos de imagens de cachorros
    :param limite: Limite de imagens de cada categoria
    :return: Lista alternada com caminhos de gatos e cachorros
    """
    lista_final = []
    labels = []
    for i in range(min(limite, len(caminhos_gato), len(caminhos_cachorro))):
        lista_final.append(caminhos_gato[i])
        labels.append(1)
        lista_final.append(caminhos_cachorro[i])
        labels.append(0)
    return lista_final, labels


def main():
    diretorio_gato = "src/images/cat/"
    diretorio_cachorro = "src/images/dog/"

    # Listar os arquivos de gatos e cachorros (até 25 de cada)
    caminhos_gato = listar_arquivos(diretorio_gato, limite=25)
    caminhos_cachorro = listar_arquivos(diretorio_cachorro, limite=25)
    # label 0 - gato
    # label 1 - cachorro

    # Gerar a lista alternada de caminhos
    lista_caminhos, labels = gerar_lista_alternada(
        caminhos_gato, caminhos_cachorro, limite=25
    )

    # Extrair características para todas as imagens
    dados = np.array(
        [extrair_caracteristicas(file_path) for file_path in lista_caminhos]
    )

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(dados, labels, test_size=0.5)
    # Escalar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinar o KNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

    # Fazer previsões
    y_pred = knn.predict(X_test)
    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do KNN: {accuracy * 100:.2f}%")

    plot_knn(X_train, X_test, y_train, y_test, y_pred)


if __name__ == "__main__":
    main()
