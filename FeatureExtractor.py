import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score




import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern


class FeatureExtractor:
    def __init__(self) -> None:
        pass

    @classmethod
    def histograma_rgb(cls, path_image: str):
        image = Image.open(path_image)
        canal_vermelho, canal_verde, canal_azul = image.split()
        hist_vermelho = canal_vermelho.histogram()
        hist_verde = canal_verde.histogram()
        hist_azul = canal_azul.histogram()
        return hist_vermelho, hist_verde, hist_azul

    @classmethod
    def momento_simples(cls, path_image: str):
        image2 = Image.open(path_image)
        image2 = image2.convert("L")

        array_image = np.array(image2)
        media = np.mean(array_image)
        variancia = np.mean((array_image - media) ** 2)
        desvio_padrao = np.std(array_image)
        skewness = np.mean(((array_image - media) / desvio_padrao) ** 3)
        curtose = np.mean((((array_image - media) / desvio_padrao) ** 4) - 3)
        return media, variancia, desvio_padrao, skewness, curtose

    @classmethod
    def momento_geometrico(cls, path_image: str):
        image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        _, image_binaria = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        momentos_geometricos = cv2.moments(image_binaria)
        hu_momentos = cv2.HuMoments(momentos_geometricos)
        return momentos_geometricos, hu_momentos

    @classmethod
    def lbp(cls, path_image: str):
        imagem = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        raio = 1
        num_pontos = 8 * raio
        lbp = local_binary_pattern(imagem, num_pontos, raio, "uniform")
        n_bins = int(lbp.max() + 1)
        histograma = cv2.calcHist(
            [lbp.astype(np.uint8)], [0], None, [n_bins], [0, n_bins]
        )
        histograma = histograma / histograma.sum()
        return lbp, histograma

    @classmethod
    def histograma_cinza(cls, image=None, path_image=""):
        if image is None:
            image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        return cv2.calcHist([image], [0], None, [256], [0, 255])

    @classmethod
    def canny(cls, path_image: str):
        imagem = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        bordas = cv2.Canny(imagem, 100, 200)
        coordenadas_bordas = np.column_stack(np.where(bordas > 0))
        num_bordas = min(100, coordenadas_bordas.shape[0])
        vetor_caracteristicas = coordenadas_bordas[:num_bordas].flatten()
        return vetor_caracteristicas


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
    for i in range(500):
        lista_final.append(caminhos_gato[i])
        labels.append(1)
        lista_final.append(caminhos_cachorro[i])
        labels.append(0)
    return lista_final, labels


def main():
    diretorio_gato = "C:\\Users\\019.705313\\Downloads\\image-classification-main\\image-classification-main\\src\\images\\cat"
    diretorio_cachorro = "C:\\Users\\019.705313\\Downloads\\image-classification-main\\image-classification-main\\src\\images\\dog"

    # Listar os arquivos de gatos e cachorros (até 25 de cada)
    caminhos_gato = listar_arquivos(diretorio_gato, 500)
    caminhos_cachorro = listar_arquivos(diretorio_cachorro)
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
    X_train, X_test, y_train, y_test = train_test_split(dados, labels, test_size=0.3)
    # Escalar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinar o KNN
    maior = float('-inf')
    i_maior = 0
    for i in range (100):
        print(i + 1)
        knn = KNeighborsClassifier(n_neighbors=i + 1)
        scores = cross_val_score(knn, dados, labels, cv=6)  # 5-fold cross-validation´
        if float(scores.mean() * 100) > float(maior):
            maior = float(scores.mean() * 100)
            i_maior = i + 1
        print(f"Acurácia média da validação cruzada: {scores.mean() * 100:.2f}%")
        print(f"Desvio padrão da acurácia da validação cruzada: {scores.std() * 100:.2f}%")
    print(f'acuracia: {maior=}')
    print(f'neighbor maior: {i_maior=}')


if __name__ == "__main__":
    main()
