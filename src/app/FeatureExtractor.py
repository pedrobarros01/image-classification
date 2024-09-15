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
