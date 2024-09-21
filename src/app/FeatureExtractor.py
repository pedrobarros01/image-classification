import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from src.helper.log import log
from src.helper.arquivos import gerar_lista_alternada, listar_arquivos

class FeatureExtractor:
    def __init__(self) -> None:
        pass

    def __histograma_rgb(self, path_image: str):
        image = Image.open(path_image)
        canal_vermelho, canal_verde, canal_azul = image.split()
        hist_vermelho = canal_vermelho.histogram()
        hist_verde = canal_verde.histogram()
        hist_azul = canal_azul.histogram()
        return hist_vermelho, hist_verde, hist_azul

    def __momento_simples(self, path_image: str):
        image2 = Image.open(path_image)
        image2 = image2.convert("L")

        array_image = np.array(image2)
        media = np.mean(array_image)
        variancia = np.mean((array_image - media) ** 2)
        desvio_padrao = np.std(array_image)
        skewness = np.mean(((array_image - media) / desvio_padrao) ** 3)
        curtose = np.mean((((array_image - media) / desvio_padrao) ** 4) - 3)
        return media, variancia, desvio_padrao, skewness, curtose

    def __momento_geometrico(self, path_image: str):
        image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        _, image_binaria = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        momentos_geometricos = cv2.moments(image_binaria)
        hu_momentos = cv2.HuMoments(momentos_geometricos)
        return momentos_geometricos, hu_momentos

    def __lbp(self, path_image: str):
        imagem = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        raio = 1
        num_pontos = 8 * raio
        lbp = local_binary_pattern(imagem, num_pontos, raio, "uniform")
        n_bins = int(lbp.max() + 1)
        histograma = cv2.calcHist(
            [lbp.astype(np.uint8)], [0], None, [n_bins], [0, n_bins]
        )
        histograma = histograma / histograma.sum()
        return lbp, np.asarray(histograma).flatten()

    def __canny(self, path_image: str):
        imagem = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        bordas = cv2.Canny(imagem, 100, 200)
        coordenadas_bordas = np.column_stack(np.where(bordas > 0))
        num_bordas = min(100, coordenadas_bordas.shape[0])
        vetor_caracteristicas = coordenadas_bordas[:num_bordas].flatten()
        return vetor_caracteristicas
    
    def __extrair_caracteristicas_hist(self, tipo_ec, file_path):
        hist_vermelho, hist_verde, hist_azul = self.__histograma_rgb(file_path)
        if tipo_ec == 'hist_r':
            return hist_vermelho
        elif tipo_ec == 'hist_g':
            return hist_verde
        elif tipo_ec == 'hist_b':
            return hist_azul
    
    def __extrair_momento_simples(self, file_path):
            (
                media,
                variancia,
                desvio_padrao,
                skewness,
                curtose,
            ) = self.__momento_simples(file_path)
            momentos_simples = np.array(
                [media, variancia, desvio_padrao, skewness, curtose]
            )  # Garantindo que seja um array
            return momentos_simples
    
    def __extrair_momento_complexo(self, tipo_ec, file_path):
            # Momentos geométricos
            momentos_geometricos, hu_momentos = self.__momento_geometrico(file_path)
            hu_momentos = np.asarray(hu_momentos).flatten()
            if tipo_ec == 'mgeo':
                return [momentos_geometricos[chave] for chave in momentos_geometricos.keys()]
            elif tipo_ec == 'mhu':
                return hu_momentos
    
    def __extrair_lbp(self, file_path):
        # Local Binary Pattern (LBP)
        _, hist_lbp = self.__lbp(file_path)
        return hist_lbp
    
    def __extrair_canny(self, file_path):
        # Vetor de bordas com Canny
        canny = self.__canny(file_path)
        return canny

    def __extrair_caracteristicas(self, file_path, tipo_ec):
        if tipo_ec in ['hist_r', 'hist_g', 'hist_b']:
            return self.__extrair_caracteristicas_hist(tipo_ec, file_path)
        elif tipo_ec == 'msimples':
            return self.__extrair_momento_simples(file_path)
        elif tipo_ec in ['mgeo', 'mhu']:
            return self.__extrair_momento_complexo(tipo_ec, file_path)
        elif tipo_ec == 'lbp':
            return self.__extrair_lbp(file_path)
        elif tipo_ec == 'canny':
            return self.__extrair_canny(file_path)
        else:
            log("Erro de comando de tipo_ec")
            raise Exception()
    def extracao(self, tipo_ec):
        log("Capturando imagens...")
        caminhos_cachorro, caminhos_gato = listar_arquivos(500)
        lista_caminhos, labels = gerar_lista_alternada(caminhos_gato, caminhos_cachorro)
        log("Imagens capturadas començando extração de caracteristicas...")
        dados = np.array(
            [self.__extrair_caracteristicas(file_path, tipo_ec) for file_path in lista_caminhos]
        )

        log("Extração de caracteristicas concluida...")
        log(dados)
        #print(dados)
        return dados, labels
