from src.helper.arquivos import extrair_datasset, substituir_dado_json, ler_json
from src.helper.plot import plot_knn, plotar_resultado
from src.app.FeatureExtractor import FeatureExtractor
from src.helper.log import log
from src.app.KNN import KNN
import argparse


def main(tipo_ec='hist_r',modo='KNN'):
    if modo not in ['KNN', 'EC', 'RES']:
        log(f"Modo escolhido errado: {modo} - Escolha KNN ou EC")
        raise Exception()
    if modo == 'RES':
        log("Esperando resultado")
        dados = ler_json()
        plotar_resultado(dados)
        log("Resultado concluido")
        return
    carac = FeatureExtractor()
    dados, labels = carac.extracao(tipo_ec)
    if modo == 'KNN':
        X_train, X_test, y_train, y_test, y_pred, acuracia = KNN.execucao(dados, labels, 0.2)
        substituir_dado_json(acuracia, tipo_ec)
        log("Plotando...")
        # Plotar os resultados
        plot_knn(X_train, X_test, y_train, y_test, y_pred)
        log("Analise concluída")


if __name__ == "__main__":
    extrair_datasset()
    parser = argparse.ArgumentParser(description="Extração de caracteristicas de imagens e Classificação em KNN")
    parser.add_argument('--modo', action='store', dest='modo', required=True, help='Modo a ser escolhido\nEC - Somente extração de características\nRES - Plotar resultado\nKNN - Classificar')
    parser.add_argument('--tipo', action='store', dest='tipo', required=False, help='Tipo de extração a ser escolhido\nPor padrao o tipo é hist_r\nhist_r - histograma vermelho\nhist_g - histograma verde\nhist_b - histograma azul\nmsimples - momento simples\nmgeo - momento geometrico\nmhu - momento de hu\nlbp - lbp\ncanny - canny')
    arg = parser.parse_args()
    main(arg.tipo, arg.modo)
    
