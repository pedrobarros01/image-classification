import matplotlib.pyplot as plt

from src.app.FeatureExtractor import FeatureExtractor


def main():
    """
    Main entry point of the application. This is where the application starts.
    """
    print("Hello, World!")
    return 0


if __name__ == "__main__":
    file_path = "src\\images\\train\\download.png"
    hist_vermelho, hist_verde, hist_azul = FeatureExtractor.histograma_rgb(file_path)
    (
        media,
        variancia,
        desvio_padrao,
        skewness,
        curtose,
    ) = FeatureExtractor.momento_simples(file_path)
    print(f"{media=}")
    print(f"{variancia=}")
    print(f"{desvio_padrao=}")
    print(f"{skewness=}")
    print(f"{curtose=}")
    momentos_geometricos, hu_momentos = FeatureExtractor.momento_geometrico(file_path)
    print(f"{momentos_geometricos=}")
    print(f"{hu_momentos=}")
    lbp, hist_lbp = FeatureExtractor.lbp(file_path)
    print(f"{hist_lbp=}")
    plt.title("LBP")
    plt.imshow(lbp)
    plt.show()
