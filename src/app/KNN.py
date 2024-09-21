from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.helper.log import log

class KNN:
    def __init__(self) -> None:
        pass

    @classmethod
    def execucao(clas, dados, labels, test_size):
        X_train, X_test, y_train, y_test = train_test_split(dados, labels, test_size=test_size)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        melhor_acuracia = float("-inf")
        melhor_n_neighbors = 0
        log("Executando KNN para achar melhor vizinho...")
        for i in range(1, 41):
            log(f"{i} de 40")
            knn = KNeighborsClassifier(n_neighbors=i)
            scores = cross_val_score(knn, dados, labels, cv=5)
            acuracia = scores.mean() * 100
            log(f"Acurácia: {acuracia:.2f}%")
            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_n_neighbors = i

        log(f"Número de vizinhos ótimo: {melhor_n_neighbors}")
        log(f"Acurácia: {melhor_acuracia:.2f}%")
        log("Iniciando plot dos resultados...")

        knn = KNeighborsClassifier(n_neighbors=melhor_n_neighbors)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        return X_train, X_test, y_train, y_test, y_pred, melhor_acuracia