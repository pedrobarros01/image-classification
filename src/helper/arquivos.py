import os
from zipfile import ZipFile
from src.helper.log import log
import json
def listar_arquivos(limite=None):
    caminhos_cachorro = 'src/images/animal/dog'
    caminhos_gato = 'src/images/animal/cat'
    arquivos_cachorro = []
    arquivos_gato = []
    for nome_arquivo in os.listdir(caminhos_cachorro):
        caminho_completo = os.path.join(caminhos_cachorro, nome_arquivo)
        if os.path.isfile(caminho_completo):
            arquivos_cachorro.append(caminho_completo)
            if limite and len(arquivos_cachorro) >= limite:
                break

    for nome_arquivo in os.listdir(caminhos_gato):
        caminho_completo = os.path.join(caminhos_gato, nome_arquivo)
        if os.path.isfile(caminho_completo):
            arquivos_gato.append(caminho_completo)
            if limite and len(arquivos_gato) >= limite:
                break
    return arquivos_cachorro, arquivos_gato


def gerar_lista_alternada(caminhos_gato, caminhos_cachorro):
    lista_final = []
    labels = []
    for i in range(len(caminhos_cachorro)):
        lista_final.append(caminhos_gato[i])
        labels.append(1)
        lista_final.append(caminhos_cachorro[i])
        labels.append(0)
    return lista_final, labels


def extrair_datasset():
    path = 'src/images/dog-vs-cat.zip'
    path_resultado = 'src/images/animal/'
    if not os.path.exists(path):
        raise Exception('Erro em datasset', 'Datasset nao baixado')
    if os.path.exists(path_resultado):
        log('Datasset já extraido, não precisa extrair novamente')
        return
    with ZipFile(path, 'r') as zipObjeto:
        zipObjeto.extractall('src/images/')
        zipObjeto.close()
    log('Datasset extraido com sucesso')

def substituir_dado_json(acuracia, tipo):
    dados = None
    arquivo_json = 'src/app/resultado.json'
    with open(arquivo_json, 'r') as file:
        dados = json.load(file)
        file.close()
    dados[tipo] = acuracia
    with open(arquivo_json, 'w') as file:
        json.dump(dados, file)
        file.close()

def ler_json():
    dados = None
    arquivo_json = 'src/app/resultado.json'
    with open(arquivo_json, 'r') as file:
        dados = json.load(file)
        file.close()
    return dados
    
