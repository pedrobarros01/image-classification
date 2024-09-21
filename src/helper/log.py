import datetime
def log(mensagem):
    log_mensagem = f'[{datetime.datetime.now().date()}] - {mensagem}'
    print(log_mensagem)
    file_name = f'src/log/log_{datetime.datetime.now().date()}.txt'
    with open(file_name, 'a+') as file:
        file.write(log_mensagem + '\n')
        file.close()