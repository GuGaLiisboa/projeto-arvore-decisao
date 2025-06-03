from flask import Flask, render_template, request
from arvore_modelo import treinar_modelo
import os

# Instanciando a aplicação Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # Pasta para armazenar os arquivos enviados
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Cria a pasta se não existir

# Configurando o caminho para os arquivos estáticos
@app.route('/', methods=['GET', 'POST'])
def index():
    dataset_path = 'dataset.csv'
    max_depth = 5

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
            file.save(path) # Salva o arquivo no servidor, para evitar voltar o dataset original ao recarregar a pagina
            dataset_path = path

    # Treina o modelo e retorna as métricas e visualizações
    acc, classes, features, relatorio, total_amostras, total_features, tempo_execucao = treinar_modelo(dataset_path, max_depth=max_depth)

    # Renderiza o template HTML com os resultados
    return render_template(
        "index.html",
        acc=acc,
        algoritmo="Árvore de Decisão",
        classes=classes,
        features=features,
        relatorio=relatorio,
        total_amostras=total_amostras,
        total_features=total_features,
        tempo_execucao=tempo_execucao
    )
# Executa o servidor em modo debug (apenas para desenvolvimento)
if __name__ == '__main__':
    app.run(debug=True)
