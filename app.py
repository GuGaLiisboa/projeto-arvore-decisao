from flask import Flask, render_template, request
from arvore_modelo import treinar_modelo
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    dataset_path = 'dataset.csv'
    max_depth = 5

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
            file.save(path)
            dataset_path = path

    acc, classes, features, relatorio, total_amostras, total_features, tempo_execucao = treinar_modelo(dataset_path, max_depth=max_depth)

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

if __name__ == '__main__':
    app.run(debug=True)
