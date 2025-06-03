import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

#Função para treinar o modelo de árvore de decisão
def treinar_modelo(caminho_csv="dataset.csv", max_depth=5): #profundidade máxima da árvore
    inicio = time.time()
    # Carregar o dataset
    df = pd.read_csv(caminho_csv)
    df["Data_Pedido"] = pd.to_datetime(df["Data_Pedido"], format="%d/%m/%Y")
    df["Ano"] = df["Data_Pedido"].dt.year
    df["Mes"] = df["Data_Pedido"].dt.month

    label_encoders = {}
    for col in ["Segmento", "Pais", "Estado", "Cidade"]:
        le = LabelEncoder()
        df[f"{col}_Encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Verificar se a coluna 'Categoria' existe
    X = df[["Valor_Venda", "Segmento_Encoded", "Pais_Encoded", "Estado_Encoded", "Mes", "Ano"]]
    y = df["Categoria"]

    total_amostras = X.shape[0]
    total_features = X.shape[1]
    # Verificar se há amostras suficientes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Treinar o modelo de árvore de decisão
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Avaliar o modelo
    acc = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, output_dict=False)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Previsão")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("static/imagens/confusion_matrix.png")
    plt.close()

    # Árvore de decisão
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_, rounded=True)
    plt.title("Árvore de Decisão")
    plt.tight_layout()
    plt.savefig("static/imagens/decision_tree.png")
    plt.close()

    # Importância das features
    feature_importances = dict(zip(X.columns, model.feature_importances_))
    sorted_features = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    tempo_execucao = round(time.time() - inicio, 2)
    # Salvar o modelo treinado
    return (
        f"{acc*100:.2f}%",
        model.classes_.tolist(),
        sorted_features,
        relatorio,
        total_amostras,
        total_features,
        tempo_execucao
    )
