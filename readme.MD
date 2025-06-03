Passo a Passo para Rodar o Projeto

 1. Requisitos:
    - Python 3.10 ou superior instalado
    - Navegador web (Chrome, Firefox, etc.)

 2. Estrutura do Projeto:

    projeto_final/
    │
    ├── app.py                   # Arquivo principal para rodar a aplicação
    ├── arvore_modelo.py         # Script com a lógica do modelo para treinar e implementar a árvore de decisão
    ├── gerar_dataset.py         # (opcional) Script para gerar dados sintéticos
    ├── dataset.csv              # Dataset padrão (que foi disponibilizado pelo professor Douglas lá no AVA)
    ├── dataset_final.csv        # Dataset final, de onde os dados para o relatório foram tirados
    ├── requirements.txt         # Lista de dependências
    ├── templates/
    │   └── index.html           # Página HTML principal com o dashboard da aplicação
    ├── static/                  # Onde ficam as imagens dos gráficos obtidos no modelo
    │
    ├── uploads/                 # Onde o dataset enviado pelo usuário será salvo
    │
    └── notebooks/               # Notebook com a análise exploratória, modelagem e visualizações do projeto.

 3. Criação do Ambiente Virtual:
    No terminal, vá para dentro da pasta do projeto, e rode:
    python -m venv venv
    Após criar o ambiente virtual, ative-o:
    venv\Scripts\activate

 4. Instalação das Dependências:
    Com o ambiente ativado, instale os pacotes:
    pip install -r requirements.txt

 5. Execução do Projeto:
    Com tudo instalado, execute:
    python app.py

    Você verá algo como:
    * Running on http://127.0.0.1:5000
    Basta clicar no link ou copiar e colar no navegador.

 6. Uso da Interface Web:
    Na interface:

    - O modelo já vem com um dataset padrão(disponibilizado pelo professor), contendo 9700 registros. o dashboard já inicia com as informações dele.
    - Você pode enviar um novo dataset, no painel lateral. Desde que o dataset tenha os campos aceitos.
    - As visualizações (árvore, matriz de confusão) serão geradas automaticamente.
    - Também é exibido o relatório, acurácia, features e outras métricas.

 7. (Opcional) Gerar novo dataset
    Se quiser testar um novo dataset, basta executar no terminal o seguinte comando:
    python gerar_dataset.py
    Isso irá gerar um novo dataset_gerado.csv com dados sintéticos.

 8. Caso queira usar o dataset final(o que foi usado no relatório)
    Basta importar lá na aplicação web, o dataset_final.csv
