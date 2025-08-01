# visao-end2end

Pipeline de análise de vídeo *end‑to‑end*, com a leitura de vídeos, armazenamento, processamento, classificação e análises.


## Estrutura do projeto
**visao-end2end/**
- data/ *# Vídeo de entrada e modelo treinado*
- resultados/ *# Matriz de confusão e saídas de detecção*
- src/ *# Código fonte principal*
    - bbox_operations.py  *# Manipulações de bounding boxes*
    - database.py  *# Operações no banco de dados*
    - extract.py  *# Extração de imagem e áudio do vídeo*
    - image_operations.py  *# Manipulações de imagem*
- requirements.txt  *# Dependências do projeto*
- Pipfile  *# Dependências do projeto (pipenv)*
- video_analysis.ipynb  *# Notebook com o processamento completo*



## Tecnologias utilizadas

- Python 3.8+
- OpenCV
- Moviepy
- Astropy
- mysql-connector-python
- scikit-learn
- Whisper
- Outras dependências conforme `requirements.txt` ou `Pipfile`


## Instalação e execução

### 1. Clonar o repositório

```bash
git clone https://github.com/mariliafernandez/visao-end2end.git
cd visao-end2end
```

### 2. Criar e ativar ambiente virtual
#### 2.1: Usando `venv` 
```bash
python3 -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
```

#### 2.2: Usando `pipenv` 
```bash
pip install pipenv
pipenv shell
```

### 3. Instalar as dependências

#### 3.1: Usando `venv`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3.2: Usando `pipenv`
```bash
pipenv install
```

### 4. Executar a solução
Abra o notebook `video_analysis.ipynb` e vá executando as células. Para utilizar o modelo treinado siga as instruções dentro do notebook.