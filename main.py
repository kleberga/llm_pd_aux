# comando executados antes de executar este script:
# 1. instalar bibliotecas: pip install -q kaggle pandas transformers
# 2. copiar o arquivo de credenciais do kaggle para outra pasta: xcopy C:\Users\Kleber\Documents\curso_intel_artificial\processamento_linguagem\kaggle.json .kaggle/
# 3. baixar o dataset: kaggle datasets download --force -d marlesson/news-of-the-site-folhauol
# 4. instalar o NVIDIA Cuda para utilizar a GPU: https://developer.nvidia.com/cuda-downloads
# 5. instalar o torch, torchvision e torchaudio para usar a GPU na execução do modelo: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 6. atualizar as bibliotecas citadas no item anterior: pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ===========================================================================================
# carga da base de dados, limpeza dos dados, carga do modelo e aplicação do modelo
# ===========================================================================================

import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from transformers import BertForTokenClassification, DistilBertTokenizerFast, pipeline
import pickle

# carregar os dados
df = pd.read_csv('news-of-the-site-folhauol.zip', encoding="utf-8")

# verificar se há colunas com missing
print(df.isnull().sum())

# eliminar as linhas cuja coluna 'text' tenha missing
df_cleaned = df.dropna(subset=['text'])

# filtrar apenas a categoria "mercado"
data_filter = df_cleaned[df_cleaned['category'] == 'mercado']

# filtrar apenas o primeiro trimestre de 2015
data_filter = data_filter[(data_filter['date'] >= '2015-01-01') & (data_filter['date'] <= '2015-03-31')]

# carregar o modelo a ser utilizado (definido no projeto)
model = BertForTokenClassification.from_pretrained('monilouise/ner_pt_br')

# carregar o tokenizador (definido nas instruções do modelo)
tokenizer = DistilBertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased'
                                                    , model_max_length=512
                                                    , do_lower_case=False
                                                    )

# criar um pipeline para o modelo
nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True, device=0)

# indicar para o modelo que deve ser utilizada a GPU
model.to("cuda")

# tamanho do lote a ser utilizado
batch_size = 16 

# extrair a coluna 'text' para uma lista, a fim de possibilizar a aplicação do pipeline e do lote
texts = data_filter['text'].tolist()

# criar lista vazia para armazenar os resultados
ner_results = []

# aplicar o modelo
for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
    batch = texts[i:i+batch_size]
    ner_results.extend(nlp(batch))

# inserir os resultados em uma coluna no dataframe "data_filter"
data_filter['ner_results'] = ner_results

# salvar o dataframe "data_filter" em um arquivo pickle
with open("ner_results.pkl", "wb") as file:
    pickle.dump(data_filter, file)

# ===========================================================================================
# extracao das 10 organizacoes mais citadas - versão original
# ===========================================================================================

import pickle
import pandas as pd
with open("ner_results.pkl", "rb") as file:
    data = pickle.load(file)

ner_results = data['ner_results']

lista_origem = []
for valor in ner_results:
    for valor2 in valor:
        if valor2['entity_group'] == "ORG":
            lista_origem.append(valor2['word'])

df = pd.DataFrame(lista_origem, columns=['palavras'])
df_counts = df['palavras'].value_counts().reset_index()
df_counts.columns = ['palavras', 'contagem']

top10_original = df_counts.head(10)

top10_original.to_excel('top10_original.xlsx', index=False)

# ===========================================================================================
# extracao das organizacoes mais citadas - versão com correção da tokenização (##)
# ===========================================================================================

import pickle
import pandas as pd

with open("ner_results.pkl", "rb") as file:
    data = pickle.load(file)

ner_results = data['ner_results']
lista_origem = []
for valor in ner_results:
    for valor2 in valor:
        if valor2['entity_group'] == "ORG":
            lista_origem.append(valor2['word'])

def merge_subwords(tokens):
    merged = []
    temp_word = ""

    for token in tokens:
        if token.startswith("##"):
            temp_word += token[2:]
        else:
            if temp_word:
                merged.append(temp_word)
            temp_word = token

    if temp_word:
        merged.append(temp_word)

    return merged

lista_limpa = merge_subwords(lista_origem)

df = pd.DataFrame(lista_limpa, columns=['palavras'])
df_counts = df['palavras'].value_counts().reset_index()
df_counts.columns = ['palavras', 'contagem']

top10_fix_string = df_counts.head(10)

top10_fix_string.to_excel('top10_fix_string.xlsx', index=False)

# ===========================================================================================
# extracao das organizacoes mais citadas - versão com correção da tokenização (##), do "s" e do "O"
# ===========================================================================================

import pickle
import pandas as pd

with open("ner_results.pkl", "rb") as file:
    data = pickle.load(file)

ner_results = data['ner_results']
lista_origem = []
for valor in ner_results:
    for valor2 in valor:
        if valor2['entity_group'] == "ORG":
            lista_origem.append(valor2['word'])

def merge_subwords(tokens):
    merged = []
    temp_word = ""

    for token in tokens:
        if token.startswith("##"):
            temp_word += token[2:]
        elif token == "s" and merged:  
            merged[-1] += token
        else:
            if temp_word and temp_word != "O":
                merged.append(temp_word)
            temp_word = token

    if temp_word and temp_word != "O":
        merged.append(temp_word)

    return merged

lista_limpa = merge_subwords(lista_origem)

df = pd.DataFrame(lista_limpa, columns=['palavras'])
df_counts = df['palavras'].value_counts().reset_index()
df_counts.columns = ['palavras', 'contagem']

top10_fix_s_o = df_counts.head(10)

top10_fix_s_o.to_excel('top10_fix_s_o.xlsx', index=False)

# ===========================================================================================
# extracao das organizacoes mais citadas - versão com correção da "Sete Brasil" e da "S & P"
# ===========================================================================================

import pickle
import pandas as pd

with open("ner_results.pkl", "rb") as file:
    data = pickle.load(file)

ner_results = data['ner_results']
lista_origem = []
for valor in ner_results:
    for valor2 in valor:
        if valor2['entity_group'] == "ORG":
            lista_origem.append(valor2['word'])

def merge_subwords(tokens):
    merged = []
    s_and_p_count = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token.startswith("##"):
            if merged:
                merged[-1] += token[2:]
            else:
                merged.append(token[2:]) 
            i += 1
            continue

        if token == "Sete" and i + 1 < len(tokens) and tokens[i + 1] == "Brasil":
            merged.append("Sete Brasil")
            i += 2
            continue

        if token == "S &":
            if i + 1 < len(tokens) and tokens[i + 1] in ["P", "Ps"]:
                merged.append("S & P")
                s_and_p_count += 1
                i += 2
                continue
            elif i + 2 < len(tokens) and tokens[i + 1] in ["##", "-", "—", ""] and tokens[i + 2] in ["P", "Ps"]:
                merged.append("S & P")
                s_and_p_count += 1
                i += 3
                continue
            else:
                print(f"[SKIPPED] 'S &' at index {i}, next: {tokens[i+1] if i+1 < len(tokens) else 'N/A'}")

        if token in ["Standard &"] and i + 1 < len(tokens) and tokens[i + 1] in ["Poor 's", "Poor´s"]:
            merged.append("S & P")
            i += 2
            continue

        # Handle standalone 's' that should attach to previous word
        if token == "s" and merged:
            merged[-1] += "s"
        elif token != "O":
            merged.append(token)

        i += 1
    return merged

lista_limpa = merge_subwords(lista_origem)

lista_limpa = ['S & P' if item in ["Standard & Poor 's", "Standard & Poor ´s", "Standard & Poor's", "Standard & Poor´s"] else item for item in  lista_limpa]

lista_limpa = ['Sete Brasil' if item == 'Sete' else item for item in  lista_limpa]

df = pd.DataFrame(lista_limpa, columns=['palavras'])
df_counts = df['palavras'].value_counts().reset_index()
df_counts.columns = ['palavras', 'contagem']

top10_fix_sete_SP = df_counts.head(10)

top10_fix_sete_SP.to_excel('top10_fix_sete_SP.xlsx', index=False)

