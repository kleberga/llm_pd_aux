# comando executados antes de executar este script:
# 1. instalar bibliotecas: pip install -q kaggle pandas transformers
# 2. copiar o arquivo de credenciais do kaggle para outra pasta: xcopy C:\Users\Kleber\Documents\curso_intel_artificial\processamento_linguagem\kaggle.json .kaggle/
# 3. baixar o dataset: kaggle datasets download --force -d marlesson/news-of-the-site-folhauol
# 4. instalar o NVIDIA Cuda para utilizar a GPU: https://developer.nvidia.com/cuda-downloads
# 5. instalar o torch, torchvision e torchaudio para usar a GPU na execuç?o do modelo: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 6. atualizar as bibliotecas citadas no item anterior: pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from transformers import BertForTokenClassification, DistilBertTokenizerFast, pipeline
import pickle

df = pd.read_csv('news-of-the-site-folhauol.zip', encoding="utf-8")
print(df.head())

df_cleaned = df.dropna(subset=['text'])

data_filter = df_cleaned[df_cleaned['category'] == 'mercado']
data_filter = data_filter[data_filter['date'].isin(['2015-01-01','2015-02-01','2015-03-01'])]

model = BertForTokenClassification.from_pretrained('monilouise/ner_pt_br')
tokenizer = DistilBertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased'
                                                    , model_max_length=512
                                                    , do_lower_case=False
                                                    )
nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True, device=0)
model.to("cuda")


batch_size = 16 
texts = data_filter['text'].tolist()
ner_results = []

for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
    batch = texts[i:i+batch_size]
    ner_results.extend(nlp(batch))


data_filter['ner_results'] = ner_results



with open("ner_results.pkl", "wb") as file:
    pickle.dump(data_filter, file)




import pickle
import pandas as pd
with open("ner_results.pkl", "rb") as file:
    data = pickle.load(file)

matching_rows = data[data['ner_results'].apply(
    lambda ents: any(ent['word'] == 'Fi' for ent in ents)
)].index.tolist()

# print(matching_rows)

# print(matches)
print(data.loc[166558,'text'])

ner_results = data['ner_results']

merged_entities = []
current_word = ""
current_entity = None
for entry in ner_results:
    for entry2 in entry:
        word = entry2['word']
        if word.startswith("##"):  
            current_word += word[2:]
        else:
            if current_word:
                merged_entities.append({"word": current_word, "entity": current_entity})
            current_word = word
            current_entity = entry2['entity_group']
    
if current_word:
    merged_entities.append({"word": current_word, "entity": current_entity})

merged_entities_filtered = []
for valor in merged_entities:
    if valor['entity'] == 'ORG':
        merged_entities_filtered.append(valor['word'])

df = pd.DataFrame(merged_entities_filtered, columns=['palavras'])
df_counts = df['palavras'].value_counts().reset_index()
df_counts.columns = ['palavras', 'contagem']

df_counts = df_counts.sort_values(by='contagem', ascending=False).reset_index(drop=True)
soma = sum(df_counts['contagem'])
df_counts['proporcao'] = (df_counts['contagem'] / soma) * 100

top_10 = df_counts.head(10)



# print(data[data['text'].str.contains('Fi', na=False)])

# print(data)

top_10.loc[1,'palavras'] = 'Club Med'




print(data.loc[2,'text'])

print("Acentos funcionando? está")