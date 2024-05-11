import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text, model, tokenizer):
    text = str(text)  # Ensure text is a string
    print(text)
    encoded_text = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')
    input_ids = torch.tensor(encoded_text).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state
        averaged_embedding = torch.mean(hidden_states, dim=1)
    return averaged_embedding

bert_embeddings = []
df = pd.read_csv('train-new.csv',encoding='latin1')
for index, row in df.iterrows():
    law_text = row['Law']
    embedding = get_bert_embedding(law_text, bert_model, tokenizer)
    bert_embeddings.append((index, embedding))
torch.save(bert_embeddings, 'bert_embeddings.pt')