from email import header
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import text
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertModel
import torch.nn.functional as F
from email import header
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import text
import torch
import difflib
import model
import nltk
import os
from pyemd import emd
import networkx as nx
from langdetect import detect
from gensim import models
from transformers import BertModel, BertTokenizer
from deep_translator import GoogleTranslator
from bert_score import BERTScorer
from googletrans import Translator
from gensim.models import KeyedVectors
import gensim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(True)  # Context-manager 

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_scorer = BERTScorer(model_type='bert-base-uncased', num_layers=1)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer=tokenizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiply by 2 for bidirectional LSTM

    def forward(self, x):
        inputs = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        h0 = torch.zeros(self.num_layers*2, last_hidden_states.size(0), self.hidden_size).to(device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers*2, last_hidden_states.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(last_hidden_states, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = F.softmax(out, dim=1)
        _, predicted_class = torch.max(out, 1)
        return predicted_class

# Define a custom Dataset class
class LegalDataset(Dataset):
    def __init__(self, summaries, sides):
        self.summaries = summaries
        self.sides = sides

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        return self.summaries[idx], self.sides[idx]

# Load the dataset
dataset_path = 'supreme court cases.csv'
df = pd.read_csv(dataset_path)
summaries = df['Summary'].tolist()
sides = df['Side'].tolist()
# Tokenize the summaries (example using simple whitespace tokenization)
input_dim = len(summaries[0])
hidden_dim = 256

# Split the dataset into training and testing sets
split_ratio = 0.8  # 80% training, 20% testing
split_idx = int(len(summaries) * split_ratio)

train_summaries, train_sides = summaries[:split_idx], sides[:split_idx]
test_summaries, test_sides = summaries[split_idx:], sides[split_idx:]

# Create the datasets and data loaders
train_dataset = LegalDataset(train_summaries, train_sides)
test_dataset = LegalDataset(test_summaries, test_sides)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize the model
model = BiLSTM(768, 64, 1, 2).to(device) 

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for summaries, sides in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        # print(summaries[0],sides)
        outputs = model(summaries[0])
        print(outputs," printing tensor as output")
        # outputs = outputs.squeeze(1)  # Remove the extra dimension
        print(outputs.float(),sides.float())
        loss = criterion(outputs.float(), sides.float())
        loss.requires_grad = True
        # loss = criterion(outputs, sides)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for summaries, sides in tqdm(test_loader, desc="Testing"):
        outputs = model(summaries[0])
        print(outputs," printing tensor as output")
        # _, predicted = torch.max(outputs.data, 1)
        total += sides.size(0)
        correct += (outputs == sides).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), 'bilstm_model.pth')