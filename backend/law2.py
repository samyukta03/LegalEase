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
import law1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import sent_tokenize
glove_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d1.txt')
df=pd.read_csv('train-new-news.csv',encoding='latin1')
TEMPLATES_AUTO_RELOAD = True
use_reloader=True
app = Flask(__name__,static_folder="F:/LegalEase/build",static_url_path='/')
app.debug = False
app.config["DEBUG"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = False
app.config.update(
    TEMPLATES_AUTO_RELOAD=False
)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

data = pd.read_csv('A2015-22.csv')
t=list(df['Law'])
# print(t)
topics = list(data['title'])
text = list(data['text'])

greetings = ['Hi', 'Hello', "Hey There", 'hi']
device = torch.device('cpu')  # or 'cuda' if you're using GPU
model_path = 'siamese_model.pth'
model_state_dict = torch.load(model_path, map_location='cpu')
num_unique_scenario = len(df['Scenario'].unique())
num_unique_law = len(df['Law'].unique())
model1 = law1.SiameseNetwork(num_unique_scenario, num_unique_law)
model1.load_state_dict(model_state_dict)
model1.eval()

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_scorer = BERTScorer(model_type='bert-base-uncased', num_layers=1)

def get_bert_embeddings(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings
df1=pd.read_csv('case.csv',encoding='latin1')
df1['Summary_BERT'] = df1['Summary'].apply(lambda x: get_bert_embeddings(x))
# model1 = torch.load(model_path, map_location=device)
# model1.eval()
def compute_bertscore(embedding1,embedding2):
    embedding1_flat = embedding1.flatten().reshape(1, -1)
    embedding2_flat = embedding2.flatten().reshape(1, -1)
    # Calculate cosine similarity between the flattened embeddings
    similarity = cosine_similarity(embedding1_flat, embedding2_flat)[0][0]
    return similarity
def compute_wmd(embedding1, embedding2):
    wmd_score = torch.sum(embedding1 * embedding2)
    return wmd_score
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Compute similarity between two texts using BERTScore and WMD
def compute_similarity(embedding1, embedding2):
    s1=jaccard_similarity(embedding1,embedding2)
    s2=glove_model.wmdistance(embedding1,embedding2)
    # Assuming you have a function to compute BERTScore
    # bert_score = compute_bertscore(embedding1, embedding2)
    # # print(bert_score,"Printing bert")
    # wmd_score = compute_wmd(embedding1, embedding2)
    # print(wmd_score," ",(1 / (1 + wmd_score)))
    # # print(wmd_score, " printing wmd")
    # similarity = (0.5 * bert_score) + (0.5 *(1 / (1 + wmd_score)))
    s=(0.8*s1)+(0.2*s2)
    return s
def textrank_similarity(G, num_iters=50, d=0.85, tol=1.0e-6):
    # Initialize node scores
    scores = {node: 1.0 for node in G.nodes()}
    # Main iteration
    for _ in range(num_iters):
        diff = 0
        # Compute new scores for each node
        for node in G.nodes():
            score = (1 - d) + d * sum(scores[neighbor] * G[node][neighbor].get('weight', 1.0)
                                       for neighbor in G.neighbors(node))
            diff += abs(scores[node] - score)
            scores[node] = score
            # print(score,G.nodes[node]['summary'],"PRINTING NEI NODES SUMMARY")
        # Check convergence
        if diff < tol:
            break
    return scores
# Create an empty graph
def find_most_similar_case(query):
    print(query,"printing in find similar")
    query_embeddings = get_bert_embeddings(query)
    similarities = {}
    main_node = None
    for node in G.nodes():
        case_text = G.nodes[node]
        similarity = compute_similarity(query, case_text['summary'])
        # print(similarity,case_text['summary'])
        # print(similarity,query,case_text)
        similarities[node] = similarity
        print(similarity," sim")
        if similarity >= 0.7:
            main_node = node
            break

    if main_node is None:
        main_node = max(similarities, key=similarities.get)

    print(G.nodes[main_node]['summary'],"PRINTING MAIN NODE")
    # Construct a subgraph with only the top-k similar nodes
    neighbors = list(G.neighbors(main_node))
    neighbor_weights = [(neighbor, G[main_node][neighbor]['weight']) for neighbor in neighbors]
    neighbor_weights.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [neighbor for neighbor, _ in neighbor_weights[:5]]
    subgraph = G.subgraph(top_neighbors)
    print(list(G.neighbors(main_node)),"NEIGHBORS")
    # Compute TextRank similarity scores on the subgraph
    textrank_scores = textrank_similarity(subgraph)
    most_similar_node = max(textrank_scores, key=textrank_scores.get)
    most_similar_case = G.nodes[most_similar_node]['summary']
    ans= "Case 1 : " + G.nodes[main_node]['summary']+"<br>"
    ans=ans+" Case 2 : " + most_similar_case+"<br>"
    return ans

G = nx.Graph()

# Add nodes to the graph with BERT embeddings as attributes
for index, row in df1.iterrows():
    summary = row['Summary']
    embedding = row['Summary_BERT']
    gen_sum=row['generated_summary']
    G.add_node(index, summary=summary, embedding=embedding,gen_sum=gen_sum)

# Calculate similarity between embeddings and add edges to the graph
for u in G.nodes():
    for v in G.nodes():
        if u != v:
            embedding_u = G.nodes[u]['summary']
            embedding_v = G.nodes[v]['summary']
            similarity = compute_similarity(embedding_u,embedding_v)
            # print(similarity,G.nodes[u]['summary'],G.nodes[v]['summary'])
            G.add_edge(u, v, weight=similarity)

# Read the dataset

# Preprocess the data
# Tokenization, padding, etc.

# Split the dataset into training and testing sets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(True)  # Context-manager 

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
dataset_path = 'case.csv'
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

bimodel_path = 'bilstm_model.pth'

# Initialize the model
model = BiLSTM(768, 64, 1, 2).to(device) 
model.load_state_dict(torch.load('bilstm_model.pth', map_location=device))

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


translator = GoogleTranslator(source='auto', target='en')

# embedding_file = 'glove.6B.300d1.txt'

# glove_model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/')
def index():
    return app.send_static_file('index.html')

# def calculate_jaccard_similarity(query, sentence):
#     query_tokens = set(nltk.word_tokenize(query.lower()))
#     sentence_tokens = set(nltk.word_tokenize(sentence.lower()))
#     intersection = query_tokens.intersection(sentence_tokens)
#     union = query_tokens.union(sentence_tokens)
#     return len(intersection) / len(union)
@app.route("/api", methods=['POST'])
@cross_origin()
def get_bot_response():
    translator1 = GoogleTranslator(source='auto', target='ta')
    translator2 = GoogleTranslator(source='auto', target='hi')
    print("in bot")
    df=pd.read_csv('train-new-news.csv',encoding='latin1')
    label_encoder_law = LabelEncoder()
# label_encoder_law.fit(df['Law'])


    df['Law'] = df['Law'].str.replace('<.*?>', '', regex=True)

# Convert 'Law' column to lowercase
    df['Law'] = df['Law'].str.lower()

# Encode 'Law' column
    label_encoder_law.fit(df['Law'])
    data = request.get_json()  # Extract the JSON data from the request body
    query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found
    language=data.get('language','')
    print(language)
    print(query)
    input_lang = ''
    if query.lower() in ['english', 'இங்கிலீஷ்', 'अंग्रेज़ी']:
        input_lang = 'en'
    elif query.lower() in ['hindi', 'ஹிந்தி', 'हिंदी']:
        input_lang = 'hi'
    elif query.lower() in ['tamil', 'தமிழ்', 'तमिल']:
        input_lang = 'ta'
    else:
        # Retain the previous language setting if not explicitly specified in the query
        pass
    if input_lang == '':
        input_lang = detect(query)
    print(input_lang,"  languageeeeeeeeeeeee")
    greetings = ['Hi', 'Hello', "Hey There", 'hi','english','இங்கிலீஷ்','अंग्रेज़ी', 'en','hindi',
    'ஹிந்தி','हिंदी', 'hi','tamil','தமிழ்','तमिल','வணக்கம்','नमस्ते']
    
    query_english=""
    if input_lang is not None:
        if input_lang == 'ta' or input_lang=='hi':
            print("translatingggg")
            query_english = translator.translate(query)
            print(query_english)
        else:
            query_english = query
    cleaned = " ".join([word for word in query_english.split()
                       if word not in nltk.corpus.stopwords.words('english')])
    print(cleaned)
    ans_temp=""
    if input_lang=="ta":
        ans_temp="Tamil-"
    elif input_lang=='hi':
        ans_temp="Hindi-"
    else:
        ans_temp="English-"
    if difflib.get_close_matches(cleaned, greetings):
        print("close matches")
        sent="Hi There! Welcome to Legal.ly\nPlease ask your legal queries."
        if input_lang=="ta":
           s=translator1.translate(sent)
           return s
        elif input_lang=='hi':
           print("heylo")
           s=translator2.translate(sent)
           return s
        else:
           print("hm")
           return sent
    case_details=find_most_similar_case(cleaned)
    ans_temp=ans_temp+" Case Details :  "+os.linesep
    print(case_details +" PRINTING CASE DETAILS")
    ans_temp= ans_temp+" "+case_details
    print(ans_temp)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        print("heyy in predict")
        output = model(cleaned)
        print(output," printing tensor output")
        print(output.item(),"printing predicted item")
        val=output.item()
        print(val," please prints")
        # prediction = predicted.item()
        # print(prediction,"printing prediction")
        if(val>=1):
            print("judge 1")
            judgement="JUDGEMENT: The judgement will be in favour of the 1st party for the given case"
        else:
            print("judge 1")
            judgement="JUDGEMENT: The judgement will be in favour of the 2nd party and the 1st party will lose the case"
        print(judgement)
    print(ans_temp, " PRINTING ANS")

    
    label_encoder_scenario = LabelEncoder()
    # encoded_laws = label_encoder_law.transform(df['Law'])
    # for encoded_value, text_value in zip(encoded_laws, df['Law']):
    #     print(f"Encoded: {encoded_value}, Text: {text_value}")

    df['Law'] = label_encoder_law.fit_transform(df['Law'])
    encoded_query = label_encoder_scenario.fit_transform([cleaned])
    scenario_text = label_encoder_scenario.inverse_transform([encoded_query.item()])[0]
    print(scenario_text)
    print(encoded_query)
    print("heloooooo")
    print(df['Law'].head())
    unique_laws = df['Law'].unique()
    # unique_laws1 =  label_encoder_law.inverse_transform(unique_laws)
    # print(unique_laws1)
    similar_laws = {}
    with torch.no_grad():
        # model.eval()
        # print("heeeuuuuyy")
        for law_id in unique_laws:
            encoded_law = law_id
            # print("hooooo")
            print(encoded_law,encoded_query)
            similarity = model1(cleaned, torch.tensor([encoded_law]))
            similar_laws[law_id] = similarity
            print(similar_laws)
            print(law_id,similarity)
            # print("haaaaa")
    # Sort the laws based on similarity scores
    sorted_laws = sorted(similar_laws.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_laws)
    # Get the top 3 most similar laws for each unique law
    top_3_similar_laws = {}
    num=0
    for law_id, _ in sorted_laws:
        if num>=3:
           break
        num+=1
        law_text = label_encoder_law.inverse_transform([law_id])[0]
        print("no probb",law_text)
        top_3_similar_laws[law_text] = similar_laws[law_id]
    print(top_3_similar_laws)
    
    if input_lang == 'ta':        
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ "--"+key +"<br>"
        print("in tamil")
        ans_temp=ans_temp+judgement+"<br>"
        print(ans_temp)
        ans_translated = translator1.translate(ans_temp)
        print(ans_translated)
        return ans_translated
    elif input_lang == 'hi':
       
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ "--"+key +"<br>"
        
        ans_temp=ans_temp+judgement+"<br>"
        ans_translated = translator2.translate(ans_temp)

        return ans_translated
    else:
        
        print("hey\n you")
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ "--"+key +"<br>"
        ans_temp=ans_temp+judgement+"<br>"
        return ans_temp

if __name__ == "__main__":
    app.run("0.0.0.0", debug=False)
