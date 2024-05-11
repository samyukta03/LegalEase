

from email import header
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from matplotlib.pyplot import text
import pandas as pd
import torch
import difflib
import model
import nltk
import os
from langdetect import detect
from langdetect import detect_langs
from transformers import BertModel, BertTokenizer
from deep_translator import GoogleTranslator
from googletrans import Translator
from gensim.models import KeyedVectors
from gensim.summarization import keywords
import spacy
import numpy as np
import networkx as nx
import law1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset




df = pd.read_csv('train-news.csv', encoding='latin1')
device = torch.device('cpu')  # or 'cuda' if you're using GPU
model_path = 'siamese_model.pth'
model_state_dict = torch.load(model_path, map_location='cpu')
num_unique_scenario = len(df['Scenario'].unique())
num_unique_law = len(df['Law'].unique())
model1 = law1.SiameseNetwork(num_unique_scenario, num_unique_law)
model1.load_state_dict(model_state_dict, strict=False)
# model1.eval()
TEMPLATES_AUTO_RELOAD = False
use_reloader=True
app = Flask(__name__,static_folder="F:/LegalEase/build",static_url_path='/')
app.debug = True
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


t=list(df['Law'])
# print(t)

# bert embeddings 
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
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

#---

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph with BERT embeddings as attributes
for index, row in df1.iterrows():
    summary = row['Summary']
    embedding = row['Summary_BERT']
    outcome = row['Outcome']
    G.add_node(index, summary=summary, outcome = outcome, embedding=embedding)

# Calculate similarity between embeddings and add edges to the graph
for u in G.nodes():
    for v in G.nodes():
        if u != v:
            embedding_u = G.nodes[u]['embedding']
            embedding_v = G.nodes[v]['embedding']
            similarity = cosine_similarity(embedding_u, embedding_v)[0][0]
            G.add_edge(u, v, weight=similarity)

# Define PageRank function to rank nodes
def page_rank(graph):
    return nx.pagerank(graph)





# Load the English NER model from SpaCy
nlp = spacy.load("en_core_web_sm")

# Function to generate summary from case details
def generate_summary(summary, outcome):
    # Process summary and outcome texts
    summary_doc = nlp(str(summary))
    outcome_doc = nlp(str(outcome))

    # Convert generator to list and extract first two sentences
    key_info_summary = " ".join([sent.text for sent in list(summary_doc.sents)[:3]])
    key_info_outcome = " ".join([sent.text for sent in list(outcome_doc.sents)[:5]])

    # Combine key information into a summary
    case_summary = f"Key Information of this case:  {key_info_summary}. {key_info_outcome}."

    return case_summary

#---

translator = GoogleTranslator(source='auto', target='en')


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/')
def index():
    return app.send_static_file('index.html')
@app.route("/api", methods=['POST'])
@cross_origin()
def get_bot_response():
    translator1 = GoogleTranslator(source='auto', target='ta')
    translator2 = GoogleTranslator(source='auto', target='hi')
    print("in bot")
    df=pd.read_csv('train-news.csv',encoding='latin1')
    label_encoder_law = LabelEncoder()
    df['Law'] = df['Law'].str.replace('<.*?>', '', regex=True)
    df['Law'] = df['Law'].str.lower()

# Encode 'Law' column
    label_encoder_law.fit(df['Law'])
    data = request.get_json()  # Extract the JSON data from the request body
    query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found
    language=data.get('language','')
    print(language)
    print(query)
    input_lang = detect(query)
    print(input_lang,"  lang detected")
    ans_temp=''
    if input_lang is not None:
        if input_lang == 'ta' or input_lang=='hi':
            query_english = translator.translate(query)
            print(query_english)
        else:
            query_english = query
    cleaned = " ".join([word for word in query_english.split()
                       if word not in nltk.corpus.stopwords.words('english')])
    print(cleaned)
    #bert embeddings 
    query_embedding = get_bert_embeddings(cleaned)
    similarities = {}
    for node in G.nodes():
        embedding = G.nodes[node]['embedding']
        similarity = cosine_similarity(query_embedding, embedding)[0][0]
        print(similarity)
        similarities[node] = similarity
    # Rank nodes using PageRank algorithm
    node_scores = page_rank(G)

    combined_scores = {node: similarity * node_scores[node] for node, similarity in similarities.items()}

    # Sort nodes by PageRank scores
    sorted_nodes = sorted(combined_scores, key=combined_scores.get, reverse=True)

    # Retrieve case details associated with top-ranked nodes
    top_k = 2  # Number of top-ranked nodes to retrieve
    top_nodes = sorted_nodes[:top_k]
    
    case_details_set = set()  # Use a set to store unique case details
    case_details = {}

    # Retrieve case details associated with top-ranked nodes
    # for node in top_nodes:
    #     summary = G.nodes[node]['summary']
    #     case_details_set.add(summary)
    for node in top_nodes:
        case_details[node] = {
            'summary': G.nodes[node]['summary'],
            'outcome': G.nodes[node]['outcome']
        }

    # Prepare response
    # ans_temp="Case Details :  "+os.linesep+"<br> "

    # for case_detail in case_details_set:
        # ans_temp= ans_temp+"<br>"+"--"+case_detail
    """
    # Identify the node with the highest similarity to the query
    # max_similarity_node = max(similarities, key=similarities.get)
    # print("max sim")
    # # Retrieve the case details associated with the node with the highest similarity
    # case_details = G.nodes[max_similarity_node]['summary']
    # ans_temp="Case Details :  "+os.linesep +"<br> "
    # ans_temp= ans_temp+case_details
    # #----
    """

    brief_information = {}
    
    for case_index, details in case_details.items():
        summary = details['summary']
        outcome = details['outcome']
        key_info = generate_summary(summary, outcome)
        brief_information[case_index] = key_info

    ans_temp= "Brief Information for Similar Cases:" + "<br>"
    # Present brief information derived from consolidated outcomes'
    i=1
    for case_index, info in brief_information.items():
        ans_temp += "<br> Case "+ str(i) + " :" + "<br>"
        i+=1
        ans_temp += info 
        # ans_temp += "<br>" + "Outcome: " + "<br>"
        # ans_temp += case_details[case_index]['outcome']  # Display summary
        # ans_temp += "<br>"

    greetings = ['Hi', 'Hello', "Hey There", 'hi']
    if difflib.get_close_matches(cleaned, greetings):
        sent="Hi There! Welcome to LegalEase\nPlease ask your legal queries."
        if input_lang=="ta":
           s=translator1.translate(sent)
           return s
        elif input_lang!="en" and input_lang!="ta":
           s=translator2.translate(sent)
           return s
        else:
           return sent
            
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
        print("heeeuuuuyy")
        for law_id in unique_laws:
            encoded_law = law_id
            print("hooooo")
            print(encoded_law,encoded_query)
            similarity = model1(cleaned, torch.tensor([encoded_law]))
            similar_laws[law_id] = similarity
            print(similar_laws)
            print(law_id,similarity)
            print("haaaaa")
    # Sort the laws based on similarity scores
    sorted_laws = sorted(similar_laws.items(), key=lambda x: x[1], reverse=True)
    print(sorted_laws)
    # Get the top 3 most similar laws for each unique law
    top_3_similar_laws = {}
    num=0
    for law_id, _ in sorted_laws:
        if num>=3:
           break
        num+=1
        law_text = label_encoder_law.inverse_transform([law_id])[0]
        print(law_text)
        top_3_similar_laws[law_text] = similar_laws[law_id]
    print(top_3_similar_laws)
    ans_temp =ans_temp+ os.linesep + "<br> "+"Law Details" +"<br> "
    if input_lang == 'ta':
        # ans_temp="I believe the ans to your query is :  "+ "<br> " +os.linesep
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ os.linesep + "<br> " +"--"+key + os.linesep
        
         ans_translated = translator1.translate(ans_temp)
        return ans_translated
    elif input_lang == 'hi':
        # ans_temp="I believe the ans to your query is :  "+ "<br> " +os.linesep
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ os.linesep + "<br> " +"--"+key + os.linesep
        
         ans_translated = translator2.translate(ans_temp)
        return ans_translated
    else:
        # ans_temp="I believe the ans to your query is :  "+ "<br> " +os.linesep
        print("hey\n you")
        for key in top_3_similar_laws:
         print(key)
         print(ans_temp)
         ans_temp =ans_temp+ "--"+key +"<br>"
        return ans_temp

if __name__ == "__main__":
    app.run("0.0.0.0", debug=False)

