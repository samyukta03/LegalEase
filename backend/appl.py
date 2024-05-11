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
import numpy as np
import networkx as nx
import law1
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
from pyemd import emd
from gensim import models
from bert_score import BERTScorer
import gensim
from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
glove_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d1.txt')

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


def compute_wmd(embedding1, embedding2):
    wmd_score = torch.sum(embedding1 * embedding2)
    return wmd_score

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_similarity(embedding1, embedding2):
    s1=jaccard_similarity(embedding1,embedding2)
    s2=glove_model.wmdistance(embedding1,embedding2)
    s=(0.8*s1)+(0.2*s2)
    return s

def text_rank(graph, damping_factor=0.85, max_iter=50, tol=1.0e-6):
    # Initialize node scores
    scores = {node: 1.0 for node in graph.nodes()}

    # Iteratively update node scores
    for _ in range(max_iter):
        diff = 0
        for node in graph.nodes():
            score = 1 - damping_factor
            for neighbor in graph.neighbors(node):
                weight = graph[node][neighbor].get('weight', 1.0)
                score += damping_factor * scores[neighbor] * weight
            diff += abs(scores[node] - score)
            scores[node] = score

        # Check convergence
        # if sum((scores[node] - prev_scores[node]) ** 2 for node in graph.nodes()) < tol:
        if diff < tol:
            break

    return scores

#construct a graph with the cases in df

G = nx.Graph() # Create an empty graph
# Add nodes to the graph with BERT embeddings as attributes
for index, row in df1.iterrows():
    summary = row['Summary']
    embedding = row['Summary_BERT']
    G.add_node(index, summary=summary, embedding=embedding)

# Calculate similarity between embeddings and add edges to the graph
for u in G.nodes():
    for v in G.nodes():
        if u != v:
            # embedding_u = G.nodes[u]['embedding']
            embedding_u = G.nodes[u]['summary']
            embedding_v = G.nodes[v]['summary']
            # embedding_v = G.nodes[v]['embedding']
            # similarity = cosine_similarity(embedding_u, embedding_v)[0][0]
            similarity = compute_similarity(embedding_u,embedding_v)
            G.add_edge(u, v, weight=similarity)

def find_similar_cases(cleaned):
    query_embedding = get_bert_embeddings(cleaned)
    similarities = {}
    main_node = None
    for node in G.nodes():
        embedding = G.nodes[node]['embedding']
        case_det = G.nodes[node]
        # similarity = cosine_similarity(query_embedding, embedding)[0][0]
        similarity = compute_similarity(cleaned, case_det['summary'])
        print(similarity)
        similarities[node] = similarity
        if similarity >= 0.7:
            main_node = node
            break
    main_node = max(similarities, key=similarities.get)
    # Construct a subgraph with only the top-k similar nodes
    neighbors = list(G.neighbors(main_node))
    neighbor_weights = [(neighbor, G[main_node][neighbor]['weight']) for neighbor in neighbors]
    neighbor_weights.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [neighbor for neighbor, _ in neighbor_weights[:5]]
    subgraph = G.subgraph(top_neighbors)
    # Compute TextRank similarity scores on the subgraph
    textrank_scores = text_rank(subgraph)
    most_similar_node = max(textrank_scores, key=textrank_scores.get)
    most_similar_case = G.nodes[most_similar_node]['summary']
    ans = []
    ans.append(G.nodes[main_node]['summary'])
    print(G.nodes[main_node]['summary'])
    ans.append(most_similar_case)
    return ans


#--------bi lstm model for judgemnt pred

#--------bi lstm model for judgemnt pred

train_df, test_df = train_test_split(df1, test_size=0.2, random_state=42)
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
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs.float(), sides.float())
        loss.requires_grad = True
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
        # correct += (predicted == sides).sum().item()
        correct += (outputs.round() == sides).item()
        accuracy = correct / total
print(f"Accuracy: {accuracy}")
if accuracy >=0.5:
    torch.save(model.state_dict(), 'bilstm_model.pth')
#----------------


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
    input_lang = detect(query)
    print(input_lang,"  languageeeeeeeeeeeee")
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
    """
    query_embedding = get_bert_embeddings(cleaned)
    similarities = []
    for summary_embedding in df1['Summary_BERT']:
       similarity = cosine_similarity(query_embedding, summary_embedding)
       similarities.append(similarity.item())
    print(similarities)
    max_index = similarities.index(max(similarities))

    max_summary = df1.loc[max_index, 'Summary']
    if pd.isna(max_summary):
        print("max_summary is null")
    else:
        print(max_summary)
    ans_temp="Case Details :  "+os.linesep
    ans_temp= ans_temp+max_summary
    """
    case_details = [] 
    case_details = find_similar_cases(cleaned)
    ans_temp="Case Details :  "+os.linesep
    i = 1 
    for case in case_details:
        ans_temp += "<br>" + "Case " + str(i) + ": " + "<br>"
        i+=1
        ans_temp += case

    model.eval()  # Set the model to evaluation mode
    ans_temp = "JUDGMENT Prediction"+"<br>"
    with torch.no_grad():
        print("heyy in predict")
        output = model(cleaned)
        # print(output," printing tensor output")
        # print(output.item(),"printing predicted item")
        val=output.item()
        # print(val," please prints")
        if(val>=1):
            print("judge 1")
            judgement="JUDGEMENT: The judgement will be in favour of the appelant for the given case"
        else:
            print("judge 1")
            judgement="JUDGEMENT: The judgement will be in favour of the respondent and the appelant will lose the case"
        print(judgement)
    ans_temp=ans_temp+judgement
    #----

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
    ans_temp =ans_temp+ os.linesep + "<br><br> "+"Law Details" +"<br> "
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


# from email import header
# from flask import Flask, request, render_template
# from flask_cors import CORS, cross_origin
# from matplotlib.pyplot import text
# import pandas as pd
# import torch
# import difflib
# import model
# import nltk
# from langdetect import detect
# from deep_translator import GoogleTranslator
# from googletrans import Translator
# from gensim.models import KeyedVectors
# import numpy as np
# import law1
# from sklearn.preprocessing import LabelEncoder
# model = law1.model
# df=pd.read_csv('train.csv')
# TEMPLATES_AUTO_RELOAD = True
# use_reloader=True
# app = Flask(__name__,static_folder="F:/Project-Legal.ly/build",static_url_path='/')
# app.debug = True
# app.config["DEBUG"] = True
# app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config.update(
#     TEMPLATES_AUTO_RELOAD=True
# )
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# data = pd.read_csv('A2015-22.csv')
# t=list(df['Law'])
# # print(t)
# topics = list(data['title'])
# text = list(data['text'])

# greetings = ['Hi', 'Hello', "Hey There", 'hi']

# translator = GoogleTranslator(source='auto', target='en')

# # embedding_file = 'glove.6B.300d1.txt'

# # glove_model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)


# @app.errorhandler(404)
# def not_found(e):
#     return app.send_static_file('index.html')

# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# # def calculate_jaccard_similarity(query, sentence):
# #     query_tokens = set(nltk.word_tokenize(query.lower()))
# #     sentence_tokens = set(nltk.word_tokenize(sentence.lower()))
# #     intersection = query_tokens.intersection(sentence_tokens)
# #     union = query_tokens.union(sentence_tokens)
# #     return len(intersection) / len(union)
# @app.route("/api", methods=['POST'])
# @cross_origin()
# def get_bot_response():
#     print("in bot")
#     df=pd.read_csv('train.csv')
#     label_encoder_law = LabelEncoder()
# # label_encoder_law.fit(df['Law'])


#     df['Law'] = df['Law'].str.replace('<.*?>', '', regex=True)

# # Convert 'Law' column to lowercase
#     df['Law'] = df['Law'].str.lower()

# # Encode 'Law' column
#     label_encoder_law.fit(df['Law'])
#     data = request.get_json()  # Extract the JSON data from the request body
#     query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found
#     print(query)
#     input_lang = detect(query)
#     if input_lang is not None:
#         if input_lang == 'ta':
#             query_english = translator.translate(query)
#             print(query_english)
#         else:
#             query_english = query
#     cleaned = " ".join([word for word in query_english.split()
#                        if word not in nltk.corpus.stopwords.words('english')])
#     label_encoder_scenario = LabelEncoder()
#     # encoded_laws = label_encoder_law.transform(df['Law'])
#     # for encoded_value, text_value in zip(encoded_laws, df['Law']):
#     #     print(f"Encoded: {encoded_value}, Text: {text_value}")

#     df['Law'] = label_encoder_law.fit_transform(df['Law'])
#     encoded_query = label_encoder_scenario.fit_transform([cleaned])
#     print(encoded_query)
#     print("heloooooo")
#     print(df['Law'].head())
#     unique_laws = df['Law'].unique()
#     # unique_laws1 =  label_encoder_law.inverse_transform(unique_laws)
#     # print(unique_laws1)
#     similar_laws = {}
#     with torch.no_grad():
#         model.eval()
#         print("heeeuuuuyy")
#         for law_id in unique_laws:
#             encoded_law = law_id
#             print("hooooo")
#             print(encoded_law,encoded_query)
#             similarity = model(torch.tensor([encoded_query]), torch.tensor([encoded_law]), torch.tensor([0.0]))
#             similar_laws[law_id] = similarity.item()
#             print(law_id,similarity)
#             print("haaaaa")
#     # Sort the laws based on similarity scores
#     sorted_laws = sorted(similar_laws.items(), key=lambda x: x[1], reverse=True)
#     print(sorted_laws)
#     # Get the top 3 most similar laws for each unique law
#     top_3_similar_laws = {}
#     num=0
#     for law_id, _ in sorted_laws:
#         if num>=3:
#            break
#         num+=1
#         law_text = label_encoder_law.inverse_transform([law_id])[0]
#         print("no probb",law_text)
#         top_3_similar_laws[law_text] = similar_laws[law_id]
#     print(top_3_similar_laws)
    

#     # if difflib.get_close_matches(query, greetings):
#     #     return "Hi There! Welcome to Legal.ly\nPlease type 'topics' to get a list of the topics I have knowledge on."

#     # if query.lower().strip() in ['topics', 'topic']:
#     #     response = 'You can ask me anything about the following topics:\n' + \
#     #         " | ".join(topics)
#     #     return response

    
    
#     # query_tokens = nltk.word_tokenize(query_english.lower())
#     # other_sentence = "THE DOWRY PROHIBITION (MAINTENANCE OF LISTS OF PRESENTS TO THE BRIDE AND BRIDEGROOM) RULES, 1985 G.S.R. 664 (E), dated 19th August, 1985.- In exercise of the powers conferred by Sec.9 of the Dowry Prohibition Act, 1961 (28 of 1961), the Central Government hereby makes the fspanlowing rules, namely: Short title and commencement:(1) These rules may be called the Dowry Prohibition (Maintenance of Lists of Presents to the Bride and Bridegroom) Rules, 1985."
#     # other_tokens = nltk.word_tokenize(other_sentence.lower())

#     # # Create embeddings for query_english and other_sentence
#     # query_embedding = sum(glove_model[word] for word in query_tokens if word in glove_model) / len(query_tokens)
#     # other_embedding = sum(glove_model[word] for word in other_tokens if word in glove_model) / len(other_tokens)

#     # # Calculate the cosine similarity between the embeddings
#     # similarity = query_embedding.dot(other_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(other_embedding))
#     # similarity1 = jaccard_similarity_with_tolerance(set_1, set_2, tolerance=0.02) 
#     # print(similarity)
#     # print(similarity1)
#     translator1 = GoogleTranslator(source='auto', target='ta')
#     if input_lang == 'ta':
#         ans_temp="I believe the ans to your query is "
#         for law in top_3_similar_laws.keys():
#          ans_temp += text[law] + " "
        
#         ans_translated = translator1.translate(ans_temp)
#         return ans_translated
#     else:
#         ans_temp="I believe the ans to your query is "
#         for key in top_3_similar_laws:
#          print(key)
#          print(ans_temp)
#          ans_temp =ans_temp+ key + " "
#         return ans_temp
#         # return ans_temp
#         # match = difflib.get_close_matches(query_english, topics, n=1)
#         # if match:
#         #     match = match[0]
#         #     print(query_english, " fetching data")
#         #     ans = model.get_answer(query_english, text[topics.index(match)])
#         #     translator1 = GoogleTranslator(source='auto', target='ta')
#         #     if input_lang == 'ta':
#         #         topic_match=text[topics.index(match)]
#         #         ans_temp = "I believe the answer to your query is:" +ans+" . \n For more context, please refer to the following text:  "+topic_match
#         #         ans_translated = translator1.translate(ans_temp)
#         #         print("in tamil",ans)
#         #         return ans_translated
#         #     else:
#         #         topic_match=text[topics.index(match)]
#         #         ans_temp = "I believe the answer to your query is:" +ans+" . \n For more context, please refer to the following text:  "+topic_match
#         #         return ans_temp
        
    
#     # return "Sorry, I didn't understand that."

# if __name__ == "__main__":
#     app.run("0.0.0.0", debug=True)
    


# # Makeshift script for chatbot backend as the Model couldn't train in time :')
# # Please read the README file to check our actual approach towards the problem

# from email import header
# from flask import Flask, request, render_template
# from flask_cors import CORS, cross_origin
# from matplotlib.pyplot import text
# import pandas as pd
# import difflib
# import model
# import nltk

# app = Flask(__name__,static_folder="F:/Project-Legal.ly/build",static_url_path='/')
# app.config["TEMPLATES_AUTO_RELOAD"] = True
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


# data = pd.read_csv('A2015-22.csv')

# topics = list(data['title'])
# text = list(data['text'])


# greetings = ['Hi', 'Hello', "Hey There", 'hi']

# # @app.route("/", methods=['GET'])
# # def index():
# #     return "Welcome to Legal.ly Bot API"
# @app.errorhandler(404)
# def not_found(e):
#     return app.send_static_file('index.html')


# @app.route('/')
# def index():
#     return app.send_static_file('index.html')

# @app.route("/api", methods=['POST'])
# @cross_origin()
# def get_bot_response():
#     print("heyyyyloo")
#     data = request.get_json()  # Extract the JSON data from the request body
#     query = data.get('msg', '')  # Get the value of 'msg' parameter, defaulting to an empty string if not found
#     print(query)
#     # query = request.form['msg']

#     cleaned = " ".join([word for word in query.split()
#                        if word not in nltk.corpus.stopwords.words('english')])

#     if difflib.get_close_matches(query, greetings):
#         return "Hi There! Welcome to Legal.ly\nPlease type 'topics' to get a list of the topics I have knowledge on."

#     if query.lower().strip() in ['topics', 'topic']:
#         response = 'You can ask me anything about the following topics:\n' + \
#             " | ".join(topics)
#         return response

#     match = difflib.get_close_matches(cleaned, topics, n=1)
#     if match:
#         match = match[0]
#         ans = model.get_answer(query, text[topics.index(match)])
#         return "I believe the answer to your query is: {}. \n For more context, please refer to the following text: {}".format(ans, text[topics.index(match)])
#     else:
#         return "Sorry, I didn't understand that."


# if __name__ == "__main__":
#     app.run("0.0.0.0", debug=True)
