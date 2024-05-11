

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Load train and test data
df = pd.read_csv('train.csv')
test_data_path = 'test.csv'

# Assuming you have loaded your dataset into a DataFrame 'df'
# and it has columns 'Scenario', 'Law', and 'Similarity'

# Encode scenarios and laws
label_encoder = LabelEncoder()
df['Scenario'] = label_encoder.fit_transform(df['Scenario'])
df['Law'] = df['Law'].str.replace('<.*?>', '', regex=True)

# Convert 'Law' column to lowercase
df['Law'] = df['Law'].str.lower()

# Encode 'Law' column
df['Law'] = label_encoder.fit_transform(df['Law'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Scenario', 'Law']], df['Similarity'], test_size=0.33, random_state=42, stratify=df['Similarity'])

# Convert data to PyTorch tensors
X_train_scenario = torch.tensor(X_train['Scenario'].values, dtype=torch.long)
X_train_law = torch.tensor(X_train['Law'].values, dtype=torch.long)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test_scenario = torch.tensor(X_test['Scenario'].values, dtype=torch.long)
X_test_law = torch.tensor(X_test['Law'].values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Define Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, num_embeddings):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=128)
        self.fc = nn.Linear(128 * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward_once(self, x):
        embedded = self.embedding(x)
        return embedded.view(embedded.size(0), -1)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), dim=1)
        output = self.fc(output)
        return self.sigmoid(output)

# Initialize the model
print("hey")
num_unique_scenario = len(X_train_scenario.unique())
num_unique_law = len(X_train_law.unique())
num_unique_scenario = len(X_train_scenario.unique())
num_unique_law = len(X_train_law.unique())
num_embeddings = len(label_encoder.classes_)
# Check the maximum index in X_train_scenario and X_train_law
max_index_scenario = X_train_scenario.max()
max_index_law = X_train_law.max()

# Verify if the number of unique values is within the range of the embedding matrix size
if num_unique_scenario > num_embeddings:
    print(f"Number of unique values in X_train_scenario ({num_unique_scenario}) exceeds the embedding matrix size ({num_embeddings}).")
if num_unique_law > num_embeddings:
    print(f"Number of unique values in X_train_law ({num_unique_law}) exceeds the embedding matrix size ({num_embeddings}).")

# Verify if the maximum index is within the range of the embedding matrix size
if max_index_scenario >= num_embeddings:
    print(f"Maximum index in X_train_scenario ({max_index_scenario}) is out of range for embedding matrix with {num_embeddings} embeddings.")
if max_index_law >= num_embeddings:
    print(f"Maximum index in X_train_law ({max_index_law}) is out of range for embedding matrix with {num_embeddings} embeddings.")

if max_index_scenario >= num_embeddings:
    num_embeddings = max_index_scenario + 1
if max_index_law >=num_embeddings:
    num_embeddings=max_index_law+1    
model = SiameseNetwork(num_embeddings)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train_scenario, X_train_law)
    loss = criterion(output, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
# Evaluate the model
with torch.no_grad():
    model.eval()
    test_output = model(X_test_scenario, X_test_law)
    test_output_np = test_output.numpy()

    for i, (scenario, law, output_similarity) in enumerate(zip(X_test_scenario, X_test_law, test_output_np)):
        scenario_text = label_encoder.inverse_transform([scenario.item()])[0]
        law_text = label_encoder.inverse_transform([law.item()])[0]
        actual_similarity = df.iloc[i]['Similarity']
        print(f"Test {i+1}: Scenario - {scenario_text}, Law - {law_text}, Predicted Similarity - {output_similarity[0]}, Actual Similarity - {actual_similarity}")

    test_loss = criterion(test_output, y_test.unsqueeze(1))
    print(f'Test Loss: {test_loss.item()}')

    # Calculate accuracy
    predictions = (test_output > 0.5).float()
    correct = (predictions == y_test.unsqueeze(1)).sum().item()
    total = y_test.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')



# Example usage
# new_scenario = "I want to know about the law related to prohibition or restriction of dowry for the bride"
# new_scenario = re.sub(r'<.*?>', '', new_scenario)
# new_scenario=new_scenario.lower()
# new_law = "THE DOWRY PROHIBITION (MAINTENANCE OF LISTS OF PRESENTS TO THE BRIDE AND BRIDEGROOM) RULES, 1985 G.S.R. 664 (E), dated 19th August, 1985.- In exercise of the powers conferred by Sec.9 of the Dowry Prohibition Act, 1961 (28 of 1961), the Central Government hereby makes the fspanlowing rules, namely: Short title and commencement:(1) These rules may be called the Dowry Prohibition (Maintenance of Lists of Presents to the Bride and Bridegroom) Rules, 1985."
# new_law = re.sub(r'<.*?>', '', new_law)
# new_law=new_law.lower()
# similarity = predict_similarity(new_scenario, new_law)
# print(f"Predicted similarity: {similarity}")

# train_data = pd.read_csv(train_data_path)
# test_data = pd.read_csv(test_data_path)

# print("Train Data:")
# print(train_data.head())

# print("\nTest Data:")
# print(test_data.head())

# # Load GloVe embeddings
# embedding_file = 'glove.6B.300d1.txt'
# glove_model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)

# # Define Siamese Neural Network model
# class SiameseNetwork(nn.Module):
#     def __init__(self, embedding_dim, hidden_size):
#         super(SiameseNetwork, self).__init__()
#         self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_model.vectors), freeze=True)
#         self.fc = nn.Linear(embedding_dim, hidden_size)
#         self.regularization = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(0.5)  
#     def forward(self, input_1, input_2):
#         embedded_1 = self.embedding(input_1.long())
#         embedded_2 = self.embedding(input_2.long())
#         embedded_1 = torch.mean(embedded_1, dim=1)
#         embedded_2 = torch.mean(embedded_2, dim=1)
#         output_1 = torch.relu(self.fc(embedded_1))
#         output_2 = torch.relu(self.fc(embedded_2))
#         return output_1, output_2

# # Preprocess data
# class LegalDataset(Dataset):
#     def __init__(self, data, word_to_idx):
#         self.data = data
#         self.word_to_idx = word_to_idx

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         scenario = self.data.iloc[idx]['Scenario']
#         law = self.data.iloc[idx]['Law']
#         similarity = self.data.iloc[idx]['Similarity']

#         scenario_tokens = [self.word_to_idx.get(word, 0) for word in scenario.split()]
#         law_tokens = [self.word_to_idx.get(word, 0) for word in law.split()]
        
#         scenario_tokens = self.pad_sequence(scenario_tokens, 50)
#         law_tokens = self.pad_sequence(law_tokens, 50)

#         return {'scenario_tokens': torch.tensor(scenario_tokens),
#                 'law_tokens': torch.tensor(law_tokens),
#                 'similarity': torch.tensor(similarity, dtype=torch.float)}

#     def pad_sequence(self, sequence, max_length):
#         if len(sequence) < max_length:
#             sequence = sequence + [0] * (max_length - len(sequence))
#         else:
#             sequence = sequence[:max_length]
#         return sequence

# word_to_idx = {word: idx for idx, word in enumerate(glove_model.index_to_key)}
# train_dataset = LegalDataset(train_data, word_to_idx)
# test_dataset = LegalDataset(test_data, word_to_idx)

# # Train the Siamese Neural Network
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SiameseNetwork(300, hidden_size=128).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CosineEmbeddingLoss()

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# model.train()
# for epoch in range(3):  # Adjust number of epochs as needed
#     for batch in train_loader:
#         input_1, input_2 = batch['scenario_tokens'].to(device), batch['law_tokens'].to(device)
#         similarity = batch['similarity'].to(device)

#         optimizer.zero_grad()
#         output_1, output_2 = model(input_1, input_2)
#         loss = criterion(output_1, output_2, torch.ones(input_1.size(0)).to(device))
#         loss.backward()
#         optimizer.step()

# # Test the Siamese Neural Network
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# def jaccard_similarity_with_tolerance(set_1, set_2, tolerance):
#     intersection = len([value for value in set_1 if any(abs(value - other_value) <= tolerance for other_value in set_2)])
#     print(intersection)
#     union = len(set_1) + len(set_2)
#     return intersection / union if union > 0 else 0

# model.eval()
# with torch.no_grad():
#     correct_predictions = 0
#     correct_predictions1 = 0
#     total_predictions = 0
#     for batch in test_loader:
#         input_1, input_2 = batch['scenario_tokens'].to(device), batch['law_tokens'].to(device)
#         similarity = batch['similarity'].to(device)
#         # print(input_1,input_2,similarity,similarity.item(),"hhhhhhhhhh")
#         output_1, output_2 = model(input_1, input_2)
#         set_1 = set(output_1.cpu().numpy().tolist()[0])
#         set_2 = set(output_2.cpu().numpy().tolist()[0])
#         similarity_score = cosine_similarity(output_1.cpu().numpy(), output_2.cpu().numpy())
#         binary_similarity = 1 if similarity.item() > 0.5 else 0
#         # Compute Jaccard similarity
#         intersection_size = len(set_1.intersection(set_2))
#         union_size = len(set_1.union(set_2))
#         print(intersection_size,union_size," sizeee")
#         similarity_score1 = jaccard_similarity_with_tolerance(set_1, set_2, tolerance=0.02) 
#         # print(output_1,output_2)
#         # print(similarity_score,similarity_score1)
#         manhattan_distance = torch.sum(torch.abs(output_1 - output_2), dim=1).cpu().numpy()  # Calculate Manhattan distance
#         similarity_score2 = 1 / (1 + manhattan_distance)
#          # Compute Euclidean distance
#         euclidean_distance = torch.dist(output_1, output_2, p=2)
        
#         # Normalize Euclidean distance
#         normalized_distance = euclidean_distance / (1 + euclidean_distance)
#         print(normalized_distance,manhattan_distance,similarity_score1,similarity.item()," distancee")
#         # Convert similarity to binary
#         binary_similarity1 = 1 if similarity.item() > 0.5 else 0
        
#         if normalized_distance < 0.3:  # Threshold for similarity
#             prediction1 = 1
#         else:
#             prediction1 = 0

#         if manhattan_distance < 4:  # Threshold for similarity
#             prediction = 1
#         else:
#             prediction = 0

#         correct_predictions += (prediction == batch['similarity'].item())
#         correct_predictions1+=(prediction1==batch['similarity'].item())
#         total_predictions += 1

# accuracy = correct_predictions / total_predictions
# accuracy2= correct_predictions1/total_predictions
# print(f"Accuracy: {accuracy}")
# print(f"Accuracy: {accuracy2}")


# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel

# # Load train and test data
# train_data_path = 'train.csv'
# test_data_path = 'test.csv'

# train_data = pd.read_csv(train_data_path)
# test_data = pd.read_csv(test_data_path)

# print("Train Data:")
# print(train_data.head())

# print("\nTest Data:")
# print(test_data.head())

# # Define Siamese Neural Network model
# class SiameseNetwork(nn.Module):
#     def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768):
#         super(SiameseNetwork, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]
#         similarity_score = self.fc(pooled_output)
#         return similarity_score

# # Preprocess data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# class LegalDataset(Dataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         print(self.data,idx)
#         print(type(idx))
#         idx=int(idx)
#         print(self.data.iloc[idx],self.data.iloc[idx]['Law'],self.data.iloc[idx]['Scenario'])
#         scenario = self.data.iloc[idx]['Scenario']
#         law = self.data.iloc[idx]['Law']
#         similarity = self.data.iloc[idx]['Similarity']

#         scenario_tokens = self.tokenizer.encode(scenario, add_special_tokens=True, truncation=True)
#         law_tokens = self.tokenizer.encode(law, add_special_tokens=True, truncation=True)
        
#         scenario_tokens = self.pad_sequence(scenario_tokens, 50)
#         law_tokens = self.pad_sequence(law_tokens, 50)

#         return {'scenario_tokens': torch.tensor(scenario_tokens),
#                 'law_tokens': torch.tensor(law_tokens),
#                 'similarity': torch.tensor(similarity, dtype=torch.float)}
    
#     def pad_sequence(self, sequence, max_length):
#         if len(sequence) < max_length:
#             sequence = sequence + [0] * (max_length - len(sequence))
#         else:
#             sequence = sequence[:max_length]
#         return sequence
# train_dataset = LegalDataset(train_data, tokenizer)
# test_dataset = LegalDataset(test_data, tokenizer)

# # Train the Siamese Neural Network
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SiameseNetwork().to(device)
# optimizer = optim.Adam(model.parameters(), lr=2e-5)
# criterion = nn.BCEWithLogitsLoss()

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# model.train()
# for epoch in range(3):  # Adjust number of epochs as needed
#     for batch in train_loader:
#         scenario_tokens = batch['scenario_tokens'].to(device)
#         law_tokens = batch['law_tokens'].to(device)
#         similarity = batch['similarity'].to(device)

#         optimizer.zero_grad()
#         similarity_score = model(scenario_tokens, attention_mask=(scenario_tokens > 0))
#         loss = criterion(similarity_score.view(-1), similarity)
#         loss.backward()
#         optimizer.step()

# # Test the Siamese Neural Network
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model.eval()
# with torch.no_grad():
#     correct_predictions = 0
#     total_predictions = 0
#     for batch in test_loader:
#         scenario_tokens = batch['scenario_tokens'].to(device)
#         law_tokens = batch['law_tokens'].to(device)
#         similarity = batch['similarity'].to(device)

#         similarity_score = model(scenario_tokens, attention_mask=(scenario_tokens > 0))
#         predictions = (torch.sigmoid(similarity_score) > 0.5).int().view(-1)

#         correct_predictions += torch.sum(predictions == similarity).item()
#         total_predictions += len(similarity)

# accuracy = correct_predictions / total_predictions
# print(f"Accuracy: {accuracy}")



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import nltk
# from gensim.models import KeyedVectors

# # Load dataset and GloVe embeddings
# data = pd.read_csv('train.csv')
# embedding_file = 'glove.6B.300d1.txt'

# glove_model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)

# # Custom dataset class
# class SiameseDataset(Dataset):
#     def __init__(self, scenarios, laws, similarities, glove_model):
#         self.scenarios = scenarios
#         self.laws = laws
#         self.similarities = similarities
#         self.glove_model = glove_model

#     def __len__(self):
#         return len(self.scenarios)

#     def __getitem__(self, idx):
#         print(idx)
#         scenario = self.scenarios[idx]
#         law = self.laws[idx]
#         similarity = self.similarities[idx]
#         print(scenario,law,similarity)
#         scenario_embedding = self.get_embedding(scenario)
#         law_embedding = self.get_embedding(law)

#         return scenario_embedding, law_embedding, similarity

#     def get_embedding(self, text):
#         tokens = nltk.word_tokenize(text.lower())
#         embedding = sum(self.glove_model[word] for word in tokens if word in self.glove_model) / len(tokens)
#         return torch.tensor(embedding, dtype=torch.float32)

# # Split dataset into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Create DataLoader instances for training and testing sets
# train_dataset = SiameseDataset(train_data['Scenario'], train_data['Law'], train_data['Similarity'], glove_model)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# test_dataset = SiameseDataset(test_data['Scenario'], test_data['Law'], test_data['Similarity'], glove_model)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# # Siamese network architecture
# class SiameseNetwork(nn.Module):
#     def __init__(self, embedding_dim=300, hidden_dim=128):
#         super(SiameseNetwork, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, input1, input2):
#         batch_size1, seq_len1, _ = input1.size()
#         batch_size2, seq_len2, _ = input2.size()

#         _, (h_n1, _) = self.lstm(input1.view(batch_size1, seq_len1, -1))
#         _, (h_n2, _) = self.lstm(input2.view(batch_size2, seq_len2, -1))

#         h_n1 = h_n1.squeeze(0)
#         h_n2 = h_n2.squeeze(0)

#         combined = torch.cat((h_n1, h_n2), dim=1)

#         output = self.fc(combined)

#         return output

# # Instantiate Siamese network, define loss function, and optimizer
# model = SiameseNetwork()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for batch_idx, (scenario, law, similarity) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = model(scenario, law)
#         loss = criterion(output, similarity.view(-1, 1))
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

# # Testing loop
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for scenario, law, similarity in test_loader:
#         output = model(scenario, law)
#         predictions = torch.round(output)
#         correct += (predictions == similarity.view(-1, 1)).sum().item()
#         total += len(similarity)

# accuracy = correct / total
# print(f"Accuracy: {accuracy}")
