import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
# Load train and test data
df = pd.read_csv('train-news.csv',encoding='latin1')
test_data_path = 'test.csv'
embedding_file = 'glove.6B.300d1.txt'
best_accuracy = 0.0
glove_model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
# Assuming you have loaded your dataset into a DataFrame 'df'
# and it has columns 'Scenario', 'Law', and 'Similarity'

# Encode scenarios and laws
label_encoder_scenario = LabelEncoder()
label_encoder_law = LabelEncoder()
df['Scenario'] = label_encoder_scenario.fit_transform(df['Scenario'])
df['Law'] = label_encoder_law.fit_transform(df['Law'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Scenario', 'Law']], df['Similarity'], test_size=0.33, random_state=42, stratify=df['Similarity'])

# Convert data to PyTorch tensors
X_train_scenario = torch.tensor(X_train['Scenario'].values, dtype=torch.long)
X_train_law = torch.tensor(X_train['Law'].values, dtype=torch.long)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test_scenario = torch.tensor(X_test['Scenario'].values, dtype=torch.long)
X_test_law = torch.tensor(X_test['Law'].values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

def get_average_embedding(words, model):
    embeddings = [model[word] for word in words if word in model]
    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)
# Calculate cosine similarity
def cosine_simm(set1, set2, model):
    embedding1 = get_average_embedding(set1, model)
    embedding2 = get_average_embedding(set2, model)
    if embedding1 is None or embedding2 is None:
        return 0
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

    
def jaccard_similarity(set1, set2,model):
    embedding1 = get_average_embedding(set1, model)
    embedding2 = get_average_embedding(set2, model)
    if embedding1 is None or embedding2 is None:
        return 0
    # return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    intersection = len(set1.intersection(set2))
    cosine_sim = 2 * intersection / (len(set1) + len(set2))
    jaccard_sim  = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    pearson_corr, _ = pearsonr(embedding1, embedding2)
    print(cosine_sim,"cosine sim",jaccard_sim, "jaccard sim",pearson_corr,"pearson_corr")
    comb=(jaccard_sim*0.7)+(pearson_corr*0.2)+(cosine_sim*0.1)
    print(comb,"combined sim")
    return comb

# Define Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, num_embeddings_scenario, num_embeddings_law):
        super(SiameseNetwork, self).__init__()
        # self.embedding_scenario = nn.Embedding(num_embeddings=1398, embedding_dim=1398)
        # self.embedding_law =  nn.Embedding(num_embeddings=1398, embedding_dim=1398)
        num_embeddings_scenario = max(num_embeddings_scenario, X_test_scenario.max().item() + 1)
        self.embedding_scenario = nn.Embedding(num_embeddings=num_embeddings_scenario, embedding_dim=1398)
        num_embeddings_law = max(num_embeddings_law, X_test_law.max().item() + 1)
        self.embedding_law = nn.Embedding(num_embeddings=num_embeddings_law, embedding_dim=1398)
        # print("Embedding scenario size:", self.embedding_scenario.weight.size())
        # print("Embedding law size:", self.embedding_law.weight.size())
        
        self.fc = nn.Linear(1398 * 2+1, 1)  # Adjusted input size for embeddings and similarity
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward_once(self, x, embedding):
        # print("Input x shape:", x.shape)
        # print("Embedding weight shape:", embedding.weight.shape)
        embedded = embedding(x)
        # print("Embedded shape:", embedded.shape)
        return embedded.view(embedded.size(0), -1)

    def forward(self, input1, input2):
        print(input1,input2," this is input ")
        if isinstance(input1, str):
            l=list(input1)
            le=LabelEncoder()
            t=le.fit_transform(l)
            # output1 = self.forward_once(torch.tensor(t, dtype=torch.long).unsqueeze(0), self.embedding_scenario)
            # output2 = self.forward_once(input2, self.embedding_law)
            law_text = label_encoder_law.inverse_transform(input2.numpy())[0]
        
            similarities = []
            scenario_tokens = set(re.findall(r'\w+', input1.lower()))
            law_tokens = set(re.findall(r'\w+', law_text.lower()))
            similarity = jaccard_similarity(scenario_tokens, law_tokens,glove_model)
            similarities.append(similarity)
            similarities_tensor = torch.tensor(similarities, dtype=torch.float32)
            return similarity

        else :
            output1 = self.forward_once(input1, self.embedding_scenario)
            output2 = self.forward_once(input2, self.embedding_law)
            
            # Calculate Jaccard similarity
            scenario_texts = label_encoder_scenario.inverse_transform(input1.numpy())
            law_texts = label_encoder_law.inverse_transform(input2.numpy())
            # print(scenario_texts,law_texts)
            similarities = []
            for scenario_text, law_text in zip(scenario_texts, law_texts):
                scenario_tokens = set(re.findall(r'\w+', scenario_text.lower()))
                law_tokens = set(re.findall(r'\w+', law_text.lower()))
                similarity = jaccard_similarity(scenario_tokens, law_tokens,glove_model)
                similarities.append(similarity)
    
            similarities_tensor = torch.tensor(similarities, dtype=torch.float32)
            output = torch.cat((output1, output2, similarities_tensor.unsqueeze(1)), dim=1)  # Concatenate embeddings and similarities
            # print(output)
            output = self.fc(output)
            return self.sigmoid(output)


# Initialize the model
num_unique_scenario = len(X_train_scenario.unique())
num_unique_law = len(X_train_law.unique())
max_index_scenario = X_train_scenario.max()
max_index_law = X_train_law.max()
if max_index_scenario >= num_unique_scenario:
    num_unique_scenario = max_index_scenario + 1
if max_index_law >=num_unique_law:
    num_unique_law=max_index_law+1    
model = SiameseNetwork(num_unique_scenario, num_unique_law)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_train_scenario, X_train_law)  # Assuming no similarity for training
    loss = criterion(output, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    # In the testing part before passing data to the model
    # print("X_test_scenario shape:", X_test_scenario.shape)
    # print("X_test_law shape:", X_test_law.shape)
    test_output = model(X_test_scenario, X_test_law)
    test_loss = criterion(test_output, y_test.unsqueeze(1))
    print(f'Test Loss: {test_loss.item()}')

    # Calculate accuracy
    predictions = (test_output > 0.5).float()
    correct = (predictions == y_test.unsqueeze(1)).sum().item()
    total = y_test.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
    if accuracy>=0.61:
         best_accuracy = accuracy
         torch.save(model.state_dict(), 'siamese_model.pth')
         print("Model saved with accuracy:", accuracy)
def main():
    model = SiameseNetwork()

# Conditional check to execute main() function when law1.py is run as a script
if __name__ == "__main__":
    main()