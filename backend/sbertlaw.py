# """
# acc - 75 % but test data == 33% of training data, training data = 67% of training data 
 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

# Load train and test data
df = pd.read_csv('train-news.csv', encoding='latin1')
test_data_path = 'test.csv'
best_accuracy = 0.0

# Load GloVe embeddings
def load_glove_embeddings(embeddings_file):
    embeddings_index = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings_file = 'glove.6B.300d.txt'
glove_embeddings_index = load_glove_embeddings(glove_embeddings_file)

# Assuming you have loaded your dataset into a DataFrame 'df'
# and it has columns 'Scenario', 'Law', and 'Similarity'

# Encode scenarios and laws
label_encoder_scenario = LabelEncoder()
label_encoder_law = LabelEncoder()
label_encoder_similarity = LabelEncoder()
df['Scenario'] = label_encoder_scenario.fit_transform(df['Scenario'])
df['Law'] = label_encoder_law.fit_transform(df['Law'])
df['Similarity'] = label_encoder_similarity.fit_transform(df['Similarity'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Scenario', 'Law']], df['Similarity'], test_size=0.33,
                                                  random_state=42, stratify=df['Similarity'])

# Define Siamese Network architecture
class SiameseNetwork(nn.Module):
    
    def __init__(self, embedding_matrix):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix), padding_idx=0)
        self.fc = nn.Linear(30000,1)   # Adjusted input size for concatenated embeddings
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        # Embed input sequences
        embedded_input1 = self.embedding(input1)
        embedded_input2 = self.embedding(input2)
        
        # Flatten embeddings
        embedded_input1 = embedded_input1.view(embedded_input1.size(0), -1)
        embedded_input2 = embedded_input2.view(embedded_input2.size(0), -1)
        
        # Concatenate embeddings
        concatenated_output = torch.cat((embedded_input1, embedded_input2), dim=1)
        # Convert the concatenated_output tensor to float32 (if it's not already)
        concatenated_output = concatenated_output.float()

        # Pass through fully connected layer and sigmoid
        output = self.fc(concatenated_output)
        return self.sigmoid(output)

# Initialize tokenizer
tokenizer = Tokenizer()

# Create sequences from text data
X_train_scenario_text = label_encoder_scenario.inverse_transform(X_train['Scenario'])
tokenizer.fit_on_texts(X_train_scenario_text)
X_train_scenario_sequences = tokenizer.texts_to_sequences(X_train_scenario_text)

X_train_law_text = label_encoder_law.inverse_transform(X_train['Law'])
tokenizer.fit_on_texts(X_train_law_text)
X_train_law_sequences = tokenizer.texts_to_sequences(X_train_law_text)

# Pad sequences
max_len = 50
X_train_scenario_padded = pad_sequences(X_train_scenario_sequences, maxlen=max_len, padding='post', truncating='post')
X_train_law_padded = pad_sequences(X_train_law_sequences, maxlen=max_len, padding='post', truncating='post')

# Initialize embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))  # Assuming 100-dimensional GloVe embeddings

# Map words to GloVe embeddings
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create an instance of the SiameseNetwork model
model = SiameseNetwork(embedding_matrix)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer here

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Convert to PyTorch tensors
    input1 = torch.tensor(X_train_scenario_padded, dtype=torch.long)
    input2 = torch.tensor(X_train_law_padded, dtype=torch.long)
    targets = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # Add unsqueeze(1) here

    # Forward pass
    output = model(input1, input2)
    print("Shape of output tensor:", output.shape)
    print("Shape of target tensor:", targets.shape)

    # Calculate loss
    loss = criterion(output, targets)


    # Calculate loss
    loss = criterion(output, targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    # Calculate accuracy
    predictions = (output > 0.5).float()
    correct = (predictions == targets).sum().item()
    total = len(targets)
    accuracy = correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')

    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    # Convert test data to PyTorch tensors
    X_test_scenario_text = label_encoder_scenario.inverse_transform(X_test['Scenario'])
    X_test_law_text = label_encoder_law.inverse_transform(X_test['Law'])

    # Convert test data to sequences using the same Tokenizer used for training
    X_test_scenario_sequences = tokenizer.texts_to_sequences(X_test_scenario_text)
    X_test_law_sequences = tokenizer.texts_to_sequences(X_test_law_text)
    
    # Pad sequences using the same maxlen and padding index used during training
    X_test_scenario_padded = pad_sequences(X_test_scenario_sequences, maxlen=max_len, padding='post', truncating='post')
    X_test_law_padded = pad_sequences(X_test_law_sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Convert padded sequences to PyTorch tensors
    test_input1 = torch.tensor(X_test_scenario_padded, dtype=torch.long)
    test_input2 = torch.tensor(X_test_law_padded, dtype=torch.long)

    # Get test predictions
    test_output = model(test_input1, test_input2)

    # Calculate test loss
    test_loss = criterion(test_output, torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1))
    print(f'Test Loss: {test_loss.item()}')

    X_test['Scenario_Text'] = label_encoder_scenario.inverse_transform(X_test['Scenario'])
    X_test['Law_Text'] = label_encoder_law.inverse_transform(X_test['Law'])
    # X_test['Similarity_Text'] = label_encoder_similarity.inverse_transform(X_test['Similarity'])
    
    # Convert test_output to a list of predicted similarities
    predicted_similarities = test_output.squeeze().tolist()

    # Add the predicted similarities to the test DataFrame
    X_test['Predicted_Similarity'] = predicted_similarities
    
    # Combine scenario and law text with predicted similarities
    X_test_with_predictions = X_test[['Scenario_Text', 'Law_Text' , 'Predicted_Similarity']]

    
    # Save the test DataFrame with the predicted similarities
    X_test.to_csv('test_with_predictions_1.csv', index=False)

    # Calculate accuracy
    predictions = (test_output > 0.5).float()
    correct = (predictions == torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)).sum().item()
    total = y_test.size
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')

    if accuracy >= 0.6:
        best_accuracy = accuracy
        torch.save(model.state_dict(), '77siamese.pth')
        print("Model saved with accuracy:", accuracy)

# acc - 75 % but test data == 33% of training data, training data = 67% of training data 
"""



 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# Load train and test data
df = pd.read_csv('train-new1.csv', encoding='latin1')

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Encode scenarios and laws
label_encoder_scenario = LabelEncoder()
label_encoder_law = LabelEncoder()
df['Scenario'] = label_encoder_scenario.fit_transform(df['Scenario'])
df['Law'] = label_encoder_law.fit_transform(df['Law'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Scenario', 'Law']], df['Similarity'], test_size=0.33, random_state=42, stratify=df['Similarity'])

# Define Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768 * 2 + 1, 1)  # Adjusted input size for BERT embeddings and similarity
        
    def forward(self, input1, input2):
        output1 = self.bert_model(**input1)[1]  # Last hidden state pooled output
        output2 = self.bert_model(**input2)[1]
        
        # Calculate Jaccard similarity
        similarity = self.calculate_similarity(output1, output2)
        return similarity
        
    def calculate_similarity(self, embedding1, embedding2):
        similarity = torch.cosine_similarity(embedding1, embedding2, dim=-1)
        return similarity

# Tokenize and prepare inputs
def prepare_input(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    return inputs

# Initialize the model
model = SiameseNetwork()

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in zip(X_train['Scenario'], X_train['Law'], y_train):
        scenario, law, similarity = batch
        
        input1 = prepare_input(str(scenario))
        input2 = prepare_input(str(law))
        similarity = torch.tensor(similarity).float()
        optimizer.zero_grad()
        output = model(input1, input2)
        loss = criterion(output, similarity.unsqueeze(0).float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}')

# Evaluate the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for batch in zip(X_test['Scenario'], X_test['Law'], y_test):
        scenario, law, similarity = batch
        input1 = prepare_input(str(scenario))
        input2 = prepare_input(str(law))
        similarity = torch.tensor(similarity).float()
        output = model(input1, input2)
        predictions = (output > 0.5).float()
        test_loss += criterion(output, similarity.unsqueeze(0).float())
        correct += (predictions == similarity).sum().item()
        total += 1
    
    print(f'Test Loss: {test_loss}')

    # Calculate accuracy
    # predictions = (output > 0.5).float()
    # correct = (predictions == similarity.unsqueeze(0)).sum().item()
    # total = len(y_test)
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')
    # if accuracy>=0.5:
    #      best_accuracy = accuracy
    #      torch.save(model.state_dict(), 'bert_model.pth')
    #      print("Model saved with accuracy:", accuracy)
    """