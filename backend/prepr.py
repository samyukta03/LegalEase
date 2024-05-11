# Define the path to your original GloVe embeddings file
original_file_path = 'glove.6B.300d.txt'

# Define the path to save the preprocessed file
preprocessed_file_path = 'glove.6B.300d1.txt'

# Read the original file to get the number of lines (vocab size) and vector size
with open(original_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    vocab_size = len(lines)
    vector_size = len(lines[0].strip().split()) - 1

# Write the vocab size and vector size as the first line of the preprocessed file
with open(preprocessed_file_path, 'w', encoding='utf-8') as f:
    f.write(f"{vocab_size} {vector_size}\n")

# Append the original lines to the preprocessed file
with open(original_file_path, 'r', encoding='utf-8') as f_in:
    with open(preprocessed_file_path, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(line)
