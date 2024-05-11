import pandas as pd
import spacy

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration


# """
# method 1 using spacy 

# Load pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to generate summary from case details
def generate_summary(summary, outcome):
    # Process summary and outcome texts
    summary_doc = nlp(str(summary))
    outcome_doc = nlp(str(outcome))
    input_text = "case summary is " + str(summary) + "outcome is: " + str(outcome) 

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Generate the summary using the input text
    # generated_summary = model.generate(input_ids)
    # generated_summary = tokenizer.decode(generated_summary[0], skip_special_tokens=True)

    # Convert generator to list and extract first two sentences
    key_info_summary = " ".join([sent.text for sent in list(summary_doc.sents)[:2]])
    key_info_outcome = " ".join([sent.text for sent in list(outcome_doc.sents)[:1]])
    # Combine key information into a summary
    case_summary = f"Key info:  {key_info_summary} {key_info_outcome}"
    # case_summary = f"Key info:  {generated_summary}"

    return case_summary
"""
# method 2 using BART



# Load pre-trained BART model and tokenizer
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


def generate_summary(summary, outcome):
    input_text = "Case summary: " + summary + ". Outcome: " + outcome

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return generated_summary
"""
# Read CSV file
df = pd.read_csv("case.csv")

# # Generate summaries for each row
summaries = []
for index, row in df.iterrows():
    print(index)
    summary = row['Summary']
    outcome = row['Outcome']
    case_summary = generate_summary(summary, outcome)
    summaries.append(case_summary)

df['generated_summary'] = summaries

# Save DataFrame back to CSV file
df.to_csv("case.csv", index=False)

