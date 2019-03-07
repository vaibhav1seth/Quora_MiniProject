import pandas as pd
import string
from nltk.tokenize import word_tokenize

data = pd.read_csv("train.csv")


# removes punctuation symbols

def punctuation():
    data['question_text'] = data['question_text'].apply(lambda x: x.lower())
    data['question_text'] = data['question_text'].apply(lambda x: x.translate(string.punctuation))
    data['question_text'] = data['question_text'].apply(lambda x: x.translate(string.digits))
    #print(data['question_text'])

#tokenizing the given texts
def tokenization():
    data['tokenized_text'] = data['question_text'].apply(word_tokenize)
    print(data["tokenized_text"].head())


# for row in data["question_text"]:
#   punctuation(row)
punctuation()
tokenization()
# print(data["question_text"].head())
