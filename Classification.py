import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

data = pd.read_csv("train.csv")


# removes punctuation symbols

def punctuation():
    data['question_text'] = data['question_text'].apply(lambda x: x.lower())
    data['question_text'] = data['question_text'].apply(lambda x: x.translate(string.punctuation))
    #data['question_text'] = data['question_text'].apply(lambda x: x.translate(string.digits))
    #print(data['question_text'])

#tokenizing the given texts
def tokenization():
    data['tokenized_text'] = data['question_text'].apply(word_tokenize)
    print(data["tokenized_text"].head())


def lemmatize_text(s):

    s = [lemmatizer.lemmatize(word) for word in s]
    return s





# for row in data["question_text"]:
#   punctuation(row)
punctuation()
tokenization()
#lemmatize_text()
# print(data["question_text"].head())
data = data.assign(col_lemma=data['tokenized_text'].apply(lambda x: lemmatize_text(x)))
print(data)

