import pandas as pd

data = pd.read_csv("train.csv")

#removes punctuation symbols

def punctuation(string):
    punctuation = '''''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuation:
            no_punct = no_punct + char
    print(no_punct)


for row in data["question_text"]:
    punctuation(row)

#print(data["question_text"].head())
