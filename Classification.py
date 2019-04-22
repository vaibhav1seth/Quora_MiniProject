import pandas as pd
import string
import word2vec
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

data = pd.read_csv("train.csv")
settings = {
	'window_size': 2,	# context window +- center word
	'n': 10,		# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 50,		# number of training epochs
	'learning_rate': 0.01	# learning rate
}

# Initialise object

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
#print(data)

w2v = word2vec()
# Numpy ndarray with one-hot representation for [target_word, context_words]
training_data = w2v.generate_training_data(settings,data['col_lemma'])


class word2vec():
  def __init__(self):
    self.n = settings['n']
    self.lr = settings['learning_rate']
    self.epochs = settings['epochs']
    self.window = settings['window_size']

  def generate_training_data(self, settings, corpus):
    # Find unique word counts using dictonary
    word_counts = defaultdict(int)
    for row in corpus:
      for word in row:
        word_counts[word] += 1
    ## How many unique words in vocab? 9
    self.v_count = len(word_counts.keys())
    # Generate Lookup Dictionaries (vocab)
    self.words_list = list(word_counts.keys())
    # Generate word:index
    self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
    # Generate index:word
    self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

    training_data = []
    # Cycle through each sentence in corpus
    for sentence in corpus:
      sent_len = len(sentence)
      # Cycle through each word in sentence
      for i, word in enumerate(sentence):
        # Convert target word to one-hot
        w_target = self.word2onehot(sentence[i])
        # Cycle through context window
        w_context = []
        # Note: window_size 2 will have range of 5 values
        for j in range(i - self.window, i + self.window+1):
          # Criteria for context word 
          # 1. Target word cannot be context word (j != i)
          # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
          # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
          if j != i and j <= sent_len-1 and j >= 0:
            # Append the one-hot representation of word to w_context
            w_context.append(self.word2onehot(sentence[j]))
            # print(sentence[i], sentence[j]) 
            # training_data contains a one-hot representation of the target word and context words
        training_data.append([w_target, w_context])
    return np.array(training_data)

  def word2onehot(self, word):
    # word_vec - initialise a blank vector
    word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
    # Get ID of word from word_index
    word_index = self.word_index[word]
    # Change value from 0 to 1 according to ID of the word
    word_vec[word_index] = 1
    return word_vec



