import numpy as np

from gensim.models import FastText
from gensim.utils import tokenize
from gensim import utils
from gensim.test.utils import datapath
from collections import Counter
from stopwords import IT_SW
from sklearn.metrics.pairwise import cosine_similarity


stopwords_dict = Counter(IT_SW)

def clean(s):
  import re
  s = re.sub(r'[^a-zA-Z0-9]',' ', s).lower()
  return s

def rem_common_and_short_words(text):
  text = text.split()
  new_text = ''
  for word in text:    
    if (len(word) > 3) and (word not in stopwords_dict):
      new_text = new_text + ' ' + word
  return new_text

class MyIter:
    def __init__(self, filepath):
        self.filepath = filepath   
    def __iter__(self):
        path = self.filepath
        with utils.open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(rem_common_and_short_words(clean(line)))) 

def train_nn(filepath, vector_size = 16, window = 5, min = 3, epochs = 5): 
    model = FastText(vector_size=vector_size, window=window, min_count=min)
    model.build_vocab(corpus_iterable=MyIter(filepath))
    total_examples = model.corpus_count
    model.train(corpus_iterable=MyIter(filepath), total_examples=total_examples, epochs=epochs)
    return model

def save_model(nn, filepath = 'autopic_model.nn'):
   nn.save(filepath)
   

def load_model(filepath = 'autopic_model.nn'):
   return FastText.load(filepath)

def get_topic_distance(tw, topic_words, nn_model):   
  l = []
  for word in tw:    
    if (len(word) > 3) and (word not in stopwords_dict):
      v = np.array([nn_model.wv[word]])
      max_s = 0
      max_t = ''
      for topic_word in topic_words:
        t = np.array([nn_model.wv[topic_word]])      
        #Cosine similarity
        similarity = cosine_similarity(v,t)[0][0]       
        if similarity > max_s:
          max_s = similarity
          max_t = topic_word
      #print(max_s, word, max_t)      
      l.append([max_s, max_t, word])
  return l

def get_topic(t, topics, nn_model, alpha = 0.7, beta = 0.65, gamma = 0.75): 
  t = rem_common_and_short_words(clean(t)).split()
  tweet_topics = []  
  for topic in topics:     
    l = get_topic_distance(t, topic[0], nn_model)    
    l = sorted(l, reverse=True)[0:2]        
    # Assign topic based on score
    # for every word in tweet check how similar is to words in topics
    # then assign topic to tweet based of two of most similar words in topic   
    if (l[0][0] > alpha) and (l[1][0] > beta) and ((l[0][0] + l[1][0])/2 > gamma):
      score = (l[0][0] + l[1][0])/2              
      tweet_topics.append([score, topic[1]])  
    else:
      #If no topic found -> topic = 'Altro'
      tweet_topics.append([1 - (l[0][0] + l[1][0])/2, 'Altro'])        
  return sorted(tweet_topics, reverse= True)[0:3]




