import autopic

#Please note stop words are automatically imported, you can use your own stopwords 
#by simply changing stopwords.py according your needs

#before all are we need to trian a neural networks model using FastText method.
# We can begin by calling the train function 

model = autopic.train_nn('twitter_data.csv')

#Once trained we can save it for later use
autopic.save_model(model)

#to load a model use:
topic_model = autopic.load_model()

#Now we are ready to use our model to classify topics.
#Here we use the embeddings as a parameter to identify predefined topics, so we will need to define some topics
#according this structure:

#Topics are defined using keywords, if a sentence don't fit a predefined topic than will be classified as 'Altro'
t_sicurezza = ["rapina","furto","omicidio","uccisione","violenza","arresto", "denuncia", "carcere"]
t_qvita = ["urbano", "mangiare", "shopping", "servizi", "trasporti", "scuole", "ospedali", "viabilità", "traffico", "prezzo", "costo"]
t_ambiente = ["smog", "inquinamento", "natura", "bosco", "oasi", "parco", "spiaggia", "montagna", "ambiente", "meteo", "contagi", "epidemia", "immondizia" ,"salute"]
t_tempo_libero = ["sport", "calcio", "palestra", "atletica", "fisico", "tennis", "basket", "saldi", "shopping", "negozi"]
t_cultura = ["cinema", "teatro", "museo", "escursione", "storia", "letteratura", "filosofia", "concerto", "biblioteca", "libro"]
t_politica = ["partito", "politica", "elezioni", "votazioni"]

topics_names = [[t_sicurezza, 'Sicurezza'], 
          [t_qvita, 'Qualità della vita e servizi'], 
          [t_ambiente, 'Ambiente e salute'],
          [t_tempo_libero, 'Tempo libero'],
          [t_cultura, 'Cultura'],
          [t_politica, 'Politica']
          ]

#Now we ready to identify a topic. Please note that as the training set is based on italian tweets 
#the sentence must have a similar fashion. Changing dataset will ovecome this limitation.

#Example 1
sentence = "Diagnosticati casi positivi in una RSA a Sciacca."
r = autopic.get_topic(sentence, topics_names, topic_model)
print(sentence, r[0])

#Example 2: 
sentence = "La partita di ieri è stata bruttissima. I giocatori non si reggevano in piedi"
r = autopic.get_topic(sentence, topics_names, topic_model)
print(sentence, r[0])

#Example 3: for a higher precision we can overwrite default values of alpha, beta and gamma params. 
# defalut values are: alpha = 0.75, beta (must be < alpha) = 0.65 and gamma = 0.75.  
sentence = "La partita di ieri è stata bruttissima. I giocatori non si reggevano in piedi"
r = autopic.get_topic(sentence, topics_names, topic_model, alpha = 0.98, beta = 0.9, gamma = 0.95)
print(sentence, r[0])



