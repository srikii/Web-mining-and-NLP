#!/usr/bin/env python
# coding: utf-8

# # Assignment 5: Clustering and Topic Modeling

# In this assignment, you'll need to use the following dataset:
# - text_train.json: This file contains a list of documents. It's used for training models
# - text_test.json: This file contains a list of documents and their ground-truth labels. It's used for testing performance. This file is in the format shown below. Note, each document has a list of labels.
# You can load these files using json.load()
# 
# |Text| Labels|
# |----|-------|
# |paraglider collides with hot air balloon ... | ['Disaster and Accident', 'Travel & Transportation']|
# |faa issues fire warning for lithium ... | ['Travel & Transportation'] |
# | .... |...|

# ## Q1: K-Mean Clustering
# 
# Define a function **cluster_kmean()** as follows: 
# - Take two file name strings as inputs: $train\_file$ is the file path of text_train.json, and $test\_file$ is the file path of text_test.json
# - When generating tfidf weights, set the min_df to 5.
# - Use **KMeans** to cluster documents in $train\_file$ into 3 clusters by **cosine similarity**  and **Euclidean distance** separately. Use sufficient iterations with different initial centroids to make sure clustering converge 
# - Test the clustering model performance using $test\_file$: 
#   * Predict the cluster ID for each document in $test\_file$.
#   * Let's only use the **first label** in the ground-truth label list of each test document, e.g. for the first document in the table above, you set the ground_truth label to "Disaster and Accident" only.
#   * Apply **majority vote** rule to dynamically map the predicted cluster IDs to the ground-truth labels in $test\_file$. **Be sure not to hardcode the mapping** (e.g. write code like {0: "Disaster and Accident"}), because a  cluster may corrspond to a different topic in each run. (hint: if you use pandas, look for "idxmax" function) 
#   * Calculate **precision/recall/f-score** for each label, compare the results from the two clustering models, and write your analysis in a pdf file 
# - This function has no return. Print out confusion matrix, precision/recall/f-score. 

# ## Q2: LDA Clustering 
# 
# Q2.1. Define a function **cluster_lda()** as follows: 
# 1. Take two file name strings as inputs: $train\_file$ is the file path of text_train.json, and $test\_file$ is the file path of text_test.json
# 2. Use **LDA** to train a topic model with documents in $train\_file$ and the number of topics $K$ = 3. Keep min_df to 5 when generating tfidf weights, as in Q1.  
# 3. Predict the topic distribution of each document in  $test\_file$ and select the topic with highest probability. Similar to Q1, apply **majority vote rule** to map the topics to the labels and show the classification report. 
# 4. Return the array of topic proportion array
# 
# Q2.2. Find similar documents
# - Define a function **find_similar_doc(doc_id, topic_mix)** to find **top 3 documents** that are the most similar to a selected one with index **doc_id** using the topic proportion array **topic_mix**. 
# - You can calculate the cosine or Euclidean distance between two documents using the topic proportion array
# - Return the IDs of these similar documents.
# 
# Q2.3. Provide a pdf document which contains: 
#   - performance comparison between Q1 and Q2.1
#   - describe how you tune the model parameters, e.g. alpha, max_iter etc. in Q2.1.
#   - discuss how effective the method in Q2.2 is to find similar documents, compared with the tfidf weight cosine similarity we used before.

# ## Q3 (Bonus): Biterm Topic Model (BTM)
# - There are many variants of LDA model. BTM is one designed for short text, while lDA in general expects documents with rich content.
# - Read this paper carefully http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf and try to understand the design
# - Try the following experiments:
#     - Script a few thousand tweets by different hastags
#     - Run LDA and BTM respectively to discover topics among the collected tweets. BTM package can be found at https://pypi.org/project/biterm/
#     - Compare the performance of each model. If one model works better, explain why it works better,
# - Summarize your experiment in a pdf document.
# - Note there is no absolute right or wrong answer in this experiment. All you need is to give a try and understand how BTM works and differences between BTM and LDA

# **Note: Due to randomness involved in these alogorithms, you may get the same result as what I showed below. However, your result should be close after you tune parameters carefully.**

# In[3]:


import json
from numpy.random import shuffle
from scipy import spatial
import numpy as np
import pandas as pd
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer,cosine_distance
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer
#from google.colab import files
# add your import


# In[12]:


# Q1
def cluster_kmean(train_file, test_file):
    
    data = pd.read_json(train_file, orient='columns')
    data.columns = ["text"]
    tfidf_vect = TfidfVectorizer(min_df=5,stop_words='english')
    dtm= tfidf_vect.fit_transform(data["text"])
    
    num_clusters=3
    clusterer = KMeansClusterer(num_clusters, cosine_distance,repeats=5)
    clusters = clusterer.cluster(dtm.toarray(),assign_clusters=True)
    
    test = pd.read_json(test_file, orient='columns')
    test.columns = ["text","label"]

    #to convert dataframe with multiple targets to the first target
    x=test["label"]
    truth = []
    for item in x:
        truth.append(item[0])
    test["label"] = truth
    
    test_dtm = tfidf_vect.transform(test["text"])
    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    confusion_df = pd.DataFrame(list(zip(test["label"].values, predicted)),                            columns = ["label", "cluster"])
    crosstab=pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label)
    print("using cosine: ")
    print(crosstab)
    dfmax = crosstab.idxmax(axis=1)
    print(dfmax)
    cluster_dict = {0: dfmax[0], 1: dfmax[1], 2: dfmax[2]}
    predicted_target=[cluster_dict[i]                       for i in predicted]

    print(metrics.classification_report          (test["label"], predicted_target))
    
    
    
    
    
    
    # Kmeans with 20 different centroid seeds
    num_clusters=3
    km = KMeans(n_clusters=num_clusters, n_init=20).fit(dtm)
    clusters = km.labels_.tolist()
    predicted2 = km.predict(test_dtm)
    confusion_df2 = pd.DataFrame(list(zip(test["label"].values, predicted2)),                            columns = ["label", "cluster"])
    
    crosstab2=pd.crosstab( index=confusion_df2.cluster, columns=confusion_df2.label)
    print("using Euclidean distance")
    print(crosstab2)
    dfmax = crosstab2.idxmax(axis=1)
    print(dfmax)
    cluster_dict={0: dfmax[0], 1: dfmax[1], 2: dfmax[2]}
    
    predicted_target2=[cluster_dict[i]                       for i in predicted2]
    print(metrics.classification_report          (test["label"], predicted_target2))

    
    
    
    return None
        


# In[16]:


# Q2
def cluster_lda(train_file, test_file):
    
    data = pd.read_json(train_file, orient='columns')
    data.columns = ["text"]

    test = pd.read_json(test_file, orient='columns')
    test.columns = ["text","label"]

    x=test["label"]
    truth = []
    for item in x:
        truth.append(item[0])
        
    test["label"] = truth
    
    tf_vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(data["text"])

    tf_feature_names = tf_vectorizer.get_feature_names()
    
    
    num_topics = 3
    #verbose gives the detailed information for every iteration can be 0 ,1 or 2. 0 for no details 2 for maximum details.
    lda = LatentDirichletAllocation(n_components=num_topics,                                 max_iter=20,verbose=1,evaluate_every=1, n_jobs=1,random_state=0).fit(tf)
    
    test_tf = tf_vectorizer.transform(test["text"])
    topic_assign = lda.transform(test_tf)
    topics = pd.DataFrame(topic_assign)
    dfmax = topics.idxmax(axis=1)
    crosstab = pd.crosstab(index=dfmax, columns=test["label"])
    print(crosstab)
    crosstab['max'] = crosstab.idxmax(axis=1)
    print(crosstab['max'])
    
    cluster_dict={0:(crosstab['max'][0]),1:(crosstab['max'][1]),2:(crosstab['max'][2])}
    predicted_target=[cluster_dict[i] for i in dfmax]
    print(metrics.classification_report(test["label"],predicted_target))
    
    return topic_assign

def find_similar(doc_id, topic_assign):
    
    i=doc_id
    p=0
    a=[]
    b=[]
    for index,values in enumerate(topic_assign):
        similarity = 1 - spatial.distance.cosine(topic_assign[i], topic_assign[p])
        p+=1
        a.append(similarity)
        b.append(index)
    
    c=sorted(zip(a, b), reverse=True)[:4]
    docs=[c[1][1],c[2][1],c[3][1]]
    
    return docs


# In[17]:


if __name__ == "__main__":  
    
    # Due to randomness, you won't get the exact result
    # as shown here, but your result should be close
    # if you tune the parameters carefully
    
    # Q1
    print("Q1")
    cluster_kmean("train_text.json","test_text.json")
            
    # Q2
    print("\nQ2")
    topic_assign =cluster_lda('train_text.json','test_text.json')
    doc_ids = find_similar(10, topic_assign)
    print ("docs similar to {0}: {1}".format(10, doc_ids))


# In[ ]:




