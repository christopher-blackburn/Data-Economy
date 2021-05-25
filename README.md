# Data-Economy

This repository is dedicated to the project "Valuing the Data Economy: A Labor Costs Approach using Unsupervised Machine Learning". 

# Pre-Processing Steps

## 1. Unpack the Burning Glass Database

The Burning Glass database consists of more than 200 million job postings from 2010 to 2019. The structured databases are contained in several ``zip`` files, while the unstructured data is contained in XML files. 

## Doc2Vec Model Training and Evaluation

To proxy time-use allocation factors, we need to come up with an estimate for the similarity between occupations based on their relationship with data-intensive responsibilities. One way of going about this estimation is by comparing the full set of occupations with known data-intensive occupations using only the language contained in the job posting text. There are a variety of ways to compute document similarities, e.g. TF-IDF, Jaccard Distance, etc., but these bag-of-words methods may miss important semantic similarities between job postings. To this end, we use Doc2Vec to perform our similarity task.

In summary, Doc2vec builds on the Word2Vec algorithm that uses word "contexts" to train a shallow neural network. There are two approaches, continuous bag-of-words and skip-gram. Each approach trains the weights of the neural network to either predict a word's context (skip-gram) or using a word's context to predict itself (continuous bag-of-words). In either case, the trained weights of the neural network correspond to the vector representations of each word in the vocabulary. Doc2Vec builds on this approach by adding a "document tag" that encodes document-level attributes into the Word2Vec model. As a consequence, document vectors are trained alongside word vectors, resulting in a dense vector representation for each document tag.

To dampen the effects of job posting outliers, I train the Doc2Vec model using ONET categories as the document tags. In this sense, the model trains a vector representation for each ONET category. After experimenting with different parameter combinations, I find training the model using the CBOW algorithm, with 1000 dimension vectors, and at least 15 epochs yields the best results. 

The code [doc2vec_training_test.py](doc2vec/doc2vec_training_test.py)

## 2. Task Parsing
One issue with using all visible text from the scraped job postings to train the LDA model is that other job related information may be comingled with the task data. In an effort to reduce commingled data, we combine two natural language processing techniques to extract task information from job postings. The first technique, known as <i>sentence boundary disambiguation</i>, is a technique used to parse sentences within a document by detecting the beginning and end of a sentence. The second technique is a part-of-speech tagging algorithm that classifies the part-of-speech, e.g. noun, verb, adverb, a word token belongs to in a sentence. 

The code [get_tasks.py](get_tasks.py) 
