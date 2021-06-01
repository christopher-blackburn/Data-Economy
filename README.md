# Data-Economy

This repository is dedicated to the project "Valuing the Data Economy: A Labor Costs Approach using Unsupervised Machine Learning". After we received the full Burning Glass Dataset on October 7th, 2020, the methodology changed from using Latent Dirichlet Allocation to Doc2Vec. The description contained in this is README pertains to the Doc2Vec version of the methodology that uses the Burning Glass Database. 

# Pre-Processing Steps

## 1. Unpack the Burning Glass Database

The Burning Glass database consists of more than 200 million job postings from 2010 to 2019. The structured databases are contained in several ``zip`` files, while the unstructured data is contained in XML files. 

## Doc2Vec Model Training and Evaluation

To proxy time-use factors, we need to come up with an estimate for the similarity between occupations based on their relationship with data-intensive responsibilities. One way of going about this estimation is by comparing the full set of occupations with known data-intensive occupations using only the language contained in the job posting text. There are a variety of ways to compute document similarities, e.g. TF-IDF, Jaccard Distance, etc., but these bag-of-words methods may miss important semantic similarities between job postings. To this end, we use Doc2Vec to perform our similarity task.

In summary, Doc2vec builds on the Word2Vec algorithm that uses word "contexts" to train a shallow neural network. There are two approaches, continuous bag-of-words and skip-gram. Each approach trains the weights of the neural network to either predict a word's context (skip-gram) or using a word's context to predict itself (continuous bag-of-words). In either case, the trained weights of the neural network correspond to the vector representations of each word in the vocabulary. Doc2Vec builds on this approach by adding a "document tag" that encodes document-level attributes into the Word2Vec model. As a consequence, document vectors are trained alongside word vectors, resulting in a dense vector representation for each document tag.

To dampen the effects of job posting outliers, I train the Doc2Vec model using ONET categories as the document tags. As a consequence, the model trains a vector representation for each ONET category rather than each document. After experimenting with different parameter combinations, I find training the model using the CBOW algorithm, with 1000 dimensions, 15 epochs, a window size of 3 words, and a hierarchical softmax classifier yields the best results. 



Due to limits on our computational resources, I limit the size of the training sample and elect to train a Doc2Vec model for each year in the data. To construct the annual training sample, I start by randomly choosing a week of job postings for a given month. I repeat this sampling process for each month so each month is represented by a weekly job posting file. In other words, our annual training sample is a composite of 12 weekly job postings files. 

After the monthly representatives are sampled and combined into an annual dataset, we split the annual dataset into a training and test sample. As before, our computational resources are limited, and therefore, we randomly sample 1 million job postings from the annual dataset to construct the training sample. We note that for each year we ensure the training and test split is such that the size of the test dataset is at least two-thirds of the training dataset. This is done to ensure an adequate sample remains for validating the trained Doc2Vec model.

After training the Doc2Vec model, I evaluate the model's performance using two common approaches. For each validation procedure, we estimate the performance metric by randomly sampling 50,000 observations with replacement from the test dataset across 5 iterations. Our first metric uses a nearest neighbors criterion to judge the accuracy of the model. For each job posting in the bootstrapped sample, we compute the cosine distance between the inferred job posting's document vector and the representative ONET category embeddings trained by the model. Once computed, we compute a binary metric based on whether the job posting's ONET category is in the 10 nearest ONET category embeddings. We estimate the performance metric by summing up these binary metrics and dividing by the number of job postings in the sample.

Our second approach uses a similarity triplet method. In this approach, we construct a triplet consisting of three job postings. Two of the three job postings are chosen to be in the same ONET category, and the final posting is chosen at random from a different ONET category. To compute the accuracy metric, we assign a binary score to each triplet. The score is equal to one when the cosine distance between the two job postings in the same ONET category is less than the distance between these postings and the job posting from a separate ONET category. The aggregate performance metric is then the sum of all binary scores divided by the total number of triplets constructed from the sample. 

As a complimentary check on the model, I also visualize the trained ONET embeddings using t-SNE (t-distributed Stochastic Neighbor Embeddings). I follow recommended guidelines (such as as PCA to reduce the embedding dimensionality and a large number of iterations) when applying t-SNE to the estimated ONET embeddings for 2011. Visualizing the embeddings in 2 dimensions shows the model appears to be performing reasonably well when preserving local relationships in the higher dimensional space. 


### The Code and Relevant Output

With broad brush strokes laid out above, the details of the program are contained in the code [doc2vec_training_test.py](doc2vec/doc2vec_training_test.py). The sections of the code are pretty self-contained and I have provided detailed to comments that highlight what each line of code is doing. For data on the the size of the training/test datasets for each year, consult the [train_file.csv](Data/train_file.csv) file. A feature file containing vocabulary size and the number of trained embeddings is in [feature_file.csv](Data/feature_file.csv). 

Lastly, the performance file containing the performance metrics is in [performance_file.csv](Data/performance_file.csv). There are four columns of data in the file: ``year, nn_score, nn_rank, triplet_score``. The ``year`` column is self-explanatory. The ``nn_score`` and ``nn_rank`` columns correspond to the nearest neighbor evaluation method discussed above. The ``nn_score`` is the bootstrapped performance metric using the nearest neighbor criteria, and the ``nn_rank`` column gives the average position for instances where the ONET category is in the Top 10 nearest neighbors. Lastly, the ``triplet_score`` column is the bootstrapped performance metric using the triplet method. 

Each trained model is saved on ``serv570`` in the directory ``prediction/burning_glass/Chris/Word2Vec`` (I know the folder name is misleading, but Doc2Vec is basically Word2Vec with a twist!). Unless you have a strong desire to re-train the models, you can use these files to load the models for yourself. 

Finally, the code for constructing the ONET landmark embeddings figure can be found in [tsneLandmarks.py](doc2vec_viz/tsneLandmarks.py), and the figure produced from the code is [doc2vec_landmarks_inset.png](figures/doc2vec_landmarks_inset.png).


## Data-Interfacing Occupations

The second component of the time-use proxy is an estimate for the fraction of jobs in an occupation that interface with data. To identify these jobs, I use Burning Glass' Skills Database. Using the database, we can identify jobs that are likely interfacing with data based on the skills required to perform the job's duties and responsibilities. 
