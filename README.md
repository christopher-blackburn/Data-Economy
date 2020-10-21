# Data-Economy

This repository is dedicated to the project "Valuing the Data Economy: A Labor Costs Approach using Unsupervised Machine Learning". 

# Pre-Processing Steps

## 1. Scrape job postings from Indeed.com

## 2. Task Parsing
One issue with using all visible text from the scraped job postings to train the LDA model is that other job related information may be comingled with the task data. In an effort to reduce commingled data, we combine two natural language processing techniques to extract task information from job postings. The first technique, known as <i>sentence boundary disambiguation</i>, is a technique used to parse sentences within a document by detecting the beginning and end of a sentence. The second technique is a part-of-speech tagging algorithm that classifies the part-of-speech, e.g. noun, verb, adverb, a word token belongs to in a sentence. 

The code [get_tasks.py](get_tasks.py) 
