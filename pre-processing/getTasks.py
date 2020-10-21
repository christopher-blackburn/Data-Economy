#File: sentence_parser.py

import csv
import os
import glob 
import re
import random 
import spacy 
import nltk.data

# Load required resources
nlp = spacy.load('en_core_web_sm')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def task_parser(file_name):
	os.chdir('/Users/cblackburn/Dropbox/Data Economy/Data')
	f = open(file_name,'r')
	text_data = f.readlines()
	text_string = text_data[0].lstrip()

	# These words appear in each job posting because the scraper picked them up
	additional_stopwords = ['Find jobs','Find Jobs','Advanced Job Search','Company reviews','Find salaries','Upload your resume','Sign in |', 'Employers /','Post Job','what job title, keywords, or company',
								'where city, state, or zip code','Find Jobs Advanced Search', 'Hiring Lab','Career Advice','Browse Jobs','Browse Companies','Salaries Find',
								'Certifications Employer','Events Work at Indeed','Countries About','Help Center','© 2019 Indeed','Do Not Sell My Personal Information',
								'Cookies, Privacy and Terms','Let Employers Find You','Upload Your Resume', 'Our m...','Let employers find you','Thousands of employers search for candidates on Indeed','']

	# Remove the nuisance words from the text data
	for word in additional_stopwords:
		text_string = text_string.replace(word,'')

	# Remove leading whitespaces
	text_string = text_string.lstrip()

	# Convert proper nouns into lower case
	for token in nlp(text_string):
		if token.pos_ == 'PROPN':
			text_string = text_string.replace(str(token),str(token).lower())
	print(text_string)


	# Find the words with upper case letters
	r = re.findall('([A-Z][a-z]+)',text_string)

	# Split the text into tokens
	token_string = text_string.split()

	# Find the positions of words with upper case letters
	start_points = []
	for idx,token in enumerate(token_string):
		if token in r:
			start_points.append(idx)

	# Parse the sentences based on upper case words
	parsed_sentences = []
	for idx,start in enumerate(start_points):
		try:
			end = start_points[idx+1] 
		except IndexError:
			end = len(token_string) 
		sentence = ' '.join(token_string[start:end]) + '.'
		parsed_sentences.append(sentence)

	lemma_parsed_sentences = []
	# For each parsed sentence...
	for sent in parsed_sentences:
		#Lemmatize the words in the sentence
		lemma_parsed_sentences.append(' '.join([token.lemma_ for token in nlp(sent)]))


	task_sentences = []
	# For each lemmatized, parsed sentence...
	for lem_sent in lemma_parsed_sentences:
		# Apply the rule for identifying task statements 
		if nlp(lem_sent)[0].pos_ == 'VERB' and nlp(lem_sent)[0].dep_ == 'ROOT':
			task_sentences.append(lem_sent)

	cleaned_tasks = []
	# For each task...
	for task in task_sentences:
		task = task.replace('\uf0b7','')
		task = task.replace('..','.')
		cleaned_tasks.append(task)

	# Some tasks may end up being embedded in a single string 
	# The follow code will break up embedded tasks into different task statements
	final_task_list = []
	# For each task...
	for task in cleaned_tasks:
		task_list = tokenizer.tokenize(task)
		if len(task_list) == 1:
			final_task_list.append(task_list[0])
		else:
			for sub_task in range(0,len(task_list)):
				final_task_list.append(sub_task)
	# The last step is to filter the resulting tasks a bit
	# There are some tasks that are short, such as 'apply .'
	
	ftasks = []
	for task in final_task_list:
		if type(task) == str:
			ftasks.append(task)

	return ftasks 


# A function to remove nonsense tasks
nonsense_task_list = ['pay time off','learn more about federal benefit','save this job','apply .',
						'read what people be say about work here .','click here to view more .','apply today','year','years', 'work from home ','save job report job',
						'require travel : ','company info follow','https:','for more information','www.','click here',"'s degree"]
def remove_nonsense(task):
	i = 0
	for nonsense in nonsense_task_list:
		if nonsense in task or len(task.split()) < 4:
			i+=1
	if i == 0:
		return task.strip()



# Grab the text files 
text_files = glob.glob('/Users/cblackburn/Dropbox/Data Economy/Data/*text.txt')

# For each job posting...
for text in text_files:
	# Extract the jk
	pat = '(?s:.*)_(.+?)text'
	jk = re.search(pat,text)[1]

	# Job postings task list 
	task_list = []
	# For each task in the job posting...
	for task in task_parser(text):
		no_nonsense_task = remove_nonsense(task)
		if no_nonsense_task is None:
			pass 
		else:
			task_list.append(no_nonsense_task)
	
	# Save the data 
	f_name = '/Users/cblackburn/Dropbox/Data Economy/Data/' + jk + 'task.txt'

	with open(f_name,'w') as f:
		for t in task_list:
			f.write('%s\n' % t)
	print('Successfully saved job {}...'.format(jk))



# Save the general task data
print('Saving task data...')
task_data = [[remove_nonsense(task)] for text in text_files for task in task_parser(text)]
with open('/Users/cblackburn/Dropbox/Data Economy/Output/evaluate_task_parser.csv','w+') as f:
	writer = csv.writer(f)
	writer.writerows(task_data)