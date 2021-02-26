# Text preprocessing module 
from gensim.utils import simple_preprocess

# Identifying data-related titles
def dataTitle(x):
	
	data = 0 
	
	for token in x.split():
		
		if token.lower() in ['data','database','dataset']: data +=1 
	
	return data 


# Identifying data-related job posting text
def dataText(x):

	data = 0 

	kws = ['data','database','databases','dataset','datasets']

	text = simple_preprocess(x)

	for word in text:

		if word in kws:

			data += 1

	if data > 0:

		return 1

	else:

		return 0

# A weighted text version 
def dataText_weighted(x):

	data = 0 

	kws = ['data','database','databases','dataset','datasets']

	text = simple_preprocess(x)

	for word in text:

		if word in kws:

			data += 1

	if data > 0:

		return data

	else:

		return 0

# Combining both models
def dataJob(x):

	if x >= 1:

		return 1

	else:

		return 0