#import umap.umap_ as umap # Alternative...see below
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(15,12)})
sns.set_style('white')


# Import the trained document embeddings from 2011 (could do other years)
model = Doc2Vec.load('w2v2011')

# Get the ONET embedding tags
tags = model.docvecs.index2entity

# Compute the document vectors for the ONET tags
landmarks = [model.docvecs[tag] for tag in tags]

# Principal Components Analysis (recommended for t-SNE)
pca = PCA(n_components=25)
pca_vectors = pca.fit_transform(landmarks)

# Run the t-SNE algorithm for dimensionality reduction
num_jobs = cpu_count()
X_embedded = TSNE(n_components=2,perplexity=15,n_jobs=num_jobs,n_iter=15000).fit_transform(pca_vectors)

# Alternatively, could use UMAP
#reducer = umap.UMAP(n_neighbors=20,min_dist=0.1)
#X_embedded = reducer.fit_transform(landmarks) 

# X-Y coordinates from TSNE (end up switching these, doesn't matter)
x_tsne = list(X_embedded[:,0])
y_tsne = list(X_embedded[:,1])


# Pandas DataFrame from t-SNE output
tsneDF = pd.DataFrame({'onet_codes':tags, 'x_tsne':x_tsne, 'y_tsne':y_tsne})

# Create a function to label the variables based on broad occupations

def major_group(x):

    if x[0:2] == '11':

        return 'Management'

    elif x[0:2] == '13':

        return 'Business and Financial Operations'

    if x[0:2] == '15':

        return 'Computer and Mathematical'

    elif x[0:2] == '17':

        return 'Architecture and Engineering'

    elif x[0:2] == '19':

        return 'Life, Physical, and Social Sciences'

    elif x[0:2] == '21':

        return 'Community and Social Services'

    elif x[0:2] == '23':

        return 'Legal'

    elif x[0:2] == '25':

        return 'Education, Training, Library'

    elif x[0:2] == '27':

        return 'Arts, Design, Entertainment, Sports, Media'

    elif x[0:2] == '29':

        return 'Healthcare Practictioners and Technical'

    elif x[0:2] == '31':

        return 'Healthcare Support'

    elif x[0:2] == '33':

        return 'Protective Service'

    elif x[0:2] == '35':

        return 'Food Preparation and Serving'

    elif x[0:2] == '37':

        return 'Building and Grounds Cleaning and Maintenance'

    elif x[0:2] == '39':

        return 'Personal Care and Service'

    elif x[0:2] == '41':

        return 'Sales and Related'

    elif x[0:2] == '43':

        return 'Office and Administrative Support'

    elif x[0:2] == '45':

        return 'Farming, Fishing, Forestry'

    elif x[0:2] == '47':

        return 'Construction and Extraction'

    elif x[0:2] == '49':

        return 'Installation, Maintenance, and Repair'

    elif x[0:2] == '51':

        return 'Production'

    elif x[0:2] == '53':

        return 'Transportation and Material Moving'

    elif x[0:2] == '55':

        return 'Military Specific'   

    else:

        return 'Other'

# Label selected occupations 
tsneDF['major_group'] = tsneDF['onet_codes'].apply(lambda x: major_group(x))

'''
------------------------------------
The ONET Landmarks figure
------------------------------------
'''
fig,ax=plt.subplots(figsize=(15,12))

# Extending the color palette
color_list = sns.color_palette('bright')
color_list.extend(sns.color_palette('colorblind'))
color_list.extend(sns.color_palette('pastel'))
color_list.extend(sns.color_palette('muted'))


 
# The base plots
sns.scatterplot(x='y_tsne',y='x_tsne',data=tsneDF[tsneDF['major_group']=='Other'],color='gray',alpha=0.5,ax=ax)

sns.scatterplot(x='y_tsne',y='x_tsne',data=tsneDF[tsneDF['major_group']!='Other'],

                hue='major_group',alpha=1,ax=ax,edgecolor='black',s=60,palette=color_list[0:23])

# Axis and legend properties
ax.axis('off')
ax.get_legend().set_title(None)
ax.legend(prop={'size':16},frameon=False,markerscale=1.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':20},frameon=False,markerscale=2)

'''
--------------------------------------------------------
The Health and Community Services Cluster
--------------------------------------------------------
'''
axins = ax.inset_axes([-0.4,0, 0.5, 0.5])
sns.scatterplot(x='y_tsne',y='x_tsne',data=tsneDF[tsneDF['major_group']!='Other'],
                hue='major_group',alpha=1,ax=axins,edgecolor='black',s=150,palette=color_list[0:23])
# sub region of the original image
x1, x2, y1, y2 = -82, -50.4, -28, 7
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.set_ylabel('')
axins.set_xlabel('')
axins.get_legend().remove()



# Okay, now we are going to label select observations
def label_near_health(DF):
    if (DF['y_tsne'] >= -82 and DF['y_tsne'] <= -50.4) and (DF['x_tsne'] >=-28 and DF['x_tsne'] <=7):
        if DF['onet_codes'][0:2] == '19':
            if DF['onet_codes'] == '19-3039.00':
                return 'Psychologists, All Other'
            elif DF['onet_codes'] == '19-3039.01':
                return 'Neuropsychologists'
            elif DF['onet_codes'] == '19-3031.02':
                return 'Clinical Psychologists'
            elif DF['onet_codes'] == '19-3031.03':
                return 'Counseling Psychologists'
        elif DF['onet_codes'][0:2] == '15':
            if DF['onet_codes'] == '15-1121.01':
                return 'Health Informatics Specialist'
        else:
            return ''
    else:
        return ''
    
tsneDF['inset_health'] = tsneDF.apply(label_near_health,axis=1)




def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) == 'Psychologists, All Other':
            ax.text(point['x']-10.5, point['y'], str(point['val']))
        elif str(point['val']) == 'Neuropsychologists':
            ax.text(point['x']-8.75, point['y'], str(point['val']))
        elif str(point['val']) == 'Clinical Psychologists':
            ax.text(point['x']-10, point['y']-0.5, str(point['val']))
        elif str(point['val']) == 'Counseling Psychologists':
            ax.text(point['x']-11, point['y']-1.5, str(point['val']))
            
        elif str(point['val']) == 'Health Informatics Specialist':
            ax.text(point['x']+1, point['y'], 'Health Informatics Spec.')
        else:
            ax.text(point['x']-8.75, point['y'], str(point['val']))


label_point(tsneDF.y_tsne, tsneDF.x_tsne, tsneDF.inset_health, axins)         

ax.indicate_inset_zoom(axins)


'''
--------------------------------------------------------
Computer and Mathematical Occupations Cluster
--------------------------------------------------------
'''
axins = ax.inset_axes([-0.25,0.7, 0.5, 0.5])
sns.scatterplot(x='y_tsne',y='x_tsne',data=tsneDF[tsneDF['major_group']!='Other'],
                hue='major_group',alpha=1,ax=axins,edgecolor='black',s=150,palette=color_list[0:23])
# sub region of the original image
x1, x2, y1, y2 = -2.8, 6.6, 9, 22.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.set_ylabel('')
axins.set_xlabel('')
axins.get_legend().remove()



# Okay, now we are going to label select observations
def label_near_comps(DF):
    if (DF['y_tsne'] >= -2.8 and DF['y_tsne'] <= 6.6) and (DF['x_tsne'] >=9 and DF['x_tsne'] <=22.5):
        if DF['onet_codes'] == '13-1111.00':
            return 'Management Analysts'
        elif DF['onet_codes'] == '41-9031.00':
            return 'Sales Engineers'
        elif DF['onet_codes'] == '13-2099.03':
            return 'Financial Analyst'
        elif DF['onet_codes'] == '15-1199.08':
            return 'Business Intelligence Analysts'

        #elif DF['onet_codes'] == '15-1151.00':
        #    return 'Computer User Support Specialists'
        
        #elif DF['onet_codes'] == '15-1152.00':
        #    return 'Computer Network Support Specialists'

        elif DF['onet_codes'] == '15-1121.00':
            return 'Computer Systems Analysts'
        
        elif DF['onet_codes'] == '15-1199.09':
            
            return 'IT Project Managers'
        
        elif DF['onet_codes'] == '15-1199.02':
            
            return 'Computer Systems Engineers'
        
    else:
        return ''
    
tsneDF['inset_comps'] = tsneDF.apply(label_near_comps,axis=1)




def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) == 'Financial Analyst':
            ax.text(point['x']+0.2, point['y'], str(point['val']))
            
        elif str(point['val']) == 'Management Analysts':
            ax.text(point['x']-1.5, point['y']-0.65, str(point['val']))
            
        elif str(point['val']) == 'Business Intelligence Analysts':
            ax.text(point['x']-2.5, point['y']-0.65, str(point['val']))
            
        elif str(point['val']) == 'IT Project Managers':
            ax.text(point['x']-2, point['y']+0.65, str(point['val']))
            
            
        elif str(point['val']) == 'Computer Systems Engineers':
            ax.text(point['x']-1, point['y']+0.65, str(point['val']))
            
        elif str(point['val']) == 'Sales Engineers':
            ax.text(point['x']+0.2, point['y'], str(point['val']))

        


label_point(tsneDF.y_tsne, tsneDF.x_tsne, tsneDF.inset_comps, axins)         
ax.indicate_inset_zoom(axins)

fig.savefig('doc2vec_landmarks_inset.pdf',bbox_inches='tight')
fig.savefig('doc2vec_landmarks_inset.png',bbox_inches='tight')