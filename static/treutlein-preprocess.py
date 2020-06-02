#!/usr/bin/env python
# coding: utf-8

# # Treutlein lab ape organoid data
# The data are here: https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-7552/
# 
# Download files 1 to 7  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.1.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.2.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.3.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.4.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.5.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.6.zip  
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.7.zip  
# and unpack
# 
# Download supplementary informationfrom the Nature paper https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1654-9/MediaObjects/41586_2019_1654_MOESM3_ESM.zip, extract `Supplementary_Table_5.txt` and rename it into `metadata_macaque_cells_suppl.tsv`. The metadata file on arrayexpress seems to be wrong. I wrote to the authors to clarify.

# In[3]:


# Prepare

get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
import matplotlib
import pickle
import scipy
from scipy import sparse

sns.set_style('ticks')

import openTSNE

# get rnaseqTool.py from berenslab/rna-seq-tsne
import rnaseqTools


# In[37]:


# Uncomment what you want

metafile  = "metadata_chimp_cells_suppl.tsv"
countfile = "chimp_cell_counts_consensus.mtx"
line = None
outputfile = "chimp.pickle"

# metafile  = "metadata_human_cells.tsv"
# countfile = "human_cell_counts_consensus.mtx"
# line = '409b2'
# outputfile = "human-409b2.pickle"

# metafile  = "metadata_human_cells.tsv"
# countfile = "human_cell_counts_consensus.mtx"
# line = 'H9'
# outputfile = "human-h9.pickle"


# In[39]:


meta = pd.read_csv('../../../../Downloads/' + metafile, sep='\t')

print(meta['in_FullLineage'].sum())
print(meta['Stage'].unique())
print(np.unique(meta['Line'].values[meta['in_FullLineage'].values]))


# In[40]:


# genes = pd.read_csv('../../../../Downloads/genes_consensus.txt', sep='\t', names=['id','name'])
# genes = genes['name'].values.squeeze().astype(str)
# print(genes)


# In[7]:


from scipy.io import mmread

get_ipython().run_line_magic('time', "counts = mmread('../../../../Downloads/' + countfile)")
counts = scipy.sparse.csc_matrix(counts).T


# In[8]:


ind = meta['in_FullLineage'].values
if line is not None:
    ind = ind & (meta['Line'].values == line)

seqDepths = np.array(counts[ind,:].sum(axis=1))
stage = meta['Stage'].values[ind].astype('str')

impGenes  = rnaseqTools.geneSelection(counts[ind,:], n=1000, decay=1.5, plot=True)


# In[9]:


# Transformations

X = np.log2(counts[:, impGenes][ind,:] / seqDepths * np.median(seqDepths) + 1)  
X = np.array(X)
X = X - X.mean(axis=0)
get_ipython().run_line_magic('time', 'U,s,V = np.linalg.svd(X, full_matrices=False)')
X = np.dot(U, np.diag(s))
X = X[:, np.argsort(s)[::-1]][:,:50]

print(X.shape)


# In[14]:


pickle.dump([X, stage], open(outputfile, 'wb'))


# In[33]:


# %time Z = fast_tsne(X)   


# In[32]:


# plt.figure(figsize=(5,5))
# for stage in np.unique(meta['Stage'].values[ind]):
#     subset = meta['Stage'].values[ind] == stage
#     plt.scatter(Z[subset,0], Z[subset,1], s=1, label=stage)

# plt.legend()
# sns.despine()
# plt.tight_layout()

