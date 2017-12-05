
# coding: utf-8

# In[2]:


import numpy as np
import itertools


# In[4]:


"""
Algorithms from the paper:

E - set of elements (in current code is nodes)
C - set of constrains (in current code is W, or we refer to it as edges)
W - sum of weights of satisfied constrains (it implies that C are weighted and in the current code we don't have that)
Maximum coherence - it doesn't exist a partition of accepted and rejected E that has larger W

Algorithm 1: Exaustive
1. Generate all possible ways of dividing elements into accepted and rejected.
2. Evaluate each of these for the extent to which it achieves coherence. (in other words compute W for each partition)
3. Pick the one with highest value of W.

Algorithm 3: Connectionists
1. Initialize units (U for each element of E) to small positive value
2. Initialize weighted edges in accordance with the constrain rules
3. while units are not settled:
    update units according to the harmony equation
4. theshold the units with 0 - accepted are positive, rejected are negative

"""


# In[33]:


def initW(n=5, seed=4040):
    # Generate matrix with values -1,0,1
    np.random.seed(seed)
    W = np.random.randint(-1,2,n**2).reshape(n,n)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i == j:
                W[i][j] = 0   # make diagonal zeros
            else:
                W[i][j] = W[j][i] # make symmetric
    return W

def subsets(nodes):
    subsets_raw = []
    # give all possible subsets of accepted nodes
    for i in range(nodes.shape[0]+1):
        subsets_raw.append(list(itertools.combinations(nodes, i)))
    subs = []    
    for l in subsets_raw:
        for e in l:
            subs.append(e)
    return subs


# In[34]:


n = 5

W = initW(n)          # initialize edges
nodes = np.arange(n)  # init nodes

s = subsets(nodes)
print(s)


# In[35]:





# In[66]:


# Cleaner way to make subsets of 1s and -1s
from itertools import combinations_with_replacement

subsets = []
for comb in combinations_with_replacement([-1,1],4):
    subsets.append(comb)
print(subsets)


# In[49]:


# make nodes an array of -1 for rejected, +1 for accepted
nodes = np.random.randint(2,size=5) * 2 - 1


# compute coherence
coh = 0
# How to get the lower diagonal of W
for i in range(W.shape[0]):
    for j in range(i):
        print("w = {}, vi = {}, vj = {}".format(W[i][j], i, j))
        coh += W[i][j] * nodes[i] * nodes[j]
print(coh)

