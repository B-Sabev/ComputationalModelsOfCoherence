
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import matplotlib.pyplot as plt


# In[2]:


class CoherenceGraph(object):
    """
    The CoherenceGraph consists of nodes that are either true or false and have negative and positive constrains in between.
    Accepts a list of nodes (s) that are accepted to be true from the start
    Can compute its coherence (W), which is a weighted sum of satisfied constrains
    """
    def __init__(self,n,s=[], seed=89):
        self.n = n               # total number of nodes
        self.v = np.zeros((n))   # truth assignment of all nodes 1 for True, -1 for False
        self.s = [e for e in s if e < self.n] # list of indecies of nodes that are set to be true
        self.c = self.initConnections(seed)   # connections between nodes
        self.W = None
            
    def initConnections(self, seed):
        """
        Randomly select from positive or negative constrains, weighted between -1,1
        All excitatory connections are set to 0.4 (except for special elements s to 0.5 )
        All inhibitory connections to -0.6
        """
        np.random.seed(seed)
        c = np.random.rand(self.n, self.n) * 2 - 1  # init with random between -1, 1
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = c[j][i] # make symmetric
        c = np.where(c>0, 0.4, c)  # set all excitatory connections to 0.4
        c = np.where(c<0,-0.6, c)   # set all inhibitory connections to -0.6
        c[self.s,:] = 0.5       # Set all d to s connections to 0.5 
        c[:,self.s] = 0.5
        diag = np.arange(self.n)
        c[diag,diag] = 0                   # Remove connections between d and d
        return c
        
    def computeW(self):
        """
        Compute the coherence(W/W*) for one assignment of nodes
        """
        E = np.where(self.v > 0, 1, -1)
        W = np.sum(self.c * np.dot(E.reshape(-1,1), E.reshape(1,-1)))   # W = C * E * E
        self.W = W / np.sum(self.c)
        return self.W  # W / W* 
        
    def setV(self, v):
        self.v = v
        
    def setS(self, s):
        self.s = [e for e in s if e < self.n] # set only valid elements for s
        


# In[3]:


class ExhaustiveSearch(object):
    """
    Exhaustive search - algorithm to compute coherence
    1. Generate all possible ways of dividing elements into accepted and rejected.
    2. Evaluate each of these for the extent to which it achieves coherence. (in other words compute W for each partition)
    3. Pick the one with highest value of W.
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.subsets = self.finalSubsets()
        self.Ws = None
        self.Wmax = None
        self.Emax = None
        
    def allSubsets(self):
        """
        Generate all possible subsets of accepted(+1) and rejected(-1) elements
        """
        n = self.graph.n
        subsets = np.zeros((2**n,n))
        for i in range(2**n):
            binary = np.array(list(bin(i)[2:])).astype(float)
            if binary.shape[0] < n:
                padding  = np.zeros(n-binary.shape[0])
                subsets[i,:] = np.append(padding, binary)
            else:
                subsets[i,:] = binary
        return np.where(subsets > 0, 1, -1)
        
    def finalSubsets(self):
        """
        Remove subsets that are inconsistent with what we know to be true (s)
        """
        subs = self.allSubsets()
        for s in self.graph.s:
            subs = subs[subs[:,s] == 1,] # remove subsets where values in s are not True
        return subs
    
    def search(self):
        """
        For all possible subsets, compute the cohence and store it in array
        """
        W = np.zeros((self.subsets.shape[0],))  
        for i,E in enumerate(self.subsets):
            self.graph.setV(E)  # set the nodes to their values
            W[i] = self.graph.computeW()
        self.Ws = W
        
    def getOptimalSolution(self):
        """
        Get W and truth assignmnets of nodes for max W
        """
        max_index = np.argmax(self.Ws)
        self.Wmax = self.Ws[max_index]
        self.Emax = self.subsets[max_index]
        return (self.Wmax, self.Emax)
            


# In[4]:


class ConnectionistModel(object):
    """
    Algorithm 3: Connectionists
    1. Initialize units (U for each element of E) to small positive value
    2. Initialize weighted edges in accordance with the constrain rules
    3. while units are not settled:
        update units according to the harmony equation
    4. theshold the units with 0 - accepted are positive, rejected are negative
    """
    
    def __init__(self, graph, initState=0.05, numCycles=200, min_max=(-1,1), decay=0.05):
        self.graph = graph
        self.initState = initState   # the start value of the undecided units
        self.numCycles = numCycles   # total number of cycles 
        self.min_max = min_max       # min and max value of a unit
        self.decay = decay           # how much the previous decays during a cycle
        self.units = None            # initial units in updateGraph taking into account the initState and s,       
        
    def initUnits(self):
        v_init = np.repeat(self.initState, self.graph.n) # make array lenght n filled with unit_value
        v_init[self.graph.s] = 1                     # set the special elements of s to true
        self.units = v_init
        
    def updateGraph(self):
        """
        Implement the connectionist network update into a stable state
        """
        self.initUnits()
        v = self.units.copy()
        for _ in range(self.numCycles): # for total number of cycles
            for i in range(self.graph.n): # for every unit in the graph
                if i not in self.graph.s: # if the unit is not a special fixed value s
                    net = np.dot(v, self.graph.c[i]) # compute total flow to the unit
                    if net > 0:
                        gradient = net*(1-v[i])
                    else:
                        gradient = net*(v[i]-(-1))
                    v[i] = v[i]*(1-self.decay) + gradient
            # should this be after every unit update, or after the whole graph updates ??
            v = np.where(v>1, 1, v)
            v = np.where(v<-1,-1,v)
        self.units = v
        
    def getSolution(self):
        # theshold and compute final W
        v = np.where(self.units > 0, 1, -1)
        self.graph.setV(v)
        self.graph.computeW()
        return (self.graph.W, self.graph.v)
        


# In[5]:


"""
Test and compare the 2 algorithms

TODO - decide on measures for comparison
"""

for n in range(5, 15):  # for different number of nodes        
    for seed in range(45,700, 100): # for different seeds
        
        # Create a graph with n number of nodes, where the s nodes are always true, with random seed for constrains seed
        G = CoherenceGraph(n=n, s=[1,4], seed=seed)

        # Exhaustive search
        ExCoh = ExhaustiveSearch(G) # create exhausive seach with the graph
        ExCoh.search()              # compute W for all possible subsets
        Wmax, Emax = ExCoh.getOptimalSolution() # get the max W and the truth assigments that achieves it
        print("{} {}".format(Wmax, Emax))

        # Connectionist model
        ConCoh = ConnectionistModel(G,               
                                    initState=0.1,   
                                    numCycles=200,   
                                    min_max=(-1,1),  
                                    decay=0.05) 
        ConCoh.updateGraph()
        Wcon, Econ = ConCoh.getSolution() # get the final state of the 
        print("{} {}".format(Wcon, Econ))
        print("")

