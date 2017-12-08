import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G=nx.Graph()
n = 10 
nodes = np.append(np.arange(5), np.arange(-5))
W = np.random.rand(n,n)
W = np.where(W > 0.5, 1, 0)

print(nodes)

G.add_nodes_from([str(node) for node in nodes])

for i in range(n):
	for j in range(n):
		if W[i][j] > 0:
			G.add_edge(i,j)
			
			
#val_map = {[val:0.5 if int(val) > 0 else 1.0 for val in nodes]}
			
#values = [val_map.get(node, 0.25) for node in G.nodes()]
#nx.draw(G, cmap=plt.get_cmap('jet'), node_color=values)

#plt.savefig("simple_path.png") # save as png
#plt.show() # display