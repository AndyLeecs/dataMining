import sys
import networkx as nx
import matplotlib.pyplot as plt

g = nx.DiGraph()

nodes = []
for i in range(666):
    nodes.append(i+1)

g.add_nodes_from(nodes)
node_pref = []
node_sizes = [];
with open('D:/dataMining/newlearner/.idea/scc.txt') as f1:
    for line in f1:
        size = float(line)
        real = size/10
        node_sizes.append(real)

edges = []
with open('D:/dataMining/newlearner/.idea/edge.txt')as f1:
    for line in f1:
        head,tail = [int(x) for x in line.split()]

        edge = (head,tail)
        edges.append(edge)

newedges = list(set(edges))
g.add_edges_from(newedges)

nx.draw(g,node_size = node_sizes,pos = nx.random_layout(g),width = 0.0000005)
plt.savefig("SCC.pdf")