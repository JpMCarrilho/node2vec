import numpy as np
import networkx as nx
import tensorflow as tf
import utils
##loads graph as a directed graph
G = nx.read_edgelist(path = "./karate.edgelist", create_using = nx.DiGraph(), data = 'weights')
ebd_dim = 128 ##embeddings dimension
r = 10 ##number of walks per nodes
l = 40 ## length of r-walk
k = 20 ##context size
p = 0.5 ##Return parameter of walk
q = 0.5 ##In-Out parameter of walk
walks = []
n_G = utils.start_weights(G)
n_G.to_undirected()
   


def __main__():
    n_G = PreProcessModifedWeights(G,p,q)  ##updated Graphs weights
    for n_walk in range(r):
        for node in G.nodes():
            walk = node2VecWalk(n_G,node,l)
            walks.append(walk)
    embeddings = None ##word2vec
    return embeddings

#if __name__ == '__main__':
#    main()
