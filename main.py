import numpy as np
import networkx as nx
import tensorflow as tf
import utils
import random
from gensim.models import Word2Vec
def main():
    ##loads graph as a directed graph (DiGraph)
    G = nx.read_edgelist(path = "./karate.edgelist", create_using = nx.DiGraph(),nodetype = int)
    r  = 10 ## number of walks per nodes
    l = 80 ## length of r-walk
    k = 10 ## context size
    p = 1 ## Return parameter of walk
    q = 1 ## In-Out parameter of walk
    n_G = utils.start_weights(G)
    n_G.to_undirected()
    #print(n_G.nodes())
    walks = []
    nodes = list(n_G.nodes())
    alias_nodes,alias_edges = utils.PreProcessModifedWeights(n_G,p,q)  ##updated Graphs weights
    for n_walk in range(r):
        print("walk number:", str(n_walk+1))
        random.shuffle(nodes)
        for node in nodes:
            walk = utils.node2VecWalk(n_G,node,l,alias_nodes,alias_edges)
            
            walks.append(walk)
    walks = [map(str,walk) for walk in walks]
    model = Word2Vec(walks,size = 128, window = 10, min_count = 0, sg = 1,workers = 8, iter = 1 )
    model.wv.save_word2vec_format("./karate.emd")
    

if __name__ == '__main__':
    main()
