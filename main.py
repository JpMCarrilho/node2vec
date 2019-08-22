import numpy as np
import networkx as nx
from random import shuffle
from gensim.models import Word2Vec
import node2vec


def main():
    #loads graph via edgelist file
    n_G = nx.read_edgelist(path ="karate.edgelist",create_using=nx.DiGraph(),nodetype = int, data =(('weight',float)))
    n_G = n_G.to_undirected()
    
    l = 40 #length of random walk
    n_walks = 10 #number of random walks
    context_size = 80
    embedding_dim  = 1
    p = 0.5
    q = 0.5
    walks = []
    nodes = list(n_G.nodes())
    
    n2v = node2vec.node2vec(n_G,p,q,l)
    for r in range(n_walks):
        print("Walk " + str(r))
        shuffle(nodes)
        for node in nodes:
            walk = n2v.node2vecWalk(n_G,node,l)
            walks.append(walk)

    walks = [map(str,walk) for walk in walks]
    model = Word2Vec(walks,embedding_dim,context_size)
    model.wv.save_word2vec_format("karate2.emd")

if __name__ == '__main__':
        main()