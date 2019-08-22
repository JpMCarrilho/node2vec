import networkx as nx
import numpy as np
from utils2 import alias_draw, alias_setup
class node2vec():
    def __init__(self,G,p,q,l):
        self.G = G
        self.p = p
        self.q = q
        self.l = l

    def start_weights(self,G):
        """
        Initializes the weights of the graph's edges to 1
        
        Arguments:
            G {graph} -- graph being analyzed by the node2vec algorithm
        """ 
        
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        
        self.G = G
        return G

    
    def PreProcessModifiedWeights(self,G,p,q):
        """
        Prepares the lookuptable probabilites for the Biased Random Walk sampling method (Alias Method)
        
        Arguments:
            G {graph} -- graph being analyzed by the algorithm
            p {float} -- "Return" hyperparameter 
            q {float} -- "In-Out" hyperparameter
        """ 
        self.G = G
        self.p = p
        self.q = q
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            Z = sum(unnormalized_probs)
            normalized_probs = [float(prob)/Z for prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
            #print(alias_nodes[node])

        alias_edges = {}
        
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(G,edge[0],edge[1],p,q)
            alias_edges[edge[1],edge[0]] = self.get_alias_edge(G,edge[1],edge[0],p,q)
        
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        #print(alias_edges)
        #print(alias_nodes)

    def get_alias_edge(self,G,src,dst,p,q):
        """
        Prepare probabilities of edges for Second Order random walk
        
        Arguments:
            src {int} -- source node of the first step
            dst {int} -- destination node of the first step
        """
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for nbr in sorted(G.neighbors(dst)):
            if nbr == src:
                G[dst][nbr]['weight'] = 1/p
            elif G.has_edge(src,nbr):
                G[dst][nbr]['weight'] = 1
            else:
                G[dst][nbr]['weight'] = 1/q
            unnormalized_probs.append(G[dst][nbr]['weight'])
        
        Z = sum(unnormalized_probs)
        normalized_probs = [float(prob)/Z for prob in unnormalized_probs]
        
        return alias_setup(normalized_probs)

    def node2vecWalk(self,G,u,l):
        G = self.G
        l =  self.l
        p = self.p
        q = self.q


        G = self.start_weights(G)
        
        self.PreProcessModifiedWeights(G,p,q)
        walk = [u]
        for step in range(l):
            curr = walk[-1]
            
            neighbors = sorted(G.neighbors(curr)) 
            
            if len(neighbors) > 0:
                if len(walk) == 1:
                    s = neighbors[alias_draw(self.alias_nodes[curr][0],self.alias_nodes[curr][1])]
                else:
                    src = walk[-2]
                
                    s = neighbors[alias_draw(self.alias_edges[src,curr][0],self.alias_edges[src,curr][1])]
            else:
                break
            walk.append(s)
        return walk
            





