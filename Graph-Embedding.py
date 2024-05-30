# graph_embedding.py

"""
Graph Embedding Module for Graph Embedding for Social Network Analysis

This module contains functions for generating graph embeddings using Node2vec to capture
the topological structure of the graph.

Techniques Used:
- Node2vec
- Random walks
- Skip-gram model

Libraries/Tools:
- Node2vec
- Gensim
"""

import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import joblib

class GraphEmbedding:
    def __init__(self, dimensions=128, walk_length=80, num_walks=10, workers=4):
        """
        Initialize the GraphEmbedding class.
        
        :param dimensions: int, dimensions of the embedding
        :param walk_length: int, length of each random walk
        :param num_walks: int, number of walks per node
        :param workers: int, number of workers for parallel processing
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None

    def load_graph(self, filepath):
        """
        Load graph data from an edge list file.
        
        :param filepath: str, path to the edge list file
        :return: NetworkX Graph, loaded graph
        """
        graph = nx.read_edgelist(filepath, nodetype=int, data=(('weight', float),))
        return graph

    def generate_embeddings(self, graph):
        """
        Generate graph embeddings using Node2vec.
        
        :param graph: NetworkX Graph, input graph
        :return: Node2Vec model, trained Node2Vec model
        """
        node2vec = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.model = model
        return model

    def save_model(self, filepath):
        """
        Save the Node2Vec model to a file.
        
        :param filepath: str, path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained Node2Vec model from a file.
        
        :param filepath: str, path to the saved model
        """
        self.model = Word2Vec.load(filepath)
        print(f"Model loaded from {filepath}")

    def get_embedding(self, node):
        """
        Get the embedding vector for a specific node.
        
        :param node: int, node identifier
        :return: numpy array, embedding vector
        """
        return self.model.wv[node]

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'
    embedding_model_filepath = 'models/graph_embedding_model.model'

    embedding = GraphEmbedding()

    # Load preprocessed graph data
    graph = embedding.load_graph(preprocessed_data_filepath)

    # Generate embeddings
    model = embedding.generate_embeddings(graph)

    # Save the Node2Vec model
    embedding.save_model(embedding_model_filepath)
    print("Graph embedding completed and model saved.")
