# data_preprocessing.py

"""
Data Preprocessing Module for Graph Embedding for Social Network Analysis

This module contains functions for collecting, cleaning, normalizing, and preparing
social network data for further analysis and modeling.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_edge_list(self, filepath):
        """
        Load graph data from an edge list file.
        
        :param filepath: str, path to the edge list file
        :return: NetworkX Graph, loaded graph
        """
        graph = nx.read_edgelist(filepath, nodetype=int)
        return graph

    def graph_to_dataframe(self, graph):
        """
        Convert the graph to a DataFrame.
        
        :param graph: NetworkX Graph, input graph
        :return: DataFrame, converted graph data
        """
        df = nx.to_pandas_edgelist(graph)
        return df

    def preprocess_graph(self, graph):
        """
        Preprocess the graph by normalizing node features and handling missing data.
        
        :param graph: NetworkX Graph, input graph
        :return: NetworkX Graph, preprocessed graph
        """
        # Example: Adding node degree as a feature
        degrees = dict(graph.degree())
        nx.set_node_attributes(graph, degrees, 'degree')

        # Convert graph to dataframe for preprocessing
        df = self.graph_to_dataframe(graph)
        
        # Handle missing values
        df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        
        # Normalize features
        node_features = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
        node_features = node_features.fillna(0)  # Fill NaN values with 0
        normalized_features = pd.DataFrame(self.scaler.fit_transform(node_features), columns=node_features.columns)

        # Update graph with normalized features
        for node in graph.nodes():
            for feature in normalized_features.columns:
                graph.nodes[node][feature] = normalized_features.loc[node, feature]

        return graph

    def save_preprocessed_data(self, graph, filepath):
        """
        Save the preprocessed graph data to an edge list file.
        
        :param graph: NetworkX Graph, preprocessed graph
        :param filepath: str, path to save the preprocessed edge list
        """
        nx.write_edgelist(graph, filepath, data=True)
        print(f"Preprocessed graph data saved to {filepath}")

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/social_network.edgelist'
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'

    preprocessing = DataPreprocessing()

    # Load raw graph data
    graph = preprocessing.load_edge_list(raw_data_filepath)

    # Preprocess the graph
    preprocessed_graph = preprocessing.preprocess_graph(graph)

    # Save preprocessed graph data
    preprocessing.save_preprocessed_data(preprocessed_graph, preprocessed_data_filepath)
    print("Data preprocessing completed.")
