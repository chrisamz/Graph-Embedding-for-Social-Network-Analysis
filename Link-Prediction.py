# link_prediction.py

"""
Link Prediction Module for Graph Embedding for Social Network Analysis

This module contains functions for developing link prediction models to estimate
the likelihood of future or missing connections in the graph.

Techniques Used:
- Similarity measures
- Machine learning models

Algorithms Used:
- Logistic Regression
- Random Forest
- Graph Neural Networks

Libraries/Tools:
- scikit-learn
- NetworkX

"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

class LinkPrediction:
    def __init__(self):
        """
        Initialize the LinkPrediction class.
        """
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }

    def load_graph(self, filepath):
        """
        Load graph data from an edge list file.
        
        :param filepath: str, path to the edge list file
        :return: NetworkX Graph, loaded graph
        """
        graph = nx.read_edgelist(filepath, nodetype=int, data=(('weight', float),))
        return graph

    def extract_features(self, graph):
        """
        Extract features for link prediction.
        
        :param graph: NetworkX Graph, input graph
        :return: DataFrame, feature matrix
        """
        # Example features: Common neighbors, Jaccard coefficient, Adamic/Adar index
        node_pairs = list(nx.non_edges(graph))
        features = []
        labels = []

        for u, v in node_pairs:
            common_neighbors = len(list(nx.common_neighbors(graph, u, v)))
            jaccard_coeff = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
            adamic_adar = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
            features.append([common_neighbors, jaccard_coeff, adamic_adar])
            labels.append(1 if graph.has_edge(u, v) else 0)

        feature_matrix = pd.DataFrame(features, columns=['common_neighbors', 'jaccard_coeff', 'adamic_adar'])
        return feature_matrix, labels

    def train_model(self, X, y, model_name):
        """
        Train a link prediction model.
        
        :param X: DataFrame, feature matrix
        :param y: Series, labels
        :param model_name: str, name of the model to train
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} is not defined.")
        model.fit(X, y)
        joblib.dump(model, f'models/{model_name}_model.pkl')
        print(f"{model_name} model trained and saved.")

    def evaluate_model(self, X, y, model_name):
        """
        Evaluate a link prediction model.
        
        :param X: DataFrame, feature matrix
        :param y: Series, labels
        :param model_name: str, name of the model to evaluate
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        y_pred = model.predict(X)
        roc_auc = roc_auc_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        print(f"{model_name} model evaluation:")
        print(f"ROC-AUC Score: {roc_auc}")
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'

    lp = LinkPrediction()

    # Load preprocessed graph data
    graph = lp.load_graph(preprocessed_data_filepath)

    # Extract features and labels
    X, y = lp.extract_features(graph)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate logistic regression model
    lp.train_model(X_train, y_train, 'logistic_regression')
    lp.evaluate_model(X_test, y_test, 'logistic_regression')

    # Train and evaluate random forest model
    lp.train_model(X_train, y_train, 'random_forest')
    lp.evaluate_model(X_test, y_test, 'random_forest')

    print("Link prediction completed.")
