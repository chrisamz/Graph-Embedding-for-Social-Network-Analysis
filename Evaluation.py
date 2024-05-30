# evaluation.py

"""
Evaluation Module for Graph Embedding for Social Network Analysis

This module contains functions for evaluating the performance of community detection, link prediction,
and graph embedding models using appropriate metrics.

Techniques Used:
- Model Evaluation
- Graph Metrics

Metrics Used:
- Modularity
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score

Libraries/Tools:
- scikit-learn
- NetworkX
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
import joblib
import community as community_louvain

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_graph(self, filepath):
        """
        Load graph data from an edge list file.
        
        :param filepath: str, path to the edge list file
        :return: NetworkX Graph, loaded graph
        """
        graph = nx.read_edgelist(filepath, nodetype=int, data=(('weight', float),))
        return graph

    def evaluate_community_detection(self, graph, partition):
        """
        Evaluate the community detection model using modularity, NMI, and ARI.
        
        :param graph: NetworkX Graph, input graph
        :param partition: dict, community assignments
        :return: dict, evaluation metrics
        """
        # Compute modularity
        modularity = community_louvain.modularity(partition, graph)
        
        # Convert partition to labels
        labels_true = list(nx.get_node_attributes(graph, 'community').values())
        labels_pred = list(partition.values())

        # Compute NMI and ARI
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        
        metrics = {
            'modularity': modularity,
            'nmi': nmi,
            'ari': ari
        }
        return metrics

    def evaluate_link_prediction(self, X_test, y_test, model_name):
        """
        Evaluate a link prediction model using ROC-AUC and accuracy.
        
        :param X_test: DataFrame, feature matrix for testing
        :param y_test: Series, true labels for testing
        :param model_name: str, name of the model to evaluate
        :return: dict, evaluation metrics
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics

    def evaluate_graph_embedding(self, graph, embedding_model):
        """
        Evaluate the graph embedding model using a downstream task, such as link prediction.
        
        :param graph: NetworkX Graph, input graph
        :param embedding_model: trained graph embedding model
        :return: dict, evaluation metrics for the downstream task
        """
        # Extract features using the embedding model
        node_pairs = list(nx.non_edges(graph))
        features = []
        labels = []

        for u, v in node_pairs:
            embedding_u = embedding_model.get_embedding(u)
            embedding_v = embedding_model.get_embedding(v)
            feature_vector = np.concatenate([embedding_u, embedding_v])
            features.append(feature_vector)
            labels.append(1 if graph.has_edge(u, v) else 0)

        X = pd.DataFrame(features)
        y = pd.Series(labels)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train and evaluate logistic regression model on the embedding features
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'
    model_name = 'logistic_regression'

    evaluator = ModelEvaluation()

    # Load preprocessed graph data
    graph = evaluator.load_graph(preprocessed_data_filepath)

    # Evaluate community detection
    partition = community_louvain.best_partition(graph)
    community_metrics = evaluator.evaluate_community_detection(graph, partition)
    print("Community Detection Evaluation:", community_metrics)

    # Load feature matrix and labels for link prediction
    X_test = pd.read_csv('data/processed/link_prediction_features_test.csv')
    y_test = pd.read_csv('data/processed/link_prediction_labels_test.csv')
    link_prediction_metrics = evaluator.evaluate_link_prediction(X_test, y_test, model_name)
    print("Link Prediction Evaluation:", link_prediction_metrics)

    # Evaluate graph embedding
    embedding_model = joblib.load('models/graph_embedding_model.pkl')
    embedding_metrics = evaluator.evaluate_graph_embedding(graph, embedding_model)
    print("Graph Embedding Evaluation:", embedding_metrics)
