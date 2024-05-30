# community_detection.py

"""
Community Detection Module for Graph Embedding for Social Network Analysis

This module contains functions for implementing community detection algorithms to identify
clusters or groups within the social network.

Techniques Used:
- Modularity optimization
- Spectral clustering

Algorithms Used:
- Louvain method
- Girvan-Newman algorithm

Libraries/Tools:
- NetworkX
- community (python-louvain)
"""

import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

class CommunityDetection:
    def __init__(self):
        """
        Initialize the CommunityDetection class.
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

    def detect_communities_louvain(self, graph):
        """
        Detect communities using the Louvain method.
        
        :param graph: NetworkX Graph, input graph
        :return: dict, community assignments
        """
        partition = community_louvain.best_partition(graph)
        return partition

    def detect_communities_girvan_newman(self, graph, n_communities=2):
        """
        Detect communities using the Girvan-Newman algorithm.
        
        :param graph: NetworkX Graph, input graph
        :param n_communities: int, number of communities to find
        :return: list, list of sets of nodes representing communities
        """
        comp = nx.community.girvan_newman(graph)
        limited = nx.community.asyn_lpa_communities(graph)
        communities = []
        for communities in limited:
            if len(communities) == n_communities:
                break
        return communities

    def plot_communities(self, graph, partition):
        """
        Plot the detected communities.
        
        :param graph: NetworkX Graph, input graph
        :param partition: dict, community assignments
        """
        pos = nx.spring_layout(graph)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(partition[node]) for node in graph.nodes()]
        nx.draw(graph, pos, node_color=colors, with_labels=True, node_size=50, cmap=cmap)
        plt.title('Community Detection using Louvain Method')
        plt.show()

    def plot_girvan_newman_communities(self, graph, communities):
        """
        Plot the communities detected by the Girvan-Newman algorithm.
        
        :param graph: NetworkX Graph, input graph
        :param communities: list, list of sets of nodes representing communities
        """
        pos = nx.spring_layout(graph)
        for community in communities:
            nx.draw_networkx_nodes(graph, pos, nodelist=list(community), node_size=50)
        nx.draw_networkx_edges(graph, pos)
        plt.title('Community Detection using Girvan-Newman Algorithm')
        plt.show()

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'

    cd = CommunityDetection()

    # Load preprocessed graph data
    graph = cd.load_graph(preprocessed_data_filepath)

    # Detect communities using Louvain method
    partition = cd.detect_communities_louvain(graph)
    cd.plot_communities(graph, partition)

    # Detect communities using Girvan-Newman algorithm
    communities = cd.detect_communities_girvan_newman(graph, n_communities=2)
    cd.plot_girvan_newman_communities(graph, communities)
    print("Community detection completed.")
