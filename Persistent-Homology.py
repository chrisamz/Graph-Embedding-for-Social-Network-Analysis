# persistent_homology.py

"""
Persistent Homology Module for Graph Embedding for Social Network Analysis

This module contains functions for applying persistent homology to study the topological features
of the graph that are invariant under continuous transformations.

Techniques Used:
- Filtration
- Betti numbers
- Persistence diagrams

Libraries/Tools:
- GUDHI
- matplotlib
"""

import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

class PersistentHomology:
    def __init__(self):
        """
        Initialize the PersistentHomology class.
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

    def create_simplex_tree(self, graph):
        """
        Create a simplex tree from the graph.
        
        :param graph: NetworkX Graph, input graph
        :return: SimplexTree, GUDHI simplex tree
        """
        simplex_tree = gd.SimplexTree()
        for edge in graph.edges(data=True):
            weight = edge[2].get('weight', 1.0)
            simplex_tree.insert([edge[0], edge[1]], filtration=weight)
        simplex_tree.initialize_filtration()
        return simplex_tree

    def compute_persistence(self, simplex_tree):
        """
        Compute the persistence of the simplex tree.
        
        :param simplex_tree: SimplexTree, GUDHI simplex tree
        :return: list, persistence intervals
        """
        persistence = simplex_tree.persistence()
        return persistence

    def plot_persistence_diagram(self, persistence):
        """
        Plot the persistence diagram.
        
        :param persistence: list, persistence intervals
        """
        gd.plot_persistence_diagram(persistence)
        plt.title('Persistence Diagram')
        plt.show()

    def plot_betti_numbers(self, simplex_tree, max_dimension=2):
        """
        Plot the Betti numbers as a function of filtration value.
        
        :param simplex_tree: SimplexTree, GUDHI simplex tree
        :param max_dimension: int, maximum homology dimension to consider
        """
        filtration_values = np.linspace(0, 1, 100)
        betti_numbers = {dim: [] for dim in range(max_dimension + 1)}
        
        for filtration_value in filtration_values:
            simplex_tree.compute_persistence()
            betti_numbers_at_filtration = simplex_tree.betti_numbers()
            for dim in range(max_dimension + 1):
                betti_numbers[dim].append(betti_numbers_at_filtration.get(dim, 0))
        
        for dim, values in betti_numbers.items():
            plt.plot(filtration_values, values, label=f'Betti {dim}')
        
        plt.xlabel('Filtration Value')
        plt.ylabel('Betti Number')
        plt.title('Betti Numbers as a Function of Filtration Value')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_social_network.edgelist'

    ph = PersistentHomology()

    # Load preprocessed graph data
    graph = ph.load_graph(preprocessed_data_filepath)

    # Create simplex tree
    simplex_tree = ph.create_simplex_tree(graph)

    # Compute persistence
    persistence = ph.compute_persistence(simplex_tree)

    # Plot persistence diagram
    ph.plot_persistence_diagram(persistence)

    # Plot Betti numbers
    ph.plot_betti_numbers(simplex_tree)
    print("Persistent homology analysis completed.")
