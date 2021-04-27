import networkx as nx
import numpy as np

class Tree:
    def __init__(self, name = False):
        self.tree = nx.Graph()
        if name:
            self.tree = nx.Graph(name=name)
        self.nodes = np.array([])

    def add_node(self, point):
        if self.nodes.size == 0:
            self.nodes = np.array([point])
            self.tree.add_node(0)
            return 0
        idx = self.tree.number_of_nodes()
        self.tree.add_node(idx)
        self.nodes = np.append( self.nodes , [point], axis=0)
        return idx

    def nearest_node(self, pos):
        distances = np.linalg.norm(self.nodes - pos, axis=1)
        nearest_point_idx = np.argmin(distances)
        return nearest_point_idx, self.nodes[nearest_point_idx]

    def add_edge(self, node_from_idx, node_to_idx, weight):
        self.tree.add_edge(node_from_idx, node_to_idx, weight=weight, tree=self.tree.name)

    def print_tree(self):
        text = "Info: " + str(nx.info(self.tree))
        text += "\nNodes: " + str(self.nodes)
        return text
