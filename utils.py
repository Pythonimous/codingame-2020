import copy
import numpy as np

class Graph(object):

    def __init__(self, graph_dict=None, value_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        if value_dict is None:
            value_dict = {}
        self.__graph_dict = graph_dict
        self.__value_dict = value_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def get_neighbors(self, vertex):
        """ returns neighbors of a vertex """
        return self.__graph_dict[vertex]

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict.setdefault(vertex, [])
            self.__value_dict[vertex] = 1

    def update_value(self, vertex, value):
        self.__value_dict[vertex] = value

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nneighbors: "
        for n in self.__graph_dict:
            res += str(n)+': ' + ' '.join(self.get_neighbors(n)) + "\n"
        return res

    def find_paths(self, start):
        distances = dict() # primary distance dictionary
        distances[start] = 0

        for vertex in self.vertices():
            if vertex == start:
                distances[vertex] = 0
            else:
                distances[vertex] = 10**10
        unvisited_distances = copy.deepcopy(distances) # secondary "unvisited" distance dict

        while unvisited_distances:
            vertex = min(unvisited_distances,
                         key=unvisited_distances.get)  # Min weight vertex we haven't visited yet
            del unvisited_distances[vertex]

            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                new_distance = distances[vertex] + 1
                if new_distance < distances[neighbor]:    # Is new path shorter?
                    distances[neighbor] = new_distance
                    if neighbor in unvisited_distances:
                        unvisited_distances[neighbor] = new_distance
        return distances


width = 10
l = np.array_split(np.array(range(width)), 2)
print([set(list(sub)) for sub in l])