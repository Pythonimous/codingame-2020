from utils import *

if __name__ == "__main__":

    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'C', 'D'],
        'C': ['A', 'B', 'D', 'F'],
        'D': ['B', 'C', 'E', 'F'],
        'E': ['D', 'F'],
        'F': ['C', 'D', 'E']
    }

    weights = {
        'AB': 1, 'AC': 4, 'BC': 3,
        'BD': 2, 'CD': 5, 'CF': 7,
        'DE': 6, 'DF': 3, 'EF': 1
    }

    g = Graph(graph, weights)

    print(g)


   # print(shortest_path(g, 'A', 'D'))
