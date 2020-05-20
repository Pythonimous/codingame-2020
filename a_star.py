import copy
import numpy as np

class Grid(object):

    def __init__(self, matrix_grid):
        self.graph_dict, self.value_dict, \
        self.type_dict = self.load_matrix(matrix_grid)  # P - pellet, F - friend, E - enemy

    @staticmethod
    def load_matrix(matrix_grid):
        gd = {}
        vd = {}
        td = {}

        for y in range(len(matrix_grid)):
            for x in range(len(matrix_grid[0])):
                if matrix_grid[y][x] == ' ':
                    tile_neighbors = list()
                    tile_neighbors.append((x, y - 1))
                    tile_neighbors.append((x, y + 1))
                    left_neighbor = (x - 1, y)
                    right_neighbor = (x + 1, y)
                    if x == 0:
                        left_neighbor = (len(matrix_grid[0]) - 1, y)
                    elif x == len(matrix_grid[0]) - 1:
                        right_neighbor = (0, y)
                    tile_neighbors.append(left_neighbor)
                    tile_neighbors.append(right_neighbor)

                    tile_neighbors = [n for n in tile_neighbors if matrix_grid[n[1]][n[0]] == ' ']

                    gd[(x, y)] = tile_neighbors
                    vd[(x, y)] = 0.8
                    td[(x, y)] = 'P'
        return gd, vd, td

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.graph_dict.keys())

    def get_neighbors(self, vertex):
        """ returns neighbors of a vertex """
        return self.graph_dict[vertex]

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.graph_dict:
            self.graph_dict.setdefault(vertex, [])
            self.value_dict[vertex] = 0.8  # if it is not yet confirmed

    def update_values(self, observed_pellets, pacmans):
        for pellet in observed_pellets:
            coord, value = pellet[:2], pellet[2]
            self.value_dict[coord] = value
        observed_pellet_coordinates = set([pellet[:2] for pellet in observed_pellets])
        for tile in pacmans.lines_of_sight:
            if tile not in observed_pellet_coordinates:
                self.value_dict[tile] = 0

    def mark_pacs(self, friends, enemies):
        for f in friends.pacmans:
            self.type_dict[f.coordinates] = 'F'
        for e in enemies.values():
            self.type_dict[e['coords']] = 'E'

    def reset_types(self):
        for tile in self.type_dict.keys():
            self.type_dict[tile] = 'P'

    def find_paths(self, start):
        distances = dict()  # primary distance dictionary
        distances[start] = 0
        prev = dict()  # previous vertices in shortest paths
        prev[start] = None

        for vertex in self.vertices():
            if vertex == start:
                distances[vertex] = 0
            else:
                distances[vertex] = 10 ** 10
                prev[vertex] = None
        unvisited_distances = copy.deepcopy(distances)  # secondary "unvisited" distance dict

        while unvisited_distances:

            pointer = min(unvisited_distances,
                          key=unvisited_distances.get)  # Min weight vertex we haven't visited yet
         #   print((pointer, self.get_neighbors(pointer)), file=sys.stderr)
            del unvisited_distances[pointer]

            for neighbor in self.get_neighbors(pointer):
                new_distance = distances[pointer] + 1
                if new_distance < distances[neighbor]:  # Is new path shorter?
                    distances[neighbor] = new_distance
                    prev[neighbor] = pointer
                    if neighbor in unvisited_distances:
                        unvisited_distances[neighbor] = new_distance

        shortest_paths = {}

        for vertex in self.vertices():
            if vertex != start:
                shortest_paths[vertex] = [vertex]
                pointer = vertex
                while prev[pointer] != start:
                    shortest_paths[vertex].append(prev[pointer])
                    pointer = prev[pointer]
                shortest_paths[vertex] = shortest_paths[vertex][::-1]

        return shortest_paths

    def best_path(self, start):
        paths = self.find_paths(start)
        path_scores = {}
        for goal, path in paths.items():
            value = 0
            abort = False
            for tile in path:
                if self.type_dict[tile] == 'F':  # если на пути встречается союзник
                    abort = True  # сразу его бросаем
                    continue
                else:
                    value += self.value_dict[tile]
            if not abort:
                path_scores[goal] = value / len(path)  # в противном случае берём соотношение ценности пути к его длине
        sorted_scores = [k for k, v in sorted(path_scores.items(), key=lambda item: item[1], reverse=True)]
        return sorted_scores[:2]


with open('sample_grid.txt', 'r') as g:
    matrix_grid = [[ch for ch in l if ch != '\n'] for l in g.readlines()]
g.close()

for g in matrix_grid:
    g += [' ']*(len(matrix_grid[0])-len(g))

grid = Grid(matrix_grid)
n = grid.get_neighbors((17,5))
paths = grid.find_paths((17,5))

#print(n)


def make_n_zones(lst, n):
    l = np.array_split(np.array(lst), n)
    return [set(list(sub)) for sub in l]


a = [(1,2), (2,4), (4,3)]
a.sort(key=lambda tup: tup[1])
print(a[0:3])