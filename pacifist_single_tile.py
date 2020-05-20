import sys
import copy
import numpy as np
from itertools import chain


# TODO: assess paths better?
# TODO: shock is very much like ZA WARUDO; find out how it happens
# TODO: SINGLE INCREMENTS, NOT PATHS
# TODO: sometimes commands are not received
# TODO: поощрить разведку
# TODO: роли?
# TODO: как-то штрафовать тупики
# TODO: не только расстояние до жратвы, но и сколько там можно сожрать всего.
# TODO: может, брать не score, именно соотношение?
# TODO: если враг не напрямую на пути, а мы заходим с разных сторон, то это не учитывается, и мы умираем.
# TODO: для первых ходов смотреть, когда пакманы без скорости счёт не пополнят, в то время как у врага он изменится - это сигнал,
#  что заморозка времени активировалась, и можно тогда прикинуть, через сколько ходов нужно опять юзать скорость
# TODO: избегай врагов лучше?

# Game classes

class Grid(object):

    def __init__(self, matrix_grid, width, height):
        self.graph_dict, self.value_dict, \
        self.type_dict = self.load_matrix(
            matrix_grid)  # P - pellet, B - blocked (by friend or enemy of the wrong type), E - enemy
        self.targeted = {k: False for k in self.graph_dict.keys()}
        self.width = width
        self.height = height

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
                    vd[(x, y)] = 0.7
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

    def update_values(self, observed_pellets, pacmen):
        """
        update values depending on what we see and where our pacs are
        """
        for pellet in observed_pellets:
            coord, value = pellet[:2], pellet[2]
            self.value_dict[coord] = value
        for coord, value in self.value_dict.items():
            if value == 10:
                if (coord[0], coord[1], value) not in observed_pellets:
                    self.value_dict[coord] = 0
        observed_pellet_coordinates = set([pellet[:2] for pellet in observed_pellets])
        for tile in pacmen.lines_of_sight:
            if tile not in observed_pellet_coordinates:
                self.value_dict[tile] = 0

    def mark_pacs(self, friends, enemies):
        """Mark tiles depending on pacs' location"""
        for f in friends.pacmen:
            self.type_dict[f.coordinates] = 'F'
        for e in enemies.pacmen:
            if e.coordinates:
                self.type_dict[e.coordinates] = 'E'

    def reset_types(self):
        for tile in self.type_dict.keys():
            self.type_dict[tile] = 'P'

    def make_n_zones(self, n):
        """Make N zones"""
        l = np.array_split(np.array(range(self.width)), n)
        return [set(list(sub)) for sub in l]

    def find_paths(self, start):
        """Find paths from the start to every other tile"""
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
            if (vertex != start) and not (self.type_dict[vertex] == 'P' and self.value_dict[vertex] == 0):
                shortest_paths[vertex] = [vertex]
                pointer = vertex
                while prev[pointer] != start:
                    shortest_paths[vertex].append(prev[pointer])
                    pointer = prev[pointer]
                shortest_paths[vertex] = shortest_paths[vertex][::-1]
        shortest_paths = {goal: path for goal, path in shortest_paths.items() if not self.targeted[goal]}
        return shortest_paths


class Pacman(object):

    def __init__(self, pid, parameters=None):
        self.pac_id = pid
        self.coordinates = None
        self.type = None
        self.boost = 0
        self.cooldown = 0

        self.counter = None

    def counters_me(self):
        """Choose the type countering opponent's"""
        type_dict = {'ROCK': 'PAPER', 'PAPER': 'SCISSORS', 'SCISSORS': 'ROCK'}
        return type_dict[self.type]


class OurPacman(Pacman):

    def __init__(self, pid, parameters=None):
        super().__init__(pid, parameters)

        self.previous_coords = (None, None, None)  # what are the coordinates of our last two turns?

        self.stuck = False
        self.cycle = False

        self.los = list()  # line of sight
        self.zone = tuple()  # what zone are we fighting in?

        self.closest_enemy = None  # what enemy is closest to us?
        self.last_distance = None  # what was the distance to him last turn?

        self.fighting = False  # are we in combat?

        self.nearest_ten = None  # Is there a large pellet near us?

        self.current_order = ''
        self.current_route = []
        self.current_goal = None

    def update(self, parameters):
        self.coordinates = parameters['coords']
        self.type = parameters['type']
        self.boost = parameters['boost']
        self.cooldown = parameters['cooldown']
        self.counter = self.counters_me()

    def get_line_of_sight(self, grid):
        self.los = list()
        cross_neighbors = copy.deepcopy(grid.get_neighbors(self.coordinates))
        while cross_neighbors:
            pointer = cross_neighbors.pop(0)
            self.los.append(pointer)
            candidates = grid.get_neighbors(pointer)
            candidates = [c for c in candidates if c not in self.los + cross_neighbors]  # ещё не в списках
            candidates = [c for c in candidates if c[0] == self.coordinates[0]
                          or c[1] == self.coordinates[1]]  # валидность по координатам
            cross_neighbors += candidates

    '''Food related functions'''

    def find_nearest_ten(self, grid):
        """Is there a large pellet next to me?"""
        neighbours = [n for n in grid.get_neighbors(self.coordinates) if grid.value_dict[n] == 10]
        if neighbours:
            self.nearest_ten = neighbours[0]
            self.current_goal = (self.nearest_ten, 10)
            grid.targeted[self.nearest_ten] = True
            self.current_order = f"MOVE {self.pac_id} {self.nearest_ten[0]} {self.nearest_ten[1]}"
        else:
            self.nearest_ten = None

    def evaluate_paths(self, grid, pacmen):
        """Evaluate paths by points and lengths till nearest pellet,
        avoid crossing paths with allies"""

        paths = grid.find_paths(self.coordinates)

        # Игнорируем пути где союзник на 1 или 2 шаге на расстоянии 1 от нас
        tail_tiles = [pacman.current_route[:2] for pacman in pacmen]
        paths = {goal: path for goal, path in paths.items() if
                 not path[2:4] in tail_tiles}  # мы не следуем за союзником
        paths = {goal: path for goal, path in paths.items() if goal[0] in self.zone}  # жрём только в своей зоне

        path_scores = {}
        for goal, path in paths.items():
            value = 0
            abort = False
            for i in range(len(path)):  # проверяем врагов и пересечения с союзниками на расстоянии 8
                tile = path[i]
                others_this_step = set([pacman.current_route[i] for pacman in pacmen
                                        if i < len(pacman.current_route)])
                if (grid.type_dict[tile] in ('E', 'F')) or tile in others_this_step:
                    abort = True  # бросаем путь если там враг или мы пересечёмся с союзником на каком-то шаге
                    continue
                else:
                    value += grid.value_dict[tile]

            if not abort:
                weight = value / len(path)
                path_scores[goal] = weight  # в противном случае берём соотношение ценности пути к его длине
        return path_scores, paths

    def choose_pellet(self, grid, pacmen):
        """Find where is it the best to go"""

        path_scores, paths = self.evaluate_paths(grid, pacmen)

        if not path_scores:  # если нам в нашей зоне уже нечего жрать
            self.zone = set(list(range(width)))
            path_scores, paths = self.evaluate_paths(grid, pacmen)

        if path_scores:
            self.not_cornered(path_scores, paths)
        else:
            self.cornered_by_an_ally(grid)

    def not_cornered(self, path_scores, paths):
        """My actions if I am not cornered"""

        paths = {goal: path for goal, path in paths.items() if goal in path_scores}

        position = 0  # at what position in path to look for the tile
        my_neighbors = grid.get_neighbors(self.coordinates)
        new_neighbors = []
        if self.boost != 0:
            position = 1
            paths = {goal: path for goal, path in paths.items() if len(path) > 1}
            path_scores = {goal: score for goal, score in path_scores.items() if goal in paths}

            for n in my_neighbors:
                new_neighbors += [a for a in grid.get_neighbors(n) if a != self.coordinates]
            my_neighbors = [n for n in new_neighbors if n not in grid.get_neighbors(self.coordinates)]

        candidate_dict = {neighbor: 0 for neighbor in my_neighbors}
        for goal, path in paths.items():
            for neighbor in my_neighbors:
                if path[position] == neighbor:
                    candidate_dict[neighbor] += path_scores[goal]
                    continue

        x, y = max(candidate_dict, key=candidate_dict.get)

        remaining_paths = {goal: path for goal, path in paths.items() if path[position] == (x, y)}
        remaining_scores = {goal: score for goal, score in path_scores.items() if goal in remaining_paths}

        best_path = max(remaining_scores, key=remaining_scores.get)
        self.current_route = remaining_paths[best_path]
        self.current_goal = best_path

        grid.targeted[(x, y)] = True
        self.current_order = f"MOVE {self.pac_id} {x} {y}"

    def cornered_by_an_ally(self, grid):
        """My actions if all my paths are blocked by an ally"""
        candidates = grid.get_neighbors(self.coordinates)  # если мы заблокированы в углу
        candidates = [c for c in candidates if grid.value_dict[c] > 0]

        if candidates:
            if self.current_goal:
                grid.targeted[self.current_goal[0]] = False

            candidate = candidates[0]
            self.current_goal = (candidate, grid.value_dict[candidate])
            self.current_route = [candidate]
            x, y = candidate

            grid.targeted[(x, y)] = True  # this field is targeted, ignore it
            self.current_order = f"MOVE {self.pac_id} {x} {y}"
        else:
            self.current_goal = None
            self.current_route = []
            self.current_order = ''

    '''Fight related functions'''

    def find_enemy(self, enemies):
        """Find closest enemy within line of sight"""
        observed_enemies = [(enemy, self.find_distance(enemy.coordinates))
                            for enemy in enemies.pacmen if enemy.coordinates in self.los]  # enemies in line of sight
        if observed_enemies:
            observed_enemies.sort(key=lambda tup: tup[1])
            self.closest_enemy = observed_enemies[0]
        else:
            self.closest_enemy = None

    def find_distance(self, finish):
        """Find manhattan distance between the pacman and a point"""
        x1, y1 = self.coordinates
        x2, y2 = (x1 - width, y1 - height)
        x, y = [x1, x2], [y1, y2]
        distance = 99999
        for alt_x in x:
            for alt_y in y:
                distance = min(abs(finish[0] - alt_x) + abs(finish[1] - alt_y), distance)
        return distance

    def fight(self):
        self.fighting = True

    def flight(self):
        self.fighting = False
        self.last_distance = 0

    def combat_mode(self, grid):
        """
        select appropriate action when we see an enemy
        """
        if self.closest_enemy:
            enemy, distance = self.closest_enemy
            x, y = enemy.coordinates
            neighboring_tiles = grid.get_neighbors(enemy.coordinates)  # какие клетки примыкают

            if self.type != enemy.counter:
                if self.cooldown == 0:  # кд дольше чем нам идти полпути
                    if (distance == 1 or (distance == 2 and enemy.boost != 0)) \
                            and ((not self.last_distance) or (self.last_distance > distance)):
                        self.current_order = f"SWITCH {self.pac_id} {enemy.counter}"  # меняем тип
                        self.last_distance = distance
                        self.fight()
                else:
                    for t in neighboring_tiles:
                        if grid.type_dict != 'F':
                            grid.type_dict[t] = 'E'
                    self.flight()

            elif (distance == 1 and ((not self.last_distance)  # если враг вплотную к нам и мы его впервые увидели
                                     or (distance < self.last_distance)  # или если дистанция сократилась
                                     or enemy.coordinates == self.previous_coords[2])):  # или если он за нами гонится
                self.current_order = f"MOVE {self.pac_id} {x} {y}"
                self.last_distance = distance
                self.fight()

            elif (distance == 2 and ((not self.last_distance)
                                     or (distance < self.last_distance)
                                     or enemy.coordinates == self.previous_coords[1])):
                self.current_order = f"MOVE {self.pac_id} {x} {y}"
                self.last_distance = distance
                self.fight()

            else:
                for t in neighboring_tiles:
                    if grid.type_dict != 'F':
                        grid.type_dict[t] = 'E'
                self.flight()


class EnemyPacman(Pacman):

    def __init__(self, pid, parameters=None):
        super().__init__(pid, parameters)
        self.coordinates = None
        self.last_seen = None
        self.status = 'UNKNOWN'

    def update(self, parameters=None):
        if parameters:  # если мы врага ВИДИМ, получаем точную информацию
            self.last_seen = self.coordinates
            self.coordinates = parameters['coords']
            self.type = parameters['type']
            self.boost = parameters['boost']
            self.cooldown = parameters['cooldown']
            self.counter = self.counters_me()
            self.status = 'ALIVE'
        else:  # если мы врага НЕ видим, спекулируем
            if self.coordinates:
                self.last_seen = self.coordinates
            self.coordinates = None
            if self.boost >= 1:
                self.boost -= 1
            if self.cooldown >= 1:
                self.cooldown -= 1


class pacmen(object):

    def __init__(self, pac_dictionary):
        self.pacmen = [Pacman(pid, parameters) for pid, parameters in pac_dictionary.items()]

    def alive(self):
        return set([p.pac_id for p in self.pacmen])

    def count(self):
        return len(self.pacmen)


class Ourpacmen(pacmen):

    def __init__(self, pac_dictionary):
        super().__init__(pac_dictionary)
        self.pacmen = [OurPacman(pid, parameters) for pid, parameters in pac_dictionary.items()]

        self.lines_of_sight = None

    def update(self, grid, pac_dictionary):
        self.pacmen = [p for p in self.pacmen if p.pac_id in pac_dictionary]  # remove dead pacmen
        for pacman in self.pacmen:
            pacman.update(pac_dictionary[pacman.pac_id])  # update parameters by input
            pacman.get_line_of_sight(grid)
            pacman.find_nearest_ten(grid)
        self.lines_of_sight = list(set(list(chain.from_iterable([p.los for p in self.pacmen]))))
        self.assign_zones(grid)

    def find_enemies(self, enemies):
        for pacman in self.pacmen:
            pacman.find_enemy(enemies)  # find nearest enemy in line of sight
            pacman.fighting = False  # by default, non-combat mode

    def assign_zones(self, grid):
        """Assigns zones to pacmen"""
        pac_x = [(p, p.coordinates[0]) for p in self.pacmen]  # leftmost to rightmost
        pac_x.sort(key=lambda tup: tup[1])

        zone_ranges = grid.make_n_zones(2)  # split into two zones
        zone_ranges = [set(zone) for zone in zone_ranges]
        mid_zone = set(list(range((width // 3), (width // 3) * 2 + 1)))

        whole_field = set(list(range(width)))

        num_pacs = self.count()

        if num_pacs == 5:  # 2 слева, 2 справа, один в центре
            pac_x[0][0].zone = zone_ranges[0]
            pac_x[1][0].zone = zone_ranges[0]
            pac_x[1][0].zone = mid_zone
            pac_x[3][0].zone = zone_ranges[1]
            pac_x[4][0].zone = zone_ranges[1]
        elif num_pacs == 4:  # 2 слева, 2 справа
            pac_x[0][0].zone = zone_ranges[0]
            pac_x[1][0].zone = zone_ranges[0]
            pac_x[2][0].zone = zone_ranges[1]
            pac_x[3][0].zone = zone_ranges[1]
        elif num_pacs == 3:  # 1 слева, 1 в центре, 1 справа
            pac_x[0][0].zone = zone_ranges[0]
            pac_x[1][0].zone = mid_zone
            pac_x[2][0].zone = zone_ranges[1]
        elif num_pacs == 2:  # 1 слева, 1 справа
            pac_x[0][0].zone = zone_ranges[0]
            pac_x[1][0].zone = zone_ranges[1]
        else:  # всего один
            pac_x[0][0].zone = whole_field


class Enemypacmen(pacmen):

    def __init__(self, enemy_dictionary):
        super().__init__(enemy_dictionary)
        self.pacmen = [EnemyPacman(pid, parameters) for pid, parameters in enemy_dictionary.items()]

    def alive(self):
        return set([p.pac_id for p in self.pacmen])

    def count(self):
        return len(self.pacmen)

    def update(self, grid, enemy_dictionary, our_pacmen):
        for pacman in self.pacmen:
            if pacman.pac_id in enemy_dictionary.keys():
                pacman.update(enemy_dictionary[pacman.pac_id])  # update parameters by input
            else:
                pacman.update()

            if (not pacman.coordinates) and pacman.last_seen:  # если мы уже не видим врага

                i_can_be_in = [pacman.last_seen]  # где он может быть?
                tile_neighbors = grid.get_neighbors(pacman.last_seen)
                if pacman.boost != 0:
                    for t in tile_neighbors:
                        i_can_be_in += grid.get_neighbors(t)
                        i_can_be_in.append(t)
                i_can_be_in = list(set(i_can_be_in))

                i_can_be_in = [loc for loc in i_can_be_in
                               if loc not in our_pacmen.lines_of_sight]  # явно не в нашем поле зрения

                if not i_can_be_in:  # если таких полей нет
                    pacman.status = 'DEAD'


def turn_inputs():
    """
    yield inputs for a single turn
    """
    my_score, opponent_score = [int(i) for i in input().split()]
    scores = {'me': my_score, 'opponent': opponent_score}

    visible_pac_count = int(input())  # all your pacs and enemy pacs in sight

    pacs = {}
    my_pacs = {}
    enemy_pacs = {}
    for i in range(visible_pac_count):

        pac_id, mine, x, y, type_id, speed_turns_left, ability_cooldown = input().split()
        pac_id = int(pac_id)
        mine = mine != "0"
        x = int(x)
        y = int(y)
        speed_turns_left = int(speed_turns_left)
        ability_cooldown = int(ability_cooldown)
        pacman = {'coords': (x, y),
                  'type': type_id,
                  'boost': speed_turns_left,
                  'cooldown': ability_cooldown}
        if mine == 1:
            my_pacs[pac_id] = pacman
        else:
            enemy_pacs[pac_id] = pacman

    pacs['friends'] = my_pacs
    pacs['enemies'] = enemy_pacs

    visible_pellet_count = int(input())  # all pellets in sight
    pellets = []
    for i in range(visible_pellet_count):
        pellets.append(tuple([int(j) for j in input().split()]))
    return scores, pacs, pellets


# Game initialization

width, height = [int(i) for i in input().split()]
grid_matrix = []
for i in range(height):
    row = [ch for ch in input()]
    grid_matrix.append(row)  # one line of the grid: 1 is floor, -1 is wall
grid = Grid(grid_matrix, width, height)

my_pacs = Ourpacmen({0: {}, 1: {}, 2: {}, 3: {}, 4: {}})
enemy_pacs = Enemypacmen({0: {}, 1: {}, 2: {}, 3: {}, 4: {}})

# Game loop
turns = 1

while True:
    '''Update grid'''
    grid.reset_types()
    scores, pacs, pellets = turn_inputs()

    my_pacs.update(grid, pacs['friends'])
    enemy_pacs.update(grid, pacs['enemies'], my_pacs)

    my_pacs.find_enemies(enemy_pacs)

    grid.mark_pacs(my_pacs, enemy_pacs)
    grid.update_values(pellets, my_pacs)

    '''Give orders'''
    if turns == 1:
        for pacman in my_pacs.pacmen:
            pacman.current_order = f"SPEED {pacman.pac_id}"
    else:
        for pacman in my_pacs.pacmen:

            pacman.combat_mode(grid)  # Combat orders

            pacman.stuck = False  # Are we stuck?
            if pacman.previous_coords[2]:
                if pacman.previous_coords[2] == pacman.coordinates and not pacman.fighting:
                    pacman.stuck = True  # либо мы в той же точке, либо у нас цикл
                    pacman.current_goal = None
                    pacman.current_route = []
                elif pacman.previous_coords[2] == pacman.previous_coords[0]:
                    pacman.cycle = True

            pacman.previous_coords = (pacman.previous_coords[1], pacman.previous_coords[2], pacman.coordinates)

            if not pacman.fighting:  # If we are not in combat
                if pacman.nearest_ten:
                    pass
                elif pacman.cooldown == 0:
                    pacman.current_order = f"SPEED {pacman.pac_id}"
                else:
                    pacman.choose_pellet(grid, my_pacs.pacmen)  # то пусть собирает

    '''Print orders'''
    chain_of_command = [pacman.current_order for pacman in my_pacs.pacmen]
    print(' | '.join(chain_of_command))
    turns += 1
