from typing import Tuple, Optional

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.jit(nopython=True)
def fitness(order: NDArray[int],
            dists: np.ndarray,
            rows: int,
            columns: int) -> Tuple[float]:
    ret = 0
    order = order.reshape(rows, columns)
    for i in range(rows):
        for j in range(columns):
            for k in range(rows):
                for l in range(columns):
                    point_distance = np.sqrt(np.square(i - j) + np.square(k - l)) + 1e-6
                    value_distance = dists[order[i, j], order[k, l]]
                    ret += value_distance / point_distance
    return ret,


class HOF:
    """
    Hall of Fame: class for registering better and better individuals for the HCA.
    """

    def __init__(self, logging_freq=10, verb=True):
        """
        Initialize a HOF
        :param logging_freq: How often should the progress be saved and logged
        :param verb: Should progress be printed
        """
        self.hashC: int = 211
        self.logging_freq: int = logging_freq
        self.verb: bool = verb
        self.generation: int = 0
        self.fitnesses: list[float] = []
        self.generations: list[int] = []
        self.searched: dict[int, float] = dict()
        self.best_fitness: float = float("infinity")
        self.best_individual: np.ndarray = None

    def add(self, individual: np.ndarray, individual_fitness: float) -> bool:
        """
        Add the given individual to the HOF
        :param individual:
        :param individual_fitness:
        :return: True if it was added, False if it was there already
        """
        hashed = self.hash(individual)
        if hashed in self.searched and self.searched[hashed] == individual_fitness:
            return False
        self.searched[hashed] = individual_fitness
        if self.best_fitness > individual_fitness:
            self.best_fitness = individual_fitness
            self.best_individual = individual
            self.fitnesses.append(individual_fitness)
            self.generations.append(self.generation)
        if self.generation % self.logging_freq == 0:
            self.fitnesses.append(individual_fitness)
            self.generations.append(self.generation)
            if self.verb:
                print(self.generation, "\t", self.best_fitness)
        self.generation += 1
        return True

    def hash(self, individual) -> int:
        """
        Hash the given individual (this is then used for dictionary keys)
        :param individual: The individual to be hashed
        :return: hash
        """
        ret = 0
        for i in range(individual.shape[0]):
            ret += i * individual[i] * self.hashC
        return ret


@nb.jit(nopython=True)
def neighbours(individual: np.ndarray,
               x: int,
               y: int,
               rows: int,
               columns: int) -> list[np.ndarray]:
    """
    Return all individual's that can be obtained by swapping the (x,y) cell
        with any of its neighbours.
    :param individual: The base individual
    :param x: X coordinate of cell to be swapped
    :param y: Y coordinate of cell to be swapped
    :param rows: Number of rows for this matrix
    :param columns: Number of columns for this matrix
    :return: list of neighbouring individuals
    """
    directions = ((1, 1), (1, 0), (0, 1), (1, -1), (0, -1), (-1, 1), (-1, 0), (-1, -1))
    order = individual.reshape(rows, columns)
    ret = []
    for dx, dy in directions:
        movedX, movedY = x + dx, y + dy
        if movedX < 0 or movedX >= rows or movedY < 0 or movedY >= columns:
            continue
        new_order: np.ndarray = order.copy()
        tmp = new_order[x, y]
        new_order[x, y] = new_order[x + dx, y + dy]
        new_order[x + dx, y + dy] = tmp
        ret.append(new_order.flatten())
    return ret


@nb.jit(nopython=True)
def fitness_refined(x: NDArray[int],
                    dists: np.ndarray,
                    rows: int,
                    cols: int) -> float:
    return fitness(x, dists, rows, cols)[0]


@nb.jit(nopython=True)
def HCARefined(individual: NDArray[int],
               rows: int,
               columns: int,
               dists: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run the HCA algorithm for refined starting from the given individual
    @return: (the best REFINED individual, its REFINED score)
    """
    for i in range(100000):
        best_fitness = fitness_refined(individual, dists, rows, columns)
        orig_fitness = best_fitness
        print(i, ":", best_fitness, "->", individual)
        for x in range(rows):
            for y in range(columns):
                best_child: Optional[np.ndarray] = None
                for n in neighbours(individual, x, y, rows, columns):
                    this_fitness = fitness_refined(n, dists, rows, columns)
                    if this_fitness < best_fitness:
                        best_fitness = this_fitness
                        best_child = n
                if best_child is None:  # We didn't find a better neighbour
                    pass
                else:
                    individual = best_child
        if orig_fitness > best_fitness:
            continue
        else:
            print(f"REFINED FINISHED IN {i}-th epoch")
            return individual, best_fitness
    return individual, best_fitness
