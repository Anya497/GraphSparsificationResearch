from graph import Graph
import numpy as np


class RandomSparsifier:
    def __init__(self, prob: float):
        '''
        Remove edges from graph with some probability

        :param prob:
        Probability to remove edge from graph
        '''
        self.prob = prob

    def to_sparse(self, graph: Graph):
        for v in graph.vertices:
            for desc in v.desc:
                if np.random.choice([True, False], size=1, p=[self.prob, 1 - self.prob]):
                    v.desc.remove(desc)
