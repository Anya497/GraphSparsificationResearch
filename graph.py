import numpy as np


class Vertex:
    def __init__(self, id: int):
        self.id = id
        self.desc: list[tuple[int, float]] = list()

    def add_desc(self, desc: tuple[id, float]):
        '''
        Add descendant to vertex

        :param desc: tuple[int, float]
            Tuple with descendant id and edge weight
        :return:
            None
        '''
        self.desc.append(desc)


class Graph:
    def __init__(self, n_v: int):
        '''
        Creates graph with n_v vertices without edges

        :param n_v: int
            Number of vertices
        '''
        self.N = n_v
        self.vertices = [Vertex(id) for id in range(n_v)]

    def add_vertex(self):
        self.vertices.append(Vertex(len(self.vertices)))

    def add_edge(self, v_pair: tuple[int, int, float]):
        self.vertices[v_pair[0]].add_desc((v_pair[1], v_pair[2]))

    def adj_mat_to_adj_list(self, mat: np.array):
        for i in range(mat.shape[0]):
            for j, w in enumerate(mat[i]):
                if w != 0:
                    self.vertices[i].add_desc((j, w))
