import numpy as np
from graph import Graph


def pagerank(graph: Graph, tolerance: float = 1e-5) -> np.array:
    '''
    Computes rank of each vertex according to its input edges

    :param graph:
    Graph object
    :param tolerance:
    Value to stop the algorithm when |current_pagerank - previous_pagerank| < tolerance
    :return:
    Rank of each vertex in graph
    '''
    last_pr = np.ones([graph.N])
    pr = np.full([graph.N], 1 / graph.N)
    while np.linalg.norm(pr - last_pr) >= tolerance:
        last_pr = pr
        pr = np.zeros([graph.N])
        for i, v in enumerate(graph.vertices):
            for desc, _ in v.desc:
                pr[desc] += last_pr[i] / len(v.desc)
    return pr
