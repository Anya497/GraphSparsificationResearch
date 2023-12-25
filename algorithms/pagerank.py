import scipy as sp
import numpy as np


def pagerank(graph: sp.sparse.coo_array, tolerance: float = 1e-5, alpha: float = 0.9) -> np.array:
    '''
    Computes rank of each vertex according to its input edges

    :param graph:
    Graph object
    :param tolerance:
    Value to stop the algorithm when |current_pagerank - previous_pagerank| < tolerance
    
    :return:
    Rank of each vertex in graph
    '''
    last_pr = np.ones(graph.shape[0])
    pr = np.ones(graph.shape[0]) / graph.shape[0]
    outdegrees = np.squeeze(np.asarray(graph.sum(axis=1).T))
    while np.linalg.norm(last_pr - pr) >= tolerance:
        last_pr = pr
        pr = alpha*((pr/outdegrees) @ graph) + (1 - alpha) / graph.shape[0]
    return pr
