import numpy as np
import networkit as nk

def CSRtoGraph(matr) :
    G = nk.Graph(matr.shape[0], True)
    start = matr.indptr[0]
    for ind, end in enumerate(matr.indptr[1:]):
        for j in range(start, end) :
            G.addEdge(ind, matr.indices[j], matr.data[j])
        start = end
    return G