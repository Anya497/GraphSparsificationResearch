import numpy as np
import networkit as nk
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils


def CSRtoGraph(matr) :
    G = nk.Graph(matr.shape[0], True)
    start = matr.indptr[0]
    for ind, end in enumerate(matr.indptr[1:]):
        for j in range(start, end) :
            G.addEdge(ind, matr.indices[j], matr.data[j])
        start = end
    return G




logger = utils.build_logger(__name__)


def graph_sparsify(M, epsilon, maxiter=10, seed=None):
    r"""Sparsify a graph (with Spielman-Srivastava).

    Parameters
    ----------
    M : Graph or sparse matrix
        Graph structure or a Laplacian matrix
    epsilon : float
        Sparsification parameter, which must be between ``1/sqrt(N)`` and 1.
    maxiter : int, optional
        Maximum number of iterations.
    seed : {None, int, RandomState, Generator}, optional
        Seed for the random number generator (for reproducible sparsification).

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    Examples
    --------
    >>> from pygsp import reduction
    >>> from matplotlib import pyplot as plt
    >>> G = graphs.Sensor(100, k=20, distributed=True, seed=1)
    >>> Gs = reduction.graph_sparsify(G, epsilon=0.4, seed=1)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = G.plot(ax=axes[0], title='original')
    >>> Gs.coords = G.coords
    >>> _ = Gs.plot(ax=axes[1], title='sparsified')

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling`.
    for more informations

    """
    # Test the input parameters
    if isinstance(M, graphs.Graph):
        if not M.lap_type == 'combinatorial':
            raise NotImplementedError
        L = M.L
    else:
        L = M

    N = np.shape(L)[0]

    if not 1./np.sqrt(N) <= epsilon < 1:
        raise ValueError('GRAPH_SPARSIFY: Epsilon out of required range')

    # Not sparse
    resistance_distances = utils.resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(M, graphs.Graph):
        W = M.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    # Calculate the new weights.
    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)
    dist = stats.rv_discrete(values=(np.arange(len(Pe)), Pe), seed=seed)

    for i in range(maxiter):
        # Rudelson, 1996 Random Vectors in the Isotropic Position
        # (too hard to figure out actual C0)
        C0 = 1 / 30.
        # Rudelson and Vershynin, 2007, Thm. 3.1
        C = 4 * C0
        q = round(N * np.log(N) * 9 * C**2 / (epsilon**2))

        results = dist.rvs(size=int(q))
        ### MODIFICATION
        values2, counts2 = np.unique(results, return_counts=True)
        spin_counts = np.column_stack((values2, counts2)).astype(int)
        #spin_counts = stats.itemfreq(results).astype(int)
        per_spin_weights = weights / (q * Pe)

        counts = np.zeros(np.shape(weights)[0])
        counts[spin_counts[:, 0]] = spin_counts[:, 1]
        new_weights = counts * per_spin_weights

        sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                     shape=(N, N))
        sparserW = sparserW + sparserW.T
        sparserL = sparse.diags(sparserW.diagonal(), 0) - sparserW

        if graphs.Graph(sparserW).is_connected():
            break
        elif i == maxiter - 1:
            logger.warning('Despite attempts to reduce epsilon, sparsified graph is disconnected')
        else:
            epsilon -= (epsilon - 1/np.sqrt(N)) / 2.

    if isinstance(M, graphs.Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not M.is_directed():
            sparserW = (sparserW + sparserW.T) / 2.

        Mnew = graphs.Graph(sparserW)
        #M.copy_graph_attributes(Mnew)
    else:
        Mnew = sparse.lil_matrix(sparserL)

    return Mnew
