import numpy as np
from collections import deque
from scipy import sparse
from .base import BaseSolver


class WFCSolverEnhanced(BaseSolver):
    """
    Optimized WFC solver using sparse transition matrices + vectorized NumPy.
    
    Key optimizations over the original:
    1. Uses scipy.sparse CSR matrices for memory-efficient transition storage
       (e.g., 32MB for K=16387 at 0.7% fill vs 1GB dense)
    2. Replaces Python-level set iteration with sparse matrix-vector multiply
    3. Caches transition matrices across multiple solve() calls
    """

    def __init__(self, library):
        super().__init__(library)
        self._T_sparse = None   # list of 4 CSR matrices (K,K)

    def _get_sparse_transitions(self):
        """Build and cache sparse CSR transition matrices."""
        if self._T_sparse is not None:
            return self._T_sparse
        
        K = self.lib.K
        
        # Build from lib.T (dense bool) if available
        if hasattr(self.lib, 'T') and self.lib.T is not None:
            T_dense = self.lib.T
            self._T_sparse = [sparse.csr_matrix(T_dense[d]) for d in range(4)]
        else:
            # Build from allow (list-of-list-of-sets or numpy object array)
            allow = self.lib.allow
            if allow is None:
                raise ValueError("No compatibility data available (neither T nor allow).")
            
            T_list = []
            for d in range(4):
                rows, cols = [], []
                for i in range(K):
                    neighbors = allow[i][d]
                    for j in neighbors:
                        rows.append(i)
                        cols.append(j)
                data = np.ones(len(rows), dtype=bool)
                T_list.append(sparse.csr_matrix((data, (rows, cols)), shape=(K, K), dtype=bool))
            self._T_sparse = T_list
        
        return self._T_sparse

    def observe(self, wave, weights, rng):
        """
        Observe step: select cell with minimum entropy, collapse it.
        Returns (y, x) or (None, None).
        """
        H, W, K = wave.shape
        options = wave.sum(axis=2)
        mask = (options > 1)
        if not np.any(mask):
            return None, None

        # Entropy calculation
        p = wave * weights
        p_sum = p.sum(axis=2, keepdims=True)
        p_norm = np.divide(p, p_sum, out=np.zeros_like(p, dtype=np.float64), where=(p_sum > 0))
        log_p = np.log(p_norm, out=np.zeros_like(p_norm), where=(p_norm > 0))
        entropy = -(p_norm * log_p).sum(axis=2)
        entropy[~mask] = np.inf
        entropy += rng.random(entropy.shape) * 1e-6

        y, x = np.unravel_index(np.argmin(entropy), entropy.shape)
        allowed = np.where(wave[y, x])[0]
        if len(allowed) == 0:
            return y, x  # contradiction

        local_w = weights[allowed]
        local_w = local_w / local_w.sum()
        choice = rng.choice(allowed, p=local_w)

        wave[y, x, :] = False
        wave[y, x, choice] = True
        return y, x

    def propagate(self, wave, T_sparse, start_cells):
        """
        Constraint propagation using sparse CSR transition matrices.
        
        Uses hybrid approach:
        - Single possible pattern: direct CSR row slice (avoid submatrix overhead)
        - Multiple patterns: scipy sparse submatrix + column-OR reduction
        """
        H, W, K = wave.shape
        DIRS = self.lib.DIRS
        T_indptr = [T.indptr for T in T_sparse]
        T_col_indices = [T.indices for T in T_sparse]
        q = deque(start_cells)

        while q:
            x, y = q.popleft()
            wave_cell = wave[y, x]
            possible_indices = np.where(wave_cell)[0]
            if possible_indices.size == 0:
                return False

            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue

                neighbor = wave[ny, nx]

                if possible_indices.size == 1:
                    # Fast path: single row direct CSR access
                    idx = possible_indices[0]
                    s = T_indptr[d][idx]
                    e = T_indptr[d][idx + 1]
                    supported = np.zeros(K, dtype=bool)
                    if s < e:
                        supported[T_col_indices[d][s:e]] = True
                else:
                    # Multi-row: scipy sparse submatrix OR-reduction
                    sub = T_sparse[d][possible_indices]
                    supported = np.asarray(sub.getnnz(axis=0)) > 0

                # Ban unsupported patterns at neighbor
                to_ban = neighbor & ~supported
                if np.any(to_ban):
                    wave[ny, nx] = neighbor & supported
                    if not wave[ny, nx].any():
                        return False
                    q.append((nx, ny))
        return True

    def solve(self, H: int, W: int, seed=None, start_token=None):
        """
        Run WFC to generate a grid of shape (H, W) with pattern indices.
        Transition matrices are built once and cached for subsequent calls.
        """
        K = self.lib.K
        wave = np.ones((H, W, K), dtype=bool)
        rng = np.random.default_rng(seed)
        freqs = self.lib.freqs
        weights = (freqs / freqs.sum()).astype(np.float64)

        T_sparse = self._get_sparse_transitions()

        while True:
            oy, ox = self.observe(wave, weights, rng)
            if oy is None:
                break
            if isinstance(oy, int) and isinstance(ox, int) and not wave[oy, ox].any():
                raise RuntimeError("Contradiction while observing.")
            ok = self.propagate(wave, T_sparse, start_cells=[(ox, oy)])
            if not ok:
                raise RuntimeError("Contradiction during propagation.")
        return np.argmax(wave, axis=2)
