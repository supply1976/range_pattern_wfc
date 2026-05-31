import numpy as np
from collections import deque
from scipy import sparse
from .base import BaseSolver


class WFCSolverEnhanced2(BaseSolver):
    """
    WFC solver using AC-4 (support counting) for constraint propagation.
    
    Key difference from AC-3 (WFCSolverEnhanced):
    - Maintains a support counter array: support[y, x, d, j] = number of patterns
      at (y,x)'s neighbor-in-direction-d that currently support pattern j at (y,x)
    - When patterns are banned, only decrements affected counters (delta update)
    - No redundant full recomputation — each ban is processed exactly once
    
    Optimizations over naive AC-4:
    - Vectorized support initialization (no Python loops)
    - int16 support array (halves memory: 134MB vs 268MB for 32x32)
    - Fast-path for observe collapse: overwrites support directly (O(nnz_row) vs O(K²))
    """

    def __init__(self, library):
        super().__init__(library)
        self._T_sparse = None   # list of 4 CSR matrices (K,K)
        self._T_sparse_T = None # list of 4 CSR matrices (K,K) transposed
        self._col_nnz = None    # cached column nnz for each direction

    def _get_sparse_transitions(self):
        """Build and cache sparse CSR transition matrices + transposes."""
        if self._T_sparse is not None:
            return self._T_sparse
        
        K = self.lib.K
        
        if hasattr(self.lib, 'T') and self.lib.T is not None:
            T_dense = self.lib.T
            self._T_sparse = [sparse.csr_matrix(T_dense[d]) for d in range(4)]
        else:
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
        
        # T_sparse_T[d][j, i] = T_sparse[d][i, j]
        self._T_sparse_T = [T.T.tocsr() for T in self._T_sparse]
        
        # Cache column nnz: col_nnz[d][j] = how many patterns support j from direction d
        OPP = [2, 3, 0, 1]
        self._col_nnz = []
        for d in range(4):
            opp_d = OPP[d]
            nnz_per_col = np.diff(self._T_sparse_T[opp_d].indptr).astype(np.int32)
            self._col_nnz.append(nnz_per_col)
        
        return self._T_sparse

    def _init_support(self, H, W, K):
        """
        Initialize support counters using vectorized numpy (no Python loops).
        
        support[y, x, d, j] = number of patterns currently possible at the
        neighbor of (y,x) in direction d that support pattern j remaining at (y,x).
        
        Edge cells get sentinel value (max int16) so constraints never trigger.
        """
        DIRS = self.lib.DIRS
        SENTINEL = np.int16(32767)
        
        # Start with sentinel everywhere (edges won't be overwritten)
        support = np.full((H, W, 4, K), SENTINEL, dtype=np.int16)
        
        # For each direction, build mask of cells that HAVE a valid neighbor
        for d, (dx, dy) in enumerate(DIRS):
            # Cells (y, x) where neighbor (x+dx, y+dy) is in-bounds
            if dy == -1:    # UP: rows 1..H-1 have neighbor above
                support[1:, :, d, :] = self._col_nnz[d]
            elif dy == 1:   # DOWN: rows 0..H-2 have neighbor below
                support[:H-1, :, d, :] = self._col_nnz[d]
            elif dx == 1:   # RIGHT: cols 0..W-2 have neighbor to right
                support[:, :W-1, d, :] = self._col_nnz[d]
            elif dx == -1:  # LEFT: cols 1..W-1 have neighbor to left
                support[:, 1:, d, :] = self._col_nnz[d]
            else:           # dx==0 and dy==0 shouldn't happen
                support[:, :, d, :] = self._col_nnz[d]
        
        return support

    def propagate(self, wave, support, banned_queue):
        """
        AC-4 constraint propagation using support counters.
        
        banned_queue: deque of (y, x, banned_indices) — cells where patterns were just banned.
        
        For each banned pattern i at cell (y,x):
          For each direction d:
            For each j that i supports at neighbor (T[d][i,j]=True):
              Decrement support[ny, nx, opposite_d, j]
              If support drops to 0 and wave[ny,nx,j] is True → ban j
        """
        H, W, K = wave.shape
        DIRS = self.lib.DIRS
        OPP = [2, 3, 0, 1]
        T_sparse = self._T_sparse
        T_indptr = [T.indptr for T in T_sparse]
        T_col_indices = [T.indices for T in T_sparse]

        while banned_queue:
            y, x, banned_indices = banned_queue.popleft()
            
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                
                opp_d = OPP[d]
                
                if banned_indices.size == 1:
                    # Single ban — direct CSR row access
                    idx = banned_indices[0]
                    s = T_indptr[d][idx]
                    e = T_indptr[d][idx + 1]
                    if s >= e:
                        continue
                    affected_j = T_col_indices[d][s:e]
                    support[ny, nx, opp_d, affected_j] -= 1
                elif banned_indices.size <= 8:
                    # Small batch — iterate CSR rows directly (avoids sparse submatrix overhead)
                    for idx in banned_indices:
                        s = T_indptr[d][idx]
                        e = T_indptr[d][idx + 1]
                        if s < e:
                            support[ny, nx, opp_d, T_col_indices[d][s:e]] -= 1
                else:
                    # Larger batch — sparse submatrix + column sum
                    sub = T_sparse[d][banned_indices]
                    lost = np.asarray(sub.sum(axis=0)).ravel()
                    nonzero_j = np.where(lost > 0)[0]
                    if nonzero_j.size == 0:
                        continue
                    support[ny, nx, opp_d, nonzero_j] -= lost[nonzero_j].astype(np.int16)
                
                # Check which patterns at neighbor now have zero support AND are still in wave
                neighbor_wave = wave[ny, nx]
                sup_slice = support[ny, nx, opp_d]
                newly_dead = (sup_slice <= 0) & neighbor_wave
                
                if newly_dead.any():
                    new_bans = np.where(newly_dead)[0]
                    wave[ny, nx, new_bans] = False
                    if not wave[ny, nx].any():
                        return False  # contradiction
                    banned_queue.append((ny, nx, new_bans))
        
        return True

    def _observe_and_propagate(self, wave, support, weights, rng):
        """
        Combined observe + propagate with fast-path for the collapse step.
        
        After collapsing a cell to a single pattern `choice`, instead of queuing
        K-1 bans (expensive sparse submatrix), we OVERWRITE support at each neighbor
        directly: support[ny, nx, opp_d, j] = 1 if T[d][choice, j] else 0.
        This is O(K) per direction instead of O(K * avg_nnz).
        """
        H, W, K = wave.shape
        DIRS = self.lib.DIRS
        OPP = [2, 3, 0, 1]
        T_indptr = [T.indptr for T in self._T_sparse]
        T_col_indices = [T.indices for T in self._T_sparse]
        
        # --- Observe: find cell, collapse ---
        options = wave.sum(axis=2)
        mask = (options > 1)
        if not np.any(mask):
            return True  # fully collapsed

        p = wave * weights
        p_sum = p.sum(axis=2, keepdims=True)
        p_norm = np.divide(p, p_sum, out=np.zeros_like(p, dtype=np.float64), where=(p_sum > 0))
        log_p = np.log(p_norm, out=np.zeros_like(p_norm), where=(p_norm > 0))
        entropy = -(p_norm * log_p).sum(axis=2)
        entropy[~mask] = np.inf
        entropy += rng.random(entropy.shape) * 1e-6

        oy, ox = np.unravel_index(np.argmin(entropy), entropy.shape)
        allowed = np.where(wave[oy, ox])[0]
        if len(allowed) == 0:
            return False  # contradiction

        local_w = weights[allowed]
        local_w = local_w / local_w.sum()
        choice = rng.choice(allowed, p=local_w)

        # Collapse
        wave[oy, ox, :] = False
        wave[oy, ox, choice] = True
        
        # --- Fast-path propagation for observe collapse ---
        # Instead of queuing K-1 bans, overwrite support at neighbors directly
        q = deque()
        
        for d, (dx, dy) in enumerate(DIRS):
            nx, ny = ox + dx, oy + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            
            opp_d = OPP[d]
            
            # After collapse: only `choice` remains at (oy, ox).
            # support[ny, nx, opp_d, j] should be 1 if T[d][choice, j] else 0
            # Get row `choice` of T[d]
            s = T_indptr[d][choice]
            e = T_indptr[d][choice + 1]
            supported_j = T_col_indices[d][s:e]
            
            # Overwrite: set all to 0, then 1 for supported
            sup_slice = support[ny, nx, opp_d]
            sup_slice[:] = 0
            if supported_j.size > 0:
                sup_slice[supported_j] = 1
            
            # Find newly dead patterns (support 0 AND still in wave)
            neighbor_wave = wave[ny, nx]
            newly_dead = (sup_slice <= 0) & neighbor_wave
            
            if newly_dead.any():
                new_bans = np.where(newly_dead)[0]
                wave[ny, nx, new_bans] = False
                if not wave[ny, nx].any():
                    return False  # contradiction
                q.append((ny, nx, new_bans))
        
        # Continue propagation for cascading bans
        if q:
            ok = self.propagate(wave, support, q)
            if not ok:
                return False  # contradiction
        return None  # not done yet

    def solve(self, H: int, W: int, seed=None, start_token=None):
        """
        Run WFC with AC-4 propagation.
        """
        K = self.lib.K
        wave = np.ones((H, W, K), dtype=bool)
        rng = np.random.default_rng(seed)
        freqs = self.lib.freqs
        weights = (freqs / freqs.sum()).astype(np.float64)

        T_sparse = self._get_sparse_transitions()
        support = self._init_support(H, W, K)

        while True:
            result = self._observe_and_propagate(wave, support, weights, rng)
            if result is True:
                break  # fully collapsed
            if result is False:
                raise RuntimeError("Contradiction during propagation.")
            # result is None → continue

        return np.argmax(wave, axis=2)
