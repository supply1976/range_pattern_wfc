import numpy as np
from .base import BaseSolver

class MRFSolver(BaseSolver):
    def solve(self, H:int, W:int, seed=None, start_token=None):
        rng = np.random.default_rng(seed)
        K = self.lib.K
        weights = self.lib.weights
        allow = self.lib.allow
        if allow is None:
            raise ValueError("Compatibility table not built.")
        if start_token is None:
            start_token = rng.choice(np.arange(K), p=weights)
        grid = np.full((H, W), -1, dtype=int)
        grid[0,0] = start_token
        for r in range(H):
            for c in range(W):
                if r==0 and c==0:
                    continue
                possible = set(range(K))
                if r>0:
                    nt = grid[r-1,c]
                    if nt!=-1:
                        possible &= allow[nt][self.lib.DOWN]
                if c>0:
                    wt = grid[r,c-1]
                    if wt!=-1:
                        possible &= allow[wt][self.lib.RIGHT]
                if possible:
                    sel = rng.choice(list(possible))
                    grid[r,c]=sel
        return grid
