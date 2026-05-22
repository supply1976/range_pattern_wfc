import numpy as np
from collections import deque
from .base import BaseSolver

class WFCSolver(BaseSolver):
    def observe(self, wave, weights, rng):
        """
        Observe step: select a cell to collapse based on entropy and randomly choose a state for it.
        
        :param wave: bool array of shape (H, W, K) representing possible states for each cell
        :param weights: array of shape (K,) representing weights for each state
        :param rng: numpy random generator instance
        """
        H, W, K = wave.shape # (H, W) is the cell grid size, K is number of states (patterns)
        options = wave.sum(axis=2) # number of possible states per cell
        mask = (options > 1) # cells with more than one possible state
        if not np.any(mask):
            return None, None # fully collapsed
        p = wave * weights # (H, W, K) weighted possibilities
        p_sum = p.sum(axis=2, keepdims=True) # (H, W, 1) sum over states
        p_norm = np.divide(p, p_sum, out=np.zeros_like(p), where=(p_sum>0)) # normalized probabilities
        # calculate entropy, H = -sum(p * log(p)), log(0) treated as 0
        log_p = np.log(p_norm, out=np.zeros_like(p_norm), where=(p_norm>0))
        entropy = -(p_norm * log_p).sum(axis=2) # (H, W) entropy per cell
        entropy[~mask] = np.inf # ban cells that are already collapsed, never be chosen
        # add small random noise to entropy to break ties
        entropy = entropy + rng.random(entropy.shape)*1e-6
        y, x = np.unravel_index(np.argmin(entropy), entropy.shape) # cell index with lowest entropy
        allowed = np.where(wave[y, x])[0] # number of allowed states at (y, x)
        if len(allowed) == 0:
            return y, x # contradiction, empty domain
        local_w = weights[allowed]
        local_w = local_w / local_w.sum()
        choice = rng.choice(allowed, p=local_w) # randomly choose a state based on weights
        # ban all other states at (y, x)
        wave[y, x, :] = False
        wave[y, x, choice] = True
        return y, x

    def propagate(self, wave, allow, start_cells):
        H, W, K = wave.shape
        q = deque(start_cells)
        DIRS = self.lib.DIRS
        while q:
            x, y = q.popleft()
            for d, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                possible_here = np.where(wave[y, x])[0]
                if possible_here.size == 0:
                    return False
                neighbor_possible = np.where(wave[ny, nx])[0]
                supported = np.zeros_like(wave[ny, nx], dtype=bool)
                support_set = set()
                for i in possible_here:
                    support_set |= allow[i][d]
                for j in neighbor_possible:
                    if j in support_set:
                        supported[j] = True
                to_ban = np.where(wave[ny, nx] & ~supported)[0]
                if to_ban.size > 0:
                    wave[ny, nx, to_ban] = False
                    if not wave[ny, nx].any():
                        return False
                    q.append((nx, ny))
        return True

    def solve(self, H:int, W:int, seed=None, start_token=None):
        # returns a grid of shape (H, W) with pattern indices
        K = self.lib.K
        wave = np.ones((H, W, K), dtype=bool)
        rng = np.random.default_rng(seed)
        freqs = self.lib.freqs
        weights = freqs / freqs.sum()
        allow = self.lib.allow
        if allow is None:
            raise ValueError("Compatibility table not built.")
        while True:
            oy, ox = self.observe(wave, weights, rng)
            if oy is None:
                # fully collapsed, no more observations needed
                break
            if isinstance(oy, int) and isinstance(ox, int) and not wave[oy, ox].any():
                raise RuntimeError("Contradiction while observing.")
            ok = self.propagate(wave, allow, start_cells=[(ox, oy)])
            if not ok:
                raise RuntimeError("Contradiction during propagation.")
        return np.argmax(wave, axis=2)
