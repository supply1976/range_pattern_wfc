import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from typing import Tuple, List, Dict, Optional

class PatternLibrary:
    """Holds extracted overlapping patterns, weights, and compatibility sets."""
    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # Up, Right, Down, Left (dx, dy)
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ID2DIRS = {UP:'UP', RIGHT:'RIGHT', DOWN:'DOWN', LEFT:'LEFT'}

    def __init__(
        self, 
        N, 
        overlap, 
        idx2pattern:List[np.ndarray], 
        freqs:np.ndarray,
        allow=None, # compatibility sets, can be precomputed or computed on the fly
        index_to_width=None,
        index_to_height=None,
    ):
        # Normalize N to (Ny, Nx)
        if isinstance(N, (list, tuple)):
            self.Ny, self.Nx = int(N[0]), int(N[1])
        else:
            self.Ny, self.Nx = int(N), int(N)
        self.N = (self.Ny, self.Nx)  # tuple form
        # Normalize overlap to (overlap_y, overlap_x)
        if overlap is None:
            self.overlap_y, self.overlap_x = self.Ny - 1, self.Nx - 1
        elif isinstance(overlap, (list, tuple)):
            self.overlap_y, self.overlap_x = int(overlap[0]), int(overlap[1])
        else:
            self.overlap_y, self.overlap_x = int(overlap), int(overlap)
        self.overlap = (self.overlap_y, self.overlap_x)
        self.idx2pattern = idx2pattern
        self.freqs = freqs
        self.K = len(idx2pattern)
        self.allow = allow
        self.index_to_width = index_to_width
        self.index_to_height = index_to_height

    @staticmethod
    def pattern_key(p: np.ndarray):
        # convert pattern array to a hashable key (tuple of pixel values)
        return tuple(p.reshape(-1).tolist())

    @staticmethod
    def _rotate90(p: np.ndarray):
        return np.rot90(p, k=1)

    @staticmethod
    def _reflectX(p: np.ndarray):
        return np.flip(p, axis=1)

    @classmethod
    def from_png(
        cls, 
        png_path:str, 
        N=3, 
        overlap=None, 
        periodic_input=False, 
        augment_rot_reflect=False,
    ):
        # Normalize N to (Ny, Nx)
        if isinstance(N, (list, tuple)):
            Ny, Nx = int(N[0]), int(N[1])
        else:
            Ny, Nx = int(N), int(N)
        # Normalize overlap to (overlap_y, overlap_x)
        if overlap is None:
            overlap_y, overlap_x = Ny - 1, Nx - 1
        elif isinstance(overlap, (list, tuple)):
            overlap_y, overlap_x = int(overlap[0]), int(overlap[1])
        else:
            overlap_y, overlap_x = int(overlap), int(overlap)
        # load from a PNG file and extract patterns
        img = Image.open(png_path)
        if img.mode =='1':
            img = img.convert("L")
        sample = np.array(img)
        if sample.ndim == 2:
            sample = sample[..., None]
        if periodic_input:
            sample = np.pad(sample, ((0, Ny-1),(0, Nx-1),(0,0)), mode='wrap')
        H, W, C = sample.shape
        patterns = sliding_window_view(sample, (Ny, Nx, C))  # (H-Ny+1, W-Nx+1, 1, Ny, Nx, C)
        # count unique patterns and their frequencies, with optional augmentation
        counts = defaultdict(int)

        def add_patch(p):
            counts[cls.pattern_key(p)] += 1

        for y in range(patterns.shape[0]):
            for x in range(patterns.shape[1]):
                pattern = patterns[y, x][0]  # (Ny, Nx, C)
                add_patch(pattern)
                if augment_rot_reflect:
                    rp = pattern
                    # rotations
                    for _ in range(3):
                        rp = cls._rotate90(rp)
                        add_patch(rp)
                    # reflections
                    add_patch(cls._reflectX(pattern))
                    rp = cls._reflectX(pattern)
                    for _ in range(3):
                        rp = cls._rotate90(rp)
                        add_patch(rp)
        unique_keys = list(counts.keys())
        key2idx = {k:i for i,k in enumerate(unique_keys)}
        idx2pattern = [np.array(k).reshape(Ny,Nx,C).astype(sample.dtype) for k in unique_keys]
        freqs = np.array([counts[k] for k in unique_keys], dtype=np.int64)
        return cls((Ny, Nx), (overlap_y, overlap_x), idx2pattern, freqs)

    @classmethod
    def from_extracted_patterns(cls, data, pattern_rank):
        p_dict = data[pattern_rank][()]
        patterns = p_dict['patterns']  # (K, Ny, Nx, C)
        if patterns.ndim == 3:
            patterns = patterns[..., None]  # add channel dim if missing
        idx2pattern = list(patterns)
        Ny, Nx = idx2pattern[0].shape[0], idx2pattern[0].shape[1]
        overlap = (Ny - 1, Nx - 1)  # default to max overlap
        if 'freqs' in list(p_dict.keys()):
            freqs = p_dict['freqs']
        else:
            # if freqs not provided, assume uniform distribution
            freqs = np.ones(len(idx2pattern), dtype=np.int64)
        index_to_width = None
        index_to_height = None
        if 'index_to_width' in list(data):
            index_to_width = data['index_to_width'][()]  # dict mapping pattern index to width value
        if 'index_to_height' in list(data):
            index_to_height = data['index_to_height'][()]  # dict mapping pattern index to height value
        return cls((Ny, Nx), overlap, idx2pattern, freqs, index_to_width=index_to_width, index_to_height=index_to_height)

    def build_compatibility(self):
        # compute which patterns can be adjacent in each direction based on pixel overlap
        Ny, Nx = self.Ny, self.Nx
        overlap_y, overlap_x = self.overlap_y, self.overlap_x
        K = self.K
        # allow[i][d] = set of pattern indices that can be adjacent to pattern i in direction d
        allow: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]
        for i, A in enumerate(self.idx2pattern):
            for j, B in enumerate(self.idx2pattern):
                if np.array_equal(A[:overlap_y, :, :], B[Ny-overlap_y:, :, :]):
                    # the first overlap_y rows of A match the last overlap_y rows of B, so pattern B can be above pattern A
                    allow[i][self.UP].add(j)
                if np.array_equal(A[Ny-overlap_y:, :, :], B[:overlap_y, :, :]):
                    # the last overlap_y rows of A match the first overlap_y rows of B, so pattern B can be below pattern A
                    allow[i][self.DOWN].add(j)
                if np.array_equal(A[:, :overlap_x, :], B[:, Nx-overlap_x:, :]):
                    # the first overlap_x columns of A match the last overlap_x columns of B, so pattern B can be to the left of pattern A
                    allow[i][self.LEFT].add(j)
                if np.array_equal(A[:, Nx-overlap_x:, :], B[:, :overlap_x, :]):
                    # the last overlap_x columns of A match the first overlap_x columns of B, so pattern B can be to the right of pattern A
                    allow[i][self.RIGHT].add(j)
        self.allow = allow
        # convert allow sets to four transition matrices for faster lookup during generation
        self.transition_matrices = {'UP': np.zeros((self.K, self.K), dtype=bool),
                                    'RIGHT': np.zeros((self.K, self.K), dtype=bool),
                                    'DOWN': np.zeros((self.K, self.K), dtype=bool),
                                    'LEFT': np.zeros((self.K, self.K), dtype=bool)}
        for i in range(self.K):
            for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
                for j in self.allow[i][d]:
                    self.transition_matrices[dir_name][i, j] = True
        # example:
        # self.transition_matrices['UP'][i, j] == True means pattern j can be above pattern i
    
    def build_compatibility_for_range_pattern(self):
        # special rules for range pattern data
        # the pattern data is a 3D integer array of shape (N, N, 3)
        # where the last dimension has 3 channels: [motif, height, width]
        # the compatibility rules are:
        # 1. split pattern into 2 sub-patterns: one for motif (Ny, Nx, 1) and one for height/width (Ny, Nx, 2)
        # 2. for motif channel, run standard N-1 overlap compatibility check (exact match on the overlapping region)
        # 3. create allow_motif based on motif compatibility
        # 4. for height/width channels, create sub-patterns by dropping 1 pixel border (shape (Ny-2, Nx-2, 2))
        # 5. based on allow_motif, check patterns that are also compatible in height/width channels with N-1 overlap on the sub-patterns (exact match on the overlapping region)
        # 6. only patterns that satisfy both motif and height/width compatibility are considered compatible
        Ny, Nx = self.Ny, self.Nx
        overlap_y, overlap_x = self.overlap_y, self.overlap_x
        K = self.K
        motif_patterns = [p[:, :, 0] for p in self.idx2pattern]  # (Ny, Nx)
        hw_patterns = [p[1:Ny-1, 1:Nx-1, 1:] for p in self.idx2pattern]  # (Ny-2, Nx-2, 2)
        # step 2: motif compatibility
        allow_motif: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]
        for i, A in enumerate(motif_patterns):
            for j, B in enumerate(motif_patterns):
                if np.array_equal(A[:overlap_y, :], B[Ny-overlap_y:, :]):
                    # the first overlap_y rows of A match the last overlap_y rows of B, so pattern B can be above pattern A 
                    allow_motif[i][self.UP].add(j)
                if np.array_equal(A[Ny-overlap_y:, :], B[:overlap_y, :]):
                    # the last overlap_y rows of A match the first overlap_y rows of B, so pattern B can be below pattern A
                    allow_motif[i][self.DOWN].add(j)
                if np.array_equal(A[:, :overlap_x], B[:, Nx-overlap_x:]):
                    # the first overlap_x columns of A match the last overlap_x columns of B, so pattern B can be to the left of pattern A
                    allow_motif[i][self.LEFT].add(j)
                if np.array_equal(A[:, Nx-overlap_x:], B[:, :overlap_x]):
                    # the last overlap_x columns of A match the first overlap_x columns of B, so pattern B can be to the right of pattern A
                    allow_motif[i][self.RIGHT].add(j)
        # step 5: check height/width compatibility based on allow_motif
        overlap_sub_y = overlap_y - 2
        overlap_sub_x = overlap_x - 2
        allow: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]
        for i in range(K):
            for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
                for j in allow_motif[i][d]:
                    if overlap_sub_y <= 0 and d in (0, 2):
                        # Special case for small N (e.g. N=3): hw_pattern is (1,1,2), no overlap region.
                        # For UP/DOWN adjacency, require width values match (ignore height).
                        if np.array_equal(hw_patterns[i][:, :, 1:], hw_patterns[j][:, :, 1:]):
                            allow[i][d].add(j)
                    elif overlap_sub_x <= 0 and d in (1, 3):
                        # Special case for small N (e.g. N=3): hw_pattern is (1,1,2), no overlap region.
                        # For LEFT/RIGHT adjacency, require height values match (ignore width).
                        if np.array_equal(hw_patterns[i][:, :, 0:1], hw_patterns[j][:, :, 0:1]):
                            allow[i][d].add(j)
                    else:
                        # Standard overlap check on sub-patterns
                        if d==0 and np.array_equal(hw_patterns[i][:overlap_sub_y, :, :], hw_patterns[j][Ny-2-overlap_sub_y:, :, :]):
                            allow[i][self.UP].add(j)
                        if d==2 and np.array_equal(hw_patterns[i][Ny-2-overlap_sub_y:, :, :], hw_patterns[j][:overlap_sub_y, :, :]):
                            allow[i][self.DOWN].add(j)
                        if d==3 and np.array_equal(hw_patterns[i][:, :overlap_sub_x, :], hw_patterns[j][:, Nx-2-overlap_sub_x:, :]):
                            allow[i][self.LEFT].add(j)
                        if d==1 and np.array_equal(hw_patterns[i][:, Nx-2-overlap_sub_x:, :], hw_patterns[j][:, :overlap_sub_x, :]):
                            allow[i][self.RIGHT].add(j)
        self.allow = allow
        # print # of no compatibility patterns in all directions for debugging
        for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
            count_no_compat = sum(1 for i in range(K) if len(allow[i][d]) == 0)
            print(f"Direction {dir_name}: {count_no_compat} patterns have no compatible neighbors")

    # ---- Fast hash-based compatibility methods (O(K) instead of O(K^2)) ----

    @staticmethod
    def _edge_key(arr: np.ndarray) -> bytes:
        """Convert a contiguous array region to a hashable bytes key."""
        return np.ascontiguousarray(arr).tobytes()

    def build_compatibility_fast(self):
        """
        Fast hash-based compatibility building.
        Uses edge hashing and grouping: O(K) instead of O(K^2) comparisons.
        """
        Ny, Nx = self.Ny, self.Nx
        ov_y, ov_x = self.overlap_y, self.overlap_x
        K = self.K
        patterns = self.idx2pattern

        edge_extractors = {
            self.UP:    (lambda p: p[:ov_y, :, :],      lambda p: p[Ny-ov_y:, :, :]),
            self.DOWN:  (lambda p: p[Ny-ov_y:, :, :],   lambda p: p[:ov_y, :, :]),
            self.LEFT:  (lambda p: p[:, :ov_x, :],      lambda p: p[:, Nx-ov_x:, :]),
            self.RIGHT: (lambda p: p[:, Nx-ov_x:, :],   lambda p: p[:, :ov_x, :]),
        }

        allow: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]

        for d in range(4):
            src_extract, tgt_extract = edge_extractors[d]
            tgt_groups = defaultdict(list)
            for j, B in enumerate(patterns):
                key = self._edge_key(tgt_extract(B))
                tgt_groups[key].append(j)
            for i, A in enumerate(patterns):
                key = self._edge_key(src_extract(A))
                if key in tgt_groups:
                    allow[i][d] = set(tgt_groups[key])

        self.allow = allow

        # Build transition matrices
        self.transition_matrices = {
            'UP': np.zeros((K, K), dtype=bool),
            'RIGHT': np.zeros((K, K), dtype=bool),
            'DOWN': np.zeros((K, K), dtype=bool),
            'LEFT': np.zeros((K, K), dtype=bool),
        }
        dir_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        for i in range(K):
            for d, name in enumerate(dir_names):
                for j in allow[i][d]:
                    self.transition_matrices[name][i, j] = True

    def build_compatibility_for_range_pattern_fast(self):
        """
        Fast hash-based compatibility for range patterns.
        Uses edge hashing for both motif and hw sub-pattern checks.
        """
        Ny, Nx = self.Ny, self.Nx
        ov_y, ov_x = self.overlap_y, self.overlap_x
        K = self.K

        motif_patterns = [p[:, :, 0] for p in self.idx2pattern]
        hw_patterns = [p[1:Ny-1, 1:Nx-1, 1:] for p in self.idx2pattern]

        # Step 1: motif compatibility via hashing
        motif_extractors = {
            self.UP:    (lambda m: m[:ov_y, :],     lambda m: m[Ny-ov_y:, :]),
            self.DOWN:  (lambda m: m[Ny-ov_y:, :],  lambda m: m[:ov_y, :]),
            self.LEFT:  (lambda m: m[:, :ov_x],     lambda m: m[:, Nx-ov_x:]),
            self.RIGHT: (lambda m: m[:, Nx-ov_x:],  lambda m: m[:, :ov_x]),
        }

        allow_motif: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]
        for d in range(4):
            src_ex, tgt_ex = motif_extractors[d]
            tgt_groups = defaultdict(list)
            for j, M in enumerate(motif_patterns):
                tgt_groups[self._edge_key(tgt_ex(M))].append(j)
            for i, M in enumerate(motif_patterns):
                key = self._edge_key(src_ex(M))
                if key in tgt_groups:
                    allow_motif[i][d] = set(tgt_groups[key])

        # Step 2: filter by hw sub-pattern compatibility
        ov_sub_y = ov_y - 2
        ov_sub_x = ov_x - 2
        Hy = Ny - 2
        Hx = Nx - 2

        allow: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]

        # UP/DOWN
        if ov_sub_y <= 0:
            hw_width_keys = {}
            for idx, hw in enumerate(hw_patterns):
                hw_width_keys[idx] = self._edge_key(hw[:, :, 1:])
            width_groups = defaultdict(set)
            for idx, k in hw_width_keys.items():
                width_groups[k].add(idx)
            for i in range(K):
                compatible_hw = width_groups[hw_width_keys[i]]
                for d in (self.UP, self.DOWN):
                    allow[i][d] = allow_motif[i][d] & compatible_hw
        else:
            hw_vert_extractors = {
                self.UP:   (lambda h: h[:ov_sub_y, :, :],     lambda h: h[Hy-ov_sub_y:, :, :]),
                self.DOWN: (lambda h: h[Hy-ov_sub_y:, :, :],  lambda h: h[:ov_sub_y, :, :]),
            }
            for d in (self.UP, self.DOWN):
                src_ex, tgt_ex = hw_vert_extractors[d]
                tgt_groups = defaultdict(set)
                for j, hw in enumerate(hw_patterns):
                    tgt_groups[self._edge_key(tgt_ex(hw))].add(j)
                for i in range(K):
                    key = self._edge_key(src_ex(hw_patterns[i]))
                    if key in tgt_groups:
                        allow[i][d] = allow_motif[i][d] & tgt_groups[key]

        # LEFT/RIGHT
        if ov_sub_x <= 0:
            hw_height_keys = {}
            for idx, hw in enumerate(hw_patterns):
                hw_height_keys[idx] = self._edge_key(hw[:, :, 0:1])
            height_groups = defaultdict(set)
            for idx, k in hw_height_keys.items():
                height_groups[k].add(idx)
            for i in range(K):
                compatible_hw = height_groups[hw_height_keys[i]]
                for d in (self.RIGHT, self.LEFT):
                    allow[i][d] = allow_motif[i][d] & compatible_hw
        else:
            hw_horiz_extractors = {
                self.LEFT:  (lambda h: h[:, :ov_sub_x, :],     lambda h: h[:, Hx-ov_sub_x:, :]),
                self.RIGHT: (lambda h: h[:, Hx-ov_sub_x:, :],  lambda h: h[:, :ov_sub_x, :]),
            }
            for d in (self.RIGHT, self.LEFT):
                src_ex, tgt_ex = hw_horiz_extractors[d]
                tgt_groups = defaultdict(set)
                for j, hw in enumerate(hw_patterns):
                    tgt_groups[self._edge_key(tgt_ex(hw))].add(j)
                for i in range(K):
                    key = self._edge_key(src_ex(hw_patterns[i]))
                    if key in tgt_groups:
                        allow[i][d] = allow_motif[i][d] & tgt_groups[key]

        self.allow = allow
        for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
            count_no_compat = sum(1 for i in range(K) if len(allow[i][d]) == 0)
            print(f"Direction {dir_name}: {count_no_compat} patterns have no compatible neighbors")

    def new_build_compatibility_for_range_pattern_fast(self):
        """
        Fast compatibility for range patterns with wildcard zero matching.

        Pattern shape: (m, n, 3) where m>=3, n>=3.
        - Channel 0 (motif): binary (0 or 1), standard N-1 overlap exact match.
        - Channel 1 (Heights): integers, zeros only in first/last row of pattern.
        - Channel 2 (Widths): integers, zeros only in first/last column of pattern.
        N-1 overlap with zero matching any value for channels 1&2.

        Optimization: splits overlap into "exact" positions (no zeros possible,
        hash-based grouping) and "border" positions (zeros possible, wildcard check).
        """
        Ny, Nx = self.Ny, self.Nx
        ov_y, ov_x = self.overlap_y, self.overlap_x
        K = self.K

        motif_patterns = [p[:, :, 0] for p in self.idx2pattern]
        ch12_patterns = [p[:, :, 1:] for p in self.idx2pattern]  # shape (Ny, Nx, 2)

        # Edge extractors for motif (2D) overlap
        motif_extractors = {
            self.UP:    (lambda m: m[:ov_y, :],     lambda m: m[Ny-ov_y:, :]),
            self.DOWN:  (lambda m: m[Ny-ov_y:, :],  lambda m: m[:ov_y, :]),
            self.LEFT:  (lambda m: m[:, :ov_x],     lambda m: m[:, Nx-ov_x:]),
            self.RIGHT: (lambda m: m[:, Nx-ov_x:],  lambda m: m[:, :ov_x]),
        }

        # Edge extractors for channels 1&2 (3D) overlap
        ch12_extractors = {
            self.UP:    (lambda c: c[:ov_y, :, :],     lambda c: c[Ny-ov_y:, :, :]),
            self.DOWN:  (lambda c: c[Ny-ov_y:, :, :],  lambda c: c[:ov_y, :, :]),
            self.LEFT:  (lambda c: c[:, :ov_x, :],     lambda c: c[:, Nx-ov_x:, :]),
            self.RIGHT: (lambda c: c[:, Nx-ov_x:, :],  lambda c: c[:, :ov_x, :]),
        }

        # Build zero-possible masks for each direction.
        # Heights (ch12[:,:,0]): zeros only at row 0 and row Ny-1 of full pattern.
        # Widths (ch12[:,:,1]): zeros only at col 0 and col Nx-1 of full pattern.
        def build_zero_masks(d):
            if d in (self.UP, self.DOWN):
                shape = (ov_y, Nx, 2)
            else:
                shape = (Ny, ov_x, 2)
            src_zmask = np.zeros(shape, dtype=bool)
            tgt_zmask = np.zeros(shape, dtype=bool)

            if d == self.UP:
                # src = rows[0:ov_y] of pattern i
                src_zmask[0, :, 0] = True            # Heights: first row of pattern
                src_zmask[:, 0, 1] = True            # Widths: first col
                src_zmask[:, Nx-1, 1] = True         # Widths: last col
                # tgt = rows[Ny-ov_y:Ny] of pattern j
                tgt_zmask[ov_y-1, :, 0] = True       # Heights: last row of pattern
                tgt_zmask[:, 0, 1] = True
                tgt_zmask[:, Nx-1, 1] = True
            elif d == self.DOWN:
                # src = rows[Ny-ov_y:Ny] of pattern i
                src_zmask[ov_y-1, :, 0] = True       # Heights: last row of pattern
                src_zmask[:, 0, 1] = True
                src_zmask[:, Nx-1, 1] = True
                # tgt = rows[0:ov_y] of pattern j
                tgt_zmask[0, :, 0] = True            # Heights: first row of pattern
                tgt_zmask[:, 0, 1] = True
                tgt_zmask[:, Nx-1, 1] = True
            elif d == self.LEFT:
                # src = cols[0:ov_x] of pattern i
                src_zmask[0, :, 0] = True            # Heights: first row
                src_zmask[Ny-1, :, 0] = True         # Heights: last row
                src_zmask[:, 0, 1] = True            # Widths: first col of pattern
                # tgt = cols[Nx-ov_x:Nx] of pattern j
                tgt_zmask[0, :, 0] = True
                tgt_zmask[Ny-1, :, 0] = True
                tgt_zmask[:, ov_x-1, 1] = True      # Widths: last col of pattern
            elif d == self.RIGHT:
                # src = cols[Nx-ov_x:Nx] of pattern i
                src_zmask[0, :, 0] = True
                src_zmask[Ny-1, :, 0] = True
                src_zmask[:, ov_x-1, 1] = True      # Widths: last col of pattern
                # tgt = cols[0:ov_x] of pattern j
                tgt_zmask[0, :, 0] = True
                tgt_zmask[Ny-1, :, 0] = True
                tgt_zmask[:, 0, 1] = True            # Widths: first col of pattern

            return src_zmask, tgt_zmask

        allow: List[List[set]] = [[set() for _ in range(4)] for _ in range(K)]

        for d in range(4):
            motif_src_ex, motif_tgt_ex = motif_extractors[d]
            ch12_src_ex, ch12_tgt_ex = ch12_extractors[d]

            src_zmask, tgt_zmask = build_zero_masks(d)
            any_zmask = src_zmask | tgt_zmask
            exact_mask = ~any_zmask
            has_exact = np.any(exact_mask)
            has_border = np.any(any_zmask)

            # Build target groups keyed by (motif_edge + ch12_exact_positions)
            tgt_groups = defaultdict(list)
            for j in range(K):
                motif_key = self._edge_key(motif_tgt_ex(motif_patterns[j]))
                if has_exact:
                    ch12_tgt = ch12_tgt_ex(ch12_patterns[j])
                    exact_key = np.ascontiguousarray(ch12_tgt[exact_mask]).tobytes()
                    combined_key = motif_key + exact_key
                else:
                    combined_key = motif_key
                tgt_groups[combined_key].append(j)

            # Pre-extract border values for wildcard check
            if has_border:
                ch12_src_borders = []
                ch12_tgt_borders = []
                for idx in range(K):
                    src_edge = ch12_src_ex(ch12_patterns[idx])
                    tgt_edge = ch12_tgt_ex(ch12_patterns[idx])
                    ch12_src_borders.append(np.ascontiguousarray(src_edge[any_zmask]))
                    ch12_tgt_borders.append(np.ascontiguousarray(tgt_edge[any_zmask]))

            # Match src patterns against target groups
            for i in range(K):
                motif_key = self._edge_key(motif_src_ex(motif_patterns[i]))
                if has_exact:
                    ch12_src = ch12_src_ex(ch12_patterns[i])
                    exact_key = np.ascontiguousarray(ch12_src[exact_mask]).tobytes()
                    combined_key = motif_key + exact_key
                else:
                    combined_key = motif_key

                if combined_key in tgt_groups:
                    if has_border:
                        src_border = ch12_src_borders[i]
                        for j in tgt_groups[combined_key]:
                            tgt_border = ch12_tgt_borders[j]
                            if np.all((src_border == 0) | (tgt_border == 0) | (src_border == tgt_border)):
                                allow[i][d].add(j)
                    else:
                        # No border positions → exact hash fully determines compatibility
                        allow[i][d] = set(tgt_groups[combined_key])

        self.allow = allow
        for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
            count_no_compat = sum(1 for i in range(K) if len(allow[i][d]) == 0)
            print(f"Direction {dir_name}: {count_no_compat} patterns have no compatible neighbors")

    # ---- Transition-matrix-only compatibility builders ----
    # These produce self.T (shape (4, K, K), bool) directly without
    # intermediate Python sets. Much faster for large K.

    def build_compatibility_matrix(self):
        """
        Build compatibility as a numpy transition matrix directly.
        Uses edge hashing (O(K)) and vectorized numpy assignment.
        Result: self.T of shape (4, K, K), dtype bool.
        T[d, i, j] = True means pattern j can be in direction d from pattern i.
        """
        Ny, Nx = self.Ny, self.Nx
        ov_y, ov_x = self.overlap_y, self.overlap_x
        K = self.K
        patterns = self.idx2pattern

        edge_extractors = {
            self.UP:    (lambda p: p[:ov_y, :, :],      lambda p: p[Ny-ov_y:, :, :]),
            self.DOWN:  (lambda p: p[Ny-ov_y:, :, :],   lambda p: p[:ov_y, :, :]),
            self.LEFT:  (lambda p: p[:, :ov_x, :],      lambda p: p[:, Nx-ov_x:, :]),
            self.RIGHT: (lambda p: p[:, Nx-ov_x:, :],   lambda p: p[:, :ov_x, :]),
        }

        T = np.zeros((4, K, K), dtype=bool)
        for d in range(4):
            src_extract, tgt_extract = edge_extractors[d]
            tgt_groups = defaultdict(list)
            for j, B in enumerate(patterns):
                key = self._edge_key(tgt_extract(B))
                tgt_groups[key].append(j)
            for i, A in enumerate(patterns):
                key = self._edge_key(src_extract(A))
                if key in tgt_groups:
                    T[d, i, tgt_groups[key]] = True

        self.T = T
        for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
            no_compat = np.sum(T[d].sum(axis=1) == 0)
            print(f"Direction {dir_name}: {no_compat} patterns have no compatible neighbors")

    def build_compatibility_for_range_pattern_matrix(self):
        """
        Build range-pattern compatibility as a numpy transition matrix directly.
        Two-stage filter: motif overlap + height/width sub-pattern check.
        Result: self.T of shape (4, K, K), dtype bool.
        """
        Ny, Nx = self.Ny, self.Nx
        ov_y, ov_x = self.overlap_y, self.overlap_x
        K = self.K

        motif_patterns = [p[:, :, 0] for p in self.idx2pattern]
        hw_patterns = [p[1:Ny-1, 1:Nx-1, 1:] for p in self.idx2pattern]

        # Stage 1: motif compatibility via hashing → T_motif
        motif_extractors = {
            self.UP:    (lambda m: m[:ov_y, :],     lambda m: m[Ny-ov_y:, :]),
            self.DOWN:  (lambda m: m[Ny-ov_y:, :],  lambda m: m[:ov_y, :]),
            self.LEFT:  (lambda m: m[:, :ov_x],     lambda m: m[:, Nx-ov_x:]),
            self.RIGHT: (lambda m: m[:, Nx-ov_x:],  lambda m: m[:, :ov_x]),
        }

        T_motif = np.zeros((4, K, K), dtype=bool)
        for d in range(4):
            src_ex, tgt_ex = motif_extractors[d]
            tgt_groups = defaultdict(list)
            for j, M in enumerate(motif_patterns):
                tgt_groups[self._edge_key(tgt_ex(M))].append(j)
            for i, M in enumerate(motif_patterns):
                key = self._edge_key(src_ex(M))
                if key in tgt_groups:
                    T_motif[d, i, tgt_groups[key]] = True

        # Stage 2: filter by hw sub-pattern compatibility
        ov_sub_y = ov_y - 2
        ov_sub_x = ov_x - 2
        Hy = Ny - 2
        Hx = Nx - 2

        T = np.zeros((4, K, K), dtype=bool)

        # UP/DOWN filtering
        if ov_sub_y <= 0:
            # Group by width-key (hw[:, :, 1:])
            width_group_ids = np.empty(K, dtype=np.int32)
            key_to_id = {}
            for idx, hw in enumerate(hw_patterns):
                key = self._edge_key(hw[:, :, 1:])
                if key not in key_to_id:
                    key_to_id[key] = len(key_to_id)
                width_group_ids[idx] = key_to_id[key]
            # width_compat[i, j] = True if same width group
            width_compat = (width_group_ids[:, None] == width_group_ids[None, :])
            for d in (self.UP, self.DOWN):
                T[d] = T_motif[d] & width_compat
        else:
            hw_vert_extractors = {
                self.UP:   (lambda h: h[:ov_sub_y, :, :],     lambda h: h[Hy-ov_sub_y:, :, :]),
                self.DOWN: (lambda h: h[Hy-ov_sub_y:, :, :],  lambda h: h[:ov_sub_y, :, :]),
            }
            for d in (self.UP, self.DOWN):
                src_ex, tgt_ex = hw_vert_extractors[d]
                tgt_groups = defaultdict(list)
                for j, hw in enumerate(hw_patterns):
                    tgt_groups[self._edge_key(tgt_ex(hw))].append(j)
                for i in range(K):
                    key = self._edge_key(src_ex(hw_patterns[i]))
                    if key in tgt_groups:
                        js = tgt_groups[key]
                        # Only keep j's that also pass motif check
                        mask = T_motif[d, i, js]
                        compatible_js = [js[idx] for idx, m in enumerate(mask) if m]
                        if compatible_js:
                            T[d, i, compatible_js] = True

        # LEFT/RIGHT filtering
        if ov_sub_x <= 0:
            # Group by height-key (hw[:, :, 0:1])
            height_group_ids = np.empty(K, dtype=np.int32)
            key_to_id = {}
            for idx, hw in enumerate(hw_patterns):
                key = self._edge_key(hw[:, :, 0:1])
                if key not in key_to_id:
                    key_to_id[key] = len(key_to_id)
                height_group_ids[idx] = key_to_id[key]
            # height_compat[i, j] = True if same height group
            height_compat = (height_group_ids[:, None] == height_group_ids[None, :])
            for d in (self.RIGHT, self.LEFT):
                T[d] = T_motif[d] & height_compat
        else:
            hw_horiz_extractors = {
                self.LEFT:  (lambda h: h[:, :ov_sub_x, :],     lambda h: h[:, Hx-ov_sub_x:, :]),
                self.RIGHT: (lambda h: h[:, Hx-ov_sub_x:, :],  lambda h: h[:, :ov_sub_x, :]),
            }
            for d in (self.RIGHT, self.LEFT):
                src_ex, tgt_ex = hw_horiz_extractors[d]
                tgt_groups = defaultdict(list)
                for j, hw in enumerate(hw_patterns):
                    tgt_groups[self._edge_key(tgt_ex(hw))].append(j)
                for i in range(K):
                    key = self._edge_key(src_ex(hw_patterns[i]))
                    if key in tgt_groups:
                        js = tgt_groups[key]
                        mask = T_motif[d, i, js]
                        compatible_js = [js[idx] for idx, m in enumerate(mask) if m]
                        if compatible_js:
                            T[d, i, compatible_js] = True

        self.T = T
        for d, dir_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
            no_compat = np.sum(T[d].sum(axis=1) == 0)
            print(f"Direction {dir_name}: {no_compat} patterns have no compatible neighbors")