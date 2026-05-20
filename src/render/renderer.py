from statistics import mode

import numpy as np
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt


class RangePatternRenderer:
    """Renderer for range patterns, which are defined by width and height values rather than pixel arrays."""
    def __init__(self, library):
        self.lib = library
        self.patterns = library.idx2pattern
        self.index_to_width = library.index_to_width
        self.index_to_height = library.index_to_height
        self.index_to_width[0] = self.index_to_width[1]  # default width for pattern index 0
        self.index_to_height[0] = self.index_to_height[1]  # default height for pattern index 0
    
    def _decode_sinple(self, pat_idx):
        p = self.patterns[pat_idx] # [N, N, 3]
        motif = p[:, :, 0]
        heights = p[:, 1, 1] 
        widths = p[1, :, 2]
        heights_decode = np.array([self.index_to_height[h] for h in heights])
        widths_decode = np.array([self.index_to_width[w] for w in widths])
        heights_decode = np.insert(heights_decode, 0, 0)
        widths_decode = np.insert(widths_decode, 0, 0)
        h_cum = np.cumsum(heights_decode)
        w_cum = np.cumsum(widths_decode)
        H, W = np.meshgrid(h_cum, w_cum, indexing='ij')
        return motif, H, W, heights_decode, widths_decode
    
    def plot_patterns_mxn(self, idx_arr):
        m, n = idx_arr.shape
        fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(6, 6))
        axes = axes.flatten()
        for i, pat_idx in enumerate(idx_arr.flatten()):
            motif, H, W, heights_decode, widths_decode = self._decode_sinple(pat_idx)
            ax = axes[i]
            ax.pcolormesh(W, H, motif, edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.invert_yaxis()
            # set title text smaller and include width and height values
            ax.set_title(f"Pattern ID {pat_idx} \n w={widths_decode[2:-1]}, h={heights_decode[2:-1]}", fontsize=8)
        plt.tight_layout()
    
    def plot_encoded_patterns_mxn(self, idx_arr):
        m, n = idx_arr.shape
        fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(6, 6))
        axes = axes.flatten()
        for i, pat_idx in enumerate(idx_arr.flatten()):
            p = self.patterns[pat_idx]
            p = p.astype(np.uint8)
            ax = axes[i]
            ax.pcolormesh(p, edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.invert_yaxis()
            ax.set_title(f"Pattern ID {pat_idx}", fontsize=8)
        plt.tight_layout()
        
    def grid_to_pattern(self, grid):
        # grid is (cell_h, cell_w) of pattern indices
        cell_h, cell_w = grid.shape
        # get synthesized widths
        widths = []
        for i, pat_idx in enumerate(grid[0]):
            p = self.patterns[pat_idx]
            if i==0:
                w = p[1, 1:-1, 2]
                widths.extend(w)
            else:
                w = p[1, -2, 2]
                widths.append(w)
        widths = np.pad(widths, pad_width=1) # pad zeros on both sides
        # get synthesized heights
        heights = []
        for j, pat_idx in enumerate(grid[:, 0]):
            p = self.patterns[pat_idx]
            if j==0:
                h = p[1:-1, 1, 1]
                heights.extend(h)
            else:
                h = p[-2, 1, 1]
                heights.append(h)
        heights = np.pad(heights, pad_width=1) # pad zeros on both sides
        # get synthesized motif by blending overlapping pixels
        motifs = [p[:, :, 0] for p in self.patterns]
        motif = np.array(motifs)[grid]  # (cell_h, cell_w, Ny, Nx) fancy indexing, 4D array of motifs for each cell
        motif = np.block(
            [[motif[0, 0],         motif[0, 1:, :, -1].T],
             [motif[1:, 0, -1, :], motif[1:, 1:, -1, -1]]]
            ) # 2D array of blended motifs, shape (Ny+cell_h-1, Nx+cell_w-1)
        assert motif.shape == (self.lib.Ny + cell_h - 1, self.lib.Nx + cell_w - 1), f"Unexpected motif shape {motif.shape}"
        #print("motif:\n", motif, motif.shape)
        #print("widths:", widths)
        #print("heights:", heights) 
        #motif = motif.transpose(0,2,1,3) # (cell_h,Ny,cell_w,Nx)
        #motif = motif.reshape(cell_h*motif.shape[1], cell_w*motif.shape[3]) # (cell_h*Ny, cell_w*Nx)
        #motif = np.concatenate([motif[0::self.lib.Ny, :], motif[-(self.lib.Ny-1):, :]], axis=0) # remove duplicated rows
        #motif = np.concatenate([motif[:, 0::self.lib.Nx], motif[:, -(self.lib.Nx-1):]], axis=1) # remove duplicated cols
        
        H, W = np.meshgrid(heights, widths, indexing='ij')
        encoded_pattern = np.stack([motif, H, W], axis=-1)
        return encoded_pattern
    
    def pattern_decode_plot(self, encoded_pattern, seed, edges=False):
        motif = encoded_pattern[:, :, 0]
        heights = encoded_pattern[:, 1, 1]
        widths = encoded_pattern[1, :, 2]
        height_decode = np.array([self.index_to_height[h] for h in heights])
        width_decode = np.array([self.index_to_width[w] for w in widths])
        height_decode = np.insert(height_decode, 0, 0) # insert zero at the beginning to represent for pcolormesh, so the first pattern cell starts at height 0. The last pattern cell will end at the last cumulative height value, which is the total height of the synthesized pattern.
        width_decode = np.insert(width_decode, 0, 0)
        h_cum = np.cumsum(height_decode)
        w_cum = np.cumsum(width_decode)
        H, W = np.meshgrid(h_cum, w_cum, indexing='ij')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))
        axes = axes.flatten()
        if edges:
            axes[0].pcolormesh(W, H, motif, edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
        else:
            axes[0].pcolormesh(W, H, motif)
            
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].invert_yaxis()
        # Ticks at first and last edge only (0 and total sum)
        axes[0].set_xticks([w_cum[1], w_cum[-2]])
        axes[0].set_xticklabels([f"{0}", f"{w_cum[-2]-w_cum[1]:.1f}"], fontsize=7)
        axes[0].set_yticks([h_cum[1], h_cum[-2]])
        axes[0].set_yticklabels([f"{0}", f"{h_cum[-2]-h_cum[1]:.1f}"], fontsize=7)
        axes[0].set_title(f"WFC output, seed={seed}")
        axes[1].pcolormesh(motif, cmap='gray', edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].axis('off')
        axes[1].invert_yaxis()
        axes[1].set_title(f"w={width_decode[2:-1]}, \n h={height_decode[2:-1]}", fontsize=8)