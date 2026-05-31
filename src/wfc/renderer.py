import os
import numpy as np
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt


class RangePatternRenderer:
    """Renderer the WFC output grid to real geometry pattern."""
    def __init__(self, patterns, index_to_width, index_to_height, context_size=50):
        self.patterns = patterns
        self.index_to_width = index_to_width
        self.index_to_height = index_to_height
        self.index_to_width[0] = context_size
        self.index_to_height[0] = context_size
    
    def pattern_decode(self, pattern, remove_context=False):
        """Decode the encoded pattern into motif, heights, and widths.
        pattern: int array of shape (m, n, 3)
        """
        if remove_context:
            # remove the 1-pixel context border around the motif, and the corresponding heights and widths
            pattern = pattern[1:-1, 1:-1, :]
        motif, H, W = pattern[:, :, 0], pattern[:, :, 1], pattern[:, :, 2]
        enc_heights = H[:, 1]  # take the first column of heights, shape (m,)
        enc_widths = W[1, :]  # take the first row of widths, shape (n,)
        heights = np.array([self.index_to_height[h] for h in enc_heights]) # real dimension (nm)
        widths = np.array([self.index_to_width[w] for w in enc_widths]) # real dimension (nm)
        return motif, heights, widths
   
    def pattern_to_bbox_list(self, pattern):
        """Convert the decoded pattern into a list of bounding boxes for "on" cells in the motif, for GDS export. Each bbox is [llx, lly, urx, ury]. The coordinate system is with origin at lower left, x increases to the right, y increases upwards.
        return: list of bboxes, each bbox is [llx, lly, urx, ury]
        """
        
        motif, heights, widths = self.pattern_decode(pattern, remove_context=True)
        # up-down filp for motif and heights to match the coordinate system of GDS (origin at lower left, y increases upwards)
        motif = np.flipud(motif)
        heights = np.flipud(heights)
        heights = np.insert(heights, 0, 0) # insert zero for edge coordinates, so the first pattern cell starts at height 0. The last pattern cell will end at the last cumulative height value, which is the total height of the synthesized pattern.
        widths = np.insert(widths, 0, 0)
        h_cum = np.cumsum(heights)
        w_cum = np.cumsum(widths)
        bboxes = []
        for r, c in zip(*np.where(motif == 1)):
            llx = float(w_cum[c])
            lly = float(h_cum[r])
            urx = float(w_cum[c + 1])
            ury = float(h_cum[r + 1])
            bboxes.append([llx, lly, urx, ury])
        return bboxes
 
    def plot_output_pattern(self, pattern, seed, savedir=None):
        motif, heights, widths = self.pattern_decode(pattern)
        heights = np.insert(heights, 0, 0) # insert zero for pcolormesh
        widths = np.insert(widths, 0, 0)
        h_cum = np.cumsum(heights)
        w_cum = np.cumsum(widths)
        # meshgrid for pcolormesh, H and W are the coordinates of the edges of the cells, 
        # so they should have one more element than the number of cells in each dimension
        H, W = np.meshgrid(h_cum, w_cum, indexing='ij') 
        # Separate content (inner) and context (1-pixel border)
        # Content: motif[1:-1, 1:-1], Context: 1-pixel border
        # Render context in lighter grey: 0 -> 0.3, 1 -> 0.7; content stays 0/1
        display = np.copy(motif).astype(float)
        # Create border mask
        border_mask = np.zeros_like(motif, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        # Map border pixels to lighter grey levels
        display[border_mask & (motif == 0)] = 0.3
        display[border_mask & (motif == 1)] = 0.7
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
        axes = axes.flatten()
        axes[0].pcolormesh(W, H, display, cmap='gray', vmin=0, vmax=1)
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].invert_yaxis()
        # Ticks at first and last edge only (0 and total sum)
        axes[0].set_xticks([w_cum[1], w_cum[-2]])
        axes[0].set_xticklabels([f"{0}", f"{w_cum[-2]-w_cum[1]:.1f}"], fontsize=10)
        axes[0].set_yticks([h_cum[1], h_cum[-2]])
        axes[0].set_yticklabels([f"{0}", f"{h_cum[-2]-h_cum[1]:.1f}"], fontsize=10)
        axes[0].set_xlabel("width (nm)", fontsize=10)
        axes[0].set_ylabel("height (nm)", fontsize=10)
        axes[0].set_title(f"WFC output, seed={seed}")
        # show motif
        axes[1].axis('off') 
        axes[1].pcolormesh(display, cmap='gray', vmin=0, vmax=1, edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].invert_yaxis()
        w_str = ' '.join(f"{v:.1f}" for v in widths[2:-1])
        h_str = ' '.join(f"{v:.1f}" for v in heights[2:-1])
        axes[1].set_title(f"motif {motif.shape}\nw=[{w_str}]\nh=[{h_str}]", fontsize=8, loc='left', wrap=True)
        # save figure
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(os.path.join(savedir, f"pattern_seed{seed}.png"), dpi=300)
        return fig, axes
    
    def plot_input_patterns_mxn(self, idx_arr):
        idx_arr = np.array(idx_arr)
        m, n = idx_arr.shape
        fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(6, 6))
        axes = [axes] if m==1 and n==1 else axes.flatten()
        for i, pat_idx in enumerate(idx_arr.flatten()):
            pattern = self.patterns[pat_idx]
            motif, heights, widths = self.pattern_decode(pattern)
            heights = np.insert(heights, 0, 0) # insert zero for pcolormesh
            widths = np.insert(widths, 0, 0)
            h_cum = np.cumsum(heights)
            w_cum = np.cumsum(widths)
            display = np.copy(motif).astype(float)
            border_mask = np.zeros_like(motif, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            display[border_mask & (motif == 0)] = 0.3
            display[border_mask & (motif == 1)] = 0.7
            H, W = np.meshgrid(h_cum, w_cum, indexing='ij')
            ax = axes[i]
            ax.pcolormesh(W, H, display, cmap='gray', vmin=0, vmax=1, edgecolors='blue', linewidth=0.5, linestyle='dashed', shading='auto')
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.invert_yaxis()
            # set title text smaller and include width and height values
            w_str = ' '.join(f"{v:.1f}" for v in widths[2:-1])
            h_str = ' '.join(f"{v:.1f}" for v in heights[2:-1])
            ax.set_title(f"Pattern ID {pat_idx} \n w=[{w_str}], h=[{h_str}]", fontsize=8)
        plt.tight_layout()
    
    def grid_to_encoded_pattern(self, grid):
        # grid is (cell_h, cell_w) of pattern indices
        cell_h, cell_w = grid.shape
        # get synthesized widths
        enc_widths = []
        for i, pat_idx in enumerate(grid[0]):
            p = self.patterns[pat_idx]
            if i==0:
                w = p[1, 1:-1, 2]
                enc_widths.extend(w)
            else:
                w = p[1, -2, 2]
                enc_widths.append(w)
        enc_widths = np.pad(enc_widths, pad_width=1) # pad zeros on both sides
        # get synthesized heights
        enc_heights = []
        for j, pat_idx in enumerate(grid[:, 0]):
            p = self.patterns[pat_idx]
            if j==0:
                h = p[1:-1, 1, 1]
                enc_heights.extend(h)
            else:
                h = p[-2, 1, 1]
                enc_heights.append(h)
        enc_heights = np.pad(enc_heights, pad_width=1) # pad zeros on both sides
        # get synthesized motif by blending overlapping pixels
        motifs = [p[:, :, 0] for p in self.patterns]
        motif = np.array(motifs)[grid]  # (cell_h, cell_w, Ny, Nx) fancy indexing, 4D array of motifs for each cell
        motif = np.block(
            [[motif[0, 0],         motif[0, 1:, :, -1].T],
             [motif[1:, 0, -1, :], motif[1:, 1:, -1, -1]]]
            ) # 2D array of blended motifs, shape (Ny+cell_h-1, Nx+cell_w-1)
        #print("motif:\n", motif, motif.shape)
        #print("widths:", widths)
        #print("heights:", heights) 
        enc_H, enc_W = np.meshgrid(enc_heights, enc_widths, indexing='ij')
        encoded_pattern = np.stack([motif, enc_H, enc_W], axis=-1)
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