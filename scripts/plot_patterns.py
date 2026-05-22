from pathlib import Path
import os, sys

SRC_DIR = Path(__file__).parent.parent / "src"
if SRC_DIR not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wfc.renderer import RangePatternRenderer


def print_patterns_stats(patterns):
    motif = patterns[:, :, :, 0]
    motif_content = motif[:, 1:-1, 1:-1]
    patterns_content = patterns[:, 1:-1, 1:-1, :]
    unique_patterns_with_context = np.unique(patterns, axis=0)
    unique_patterns_content = np.unique(patterns_content, axis=0)
    unique_motifs_with_context = np.unique(motif, axis=0)
    unique_motifs_content = np.unique(motif_content, axis=0)
    
    print("unique patterns (including context):", unique_patterns_with_context.shape[0])
    print("unique patterns (content only):", unique_patterns_content.shape[0])
    print("unique motifs (including context):", unique_motifs_with_context.shape[0])
    print("unique motifs (content only):", unique_motifs_content.shape[0])

def save_all_to_pdf(png_files, savedir):
    """Save all pattern PNGs to a single PDF, 4 figures per page (2x2 grid)."""
    pdf_path = os.path.join(savedir, "all_patterns.pdf")
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(png_files), 4):
            page_imgs = png_files[page_start:page_start+4]
            fig_page, axes_page = plt.subplots(2, 2, figsize=(16, 12))
            axes_page = axes_page.flatten()
            for j, img_path in enumerate(page_imgs):
                img = plt.imread(img_path)
                axes_page[j].imshow(img)
                axes_page[j].axis('off')
            for j in range(len(page_imgs), 4):
                axes_page[j].axis('off')
            fig_page.tight_layout()
            pdf.savefig(fig_page)
            plt.close(fig_page)
    print(f"Saved all patterns to {pdf_path}")

def main():
    input_file = sys.argv[1]
    savedir = os.path.dirname(input_file)
    data = np.load(input_file, allow_pickle=True)
    print(list(data))
    patterns = data['patterns']
    seeds = data['seeds']
    index_to_width = data['index_to_width'][()]
    index_to_height = data['index_to_height'][()]
    print_patterns_stats(patterns)
    
    renderer = RangePatternRenderer(patterns, index_to_width, index_to_height, context_size=20)
    png_files = []
    for i in range(len(patterns)):
        if i > 10: break
        fig, axes = renderer.plot_output_pattern(patterns[i], seeds[i])
        #png_files.append(os.path.join(savedir, f"pattern_seed{seeds[i]}.png"))
        #plt.close(fig)

    #save_all_to_pdf(png_files, savedir)
        
    plt.show()
    
if __name__ == '__main__':
    main()