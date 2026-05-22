import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from PIL import Image
from .patterns import PatternLibrary
from .wfc_solver import WFCSolver
from .mrf_solver import MRFSolver
from .renderer import RangePatternRenderer

SOLVERS = {
    'wfc': WFCSolver,
    'mrf': MRFSolver,
}

def parse_args():
    p = argparse.ArgumentParser(description='Overlap WFC refactored CLI')
    p.add_argument('--input_mode', type=str, choices=['png', 'range_patterns'], required=True,
                   help='Input mode: "png" for PNG bitmap extraction, "range_patterns" for pre-extracted npz patterns')
    p.add_argument('--input_file', type=str, required=True,
                   help='Path to input file (PNG for png mode, npz for range_patterns mode)')
    p.add_argument('--compatibility', type=str, default=None, 
                   help='Optional precomputed compatibility lookup table npz file')
    p.add_argument('--pattern_rank', type=str, default='3x3', 
                   help='Pattern rank to use from npz file if --input_patterns is used (default 3x3)')
    p.add_argument('--pattern_size', type=int, nargs='+', default=[3], 
                   help='Pattern size: single int for square (NxN) or two ints for [Ny Nx] (default 3)')
    p.add_argument('--overlap', type=int, nargs='+', default=None, 
                   help='Pixel overlap: single int or two ints [overlap_y overlap_x] (default N-1)')
    p.add_argument('--output_size', type=int, nargs=2, default=[2,2], 
                   help='Output cell grid size in (H W), default 2 2')
    p.add_argument('--solver', choices=SOLVERS.keys(), default='wfc', 
                   help='Solver algorithm to synthesize a bigger image (default wfc)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--augment_rot_reflect', action='store_true')
    p.add_argument('--periodic_input', action='store_true')
    p.add_argument('--start_token', type=int, default=None)
    p.add_argument('--num_outputs', type=int, default=1, 
                   help='Number of outputs to generate with different random seeds (default 1; set 10 for batch)')
    p.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs') 
    return p.parse_args()

def main():
    args = parse_args()
    # Normalize pattern_size to tuple (Ny, Nx)
    if len(args.pattern_size) == 1:
        pattern_size = (args.pattern_size[0], args.pattern_size[0])
    elif len(args.pattern_size) == 2:
        pattern_size = tuple(args.pattern_size)
    else:
        raise ValueError("--pattern_size must be 1 or 2 integers")
    # Normalize overlap to tuple (overlap_y, overlap_x)
    if args.overlap is None:
        overlap = None  # let PatternLibrary decide default
    elif len(args.overlap) == 1:
        overlap = (args.overlap[0], args.overlap[0])
    elif len(args.overlap) == 2:
        overlap = tuple(args.overlap)
    else:
        raise ValueError("--overlap must be 1 or 2 integers")
    # Derive use_range_pattern from input_mode
    use_range_pattern = (args.input_mode == 'range_patterns')
    # Prepare output directory based on input filename
    input_file = args.input_file
    if args.output_dir is None:
        raise ValueError("Output directory must be specified with --output_dir to save results.")
    output_dir = os.path.join(args.output_dir, args.pattern_rank)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.input_mode == 'range_patterns':
        # load pre-extracted patterns and frequencies from npz
        data = np.load(args.input_file, allow_pickle=True)
        lib = PatternLibrary.from_extracted_patterns(data, pattern_rank=args.pattern_rank)
    else:
        # standard WFC pattern extraction from input PNG (color bitmap)
        lib = PatternLibrary.from_png(
            args.input_file,
            N=pattern_size,
            overlap=overlap,
            periodic_input=args.periodic_input,
            augment_rot_reflect=args.augment_rot_reflect,
        )
        # save extracted patterns and frequencies to npz for later reuse
        np.savez_compressed(
            os.path.join(output_dir, f'{lib.Ny}x{lib.Nx}_patterns_{lib.K}.npz'),
            patterns=np.stack(lib.idx2pattern, axis=0),
            freqs=lib.freqs,
        )

    if args.compatibility is None:
        # print time elapsed for compatibility building to console
        start_time = time.time()
        if use_range_pattern:
            print("Building compatibility using FAST range pattern representation...")
            lib.build_compatibility_for_range_pattern_fast()
        else:
            print("Building compatibility using FAST standard pattern overlap...")
            lib.build_compatibility_fast()
        end_time = time.time()
        print(f"Compatibility building completed in {end_time - start_time:.2f} seconds.")
        # save compatibility lookup table to npz for later reuse
        np.savez_compressed(
            os.path.join(output_dir, f"compatibility_{args.pattern_rank}_{lib.K}.npz"),
            allow=lib.allow,
        )
    else:
        # load precomputed compatibility lookup table
        data = np.load(args.compatibility, allow_pickle=True)
        lib.allow = data['allow']
    # print total number of patterns that has no compatible neighbors in four directions
    for d in range(4):
        no_compat_count = sum(1 for i in range(lib.K) if len(lib.allow[i][d]) == 0)
        print(f"Direction {d}: {no_compat_count} patterns have no compatible neighbors.")
    cell_h, cell_w = args.output_size  # number of pattern cells in output grid
    
    solver_cls = SOLVERS[args.solver]
    solver = solver_cls(lib)
    
    # Batch generation: generate num_outputs, each with up to max_retries attempts
    num = max(1, int(args.num_outputs))
    max_retries = 1000
    output_grids = []
    seeds = []
    time_log = pd.DataFrame(columns=['output_index', 'attempt', 'seed', 'duration_sec', 'status'])  
    # for detailed logging of each attempt

    dateID = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, dateID)
    os.makedirs(output_dir, exist_ok=True)
    for out_idx in range(num):
        success = False
        for retry in range(max_retries):
            if args.seed is not None:
                seed = int(args.seed) + out_idx * max_retries + retry
            else:
                seed = int(np.random.randint(0, 2**31 - 1))
            t_attempt = time.time()
            try:
                grid = solver.solve(cell_h, cell_w, seed=int(seed), start_token=args.start_token)
                dt = time.time() - t_attempt
                time_log.loc[len(time_log)] = [out_idx, retry+1, seed, dt, 'success']                
                output_grids.append(grid)
                seeds.append(seed)
                print(f"[{out_idx}/{num}] Success on attempt {retry+1} with seed {seed} ({dt:.3f}s)")
                success = True
                break
            except Exception:
                dt = time.time() - t_attempt
                time_log.loc[len(time_log)] = [out_idx, retry+1, seed, dt, 'fail']
        if not success:
            print(f"[{out_idx+1}/{num}] FAILED after {max_retries} attempts")

    print(time_log)
    # Print time log summary (now time_log is a detailed dataframe)
    # for each output_index, print final status, seed, the number of attempts, and accumulated time spent
    summary = time_log.groupby('output_index').agg(
        final_status=('status', lambda x: 'success' if 'success' in x.values else 'fail'),
        seed=('seed', 'last'),
        attempts=('attempt', 'max'),
        total_time_sec=('duration_sec', 'sum')
    ).reset_index()
    print("\nSummary of generation attempts:")
    print(summary)
    # save time_log to csv and summary to .txt
    time_log.to_csv(os.path.join(output_dir, "generation_time_log.csv"), index=False)
    with open(os.path.join(output_dir, "generation_summary.txt"), "w") as f:
        f.write(summary.to_string(index=False))
    
    # save output patterns and seeds to npz for later reuse
    generated_patterns = []
    if use_range_pattern:
        render = RangePatternRenderer(lib.idx2pattern, lib.index_to_width, lib.index_to_height, context_size=20)
        for grid, seed in zip(output_grids, seeds):
            encoded_pattern = render.grid_to_encoded_pattern(grid)
            generated_patterns.append(encoded_pattern)
        if generated_patterns:
            generated_patterns = np.stack(generated_patterns, axis=0)  # shape (num, H, W, 3)
            success_nums, out_m, out_n, _, = generated_patterns.shape
            np.savez_compressed(
                os.path.join(output_dir, f"wfc_outputs_{out_m}x{out_n}_{success_nums}_patterns.npz"),
                patterns=np.stack(generated_patterns),
                seeds=seeds,
                index_to_width=lib.index_to_width,
                index_to_height=lib.index_to_height,
            )
        for encoded_pattern, seed in zip(generated_patterns, seeds):
            #render.plot_input_patterns_mxn(idx_arr=grid[0:3, 0:3])
            render.plot_output_pattern(encoded_pattern, seed)
        plt.show()

if __name__ == '__main__':
    main()
