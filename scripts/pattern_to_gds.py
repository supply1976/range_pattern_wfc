from pathlib import Path
import os, sys

SRC_DIR = Path(__file__).parent.parent / "src"
if SRC_DIR not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wfc.renderer import RangePatternRenderer

try:
    from proteus import layer
    from proteus import mlo
except ImportError:
    print("Proteus library not found. GDS export functionality will be unavailable.")

def main():
    input_file = sys.argv[1]
    savedir = os.path.dirname(input_file)
    data = np.load(input_file, allow_pickle=True)
    print(list(data))
    patterns = data['patterns']
    seeds = data['seeds']
    index_to_width = data['index_to_width'][()]
    index_to_height = data['index_to_height'][()]
    
    renderer = RangePatternRenderer(patterns, index_to_width, index_to_height, context_size=20)
    L = 8000.0 * np.arange(100)
    Y, X = np.meshgrid(L, L, indexing='ij')
    origins = list(zip(map(float, X.flatten()), map(float, Y.flatten())))
    outputs = []
    covers = []
    for i in range(len(patterns)):
        #if i > 10: break
        if i==0 or (i+1)%100==0:
            print("process: {}/{}".format(i+1, len(patterns)))
        bboxes = renderer.pattern_to_bbox_list(patterns[i])
        _recs = []
        for bbox in bboxes:
            llx, lly, urx, ury = bbox
            _rec = mlo.createRectangle([llx, lly], [urx, ury])
            _recs.append(_rec)
        _output_polygons = layer.sum(_recs)
        _output_polygons = mlo.move(_output_polygons, offset=list(origins[i]))
        _layer_bbox = mlo.layerBoundingBox(_output_polygons)
        outputs.append(_output_polygons)
        covers.append(_layer_bbox)

    final_output = layer.sum(outputs)
    cover_layer = layer.sum(covers)

    layer.writeGds(
        {(0,0): final_output, (300,0): cover_layer},
        os.path.join(savedir, f"final_outputs_merged.oas")
        )
        
if __name__ == '__main__':
    main()
