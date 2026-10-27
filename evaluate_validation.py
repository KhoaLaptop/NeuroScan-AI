"""Score detector checkpoints on every held-out patient.

    python evaluate_validation.py v6=best_model_detection_v6.pth v7=best_model_detection_v7.pth

Each folder in Tumor Dataset/validation/ is one patient; a name containing "tumor"
means a tumor patient. If the folder has slice_labels.csv (columns path, mark), the
slices marked "visible" are also scored on their own, since most slices of a tumor
patient do not show the tumor.
"""

import csv
import os
import sys

import torch
from PIL import Image

from finetune_v2 import eval_tf, get_device, images_in, load

D = 'Tumor Dataset'
# The first held-out tumor patient lives outside validation/; its slice marks are here.
EXTRA_TUMOR = ('tumor 1 (held-out)', f'{D}/holdout_tumor_slice_labels.csv')


def patients():
    out = []
    labels = list(csv.DictReader(open(EXTRA_TUMOR[1])))
    out.append((EXTRA_TUMOR[0], 'tumor', [r['path'] for r in labels],
                {r['path'] for r in labels if r['mark'] == 'visible'}))
    vdir = f'{D}/validation'
    for name in sorted(os.listdir(vdir)):
        folder = os.path.join(vdir, name)
        if not os.path.isdir(folder):
            continue
        truth = 'tumor' if 'tumor' in name.lower() else 'normal'
        lab = os.path.join(folder, 'slice_labels.csv')
        visible = {r['path'] for r in csv.DictReader(open(lab)) if r['mark'] == 'visible'} if os.path.exists(lab) else set()
        out.append((name, truth, images_in(folder), visible))
    return out


def main(specs):
    device = get_device()
    tf = eval_tf()
    models = {name: load(path, device).eval() for name, path in (s.split('=', 1) for s in specs)}

    print(f'{"patient":28} {"truth":>6} {"n":>4}  ' + '  '.join(f'{m:>16}' for m in models))
    print(f'{"":28} {"":>6} {"":>4}  ' + '  '.join(f'{"flagged (visible)":>16}' for _ in models))
    for name, truth, paths, visible in patients():
        cells = []
        for m in models.values():
            flagged = vis_hit = 0
            for p in paths:
                with torch.no_grad():
                    tumor = torch.argmax(m(tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)), 1).item() == 1
                flagged += tumor
                vis_hit += tumor and p in visible
            vis = f' ({vis_hit}/{len(visible)})' if visible else ''
            cells.append(f'{100 * flagged / len(paths):5.0f}%{vis}')
        print(f'{name:28} {truth:>6} {len(paths):>4}  ' + '  '.join(f'{c:>16}' for c in cells))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
