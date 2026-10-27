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
from sklearn.metrics import f1_score, precision_score, recall_score

from finetune_v2 import eval_tf, get_device, images_in, load

D = 'Tumor Dataset'
# The first held-out tumor patient lives outside validation/; its slice marks are here.
EXTRA_TUMOR = ('tumor 1 (held-out)', f'{D}/holdout_tumor_slice_labels.csv')
# shortcut: chosen by looking at these same validation patients; re-pick on a fresh set.
PATIENT_CUTOFF = 0.10


def slice_truth(csv_path):
    """Visible-tumor slices are 1, marked tumor-free slices 0; 'unsure' slices are left out."""
    rows = list(csv.DictReader(open(csv_path)))
    return [r['path'] for r in rows], {r['path']: int(r['mark'] == 'visible') for r in rows if r['mark'] in ('visible', 'none')}


def patients():
    paths, truth = slice_truth(EXTRA_TUMOR[1])
    out = [(EXTRA_TUMOR[0], 'tumor', paths, truth)]
    vdir = f'{D}/validation'
    for name in sorted(os.listdir(vdir)):
        folder = os.path.join(vdir, name)
        if not os.path.isdir(folder):
            continue
        kind = 'tumor' if 'tumor' in name.lower() else 'normal'
        paths = images_in(folder)
        lab = os.path.join(folder, 'slice_labels.csv')
        if kind == 'normal':
            truth = {p: 0 for p in paths}
        else:
            truth = slice_truth(lab)[1] if os.path.exists(lab) else {}
        out.append((name, kind, paths, truth))
    return out


def main(specs):
    device = get_device()
    tf = eval_tf()
    models = {name: load(path, device).eval() for name, path in (s.split('=', 1) for s in specs)}

    stats = {m: {'st': [], 'sp': [], 'pt': [], 'pp': []} for m in models}
    print(f'{"patient":40} {"truth":>6} {"n":>4}  ' + '  '.join(f'{m:>16}' for m in models))
    print(f'{"":40} {"":>6} {"":>4}  ' + '  '.join(f'{"flagged (visible)":>16}' for _ in models))
    for name, kind, paths, truth in patients():
        visible = {p for p, t in truth.items() if t == 1}
        cells = []
        for mname, m in models.items():
            s = stats[mname]
            flagged = vis_hit = 0
            for p in paths:
                with torch.no_grad():
                    tumor = torch.argmax(m(tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)), 1).item() == 1
                flagged += tumor
                vis_hit += tumor and p in visible
                if p in truth:
                    s['st'].append(truth[p]); s['sp'].append(int(tumor))
            s['pt'].append(int(kind == 'tumor')); s['pp'].append(int(flagged / len(paths) >= PATIENT_CUTOFF))
            vis = f' ({vis_hit}/{len(visible)})' if visible else ''
            cells.append(f'{100 * flagged / len(paths):5.0f}%{vis}')
        print(f'{name[:40]:40} {kind:>6} {len(paths):>4}  ' + '  '.join(f'{c:>16}' for c in cells))

    print(f'\n{"":53}' + '  '.join(f'{m:>16}' for m in models))
    for label, t, p in [('slice F1 (labelled slices)', 'st', 'sp'), (f'patient F1 (cutoff {PATIENT_CUTOFF:.0%})', 'pt', 'pp')]:
        for metric, fn in [('F1', f1_score), ('precision', precision_score), ('recall', recall_score)]:
            row = [fn(stats[m][t], stats[m][p], zero_division=0) for m in models]
            print(f'{label + " " + metric:53}' + '  '.join(f'{v:>16.2f}' for v in row))
    n = stats[next(iter(models))]
    print(f'(slice: {sum(n["st"])} tumor / {len(n["st"]) - sum(n["st"])} normal slices; patient: {sum(n["pt"])} tumor / {len(n["pt"]) - sum(n["pt"])} normal)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
