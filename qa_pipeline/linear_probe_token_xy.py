"""Linear probe: does the 288-D grounding token linearly encode the selected
object's ego-frame position (x forward, y left)?

Ridge fit on train tokens (target = GT center from the export npz), evaluated
on val tokens. Intercept handled by centering targets on the train mean.
Alpha is chosen on a train-internal file split, val is untouched until the
final report. IoU per object comes from the same helper the QA-bank builder
uses, so val can be split into correctly- vs wrongly-selected tokens.
CPU-only.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/3eedqa/3EED")
from utils.eval_det import iou3d_rotated_vs_aligned

TOKENS = "/root/autodl-tmp/3eed_data/multi_grounding/others_linked_tokens"


def load(split):
    xs, ys, ious, files = [], [], [], []
    for fidx, path in enumerate(sorted(glob.glob(os.path.join(TOKENS, split, "rank00", "batch*.npz")))):
        with np.load(path) as data:
            tok = data["object_token"]
            gt = data["gt_box"]
            count = data["target_count"]
            n = tok.shape[0]
            for i in range(n):
                c = int(count[i])
                if c < 2:
                    continue
                boxes = np.concatenate((data["pred_center"][i, :c], data["pred_size"][i, :c]), axis=1)
                for j in range(c):
                    val, _ = iou3d_rotated_vs_aligned(
                        torch.as_tensor(gt[i, j:j + 1]), torch.as_tensor(boxes[j:j + 1]))
                    xs.append(tok[i, j])
                    ys.append(gt[i, j, :2])
                    ious.append(float(val[0, 0]))
                    files.append(fidx)
    return (np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64),
            np.asarray(ious), np.asarray(files))


def r2(pred, true, mu):
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - mu) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def binned(values, neg_thr, pos_thr):
    out = np.zeros(len(values), dtype=np.int8)
    out[values >= pos_thr] = 1
    out[values <= neg_thr] = -1
    return out


def fit(x, y, alpha):
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + alpha * np.eye(d), x.T @ y)


def report(tag, pred, true, mu, mask=None):
    if mask is None:
        mask = np.ones(len(true), dtype=bool)
    p, t = pred[mask], true[mask]
    r = r2(p, t, mu)
    mae = np.abs(p - t).mean(axis=0)
    print("%s n=%d  R2 x=%.3f y=%.3f  MAE x=%.2fm y=%.2fm" % (tag, mask.sum(), r[0], r[1], mae[0], mae[1]), flush=True)
    for name, thr, axis in (("lon(x)", 3.0, 0), ("lat(y)", 1.5, 1)):
        tb = binned(t[:, axis], -thr, thr)
        pb = binned(p[:, axis], -thr, thr)
        sel = tb != 0
        acc = (pb[sel] == tb[sel]).mean() if sel.any() else float("nan")
        print("   %s named-object bin acc %.3f (n=%d)" % (name, acc, int(sel.sum())), flush=True)


def main():
    x_tr, y_tr, _, fid_tr = load("train")
    x_va, y_va, iou_va, _ = load("val")
    mu_x, sd_x = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-6
    x_tr = (x_tr - mu_x) / sd_x
    x_va = (x_va - mu_x) / sd_x
    mu_y = y_tr.mean(axis=0)
    print("train tokens %s  val tokens %s  train y mean %s" % (x_tr.shape, x_va.shape, np.round(mu_y, 2)), flush=True)

    # alpha on a train-internal file split, val untouched
    inner = (fid_tr % 7) != 0
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        w = fit(x_tr[inner], y_tr[inner] - mu_y, alpha)
        inner_r2 = r2(x_tr[~inner] @ w + mu_y, y_tr[~inner], mu_y)
        if best is None or inner_r2.mean() > best[0]:
            best = (inner_r2.mean(), alpha)
    alpha = best[1]
    print("alpha=%g chosen on train-internal split (inner R2 %.3f)" % (alpha, best[0]), flush=True)

    w = fit(x_tr, y_tr - mu_y, alpha)
    pred_tr = x_tr @ w + mu_y
    pred_va = x_va @ w + mu_y

    report("TRAIN(in-sample)", pred_tr, y_tr, mu_y)
    report("VAL(all)", pred_va, y_va, mu_y)
    ok = iou_va >= 0.25
    report("VAL(selection IoU>=0.25)", pred_va, y_va, mu_y, ok)
    report("VAL(selection wrong)", pred_va, y_va, mu_y, ~ok)
    print("val selection-correct rate %.3f (%d/%d)" % (ok.mean(), ok.sum(), len(ok)), flush=True)

    # pairwise orderings within a sample, both objects correctly selected
    n_pairs = len(x_va) // 2
    okp = ok.reshape(n_pairs, 2).all(axis=1)
    yv = y_va.reshape(n_pairs, 2, 2)
    pv = pred_va.reshape(n_pairs, 2, 2)
    d_t = np.hypot(yv[:, :, 0], yv[:, :, 1])
    d_p = np.hypot(pv[:, :, 0], pv[:, :, 1])
    lr = np.sign(pv[:, 0, 1] - pv[:, 1, 1]) == np.sign(yv[:, 0, 1] - yv[:, 1, 1])
    cl = np.sign(d_p[:, 0] - d_p[:, 1]) == np.sign(d_t[:, 0] - d_t[:, 1])
    for tag, mask in (("all pairs", np.ones(n_pairs, bool)), ("both-correct pairs", okp)):
        print("pairwise %s n=%d  left/right %.3f  closer-to-ego %.3f"
              % (tag, mask.sum(), lr[mask].mean(), cl[mask].mean()), flush=True)

    # how separated are the two objects, for reference
    dy = np.abs(yv[:, 0, 1] - yv[:, 1, 1])
    print("pair |delta_y| median %.2fm  frac>=2.0m %.3f" % (np.median(dy), (dy >= 2.0).mean()), flush=True)

    # same-caliber subsets: which_is_left only asks about |delta_y| >= 2.0,
    # relative_location names a side only when the label rule fires
    sep = okp & (dy >= 2.0)
    print("pairwise both-correct & |dy|>=2.0 n=%d  left/right %.3f" % (sep.sum(), lr[sep].mean()), flush=True)
    lat = np.abs(y_va[:, 1]) >= 2.0
    tb = binned(y_va[:, 1], -1.5, 1.5)
    pb = binned(pred_va[:, 1], -1.5, 1.5)
    m = ok & (tb != 0) & lat
    print("val lat-named & |y|>=2.0 & correct-selection n=%d  probe bin acc %.3f"
          % (m.sum(), (pb[m] == tb[m]).mean()), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
