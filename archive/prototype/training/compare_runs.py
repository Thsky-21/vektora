"""
compare_runs.py — score saved runs side by side from their test_scores.csv.

Run:  ../../.venv/Scripts/python.exe compare_runs.py run1 run2 run3
(each name is a folder in prototype/data/build/ holding report.json and
test_scores.csv, as archived after each training run)

Every run is scored twice:
  own   — on the test set it was evaluated on at the time
  clean — on the CURRENT manifest's test set only (exclusions removed), so
          runs trained before a cleanup are measured on exactly the same clips
Each run keeps its own threshold (chosen on its own validation set).
"""

import csv
import json
import os
import sys

import config as C

KW = "keyword"


def metrics(rows, thr):
    tp = sum(r["true"] == KW and r["pk"] >= thr for r in rows)
    fn = sum(r["true"] == KW and r["pk"] < thr for r in rows)
    fa = sum(r["true"] != KW and r["pk"] >= thr for r in rows)
    neg = sum(r["true"] != KW for r in rows)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fa, 1)
    cm = {t: {p: sum(r["true"] == t and r["pred"] == p for r in rows) for p in C.CLASSES} for t in C.CLASSES}
    return dict(
        n=len(rows), acc3=sum(r["true"] == r["pred"] for r in rows) / len(rows),
        tp=tp, fn=fn, recall=rec, precision=prec,
        f1=2 * prec * rec / max(prec + rec, 1e-9),
        fa=fa, negatives=neg, fa_rate=fa / max(neg, 1),
        fa_noise=sum(r["source"] == "noise" and r["pk"] >= thr for r in rows),
        fa_hard=sum(r["source"] == "hard" and r["pk"] >= thr for r in rows),
        fa_everyday=sum(r["source"] in ("generic", "gsc") and r["pk"] >= thr for r in rows),
        cm=cm,
    )


def main(names):
    with open(C.MANIFEST) as f:
        clean = {(os.path.basename(r["path"]), r["source"]) for r in csv.DictReader(f) if r["split"] == "test"}
    out = {}
    for name in names:
        d = os.path.join(C.BUILD_DIR, name)
        rep = json.load(open(os.path.join(d, "report.json")))
        rows = [dict(file=r["path"], source=r["source"], true=r["true"], pred=r["pred"],
                     pk=float(r["p_keyword"])) for r in csv.DictReader(open(os.path.join(d, "test_scores.csv")))]
        thr = rep["threshold"]
        h = rep["history"]
        b = rep["best_epoch"] - 1
        out[name] = dict(
            threshold=thr,
            best_epoch=rep["best_epoch"], epochs_run=rep["epochs_run"],
            early_stopped=rep["epochs_run"] < C.EPOCHS,
            best_val_acc=h["val_accuracy"][b], best_val_loss=h["val_loss"][b],
            train_acc_at_best=h["accuracy"][b], train_acc_last=h["accuracy"][-1],
            val_acc_last=h["val_accuracy"][-1],
            own=metrics(rows, thr),
            clean=metrics([r for r in rows if (r["file"], r["source"]) in clean], thr),
            clean_rows=[r for r in rows if (r["file"], r["source"]) in clean],
        )

    def line(label, fn):
        print(f"{label:<34}" + "".join(f"{fn(out[n]):>14}" for n in names))

    print(f"{'':<34}" + "".join(f"{n:>14}" for n in names))
    line("best epoch / epochs run", lambda o: f"{o['best_epoch']}/{o['epochs_run']}")
    line("early stopping triggered", lambda o: "yes" if o["early_stopped"] else "no")
    line("best val accuracy", lambda o: f"{o['best_val_acc']:.3f}")
    line("best val loss", lambda o: f"{o['best_val_loss']:.4f}")
    line("train acc (best epoch / last)", lambda o: f"{o['train_acc_at_best']:.3f}/{o['train_acc_last']:.3f}")
    line("val acc (last epoch)", lambda o: f"{o['val_acc_last']:.3f}")
    line("threshold", lambda o: f"{o['threshold']:.3f}")
    for scope in ("own", "clean"):
        print(f"\n--- test set: {scope} ---")
        line("clips", lambda o: o[scope]["n"])
        line("3-class accuracy", lambda o: f"{o[scope]['acc3']:.1%}")
        line("Vektora TP / FN", lambda o: f"{o[scope]['tp']}/{o[scope]['fn']}")
        line("Vektora recall", lambda o: f"{o[scope]['recall']:.1%}")
        line("Vektora precision", lambda o: f"{o[scope]['precision']:.1%}")
        line("Vektora F1", lambda o: f"{o[scope]['f1']:.3f}")
        line("false accepts / negatives", lambda o: f"{o[scope]['fa']}/{o[scope]['negatives']}")
        line("false accept rate", lambda o: f"{o[scope]['fa_rate']:.1%}")
        line("  noise FA", lambda o: o[scope]["fa_noise"])
        line("  look-alike (hard) FA", lambda o: o[scope]["fa_hard"])
        line("  everyday-word FA", lambda o: o[scope]["fa_everyday"])
    for n in names:
        o = out[n]
        print(f"\n{n} confusion on clean test (rows true, cols pred {C.CLASSES}):")
        for t in C.CLASSES:
            print(f"  {t:<8}" + "".join(f"{o['clean']['cm'][t][p]:>5}" for p in C.CLASSES))
        print(f"{n} misses and false accepts on clean test (threshold {o['threshold']:.2f}):")
        for r in sorted(o["clean_rows"], key=lambda r: (r["true"] != KW, r["pk"])):
            miss = r["true"] == KW and r["pk"] < o["threshold"]
            fa = r["true"] != KW and r["pk"] >= o["threshold"]
            if miss or fa:
                print(f"  {'MISS' if miss else 'FA  '} {r['source']:<9}{r['file']:<34}P(Vektora)={r['pk']:.3f}  pred={r['pred']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["run1", "run2", "run3"])
