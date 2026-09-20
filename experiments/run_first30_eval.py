"""Local vs context-aware tagger, scored against the hand-built first-30 gold set.

Runs BOTH tagging arms over the 30 chunks in experiments/gold_first30.jsonl:
  local   = tag_one_chunk(text, prev_chunk, next_chunk)          # prev/next only
  context = tag_with_context(text, global_map, adjacent scenes)  # the V1 pipeline

Scores each arm against gold, field by field — never one "accuracy" number, per the
Phase 11 rule in docs/tagger_study_plan.html. Caches raw model output to
experiments/first30_tagger_output.json so the metrics can be recomputed (or the report
regenerated) without re-spending API calls.

Run from repo root:
    .venv/bin/python experiments/run_first30_eval.py                # tag + score
    .venv/bin/python experiments/run_first30_eval.py --score-only   # reuse cached output
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import load_chunks
from ingest import _scene_context
from scenes import render_global_map
from tag_white_nights import tag_one_chunk, tag_with_context

GOLD_PATH = ROOT / "experiments" / "gold_first30.jsonl"
CACHE_PATH = ROOT / "experiments" / "first30_tagger_output.json"

ARMS = ["local", "context"]
SINGLE_LABEL_FIELDS = ["narrating_voice", "speaking_voice", "speaker_relation", "narrative_relation"]
SET_FIELDS = ["canonical_themes", "characters_present"]


# ---------------------------------------------------------------- loading / running

def load_gold() -> dict:
    gold = {}
    for line in GOLD_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gold[d["id"]] = d
    return gold


def run_tagger(gold_ids: list) -> dict:
    chunks = {c["id"]: c for c in load_chunks()}
    gm_str = render_global_map(json.loads((ROOT / "context" / "global_map.json").read_text()))
    scenes = json.loads((ROOT / "context" / "scenes.json").read_text())
    cards = json.loads((ROOT / "context" / "scene_cards.json").read_text())
    scene_of = {cid: i for i, s in enumerate(scenes) for cid in s["chunk_ids"]}

    out = {}
    for i, cid in enumerate(gold_ids, 1):
        c = chunks[cid]
        prev = chunks[c["prev_id"]]["text"] if c["prev_id"] else None
        nxt = chunks[c["next_id"]]["text"] if c["next_id"] else None
        row = {}

        for attempt in (1, 2):
            try:
                row["local"] = tag_one_chunk(c["text"], prev, nxt).model_dump()
                break
            except Exception as e:
                row["local"] = None
                if attempt == 2:
                    print(f"  ! local tag failed for {cid}: {e}")

        for attempt in (1, 2):
            try:
                sc = _scene_context(cards, scene_of[cid])
                row["context"] = tag_with_context(c["text"], gm_str, sc).model_dump()
                break
            except Exception as e:
                row["context"] = None
                if attempt == 2:
                    print(f"  ! context tag failed for {cid}: {e}")

        out[cid] = row
        print(f"  {i:>2}/{len(gold_ids)}  {cid}  "
              f"local={row['local']['narrative_relation'] if row['local'] else 'FAIL':<12}"
              f"context={row['context']['narrative_relation'] if row['context'] else 'FAIL'}")
    return out


# ---------------------------------------------------------------- scoring

def score_single_label(field: str, gold: dict, preds: dict) -> dict:
    pairs = [(cid, gold[cid][field], preds[cid][field])
             for cid in gold if preds.get(cid) is not None]
    n = len(pairs)
    correct = sum(1 for _, g, p in pairs if g == p)
    classes = sorted({g for _, g, _ in pairs} | {p for _, _, p in pairs}, key=lambda x: str(x))
    per_class = {}
    for cls in classes:
        tp = sum(1 for _, g, p in pairs if p == cls and g == cls)
        fp = sum(1 for _, g, p in pairs if p == cls and g != cls)
        fn = sum(1 for _, g, p in pairs if g == cls and p != cls)
        support = sum(1 for _, g, _ in pairs if g == cls)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) else 0.0
        per_class[cls] = {"support": support, "precision": prec, "recall": rec, "f1": f1}
    mismatches = [{"id": cid, "gold": g, "pred": p} for cid, g, p in pairs if g != p]
    return {"n": n, "accuracy": correct / n if n else None,
            "per_class": per_class, "mismatches": mismatches}


def score_set_field(field: str, gold: dict, preds: dict) -> dict:
    tp_total = fp_total = fn_total = 0
    macro_p, macro_r, macro_f1 = [], [], []
    per_chunk = []
    extra_labels = {}   # label -> count of false positives (over-application)
    missed_labels = {}  # label -> count of false negatives (under-application)

    for cid in gold:
        p = preds.get(cid)
        if p is None:
            continue
        g_set, p_set = set(gold[cid][field]), set(p[field])
        tp = len(g_set & p_set)
        fp_set = p_set - g_set
        fn_set = g_set - p_set
        tp_total += tp
        fp_total += len(fp_set)
        fn_total += len(fn_set)
        for lbl in fp_set:
            extra_labels[lbl] = extra_labels.get(lbl, 0) + 1
        for lbl in fn_set:
            missed_labels[lbl] = missed_labels.get(lbl, 0) + 1

        if not g_set and not p_set:
            prec = rec = 1.0
        else:
            prec = tp / (tp + len(fp_set)) if (tp + len(fp_set)) else 0.0
            rec = tp / (tp + len(fn_set)) if (tp + len(fn_set)) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)
        per_chunk.append({"id": cid, "gold": sorted(g_set), "pred": sorted(p_set),
                           "precision": prec, "recall": rec})

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else None
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else None
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p and micro_r and (micro_p + micro_r) else 0.0)

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": sum(macro_p) / len(macro_p) if macro_p else None,
                  "recall": sum(macro_r) / len(macro_r) if macro_r else None,
                  "f1": sum(macro_f1) / len(macro_f1) if macro_f1 else None},
        "over_applied": dict(sorted(extra_labels.items(), key=lambda kv: -kv[1])),
        "under_applied": dict(sorted(missed_labels.items(), key=lambda kv: -kv[1])),
        "per_chunk": per_chunk,
    }


# ---------------------------------------------------------------- report

def pct(x):
    return f"{x:.0%}" if x is not None else "  n/a"


def print_report(gold: dict, results: dict):
    for arm in ARMS:
        preds = {cid: results[cid].get(arm) for cid in gold}
        n_fail = sum(1 for cid in gold if preds.get(cid) is None)
        print("\n" + "=" * 88)
        print(f"ARM: {arm.upper()}" + (f"   ({n_fail} tagging failures)" if n_fail else ""))
        print("=" * 88)

        for field in SINGLE_LABEL_FIELDS:
            s = score_single_label(field, gold, preds)
            print(f"\n-- {field}  (accuracy {pct(s['accuracy'])}, n={s['n']})")
            print(f"   {'class':<26}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>7}")
            for cls, m in s["per_class"].items():
                print(f"   {str(cls):<26}{m['support']:>8}{pct(m['precision']):>11}"
                      f"{pct(m['recall']):>9}{m['f1']:>7.2f}")
            if s["mismatches"]:
                print(f"   mismatches ({len(s['mismatches'])}):")
                for m in s["mismatches"]:
                    print(f"     {m['id']:<28} gold={str(m['gold']):<14} pred={m['pred']}")

        for field in SET_FIELDS:
            s = score_set_field(field, gold, preds)
            print(f"\n-- {field}")
            print(f"   micro  P={pct(s['micro']['precision'])}  R={pct(s['micro']['recall'])}  "
                  f"F1={s['micro']['f1']:.2f}")
            print(f"   macro  P={pct(s['macro']['precision'])}  R={pct(s['macro']['recall'])}  "
                  f"F1={s['macro']['f1']:.2f}")
            if s["over_applied"]:
                print(f"   over-applied (pred has it, gold doesn't): {s['over_applied']}")
            if s["under_applied"]:
                print(f"   under-applied (gold has it, pred misses): {s['under_applied']}")

        # targeted diagnostic flagged in experiments/gold_first30_notes.md: gold withholds
        # unrequited-longing across all 30 chunks on purpose (no unattainable person yet
        # in the story) -- any appearance here is the exact theme-collapse failure F3 flagged.
        leaks = [cid for cid in gold
                 if preds.get(cid) and "unrequited-longing" in preds[cid]["canonical_themes"]]
        print(f"\n-- diagnostic: 'unrequited-longing' applied where gold has none "
              f"({len(leaks)}/30)")
        if leaks:
            print(f"   {leaks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true",
                     help="reuse cached experiments/first30_tagger_output.json instead of re-tagging")
    args = ap.parse_args()

    gold = load_gold()

    if args.score_only:
        if not CACHE_PATH.exists():
            print(f"No cache at {CACHE_PATH} -- run without --score-only first.")
            sys.exit(1)
        results = json.loads(CACHE_PATH.read_text())
    else:
        print(f"Tagging {len(gold)} chunks with both arms (local, context)...")
        results = run_tagger(list(gold.keys()))
        CACHE_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nCached raw output -> {CACHE_PATH}")

    print_report(gold, results)


if __name__ == "__main__":
    main()
