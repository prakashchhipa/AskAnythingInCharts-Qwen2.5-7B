# src/prepare_data.py
import argparse, os, sys, time
from datasets import load_from_disk
from datasets.utils.logging import set_verbosity_error

# make local import work when run as a script
sys.path.append(os.path.join(os.path.dirname(__file__)))
from datasets_build import build_datasets

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=80000,
                    help="Cap number of training samples to keep prep fast (adjust later if desired).")
    ap.add_argument("--max-eval", type=int, default=2000)
    ap.add_argument("--out-dir", type=str, default="data/hf_cache",
                    help="Where to save the prepared datasets with save_to_disk().")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    set_verbosity_error()
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("⏬ Building & downloading datasets (this will cache under HF_DATASETS_CACHE if set)...")
    t0 = time.time()
    ds = build_datasets(max_train=args.max_train, max_eval=args.max_eval, seed=args.seed)
    train_ds, eval_ds = ds["train"], ds["eval"]

    print(f"✅ Built datasets in {time.time()-t0:.1f}s")
    print(f"   Train size: {len(train_ds):,}  |  Eval size: {len(eval_ds):,}")

    # quick sanity peek
    ex = train_ds[0]
    keys = list(ex.keys())
    print(f"Sample keys: {keys}")
    # Confirm we have an image and messages
    assert "images" in ex and "messages" in ex, "Prepared sample missing images/messages"

    # save to disk (arrow format)
    train_path = os.path.join(args.out_dir, "train")
    eval_path  = os.path.join(args.out_dir, "eval")
    print(f"💾 Saving train -> {train_path}")
    train_ds.save_to_disk(train_path)
    print(f"💾 Saving eval  -> {eval_path}")
    eval_ds.save_to_disk(eval_path)

    # reload once to prove integrity
    _train = load_from_disk(train_path)
    _eval  = load_from_disk(eval_path)
    print(f"🔁 Reload OK: train={len(_train):,}, eval={len(_eval):,}")
    print("🎉 Data preparation complete.")

if __name__ == "__main__":
    main()
