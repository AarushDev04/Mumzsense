#!/usr/bin/env bash
# =============================================================================
# MumzSense v1 — Re-index Knowledge Base with Real Semantic Embeddings
# =============================================================================
# Run this script once after cloning or after any change to corpus_validated.jsonl.
# It clears the stale fallback embeddings and rebuilds the local_store.json
# using bge-large-en-v1.5 (EN) and multilingual-mpnet (AR) sentence transformers.
#
# Prerequisites:
#   pip install sentence-transformers  (or requirements.runtime.txt)
#   GROQ_API_KEY set in env/.env (not required for indexing itself)
#
# Usage (from repo root):
#   cd backend
#   bash reindex_with_real_embeddings.sh
#
# Or with explicit env:
#   USE_REAL_EMBEDDINGS=true bash reindex_with_real_embeddings.sh
# =============================================================================

set -e

# Force real embeddings regardless of any .env default
export USE_REAL_EMBEDDINGS=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "==========================================================="
echo "  MumzSense — Re-index with Real Semantic Embeddings"
echo "==========================================================="
echo ""
echo "USE_REAL_EMBEDDINGS=$USE_REAL_EMBEDDINGS"
echo ""

# Step 1: Clear stale fallback embeddings
echo "[1/3] Clearing stale embeddings from local_store.json ..."
python3 - <<'PYEOF'
import json, pathlib
path = pathlib.Path("data/local_store.json")
if path.exists():
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        store = {}
    old_count = len(store.get("posts", []))
    store["posts"] = []
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Cleared {old_count} stale posts from {path}")
else:
    print("  local_store.json does not exist — will be created during indexing")
PYEOF

echo ""
echo "[2/3] Embedding and indexing corpus (this takes ~5–10 min on CPU) ..."
echo "      Models: BAAI/bge-large-en-v1.5 (EN), sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (AR)"
echo ""

python3 data/embed_and_index.py --corpus data/corpus_validated.jsonl --batch-size 10

echo ""
echo "[3/3] Verifying embedding quality ..."
python3 - <<'PYEOF'
import json, numpy as np, pathlib

path = pathlib.Path("data/local_store.json")
store = json.loads(path.read_text(encoding="utf-8"))
posts = store.get("posts", [])

if not posts:
    print("  ERROR: No posts indexed! Check embed_and_index.py output above.")
    exit(1)

sample = posts[:5]
for i, p in enumerate(sample):
    en_emb = p.get("en_embedding", [])
    nonzero = sum(1 for v in en_emb if v != 0)
    print(f"  Post {i}: en_embedding len={len(en_emb)}, nonzero={nonzero} ({'OK' if nonzero > 100 else 'FAIL — still looks like fallback embedding!'})")

total = len(posts)
print(f"\n  Total posts indexed: {total}")
if total < 400:
    print("  WARNING: Expected ~454–560 posts. Re-run if count is low.")
else:
    print("  Post count looks good.")

# Quick self-similarity check
emb = np.asarray(posts[0]["en_embedding"])
norm = np.linalg.norm(emb)
print(f"\n  Self-norm of first post EN embedding: {norm:.4f} (should be ~1.0 if normalised)")

print("\n=========================================================")
print("  Re-indexing complete. Restart the FastAPI server now.")
print("=========================================================")
PYEOF
