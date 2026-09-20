# pass_cache/

Per-pass output caches written by `experiments/tagging/iterate_pass.py`: `local.json`, `scene.json`, `global.json`, each `chunk id → that pass's fields` for the 30 gold chunks. They let you re-run and score ONE pass while reusing the others from cache. Delete a file to force that pass to re-tag.

Caveat: these were produced by the *adjudicating* pipeline (pass 3 reading pass 2's candidates), before the rebuild in which every pass is final.
