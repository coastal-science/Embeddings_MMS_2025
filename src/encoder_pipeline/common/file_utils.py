import hashlib
import json
import time
from pathlib import Path


def get_or_create_hashed_file(out_dir: str, suffix: str, hash_params: dict) -> str:
    """Existing file under out_dir matching hash_params's content hash
    (first 16 hex chars of its sha256, ignoring whatever's appended after
    it e.g. a timestamp) if one exists, else a fresh content-hash +
    timestamp path -- not itself created on disk, caller writes there.
    returns file and content hash"""
    content_hash = hashlib.sha256(json.dumps(hash_params, sort_keys=True, default=str).encode()).hexdigest()[:16]
    matches = sorted(Path(out_dir).glob(f"{content_hash}*{suffix}"))
    if matches:
        return str(matches[0]), content_hash
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(out_dir) / f"{content_hash}_{timestamp}{suffix}"), content_hash
