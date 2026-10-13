from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str | Path, expected_sha256: str | None = None,
             retries: int = 3, timeout: int = 120) -> dict:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            break
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    tmp.replace(dest)
    digest = sha256_file(dest)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"sha256 mismatch for {dest.name}: {digest} != {expected_sha256}")
    return {"url": url, "path": str(dest), "sha256": digest, "bytes": dest.stat().st_size}


def github_raw_url(repo_url: str, commit: str, path_in_repo: str) -> str:
    owner_repo = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    return f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path_in_repo}"
