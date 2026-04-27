"""One-off: extract synth_v3 shard_000 to dataset/output/synth_iter for aug iteration."""
import os
import sys
import tarfile

TAR = r"c:/Users/Mihajlo/Documents/Body Tracking/synth_v3_partial.tar"
OUT = r"c:/Users/Mihajlo/Documents/Body Tracking/dataset/output/synth_iter"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    with tarfile.open(TAR) as t:
        members = [m for m in t.getmembers()
                   if m.name.startswith("synth_v3/shard_000/")]
        print(f"extracting {len(members)} from shard_000")
        n_files = 0
        for m in members:
            if m.isdir():
                continue
            rel = m.name.replace("synth_v3/shard_000/", "")
            target = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with t.extractfile(m) as f, open(target, "wb") as g:
                g.write(f.read())
            n_files += 1
            if n_files % 500 == 0:
                print(f"  {n_files}")
        print(f"done: {n_files} files -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
