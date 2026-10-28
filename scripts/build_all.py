"""Rebuilds every figure: runs scripts/build.py in a background Blender for each
<kind>.py here, a few at a time, and fails if any figure reports problems.

    python3 scripts/build_all.py [--previews] [--no-save]

--previews  also render previews/<kind>.png, four views in both palettes
--no-save   skip writing the .blend and .glb files
BLENDER overrides the path to the Blender binary.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
KINDS = sorted(
    f[:-3].capitalize() for f in os.listdir(HERE) if f.endswith(".py") and f not in ("lib.py", "build.py", "build_all.py")
)


def build(kind):
    args = [BLENDER, "--background", "--factory-startup", "--python", os.path.join(HERE, "build.py"), "--", kind]
    if "--previews" in sys.argv:
        args += ["--sheet", os.path.join(ROOT, "previews", kind.lower() + ".png")]
    if "--no-save" in sys.argv:
        args.append("--no-save")
    run = subprocess.run(args, capture_output=True, text=True)
    out = run.stdout + run.stderr
    # build.py prints its report last; a figure script may print its own lines.
    line = next((l for l in reversed(out.splitlines()) if l.startswith("REPORT {")), None)
    if line:
        return json.loads(line[7:])
    return {"kind": kind, "problems": [f"Blender exited with {run.returncode}"], "log": out[-2000:]}


if "--previews" in sys.argv:
    os.makedirs(os.path.join(ROOT, "previews"), exist_ok=True)
with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as pool:
    reports = sorted(pool.map(build, KINDS), key=lambda r: r["kind"])
for r in reports:
    print(
        f"{r['kind']:<13} {r.get('triangles', '-'):>6} tris  {r.get('glb_bytes', '-'):>7} bytes  "
        + ("; ".join(r["problems"]) or "ok")
    )
failed = [r for r in reports if r["problems"]]
for r in failed:
    if r.get("log"):
        print(f"\n{r['kind']}:\n{r['log']}", file=sys.stderr)
sys.exit(1 if failed else 0)
