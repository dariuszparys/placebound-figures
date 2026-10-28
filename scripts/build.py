"""Builds one figure headless: runs scripts/<kind>.py, then finishes,
validates and exports it, and optionally renders previews.

    blender --background --factory-startup --python scripts/build.py -- Knight \\
        [--sheet out.png] [--board out.png] [--with Archer,Goblin] [--no-save]

--sheet  four views in both palettes
--board  the figure beside the --with kinds (read from models/) at board scale
--no-save  skip writing the .blend and .glb
The last line printed is `REPORT {json}` with size, bounds, role shares and problems.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if not args:
    raise SystemExit("usage: build.py -- <Kind> [--sheet out.png] [--board out.png] [--with A,B] [--no-save]")
kind = args[0]


def opt(name):
    return args[args.index(name) + 1] if name in args else None


lib.reset()
spec = importlib.util.spec_from_file_location(kind.lower(), os.path.join(HERE, kind.lower() + ".py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
figure = lib.finish(kind, module.build(), **getattr(module, "FINISH", {}))
info = lib.report(kind, figure)
if "--no-save" not in args:
    glb = lib.save(kind, figure)
    info["glb_bytes"] = os.path.getsize(glb)
if opt("--sheet"):
    lib.render_sheet(figure, os.path.abspath(opt("--sheet")))
if opt("--board"):
    others = [k for k in (opt("--with") or "").split(",") if k and k != kind]
    figures = [(kind, figure)] + [(k, o) for k in others if (o := lib.import_figure(k))]
    lib.render_board(figures, os.path.abspath(opt("--board")))
print("REPORT " + json.dumps(info))
