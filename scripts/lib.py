"""Modelling contract shared by every Placebound figure script.

Each figure lives in scripts/<kind>.py and defines `build()`, which uses the
helpers below and returns the objects that make up the figure. build.py runs it
headless; `finish()` then joins, shades, bakes, validates and exports it:

    blender --background --factory-startup --python scripts/build.py -- Knight

Conventions the game (Placebound) depends on:
- Blender units are figure units. Z is up and the figure faces -Y, which the
  glTF exporter turns into the game's +z front.
- The origin is the top of the plinth. Grounded figures stand on z = 0; flyers
  hover with their lowest point above it.
- Every face carries one role material named exactly as in ROLES. The game
  swaps in the owning side's colour per role, so the same
  mesh serves Light, Dark and a Shapeshifter's copy.
- Ambient occlusion is baked into the point colour attribute "AO", which the
  game multiplies with the role colour.
"""
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLEND_DIR = os.path.join(ROOT, "blend")
MODEL_DIR = os.path.join(ROOT, "models")

ROLES = ("body", "cloth", "skin", "trim", "metal", "eye")
# Preview colours, mirrored from the palette the game ships.
# (hex, roughness, metalness, emission hex, emission strength)
PALETTES = {
    "light": {
        "body": (0xEEF0E4, 0.55, 0.0, None, 0),
        "cloth": (0xB4CFCB, 0.8, 0.0, None, 0),
        "skin": (0xF0DAC6, 0.6, 0.0, None, 0),
        "trim": (0x35B8AA, 0.45, 0.0, None, 0),
        "metal": (0xC3CCD4, 0.32, 0.65, None, 0),
        "eye": (0x6FE0D2, 0.5, 0.0, 0x35B8AA, 0.9),
    },
    "dark": {
        "body": (0x241E29, 0.85, 0.0, None, 0),
        "cloth": (0x3E2C35, 0.9, 0.0, None, 0),
        "skin": (0x9E8E98, 0.7, 0.0, None, 0),
        "trim": (0xFF8A4C, 0.5, 0.0, 0xFF6A2A, 0.35),
        "metal": (0x5D525C, 0.42, 0.55, None, 0),
        "eye": (0xFFB27A, 0.5, 0.0, 0xFFB27A, 1.4),
    },
}
PLINTH = {"light": 0xCDD4C6, "dark": 0x3B3036}

# side: which army fields the kind; elementals serve either.
KINDS = {
    "Knight": ("troop", "light"),
    "Archer": ("troop", "light"),
    "Unicorn": ("beast", "light"),
    "Golem": ("heavy", "light"),
    "Valkyrie": ("flyer", "light"),
    "Djinni": ("flyer", "light"),
    "Phoenix": ("flyer", "light"),
    "Wizard": ("commander", "light"),
    "Goblin": ("troop", "dark"),
    "Manticore": ("troop", "dark"),
    "Basilisk": ("beast", "dark"),
    "Troll": ("heavy", "dark"),
    "Banshee": ("flyer", "dark"),
    "Dragon": ("flyer", "dark"),
    "Shapeshifter": ("flyer", "dark"),
    "Sorceress": ("commander", "dark"),
    "Earth": ("heavy", "either"),
    "Fire": ("elemental", "either"),
    "Air": ("flyer", "either"),
    "Water": ("elemental", "either"),
}
# Allowed height of the figure's top per tier. The ranges do not overlap, so the
# game's size order holds: troop < beast < heavy, and commanders tower over all.
TOP = {
    "troop": (0.64, 0.74),
    "beast": (0.80, 0.88),
    "heavy": (0.94, 1.12),
    "flyer": (0.85, 1.22),
    "commander": (1.32, 1.45),
    "elemental": (0.70, 0.86),
}
# Flyers hover: nothing below this height (solid parts should stay above ~0.34;
# the board settles flyers so the plinth top meets model z ~0.31).
FLYER_FLOOR = 0.15
# Half extents of the footprint, |x| and |y|.
FOOTPRINT = (0.46, 0.56)
TRI_BUDGET = 18000


def hexcolor(h, alpha=1.0):
    """Linear RGBA from an sRGB hex colour."""
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin((h >> 16) & 255), lin((h >> 8) & 255), lin(h & 255), alpha)


# ---------------------------------------------------------------- scene


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Saving over a .blend would otherwise leave a .blend1 backup beside it.
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    for role in ROLES:
        mat(role)
    set_palette("light")


def _link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mat(role):
    """The role material, created on first use with a palette-driven node tree."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; use one of {ROLES}")
    m = bpy.data.materials.get(role)
    if m:
        return m
    m = bpy.data.materials.new(role)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "bsdf"
    ao = nt.nodes.new("ShaderNodeVertexColor")
    ao.layer_name = "AO"
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.name = "palette"
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    sock = {s.identifier: s for s in mix.inputs}
    sock["Factor_Float"].default_value = 1.0
    nt.links.new(rgb.outputs[0], sock["A_Color"])
    nt.links.new(ao.outputs["Color"], sock["B_Color"])
    out_sock = {s.identifier: s for s in mix.outputs}["Result_Color"]
    nt.links.new(out_sock, bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return m


def set_palette(name):
    for role in ROLES:
        color, rough, metal, emit, strength = PALETTES[name][role]
        nt = mat(role).node_tree
        nt.nodes["palette"].outputs[0].default_value = hexcolor(color)
        b = nt.nodes["bsdf"]
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        b.inputs["Emission Color"].default_value = hexcolor(emit if emit is not None else 0)
        b.inputs["Emission Strength"].default_value = strength


def set_role(obj, role):
    """Gives every face of `obj` the role material."""
    obj.data.materials.clear()
    obj.data.materials.append(mat(role))
    for p in obj.data.polygons:
        p.material_index = 0
    return obj


def paint(obj, role, where):
    """Gives the faces whose local centre satisfies `where(Vector)` the role material."""
    m = mat(role)
    names = [s.name for s in obj.data.materials]
    if role not in names:
        obj.data.materials.append(m)
        names.append(role)
    idx = names.index(role)
    for p in obj.data.polygons:
        if where(p.center):
            p.material_index = idx
    return obj


def place(obj, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    """Bakes a transform into the mesh, on top of any already baked."""
    m = Matrix.LocRotScale(Vector(loc), Euler(rot), Vector(scale))
    obj.data.transform(m)
    obj.data.update()
    return obj


def _from_bmesh(name, bm, role):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = _link(bpy.data.objects.new(name, me))
    return set_role(obj, role)


def apply_modifiers(obj):
    """Applies every modifier (and converts curves/metaballs) without operators."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    if obj.type == "MESH":
        obj.modifiers.clear()
        old = obj.data
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        return obj
    # Curves and metaballs become a new mesh object; callers free the old data.
    me.transform(obj.matrix_world)
    new = _link(bpy.data.objects.new(obj.name + "_mesh", me))
    bpy.data.objects.remove(obj)
    return new


def remove(obj):
    bpy.data.objects.remove(obj, do_unlink=True)


def duplicate(obj, name=None):
    new = obj.copy()
    new.data = obj.data.copy()
    new.name = name or obj.name + "_copy"
    return _link(new)


# ---------------------------------------------------------------- primitives


def box(size, loc=(0, 0, 0), rot=(0, 0, 0), role="body", bevel=0.0, segments=2, name="box"):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    obj = _from_bmesh(name, bm, role)
    if bevel > 0:
        bevel_edges(obj, bevel, segments)
    return place(obj, loc, rot)


def cylinder(r1, r2, depth, loc=(0, 0, 0), rot=(0, 0, 0), role="body", segments=24, bevel=0.0, name="cyl"):
    """A cone frustum along local z, centred on its middle; r1 bottom, r2 top."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=r1, radius2=r2, depth=depth)
    obj = _from_bmesh(name, bm, role)
    if bevel > 0:
        bevel_edges(obj, bevel, 2)
    return place(obj, loc, rot)


def sphere(r, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), role="body", segments=24, rings=16, name="sphere"):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=r)
    obj = _from_bmesh(name, bm, role)
    return place(obj, loc, rot, scale)


def ico(r, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), role="body", subdiv=1, name="ico"):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    obj = _from_bmesh(name, bm, role)
    return place(obj, loc, rot, scale)


def lathe(profile, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), role="body", segments=24, start=0.0, sweep=math.tau, name="lathe"):
    """Turns a profile of (radius, z) points around local z. Points with radius 0
    close the shape. A partial sweep leaves it open, starting at angle `start`
    measured from -Y (the front) toward +X."""
    full = abs(sweep - math.tau) < 1e-6
    steps = segments if full else segments + 1
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        if r <= 1e-6:
            rings.append([bm.verts.new((0, 0, z))])
            continue
        ring = []
        for i in range(steps):
            a = start + sweep * i / segments
            ring.append(bm.verts.new((math.sin(a) * r, -math.cos(a) * r, z)))
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        n = steps if full else steps - 1
        for i in range(n):
            j = (i + 1) % steps
            if len(a) == 1:
                bm.faces.new((a[0], b[j], b[i]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[j], b[0]))
            else:
                bm.faces.new((a[i], a[j], b[j], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = _from_bmesh(name, bm, role)
    return place(obj, loc, rot, scale)


def _curve_object(name, curve):
    return _link(bpy.data.objects.new(name, curve))


def tube(points, radii, role="body", sides=10, smooth=True, caps=True, resolution=6, name="tube"):
    """A tube through `points` whose radius follows `radii` (one per point, or a
    single number). Smooth tubes pass through every point on an auto-handled
    Bezier curve: limbs, tails, horns, straps, staffs."""
    if isinstance(radii, (int, float)):
        radii = [radii] * len(points)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_mode = "ROUND"
    cu.bevel_depth = 1.0
    cu.bevel_resolution = max(1, (sides - 4) // 2)
    cu.resolution_u = resolution
    cu.use_fill_caps = caps
    if smooth:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(points) - 1)
        for bp, p, r in zip(sp.bezier_points, points, radii):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = "AUTO"
            bp.radius = r
    else:
        sp = cu.splines.new("POLY")
        sp.points.add(len(points) - 1)
        for pt, p, r in zip(sp.points, points, radii):
            pt.co = (*p, 1)
            pt.radius = r
    obj = apply_modifiers(_curve_object(name, cu))
    bpy.data.curves.remove(cu)
    obj.name = name
    return set_role(obj, role)


def extrude(outline, depth, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), role="body", bevel=0.004, smooth=False, resolution=8, name="slab"):
    """A flat outline of (u, v) points standing in the XZ plane (u along x, v up),
    extruded `depth` along y and centred on it, with rounded edges. `smooth`
    passes a curve through the points instead of straight edges: blades,
    wings, feathers, ears, shields, crests."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = max(depth / 2 - bevel, 0.0001)
    cu.bevel_depth = bevel
    cu.bevel_resolution = 2 if bevel > 0 else 0
    cu.resolution_u = resolution
    if smooth:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(outline) - 1)
        for bp, (u, v) in zip(sp.bezier_points, outline):
            bp.co = (u, v, 0)
            bp.handle_left_type = bp.handle_right_type = "AUTO"
    else:
        sp = cu.splines.new("POLY")
        sp.points.add(len(outline) - 1)
        for pt, (u, v) in zip(sp.points, outline):
            pt.co = (u, v, 0, 1)
    sp.use_cyclic_u = True
    obj = apply_modifiers(_curve_object(name, cu))
    bpy.data.curves.remove(cu)
    obj.name = name
    # Curve space is XY with extrusion along z; stand it up into XZ.
    obj.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))
    set_role(obj, role)
    return place(obj, loc, rot, scale)


def skin(verts, edges, radii, role="body", root=0, subdiv=2, name="skin", smooth_branches=0.6):
    """A smooth body grown around an edge skeleton: torsos, limbs, necks, tails,
    whole creatures. `radii` gives one (rx, ry) or r per vertex, the half width
    of the body around that joint."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in edges], [])
    obj = _link(bpy.data.objects.new(name, me))
    mod = obj.modifiers.new("skin", "SKIN")
    mod.use_smooth_shade = True
    mod.branch_smoothing = smooth_branches
    sk = me.skin_vertices[0].data
    for v, r in zip(sk, radii):
        v.radius = (r, r) if isinstance(r, (int, float)) else tuple(r)
        v.use_root = False
    sk[root].use_root = True
    if subdiv:
        sub = obj.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = subdiv
    obj = apply_modifiers(obj)
    return set_role(obj, role)


def metaball(elements, role="body", resolution=0.02, threshold=0.6, name="meta"):
    """An organic mass fused from metaball elements. Each element is a dict with
    co, radius, and optional type ('BALL', 'ELLIPSOID', 'CAPSULE', 'CUBE'),
    size (x, y, z half sizes for non-balls), rot (Euler) and negative."""
    mb = bpy.data.metaballs.new(name)
    mb.resolution = mb.render_resolution = resolution
    mb.threshold = threshold
    for e in elements:
        el = mb.elements.new(type=e.get("type", "BALL"))
        el.co = e["co"]
        el.radius = e["radius"]
        if "size" in e:
            el.size_x, el.size_y, el.size_z = e["size"]
        if "rot" in e:
            el.rotation = Euler(e["rot"]).to_quaternion()
        el.use_negative = e.get("negative", False)
        el.stiffness = e.get("stiffness", 2.0)
    obj = _link(bpy.data.objects.new(name, mb))
    obj = apply_modifiers(obj)
    bpy.data.metaballs.remove(mb)
    obj.name = name
    return set_role(obj, role)


# ---------------------------------------------------------------- modifiers


def bevel_edges(obj, width, segments=2, angle=math.radians(40)):
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = angle
    mod.harden_normals = False
    return apply_modifiers(obj)


def subdivide(obj, levels=1):
    mod = obj.modifiers.new("sub", "SUBSURF")
    mod.levels = mod.render_levels = levels
    return apply_modifiers(obj)


def solidify(obj, thickness, offset=0.0):
    mod = obj.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = offset
    return apply_modifiers(obj)


def mirror(obj, axis="X", bisect=True):
    """Mirrors across the figure's centre plane (x = 0 by default)."""
    mod = obj.modifiers.new("mirror", "MIRROR")
    mod.use_axis = (axis == "X", axis == "Y", axis == "Z")
    mod.use_bisect_axis = (axis == "X" and bisect, axis == "Y" and bisect, axis == "Z" and bisect)
    mod.use_clip = False
    mod.merge_threshold = 0.0005
    return apply_modifiers(obj)


def displace(obj, strength, size=0.05, noise="CLOUDS", depth=2, midlevel=0.5, seed=0):
    """Pushes the surface along its normals by procedural noise: stone, bark,
    hide, flames. Subdivide first if the mesh is coarse."""
    tex = bpy.data.textures.new(f"{obj.name}_{noise}_{seed}", noise)
    tex.noise_scale = size
    if hasattr(tex, "noise_depth"):
        tex.noise_depth = depth
    if noise == "VORONOI":
        tex.distance_metric = "DISTANCE"
    mod = obj.modifiers.new("disp", "DISPLACE")
    mod.texture = tex
    mod.strength = strength
    mod.mid_level = midlevel
    mod.texture_coords = "LOCAL"
    return apply_modifiers(obj)


def remesh(obj, voxel=0.006, smooth=0, adaptivity=0.0):
    """Fuses overlapping parts into one clean shell (voxel remesh). Keeps the
    first material only, so remesh single-role parts."""
    role = obj.data.materials[0].name if obj.data.materials else "body"
    mod = obj.modifiers.new("remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.adaptivity = adaptivity
    obj = apply_modifiers(obj)
    if smooth:
        sm = obj.modifiers.new("smooth", "SMOOTH")
        sm.iterations = smooth
        sm.factor = 0.5
        obj = apply_modifiers(obj)
    return set_role(obj, role)


def deform(obj, fn):
    """Moves every vertex to `fn(Vector) -> Vector`: bends, tapers, bulges,
    ridges, folds, anything a sculptor would push by hand."""
    for v in obj.data.vertices:
        v.co = fn(v.co.copy())
    obj.data.update()
    return obj


def smooth(obj, iterations=4, factor=0.5):
    """Relaxes lumps and creases (Laplacian smoothing that keeps volume)."""
    mod = obj.modifiers.new("smooth", "LAPLACIANSMOOTH")
    mod.iterations = iterations
    mod.lambda_factor = factor
    mod.use_volume_preserve = True
    return apply_modifiers(obj)


def decimate(obj, ratio):
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = ratio
    mod.use_collapse_triangulate = False
    return apply_modifiers(obj)


def boolean(obj, cutter, op="DIFFERENCE"):
    mod = obj.modifiers.new("bool", "BOOLEAN")
    mod.object = cutter
    mod.operation = op
    mod.solver = "EXACT"
    obj = apply_modifiers(obj)
    remove(cutter)
    return obj


def join(objs, name):
    """Joins meshes into one object whose material slots are ROLES in order."""
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    for o in objs:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), depsgraph=dg)
        me.transform(o.matrix_world)
        roles = [m.name.split(".")[0] if m else "body" for m in me.materials] or ["body"]
        before = len(bm.faces)
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        for f in bm.faces[before:]:
            role = roles[f.material_index] if f.material_index < len(roles) else roles[0]
            f.material_index = ROLES.index(role)
        bpy.data.meshes.remove(me)
    for o in objs:
        remove(o)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for role in ROLES:
        me.materials.append(mat(role))
    return _link(bpy.data.objects.new(name, me))


# ---------------------------------------------------------------- finishing


def _bounds(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(axis=0), co.max(axis=0), co


def tris(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def role_areas(obj):
    area = {r: 0.0 for r in ROLES}
    for p in obj.data.polygons:
        area[ROLES[p.material_index]] += p.area
    total = sum(area.values()) or 1
    return {r: round(a / total, 3) for r, a in area.items() if a > 0}


def validate(kind, obj):
    """Problems that would break the game or its size order; empty when fine."""
    tier, _ = KINDS[kind]
    lo, hi, _ = _bounds(obj)
    problems = []
    top_lo, top_hi = TOP[tier]
    if not top_lo <= hi[2] <= top_hi:
        problems.append(f"top {hi[2]:.3f} outside {tier} range {top_lo}-{top_hi}")
    if tier == "flyer":
        if lo[2] < FLYER_FLOOR:
            problems.append(f"lowest point {lo[2]:.3f} below flyer floor {FLYER_FLOOR}")
    elif not -0.04 <= lo[2] <= 0.02:
        problems.append(f"lowest point {lo[2]:.3f}: grounded figures stand on z = 0")
    if max(abs(lo[0]), abs(hi[0])) > FOOTPRINT[0]:
        problems.append(f"x extent {lo[0]:.3f}..{hi[0]:.3f} beyond ±{FOOTPRINT[0]}")
    if max(abs(lo[1]), abs(hi[1])) > FOOTPRINT[1]:
        problems.append(f"y extent {lo[1]:.3f}..{hi[1]:.3f} beyond ±{FOOTPRINT[1]}")
    n = tris(obj)
    if n > TRI_BUDGET:
        problems.append(f"{n} triangles over the {TRI_BUDGET} budget")
    areas = role_areas(obj)
    if areas.get("body", 0) + areas.get("cloth", 0) < 0.4:
        problems.append(f"body + cloth cover {areas.get('body', 0) + areas.get('cloth', 0):.0%} of the surface; the side's colour needs at least 40%")
    if "AO" not in obj.data.color_attributes:
        problems.append("no AO colour attribute")
    return problems


def shade(obj, angle=35):
    me = obj.data
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.set_sharp_from_angle(angle=math.radians(angle))
    me.update()


def _convexity(me, co):
    """Per-vertex convexity in [-1, 1]: positive on edges that stick out,
    negative in creases, scaled to the figure's own spread."""
    n = len(me.vertices)
    nrm = np.empty(n * 3)
    me.vertices.foreach_get("normal", nrm)
    nrm = nrm.reshape(-1, 3)
    e = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    d = co[e[:, 1]] - co[e[:, 0]]
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
    acc = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(acc, e[:, 0], -(nrm[e[:, 0]] * d).sum(1))
    np.add.at(acc, e[:, 1], (nrm[e[:, 1]] * d).sum(1))
    np.add.at(cnt, e[:, 0], 1)
    np.add.at(cnt, e[:, 1], 1)
    curv = acc / np.maximum(cnt, 1)
    # One pass of neighbour averaging takes out single-vertex noise.
    acc = curv.copy()
    np.add.at(acc, e[:, 0], curv[e[:, 1]])
    np.add.at(acc, e[:, 1], curv[e[:, 0]])
    curv = acc / (cnt + 1)
    scale = np.percentile(np.abs(curv), 96) or 1
    return np.clip(curv / scale, -1, 1)


def bake_ao(obj, grounded=True, distance=0.12, samples=128, floor=0.32, edges=0.14):
    """Bakes ambient occlusion into the point colour attribute "AO", lifted so
    crevices darken without going black. Like a painted miniature, convex edges
    stay bright while creases and the feet darken. Emissive faces stay unshaded."""
    scene = bpy.context.scene
    engine = scene.render.engine
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = "CPU"
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.light_settings.distance = distance
    ground = None
    if grounded:
        ground = box((3, 3, 0.02), loc=(0, 0, -0.01), role="body", name="bake_ground")
    me = obj.data
    if "AO" in me.color_attributes:
        me.color_attributes.remove(me.color_attributes["AO"])
    attr = me.color_attributes.new("AO", "FLOAT_COLOR", "POINT")
    me.color_attributes.active_color = attr
    me.color_attributes.render_color_index = me.color_attributes.find("AO")
    scene.render.bake.target = "VERTEX_COLORS"
    for o in bpy.context.scene.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="AO")
    if ground:
        remove(ground)
    scene.render.engine = engine

    n = len(me.vertices)
    col = np.empty(n * 4, dtype=np.float32)
    attr.data.foreach_get("color", col)
    ao = col.reshape(-1, 4)[:, 0]
    lo, hi, co = _bounds(obj)
    h = np.clip((co[:, 2] - lo[2]) / max(hi[2] - lo[2], 1e-6) * 1.6, 0, 1)
    edge = _convexity(me, co)
    shade_ = (floor + (1 - floor) * ao) * (0.78 + 0.22 * h) * (1 - edges + edges * (0.5 + 0.5 * edge))
    eye = ROLES.index("eye")
    lit = np.zeros(n, dtype=bool)
    for p in me.polygons:
        if p.material_index == eye:
            lit[list(p.vertices)] = True
    shade_[lit] = 1.0
    out = np.repeat(np.clip(shade_, 0, 1), 4).reshape(-1, 4)
    out[:, 3] = 1
    attr.data.foreach_set("color", out.ravel().astype(np.float32))
    me.update()


def finish(kind, objs, ao_distance=0.12, ao_floor=0.32, sharp_angle=35, edges=0.14):
    """Joins the parts into the figure, shades and bakes it, and reports."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    obj = join(objs, kind)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    shade(obj, sharp_angle)
    bake_ao(obj, grounded=KINDS[kind][0] != "flyer", distance=ao_distance, floor=ao_floor, edges=edges)
    return obj


def report(kind, obj):
    lo, hi, _ = _bounds(obj)
    return {
        "kind": kind,
        "tier": KINDS[kind][0],
        "triangles": tris(obj),
        "vertices": len(obj.data.vertices),
        "min": [round(float(v), 3) for v in lo],
        "max": [round(float(v), 3) for v in hi],
        "roles": role_areas(obj),
        "problems": validate(kind, obj),
    }


def save(kind, obj):
    """Writes blend/<Kind>.blend and models/<kind>.glb."""
    os.makedirs(BLEND_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_palette(KINDS[kind][1] if KINDS[kind][1] != "either" else "light")
    for o in bpy.context.scene.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    glb = os.path.join(MODEL_DIR, kind.lower() + ".glb")
    bpy.ops.export_scene.gltf(
        filepath=glb,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_texcoords=False,
        export_normals=True,
        export_tangents=False,
        export_vertex_color="NAME",
        export_vertex_color_name="AO",
        export_all_vertex_colors=False,
        export_materials="EXPORT",
        export_image_format="NONE",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_skins=False,
        export_morph=False,
        export_extras=False,
        export_yup=True,
        # Roughly halves the file; the game decodes it with three's MeshoptDecoder.
        export_meshopt_compression_enable=True,
    )
    # The startup file browser remembers the home folder; keep it out of the .blend.
    # Fill the whole buffer first, since clearing it alone leaves the old bytes.
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "FILE_BROWSER" and space.params:
                    space.params.directory = b"/" * 1023
                    space.params.directory = b""
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BLEND_DIR, kind + ".blend"), compress=True)
    return glb


# ---------------------------------------------------------------- previews

BACKDROP = 0x101B23


def _stage():
    """Lights close to the game's: a hemisphere of sky and ground light and a
    warm sun from the upper left front."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True
    scene.eevee.taa_render_samples = 32
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = -1
    ramp.inputs["From Max"].default_value = 1
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    sock = {s.identifier: s for s in mix.inputs}
    sock["A_Color"].default_value = hexcolor(0x28334C)
    sock["B_Color"].default_value = hexcolor(0xE5F5EB)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.85
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], sock["Factor_Float"])
    nt.links.new({s.identifier: s for s in mix.outputs}["Result_Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs[0])
    sun = bpy.data.objects.get("preview_sun")
    if not sun:
        data = bpy.data.lights.new("preview_sun", "SUN")
        data.energy = 3.6
        data.color = hexcolor(0xFFE1B6)[:3]
        data.angle = math.radians(3)
        sun = _link(bpy.data.objects.new("preview_sun", data))
    # The game's sun sits at (-10, 22, 12) in y-up space: left, high, in front.
    direction = Vector((-10, -12, 22)).normalized()
    sun.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.objects.get("preview_cam")
    if not cam:
        cam = _link(bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam")))
    scene.camera = cam
    return cam


def _look(cam, target, azimuth, elevation, distance):
    """Camera on a sphere round `target`; azimuth 0 looks at the figure's front
    (from -Y), positive turns toward its left side (+X)."""
    d = Vector((math.sin(azimuth) * math.cos(elevation), -math.cos(azimuth) * math.cos(elevation), math.sin(elevation)))
    cam.location = Vector(target) + d * distance
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()


def _render(path, w, h):
    scene = bpy.context.scene
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    os.remove(path)
    return px.reshape(h, w, 4)


def _over(px, color=BACKDROP):
    bg = np.array([((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255])
    a = px[:, :, 3:4]
    out = np.empty_like(px)
    out[:, :, :3] = px[:, :, :3] * a + bg * (1 - a)
    out[:, :, 3] = 1
    return out


def _write(px, path):
    h, w = px.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), w, h, alpha=True)
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


VIEWS = (
    ("board", 0.5, 40),
    ("front", 0.0, 8),
    ("side", math.pi / 2, 5),
    ("back", math.pi * 0.8, 25),
)


def render_sheet(obj, path, tile=(480, 600)):
    """Four views (board angle, front, left side, back) in the Light palette on
    the top row and the Dark palette below."""
    cam = _stage()
    cam.data.lens_unit = "FOV"
    cam.data.angle = math.radians(30)
    lo, hi, _ = _bounds(obj)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    distance = radius / math.sin(math.radians(15)) * 1.02
    rows = []
    for palette in ("light", "dark"):
        set_palette(palette)
        tiles = []
        for name, az, el in VIEWS:
            _look(cam, centre, az, math.radians(el), distance)
            tiles.append(_over(_render(path + f".{name}.png", *tile)))
        rows.append(np.concatenate(tiles, axis=1))
    # Image rows run bottom to top.
    _write(np.concatenate(rows[::-1], axis=0), path)
    set_palette("light")


# The board draws a figure 3.2 times its modelled size on a plinth 1.1 times its
# size, squares 2.65 apart, seen from 40 degrees above the horizon.
BOARD_FIGURE = 3.2
BOARD_PLINTH = 1.1
TILE = 2.65


def _plinth(palette):
    if palette == "light":
        p = cylinder(0.48, 0.44, 0.15, role="body", segments=40, name="plinth")
    else:
        p = cylinder(0.52, 0.44, 0.15, role="body", segments=6, name="plinth")
    m = bpy.data.materials.new("plinth_" + palette)
    m.diffuse_color = hexcolor(PLINTH[palette])
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hexcolor(PLINTH[palette])
    p.data.materials.clear()
    p.data.materials.append(m)
    p.scale = (BOARD_PLINTH,) * 3
    return p


def import_figure(kind):
    """Imports models/<kind>.glb with its materials mapped back to the
    role materials, so palettes switch it too. None when it does not exist."""
    path = os.path.join(MODEL_DIR, kind.lower() + ".glb")
    if not os.path.exists(path):
        return None
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=path)
    except RuntimeError as error:
        # Another build may be writing it right now; leave it out of this preview.
        print(f"skipping {kind}: {error}")
        for o in set(bpy.data.objects) - before:
            remove(o)
        return None
    new = [o for o in bpy.data.objects if o not in before]
    if not any(o.type == "MESH" for o in new):
        return None
    meshes = [o for o in new if o.type == "MESH"]
    for o in meshes:
        o.data.transform(o.matrix_world)
        o.matrix_world = Matrix.Identity(4)
        o.parent = None
        for i, m in enumerate(o.data.materials):
            if m:
                o.data.materials[i] = mat(m.name.split(".")[0])
        if "AO" not in o.data.color_attributes and o.data.color_attributes:
            o.data.color_attributes[0].name = "AO"
    for o in new:
        if o.type != "MESH":
            remove(o)
    return join(meshes, kind + "_import") if len(meshes) > 1 else meshes[0]


def render_board(figures, path, px_per_unit=90):
    """The figures in a row at board scale, Light row above Dark, as the board
    camera sees them on a desktop screen. `figures` is a list of (kind, object)."""
    cam = _stage()
    cam.data.type = "ORTHO"
    # ortho_scale spans the width whatever the strip's aspect, so one figure
    # alone is drawn at the same scale as a row of them.
    cam.data.sensor_fit = "HORIZONTAL"
    n = len(figures)
    width = n * TILE + 1.0
    w = int(width * px_per_unit)
    # Tall enough for a commander (top 1.45, 4.8 at board scale) and the plinth.
    h = int(6.6 * px_per_unit)
    cam.data.ortho_scale = width
    xs = [(i - (n - 1) / 2) * TILE for i in range(n)]
    for _, obj in figures:
        obj.hide_render = True
    copies = []
    for (kind, obj), x in zip(figures, xs):
        copy = duplicate(obj, f"board_{kind}")
        copy.hide_render = False
        copy.data.transform(Matrix.Scale(BOARD_FIGURE, 4))
        # Flyers are scaled about their hover height and settled toward the
        # plinth, as the game does.
        flyer = KINDS[kind][0] == "flyer"
        copy.location = (x, 0, -0.8395 if flyer else 0.165)
        copies.append(copy)
    rows = []
    for palette, yaw in (("light", 0.5), ("dark", -0.5)):
        set_palette(palette)
        plinths = []
        for x, copy in zip(xs, copies):
            # The board turns each piece toward the camera and the enemy side.
            copy.rotation_euler = (0, 0, yaw)
            p = _plinth(palette)
            p.location = (x, 0, 0.075 * BOARD_PLINTH)
            plinths.append(p)
        _look(cam, (0, 0, 2.2), 0.0, math.radians(40), 40)
        rows.append(_over(_render(path + f".{palette}.png", w, h)))
        for p in plinths:
            remove(p)
    for c in copies:
        remove(c)
    for _, obj in figures:
        obj.hide_render = False
    cam.data.type = "PERSP"
    _write(np.concatenate(rows[::-1], axis=0), path)
    set_palette("light")
