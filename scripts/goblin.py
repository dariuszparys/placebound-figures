"""Goblin: a wiry, hunched Dark skirmisher caught winding up a fast club strike.
An oversized head with huge ragged ears, a long hooked nose and a wide fanged
grin under beady glowing eyes; skinny knobbly limbs, big clawed hands and
long-toed feet; a crude spiked club cocked over the right shoulder, a scrap of
leather vest, a ragged loincloth and a belt with a pouch and a small knife.

The figure faces -Y; its right hand is on -X, its left on +X.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

import lib

TAU = math.tau


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a)


def frame(origin, ydir, zhint=(0, 0, 1)):
    """A matrix whose local y runs along `ydir` and z is as close to `zhint` as it can be."""
    y = Vector(ydir).normalized()
    z = Vector(zhint)
    z = (z - y * z.dot(y)).normalized()
    x = y.cross(z)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = x[i], y[i], z[i], origin[i]
    return m


def zframe(origin, zdir, xhint=(1, 0, 0)):
    """A matrix whose local z runs along `zdir` and x is as close to `xhint` as it can be."""
    z = Vector(zdir).normalized()
    x = Vector(xhint)
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = x[i], y[i], z[i], origin[i]
    return m


def put(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def flip(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def mirrored(obj, name=None):
    """A copy mirrored across x = 0, turned right way out."""
    new = lib.duplicate(obj, name or obj.name + "_m")
    new.data.transform(Matrix.Scale(-1, 4, Vector((1, 0, 0))))
    return flip(new)


def mesh_object(name, bm, slots):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    for r in slots:
        me.materials.append(lib.mat(r))
    return obj


def surface(fn, nu, nv, slots=("body",), pick=None, name="surface"):
    """A grid surface through fn(u, v) for u, v in [0, 1]; pick(u, v) gives the
    slot index of the face around (u, v)."""
    bm = bmesh.new()
    grid = [[bm.verts.new(fn(i / nu, j / nv)) for j in range(nv + 1)] for i in range(nu + 1)]
    for i in range(nu):
        for j in range(nv):
            f = bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
            if pick:
                f.material_index = pick((i + 0.5) / nu, (j + 0.5) / nv)
    return mesh_object(name, bm, slots)


def shell(obj, thickness, offset=-1.0, mat_offset=0, rim_offset=0):
    """Solidify that can give the back and rim faces other material slots."""
    mod = obj.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = offset
    mod.use_even_offset = True
    mod.material_offset = mat_offset
    mod.material_offset_rim = rim_offset
    return lib.apply_modifiers(obj)


def cone(base, tip, r, role="metal", sides=8, name="cone"):
    return lib.tube([base, tip], [r, r * 0.08], role=role, sides=sides, smooth=False, name=name)


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def keep_largest(bm):
    """Deletes every island but the largest."""
    seen = set()
    islands = []
    for f in bm.faces:
        if f in seen:
            continue
        stack, island = [f], []
        seen.add(f)
        while stack:
            g = stack.pop()
            island.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
        islands.append(island)
    islands.sort(key=len, reverse=True)
    dead = [f for isl in islands[1:] for f in isl]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")


def boundary_loops(bm):
    """The mesh's open edges chained into loops of verts."""
    left = {e for e in bm.edges if e.is_boundary}
    loops = []
    while left:
        e = left.pop()
        start, v = e.verts
        loop = [start, v]
        while True:
            nxt = [f for f in v.link_edges if f in left]
            if not nxt:
                break
            left.discard(nxt[0])
            v = nxt[0].other_vert(v)
            if v is start:
                break
            loop.append(v)
        loops.append(loop)
    return loops


def resample(points, step, closed=False, smooth=3):
    """Evenly spaced, gently smoothed points along a polyline."""
    pts = [Vector(p) for p in points] + ([Vector(points[0])] if closed else [])
    seg = [(b - a).length for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    n = max(2, round(total / step))
    out, acc, i = [], 0.0, 0
    for k in range(n if closed else n + 1):
        d = total * k / n
        while i < len(seg) - 1 and acc + seg[i] < d:
            acc += seg[i]
            i += 1
        t = 0.0 if seg[i] == 0 else min(1.0, (d - acc) / seg[i])
        out.append(pts[i].lerp(pts[i + 1], t))
    for _ in range(smooth):
        m = len(out)
        new = []
        for k in range(m):
            if not closed and k in (0, m - 1):
                new.append(out[k])
            else:
                new.append(out[k] * 0.5 + (out[k - 1] + out[(k + 1) % m]) * 0.25)
        out = new
    return out


def cut_shell(src, keep, lift, thickness, role, relax=8, name="cut", rim=None, loops=None):
    """A garment cut from a copy of `src`: the faces whose centre passes `keep`,
    with the cut edge relaxed into smooth curves, lifted off the surface and
    given thickness; `rim` gives the cut edges their own role. `loops`, a list,
    receives the cut edges as loops of (point, normal) before the thickening."""
    o = lib.duplicate(src, name)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    dead = [f for f in bm.faces if not keep(f.calc_center_median())]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    keep_largest(bm)
    for _ in range(relax):
        moves = {}
        for v in bm.verts:
            if not v.is_boundary:
                continue
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                moves[v] = v.co * 0.4 + (nb[0].co + nb[1].co) * 0.3
        for v, c in moves.items():
            v.co = c
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * lift
    if loops is not None:
        loops += [[(v.co.copy(), v.normal.copy()) for v in loop] for loop in boundary_loops(bm)]
    bm.to_mesh(o.data)
    bm.free()
    lib.set_role(o, role)
    if rim:
        o.data.materials.append(lib.mat(rim))
        return shell(o, thickness, offset=1.0, rim_offset=1)
    return lib.solidify(o, thickness, offset=1.0)


def closed_tube(points, r, role, sides=8, resolution=2, name="ring"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_mode = "ROUND"
    cu.bevel_depth = r
    cu.bevel_resolution = max(1, (sides - 4) // 2)
    cu.resolution_u = resolution
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(points) - 1)
    for bp, p in zip(sp.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    sp.use_cyclic_u = True
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj = lib.apply_modifiers(obj)
    bpy.data.curves.remove(cu)
    obj.name = name
    return lib.set_role(obj, role)


# ---------------------------------------------------------------- pose

# Small round tubes and knuckles would otherwise get hard edges between
# their facets; 50 degrees keeps them soft and the bevelled metal crisp.
FINISH = {"sharp_angle": 50}

# Skeleton of the wiry body: a hunched spine with the neck thrust forward,
# splayed crouching legs (left foot forward, right heel up), the right arm
# cocked high with the club and the left reaching forward with claws spread.
J = {
    "pelvis": V(0, 0.036, 0.19),
    "waist": V(0, 0.022, 0.23),
    "chest": V(0, 0.002, 0.27),
    "back": V(0, 0.008, 0.306),
    "neck0": V(0, -0.034, 0.33),
    "neck1": V(0, -0.068, 0.35),
    "r_hip": V(-0.046, 0.03, 0.18),
    "r_thigh": V(-0.082, 0.004, 0.162),
    "r_knee": V(-0.112, -0.024, 0.146),
    "r_shin": V(-0.111, -0.008, 0.128),
    "r_calf": V(-0.108, 0.024, 0.104),
    "r_ankle": V(-0.1, 0.07, 0.058),
    "l_hip": V(0.046, 0.026, 0.18),
    "l_thigh": V(0.08, -0.028, 0.174),
    "l_knee": V(0.106, -0.074, 0.166),
    "l_shin": V(0.105, -0.07, 0.145),
    "l_calf": V(0.103, -0.062, 0.106),
    "l_ankle": V(0.096, -0.05, 0.048),
    "r_sho": V(-0.078, 0.018, 0.324),
    "r_bicep": V(-0.122, 0.044, 0.334),
    "r_elbow": V(-0.165, 0.05, 0.33),
    "r_fore": V(-0.17, 0.046, 0.378),
    "r_wrist": V(-0.168, 0.038, 0.422),
    "l_sho": V(0.078, -0.022, 0.318),
    "l_bicep": V(0.116, -0.03, 0.302),
    "l_elbow": V(0.15, -0.05, 0.28),
    "l_elb2": V(0.156, -0.069, 0.281),
    "l_fore": V(0.162, -0.094, 0.284),
    "l_wrist": V(0.17, -0.132, 0.292),
}
BONES = [
    ("pelvis", "waist"), ("waist", "chest"), ("chest", "back"), ("back", "neck0"), ("neck0", "neck1"),
    ("pelvis", "r_hip"), ("r_hip", "r_thigh"), ("r_thigh", "r_knee"), ("r_knee", "r_shin"), ("r_shin", "r_calf"), ("r_calf", "r_ankle"),
    ("pelvis", "l_hip"), ("l_hip", "l_thigh"), ("l_thigh", "l_knee"), ("l_knee", "l_shin"), ("l_shin", "l_calf"), ("l_calf", "l_ankle"),
    ("back", "r_sho"), ("r_sho", "r_bicep"), ("r_bicep", "r_elbow"), ("r_elbow", "r_fore"), ("r_fore", "r_wrist"),
    ("back", "l_sho"), ("l_sho", "l_bicep"), ("l_bicep", "l_elbow"), ("l_elbow", "l_elb2"), ("l_elb2", "l_fore"), ("l_fore", "l_wrist"),
]
RADII = {
    "pelvis": (0.052, 0.046), "waist": (0.05, 0.048), "chest": (0.064, 0.052), "back": (0.07, 0.062),
    "neck0": 0.034, "neck1": 0.03,
    "r_hip": 0.028, "r_thigh": 0.023, "r_knee": 0.027, "r_shin": 0.015, "r_calf": 0.019, "r_ankle": 0.012,
    "l_hip": 0.028, "l_thigh": 0.023, "l_knee": 0.027, "l_shin": 0.015, "l_calf": 0.019, "l_ankle": 0.012,
    "r_sho": 0.026, "r_bicep": 0.0185, "r_elbow": 0.0195, "r_fore": 0.019, "r_wrist": 0.012,
    "l_sho": 0.026, "l_bicep": 0.0185, "l_elbow": 0.0195, "l_elb2": 0.019, "l_fore": 0.0168, "l_wrist": 0.012,
}

SCALE = 1.06
HEAD_POS = V(0, -0.108, 0.388)
# Chin up, so the face shows from the board camera; cocked a little.
HEAD_ROT = (-0.55, 0.12, 0.06)
HEAD_M = Matrix.LocRotScale(HEAD_POS, Euler(HEAD_ROT), V(1, 1, 1))


# ---------------------------------------------------------------- body


def body():
    names = list(J)
    idx = {n: i for i, n in enumerate(names)}
    verts = [J[n] for n in names]
    edges = [(idx[a], idx[b]) for a, b in BONES]
    radii = [RADII[n] for n in names]
    # Left undecimated so the vest and belt are cut and fitted to the smooth
    # surface; build() decimates it afterwards.
    return lib.skin(verts, edges, radii, role="body", name="body")


# ---------------------------------------------------------------- head

HEAD_R = (0.076, 0.072, 0.064)


def _dir(az, el):
    return V(math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))


# (azimuth, elevation, angular width, height): sculpted bumps on the skull.
HEAD_BUMPS = [
    (math.pi, 0.55, 0.75, 0.018),  # skull swelling up and back
    (0.0, 0.75, 0.35, -0.006),  # sloping forehead
    (0.0, -0.4, 0.55, 0.02),  # muzzle pushed forward
    (0.62, -0.14, 0.28, 0.016),  # cheeks bunched up by the grin
    (-0.62, -0.14, 0.28, 0.016),
    (0.95, -0.55, 0.35, 0.01),  # jaw corners
    (-0.95, -0.55, 0.35, 0.01),
    (0.0, -0.9, 0.26, 0.012),  # small pointed chin
    (0.2, 0.3, 0.15, 0.014),  # scowling brow: low inside, high outside
    (-0.2, 0.3, 0.15, 0.014),
    (0.5, 0.42, 0.18, 0.011),
    (-0.5, 0.42, 0.18, 0.011),
    (0.36, 0.17, 0.15, -0.012),  # eye sockets
    (-0.36, 0.17, 0.15, -0.012),
    (1.45, -0.2, 0.4, -0.006),  # hollow temples under the ears
    (-1.45, -0.2, 0.4, -0.006),
]
_BUMPS = [(_dir(a, e), w, h) for a, e, w, h in HEAD_BUMPS]


def head_point(d):
    d = Vector(d).normalized()
    a, b, c = HEAD_R
    r = 1 / math.sqrt((d.x / a) ** 2 + (d.y / b) ** 2 + (d.z / c) ** 2)
    for cd, w, h in _BUMPS:
        ang = math.acos(max(-1.0, min(1.0, d.dot(cd))))
        r += h * math.exp(-((ang / w) ** 2))
    return d * r


def surf(az, el, out=0.0):
    """A point on the head surface (head-local), `out` off it."""
    d = _dir(az, el)
    return head_point(d) + d * out


MOUTH_T = -0.42  # ring parameter of the grin's centre line
MOUTH_B = 0.21  # half height of the band of rings bent to follow the grin
MOUTH_SPAN = 1.16
MOUTH_INNER = "trim"


def mouth_lift(az):
    """How far the grin curls up toward the cheeks, fading round the back."""
    a = min(abs(az), 1.1)
    return 0.5 * a * a * math.exp(-(max(0.0, abs(az) - 1.0) / 0.4) ** 2)


def mouth_el(az):
    return MOUTH_T + mouth_lift(az)


def mouth_open(az):
    """Half height (radians) of the open grin, widest in the middle."""
    return 0.096 * max(0.0, 1 - (az / MOUTH_SPAN) ** 2) ** 0.6 + 0.004


def _warp(az, t):
    """Bends the rings to follow the grin: inside the mouth band the rings at
    m = +-0.5 run exactly along the lip edges and the ones between line the open
    mouth; outside it the lift fades out toward the poles, so rings never cross."""
    lift = mouth_lift(az)
    m = (t - MOUTH_T) / MOUTH_B
    if abs(m) <= 1:
        op = mouth_open(az)
        if abs(m) <= 0.5:
            off = m / 0.5 * op
        else:
            off = math.copysign(op + (abs(m) - 0.5) / 0.5 * (MOUTH_B - op), m)
        return MOUTH_T + lift + off
    top, bottom = MOUTH_T + MOUTH_B, MOUTH_T - MOUTH_B
    if t > top:
        g = math.cos(math.pi / 2 * (t - top) / (math.pi / 2 - top)) ** 2
    else:
        g = math.cos(math.pi / 2 * (bottom - t) / (bottom + math.pi / 2)) ** 2
    return t + lift * g


def _carve(az, el):
    """Radial offset that opens the grin and rolls the lips round it."""
    window = math.exp(-(max(0.0, abs(az) - MOUTH_SPAN) / 0.12) ** 2)
    if window < 1e-3:
        return 0.0
    dm = el - mouth_el(az)
    op = mouth_open(az)
    off = 0.0
    if abs(dm) < op * 0.999:
        off -= 0.02 * math.sqrt(1 - (dm / op) ** 2) * min(1.0, op / 0.03)
    upper = 1.2 if dm > 0 else 1.0
    off += 0.0045 * upper * math.exp(-(((abs(dm) - op - 0.03) / 0.028) ** 2))
    return off * window


def skull():
    na = 32
    base = [-math.pi / 2 + math.pi * k / 22 for k in range(1, 22)]
    band = [MOUTH_T + MOUTH_B * i / 6 for i in range(-6, 7)]
    ts = sorted([t for t in base if abs(t - MOUTH_T) > MOUTH_B + 0.05] + band)
    bm = bmesh.new()
    rings = []
    for t in ts:
        ring = []
        for k in range(na):
            az = -math.pi + TAU * k / na
            el = _warp(az, t)
            d = _dir(az, el)
            ring.append(bm.verts.new(head_point(d) + d * _carve(az, el)))
        rings.append(ring)
    south = bm.verts.new(head_point(V(0, 0, -1)))
    north = bm.verts.new(head_point(V(0, 0, 1)))
    for (ta, a), (tb, b) in zip(zip(ts, rings), zip(ts[1:], rings[1:])):
        inside = abs(ta - MOUTH_T) <= MOUTH_B * 0.51 and abs(tb - MOUTH_T) <= MOUTH_B * 0.51
        for k in range(na):
            j = (k + 1) % na
            f = bm.faces.new((a[k], a[j], b[j], b[k]))
            # The open grin's throat takes the accent, so the teeth stand out
            # against it from the board camera.
            if inside and abs(-math.pi + TAU * (k + 0.5) / na) < MOUTH_SPAN * 0.97:
                f.material_index = 1
    for k in range(na):
        j = (k + 1) % na
        bm.faces.new((south, rings[0][j], rings[0][k]))
        bm.faces.new((north, rings[-1][k], rings[-1][j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_object("skull", bm, ("body", MOUTH_INNER))


def head_parts():
    parts = [skull()]

    # Teeth: a crooked row hanging from the upper lip, a shorter one below,
    # and two lower tusks jutting up past the lip.
    for i in range(7):
        a = -0.8 + 1.6 * (i + 0.5) / 7
        el, op = mouth_el(a), mouth_open(a)
        length = (0.85 if i % 2 else 1.0) * op
        top = surf(a, el + op * 1.1, -0.006)
        bot = surf(a, el + op * 1.1 - length * 1.1, -0.001)
        parts.append(cone(top, bot, 0.0084, role="skin", sides=8, name="tooth_u"))
    for i in range(6):
        a = -0.62 + 1.24 * (i + 0.5) / 6
        el, op = mouth_el(a), mouth_open(a)
        bot = surf(a, el - op * 1.05, -0.006)
        top = surf(a, el - op * 1.05 + op * 0.7, -0.002)
        parts.append(cone(bot, top, 0.0062, role="skin", sides=8, name="tooth_l"))
    for s in (-1, 1):
        a = s * 0.44
        el, op = mouth_el(a), mouth_open(a)
        base = surf(a, el - op * 0.95, -0.003)
        mid = surf(a + s * 0.024, el + op * 0.75, 0.0095)
        tip = surf(a + s * 0.085, el + op * 2.7, 0.011)
        parts.append(lib.tube([base, mid, tip], [0.0094, 0.0074, 0.001], role="skin", sides=6, resolution=3, name="tusk"))

    # Long hooked nose growing from under the eyes and drooping over the grin,
    # with flared nostrils and a wart.
    # A deep, narrow bridge with a hump, hooking down to a bulbous tip.
    nose = [V(0, -0.064, 0.006), V(0, -0.083, 0.0025), V(0, -0.1, -0.0065), V(0, -0.1115, -0.016), V(0, -0.1175, -0.0255), V(0, -0.116, -0.0335)]
    n = lib.tube(nose, [0.021, 0.0175, 0.0135, 0.011, 0.0088, 0.0062], role="body", sides=10, resolution=3, name="nose")
    lib.deform(n, lambda q: V(q.x / (1 + 0.3 * smoothstep(-0.104, -0.068, q.y)), q.y, q.z))
    parts.append(n)
    parts.append(lib.sphere(0.0102, loc=(0, -0.1165, -0.031), scale=(1.12, 1.0, 1.0), role="body", segments=10, rings=7, name="nose_tip"))
    for s in (-1, 1):
        parts.append(lib.sphere(0.0085, loc=(s * 0.0095, -0.106, -0.0245), scale=(1, 1.25, 0.85), role="body", segments=10, rings=6, name="nostril"))
    parts.append(lib.sphere(0.0045, loc=(0.0065, -0.106, -0.006), role="body", segments=6, rings=5, name="wart"))
    for az, el, r in ((0.5, 0.95, 0.0065), (-2.2, 0.85, 0.006)):
        parts.append(lib.sphere(r, loc=surf(az, el, -r * 0.35), role="body", segments=6, rings=5, name="wart"))

    # Beady glowing eyes under a heavy brow slanting down to the nose.
    for s in (-1, 1):
        a, e = s * 0.37, 0.16
        parts.append(lib.sphere(0.0135, loc=surf(a, e, -0.002), scale=(1.25, 1, 0.82), role="eye", segments=10, rings=7, name="eye"))
        brow = [surf(s * 0.1, 0.17, 0.0), surf(s * 0.3, 0.27, 0.004), surf(s * 0.52, 0.33, 0.003), surf(s * 0.68, 0.32, -0.002)]
        parts.append(lib.tube(brow, [0.006, 0.0088, 0.0078, 0.004], role="body", sides=8, resolution=2, name="brow"))
    for p in parts:
        put(p, HEAD_M)
    return parts


# ---------------------------------------------------------------- ears

EAR_INNER = "trim"


EAR_L = 0.16
# Bites out of the top edge: one wide ragged tear and one small nick
# (centre u, half width in u, depth as a share of the width).
EAR_NOTCHES = ((0.55, 0.065, 0.75), (0.78, 0.025, 0.35))
# The left (+x) ear sweeps about 23 degrees up and 32 back, its cup facing
# forward and out (normal . board camera about 0.88, . side view about 0.44),
# so it reads as a cupped ear from the board, the front and the side.
EAR_ROT = (0.35, -0.7, 0.15)
# The club-side (-x) ear has its own pose: about 30 degrees up but yawed
# forward to about 5 back, a touch shorter and drooping at the tip, so the
# tip lands left of the raised wrist and fist from the board camera instead
# of under them; sweeping it back would push it into the forearm.
EAR_ROT_CLUB = (0.35, -0.55, -0.35)
EAR_L_CLUB = 0.155
CUP = 1.15


def ear_notch(u):
    """A rounded bite profile (no kink at its edges, so the solidified back
    of the ear stays smooth); torn a little off-centre so it looks ripped."""
    out = 0.0
    for c, h, d in EAR_NOTCHES:
        t = (u - c) / h
        t *= 1.15 if t < 0 else 0.85
        out = max(out, d * max(0.0, 1 - abs(t)) ** 1.6)
    return out


def ear_thick(u):
    """Thick and fleshy at the root, thin at the tip."""
    return 0.012 + (0.004 - 0.012) * u


def ear(side):
    """A huge ragged leaf-shaped ear swept out, up and back: a deep cup with a
    dipped concha, a rolled rim along both edges, a torn bite and a nick out of
    the top and a ring through the lobe. The club-side ear droops at the tip."""
    L = EAR_L if side > 0 else EAR_L_CLUB

    def top(u):
        return 0.033 * (1 - u) ** 0.6 * (1 + 0.8 * u) * (1 - ear_notch(u))

    def bottom(u):
        lobe = 0.008 * math.sin(math.pi * min(1.0, max(0.0, (u - 0.08) / 0.26))) ** 2
        return 0.044 * (1 - u) ** 1.3 * (1 + 0.6 * u) + lobe

    def fn(u, v):
        s = 2 * v - 1
        w = top(u) if s > 0 else bottom(u)
        x = u * L
        z = 0.03 * u * u + s * w
        if side < 0:
            z -= 0.04 * smoothstep(0.6, 1.0, u) ** 2
        concha = 0.006 * smoothstep(0.08, 0.2, u) * (1 - smoothstep(0.45, 0.62, u)) * max(0.0, 1 - (s / 0.75) ** 2)
        y = -CUP * s * s * w + concha + 0.012 * u * u
        return V(x, y, z)

    def normal(u, v):
        e = 1e-3
        du = fn(min(1.0, u + e), v) - fn(max(0.0, u - e), v)
        dv = fn(u, min(1.0, v + e)) - fn(u, max(0.0, v - e))
        return du.cross(dv).normalized()

    def pick(u, v):
        s = 2 * v - 1
        lim = 0.62 if s > 0 else 0.7
        return 0 if (abs(s) < lim and 0.06 < u < 0.84) else 1

    (c1, h1, _), (c2, h2, _) = EAR_NOTCHES
    dense = [round(c1 + k * h1, 4) for k in (-1.2, -0.85, -0.55, -0.25, 0, 0.3, 0.65, 1.0, 1.4)]
    dense += [round(c2 + k * h2, 4) for k in (-1.2, -0.6, 0, 0.6, 1.2)]
    base = [round(i / 13, 4) for i in range(14)]
    us = sorted(set(dense + [u for u in base if min(abs(u - d) for d in dense) > 0.012]))
    vs = [j / 8 for j in range(9)]
    bm = bmesh.new()
    grid = [[bm.verts.new(fn(u, v)) for v in vs] for u in us]
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            f = bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
            f.material_index = pick((us[i] + us[i + 1]) / 2, (vs[j] + vs[j + 1]) / 2)
    e = mesh_object("ear", bm, (EAR_INNER, "body", "body"))
    # Tapered thickness: a solidify scaled by a vertex group running root to tip.
    vg = e.vertex_groups.new(name="taper")
    for i, u in enumerate(us):
        for j in range(len(vs)):
            vg.add([i * len(vs) + j], 1 - u, "REPLACE")
    mod = e.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = ear_thick(0)
    mod.offset = -1.0
    mod.use_even_offset = True
    mod.vertex_group = "taper"
    mod.thickness_vertex_group = ear_thick(1) / ear_thick(0)
    mod.material_offset = 1
    mod.material_offset_rim = 1
    e = lib.apply_modifiers(e)
    e.vertex_groups.clear()
    parts = [e]

    # Rolled rims wrapping both edges; the bites cut through the top one.
    def rim(v, u0, u1, n, torn0=False, torn1=False, res=2):
        pts, radii = [], []
        for k in range(n + 1):
            u = u0 + (u1 - u0) * k / n
            t = ear_thick(u)
            pts.append(fn(u, v) - normal(u, v) * t * 0.5)
            r = t * 0.5 + 0.0008 * (1 - 0.6 * u)
            # The rim thins out where a bite tore through it.
            if (k == 0 and torn0) or (k == n and torn1):
                r *= 0.45
            radii.append(r)
        return lib.tube(pts, radii, role="body", sides=6, resolution=res, name="ear_rim")
    parts.append(rim(1.0, 0.02, c1 - h1 * 0.8, 5, torn1=True))
    parts.append(rim(1.0, c1 + h1 * 0.8, c2 - h2 * 0.8, 3, True, True))
    parts.append(rim(1.0, c2 + h2 * 0.8, 0.985, 3, torn0=True))
    parts.append(rim(0.0, 0.02, 0.985, 11, res=1))
    # A ring through the lobe, its plane across the edge so it visibly pierces it.
    u = 0.2
    p, n = fn(u, 0.0), normal(u, 0.0)
    tan = (fn(u + 0.01, 0.0) - fn(u - 0.01, 0.0)).normalized()
    down = tan.cross(n).normalized()
    if down.z > 0:
        down = -down
    r = 0.0115
    c = p - n * ear_thick(u) * 0.5 + down * r * 0.55
    ring = [c + (n * math.cos(a) + down * math.sin(a)) * r for a in [TAU * k / 8 for k in range(8)]]
    parts.append(closed_tube(ring, 0.0028, "trim", sides=6, resolution=2, name="earring"))
    m = HEAD_M @ Matrix.LocRotScale(V(0.06, 0.014, 0.004), Euler(EAR_ROT if side > 0 else EAR_ROT_CLUB), V(1, 1, 1))
    for p in parts:
        put(p, m)
    if side < 0:
        flipped = [mirrored(p) for p in parts]
        for p in parts:
            lib.remove(p)
        parts = flipped
    return parts


# ---------------------------------------------------------------- club and hands

CLUB_GRIP = V(-0.168, 0.036, 0.452)
CLUB_DIR = V(0.12, 0.6, 0.8).normalized()
CLUB_M = zframe(CLUB_GRIP, CLUB_DIR, xhint=(-1, 0, 0))


def club():
    parts = []
    pts = [V(0.002, 0, -0.064), V(0, 0, -0.05), V(-0.002, 0, 0.0), V(0.003, 0.002, 0.07), V(0.0, 0.004, 0.13),
           V(-0.004, 0.0, 0.178), V(0.0, -0.003, 0.222), V(0.0, 0.0, 0.25), V(0, 0, 0.26)]
    radii = [0.006, 0.015, 0.0125, 0.014, 0.02, 0.032, 0.036, 0.027, 0.007]
    zs = [p.z for p in pts]

    def rad(z):
        return float(np.interp(z, zs, radii))
    c = lib.tube(pts, radii, role="skin", sides=10, resolution=3, name="club")

    def knots(p):
        a = math.atan2(p.y, p.x)
        k = 1 + 0.07 * math.sin(5 * a + p.z * 60) + 0.05 * math.sin(3 * a - p.z * 110)
        return V(p.x * k, p.y * k, p.z)
    lib.deform(c, knots)
    parts.append(c)
    # Crude iron spikes hammered through the head at odd angles and lengths,
    # and one through the crown.
    spikes = [(0.15, 0.3, 1.0, 0.1), (0.148, 2.5, 0.8, 0.45), (0.158, 4.4, 1.15, 0.2),
              (0.188, 1.35, 1.2, 0.35), (0.184, 3.3, 0.85, 0.0), (0.194, 5.6, 1.05, 0.5),
              (0.222, 0.15, 0.9, 0.55), (0.218, 2.2, 1.2, 0.25), (0.228, 4.3, 0.95, 0.7)]
    for z, a, k, tilt in spikes:
        d = V(math.cos(a), math.sin(a), tilt).normalized()
        base = V(0, 0, z) + V(math.cos(a), math.sin(a), 0) * (rad(z) - 0.01)
        parts.append(cone(base, base + d * 0.05 * k, 0.009, role="metal", sides=6, name="spike"))
    parts.append(cone(V(0, 0, 0.245), V(0.0, 0.004, 0.288), 0.0085, role="metal", sides=6, name="spike_top"))
    # Rawhide wraps on the grip above the fist and under the head, one loose end flapping.
    for z, tilt in ((0.046, 0.004), (0.128, 0.006)):
        r = rad(z) * 1.13 + 0.0015
        parts.append(closed_tube([V(math.cos(t) * r, math.sin(t) * r, z + tilt * math.sin(t)) for t in [TAU * k / 8 for k in range(8)]], 0.0042, "trim", sides=6, resolution=2, name="binding"))
    r = rad(0.128) * 1.13 + 0.002
    parts.append(lib.tube([V(r, 0.0, 0.128), V(r + 0.009, -0.004, 0.113), V(r + 0.014, 0.002, 0.093)], [0.0042, 0.0035, 0.0022], role="trim", sides=6, resolution=3, name="binding_end"))
    for p in parts:
        put(p, CLUB_M)
    return parts


def fist():
    """The right fist clamped round the club grip (club-local frame: grip on z)."""
    parts = []
    parts.append(lib.sphere(0.031, loc=(-0.008, 0.0, 0.0), scale=(1.0, 0.95, 1.15), role="body", segments=14, rings=8, name="fist"))
    for i in range(4):
        z = -0.021 + i * 0.014
        # Finger rolls wrap the grip; knuckles stand out on the back of the hand.
        parts.append(lib.tube([V(-0.026, 0.006, z), V(-0.015, -0.022, z), V(0.009, -0.024, z), V(0.021, -0.01, z)],
                              [0.0092, 0.009, 0.0085, 0.0072], role="body", sides=6, resolution=2, name="finger"))
        parts.append(lib.sphere(0.0095, loc=(-0.03, -0.006, z), role="body", segments=6, rings=5, name="knuckle"))
        parts.append(cone(V(0.02, -0.012, z), V(0.024, 0.0, z), 0.005, role="skin", sides=6, name="claw"))
    parts.append(lib.tube([V(-0.013, 0.022, -0.022), V(0.007, 0.024, -0.004), V(0.018, 0.013, 0.012)], [0.0102, 0.0086, 0.0064], role="body", sides=8, resolution=3, name="thumb"))
    for p in parts:
        put(p, CLUB_M)
    return parts


def claw_hand():
    """The open left hand reaching forward, long knobbly fingers hooked into claws."""
    wrist = J["l_wrist"]
    m = frame(wrist, V(0.3, -1, 0.05))
    parts = [lib.sphere(0.03, loc=(0, 0.03, 0.0), scale=(0.95, 0.8, 0.5), role="body", segments=14, rings=8, name="palm"),
             lib.sphere(0.0128, loc=(0, 0.009, 0.0), scale=(1.0, 1.6, 0.8), role="body", segments=10, rings=6, name="wrist_ball"),
             # A low ridge of knuckles across the back of the hand.
             lib.tube([V(-0.021, 0.045, 0.006), V(0.0, 0.049, 0.0085), V(0.021, 0.045, 0.006)], [0.0068, 0.0074, 0.0068], role="body", sides=6, resolution=2, name="knuckle_ridge")]
    for x, spread, length in ((-0.018, -0.35, 1.0), (0.0, 0.0, 1.1), (0.018, 0.35, 1.0)):
        base = V(x, 0.05, 0.002)
        d = V(math.sin(spread), math.cos(spread), 0)
        k1 = base + d * 0.02 * length + V(0, 0, 0.005)
        k2 = k1 + d * 0.016 * length + V(0, 0, -0.009)
        tip = k2 + d * 0.005 + V(0, 0, -0.012)
        parts.append(lib.tube([base, k1, k2, tip], [0.0105, 0.0092, 0.008, 0.006], role="body", sides=6, resolution=2, name="finger"))
        parts.append(lib.sphere(0.0105, loc=base + V(0, 0, 0.003), role="body", segments=6, rings=5, name="knuckle"))
        parts.append(cone(tip + V(0, 0, 0.004), tip + d * 0.003 + V(0, 0, -0.01), 0.0055, role="skin", sides=6, name="claw"))
    thumb = [V(0.019, 0.014, -0.002), V(0.035, 0.031, -0.006), V(0.039, 0.046, -0.014)]
    parts.append(lib.tube(thumb, [0.0105, 0.0088, 0.0066], role="body", sides=8, resolution=3, name="thumb"))
    parts.append(cone(thumb[-1], thumb[-1] + V(0.002, 0.006, -0.009), 0.005, role="skin", sides=6, name="claw"))
    for p in parts:
        put(p, m)
    return parts


# ---------------------------------------------------------------- feet


def foot(ankle, heel, ball, toe_dir):
    """A bony bare foot: a slim heel tube capped by a heel ball, a sole tube
    arching up between heel and ball, an instep tube sloping from the ankle
    to the ball, a knob of ankle bone on the outside and three long knuckly
    toes ending in claws."""
    parts = []
    parts.append(lib.tube([ankle, heel], [0.013, 0.015], role="body", sides=10, resolution=2, caps=False, name="heel"))
    parts.append(lib.sphere(0.015, loc=heel, scale=(0.95, 1.1, 1.0), role="body", segments=8, rings=6, name="heel_ball"))
    # The sole arches up off the ground between heel and ball; the instep
    # slopes straight down from the ankle to the toes.
    parts.append(lib.tube([heel, (heel + ball) / 2 + V(0, 0, 0.009), ball], [0.0135, 0.0115, 0.013], role="body", sides=10, resolution=3, name="foot"))
    parts.append(lib.tube([ankle, ankle.lerp(ball, 0.5) + V(0, 0, 0.001), ball], [0.012, 0.0125, 0.0125], role="body", sides=10, resolution=2, caps=False, name="instep"))
    fwd = Vector(toe_dir).normalized()
    side = fwd.cross(V(0, 0, 1)).normalized()
    out = side if side.x * ankle.x > 0 else -side
    parts.append(lib.sphere(0.0078, loc=ankle + out * 0.0095 + V(0, 0, -0.003), scale=(1, 1, 1.25), role="body", segments=7, rings=5, name="ankle_bone"))
    for s in (-1, 0, 1):
        d = (fwd + side * s * 0.42).normalized()
        L = 0.062 if s == 0 else 0.054
        b = ball + side * s * 0.0115
        k = b + d * L * 0.55 + V(0, 0, 0.008)
        t = b + d * L + V(0, 0, -(b.z - 0.0068))
        parts.append(lib.tube([b, k, t], [0.009, 0.0075, 0.006], role="body", sides=6, resolution=2, name="toe"))
        parts.append(lib.sphere(0.0075, loc=k, role="body", segments=6, rings=5, name="toe_knuckle"))
        parts.append(cone(t + V(0, 0, 0.002), t + d * 0.011 + V(0, 0, -0.006), 0.0052, role="skin", sides=6, name="toe_claw"))
    return parts


# ---------------------------------------------------------------- clothes


def belt_ring(body_co, centre, n=28):
    """The belt line: slung low at the front, riding up the hunched back,
    snug to the body all the way round."""
    pts = []
    rel = body_co[:, :2] - np.array(centre)
    ang = np.arctan2(rel[:, 0], -rel[:, 1])
    rad = np.linalg.norm(rel, axis=1)
    for i in range(n):
        a = -math.pi + TAU * i / n
        back = max(0.0, -math.cos(a))
        front = max(0.0, math.cos(a))
        z = 0.2 - 0.008 * front + 0.015 * back
        d = np.abs((ang - a + math.pi) % TAU - math.pi)
        sel = (d < TAU / n * 0.75) & (np.abs(body_co[:, 2] - z) < 0.008)
        r = float(rad[sel].max()) if sel.any() else 0.04
        pts.append([r + 0.0045 - 0.002 * back, a, z])
    rs = [p[0] for p in pts]
    rs = [(rs[i - 1] + 2 * rs[i] + rs[(i + 1) % n]) / 4 for i in range(n)]
    return [V(centre[0] + math.sin(a) * r, centre[1] - math.cos(a) * r, z) for r, (_, a, z) in zip(rs, pts)]


def loincloth(body_co):
    parts = []
    centre = (0, 0.03)
    torso = body_co[np.abs(body_co[:, 0]) < 0.058]
    ring = belt_ring(torso, centre)
    parts.append(closed_tube(ring, 0.0064, "trim", sides=6, resolution=1, name="belt"))
    front = min(ring, key=lambda p: p.y)
    back = max(ring, key=lambda p: p.y)
    parts.append(lib.box((0.022, 0.008, 0.018), loc=front + V(0, -0.006, 0.0), role="metal", bevel=0.003, name="buckle"))

    def flap(top, length, width, dirn, flare, folds, ragged, seed):
        def fn(u, v):
            x = (u - 0.5) * width * (1 - 0.12 * v)
            hem = 1 - ragged * abs(math.sin(u * 11 + seed)) * smoothstep(0.55, 1.0, v)
            z = top.z - 0.004 - length * v * hem
            fold = 0.005 * math.sin(u * math.pi * folds + seed) * (0.3 + v)
            y = top.y + dirn * (0.006 + flare * v * v) + fold
            return (x, y, z)
        return fn

    f = surface(flap(front, 0.1, 0.072, -1, 0.022, 3, 0.25, 0.4), 9, 6, slots=("cloth",), name="flap_front")
    parts.append(lib.solidify(flip(f), 0.004))
    # The back flap hangs close behind the seat instead of flaring out like a tail.
    b = surface(flap(back, 0.095, 0.084, 1, 0.006, 4, 0.3, 1.7), 9, 6, slots=("cloth",), name="flap_back")
    parts.append(lib.solidify(b, 0.004))

    # A leather pouch hanging off the right hip: a rounded box with a cloth
    # flap folded over its mouth, tied shut with a thong.
    pm = frame(V(-0.074, -0.014, 0.17), V(0.7, 0.71, 0.0))
    box = lib.box((0.032, 0.017, 0.034), role="skin", bevel=0.0065, segments=3, name="pouch")
    lib.deform(box, lambda q: V(q.x * (1 + 0.12 * (0.017 - q.z) / 0.034), q.y * (1 + 0.25 * (1 - (q.x / 0.017) ** 2)), q.z))
    lid = lib.box((0.035, 0.02, 0.005), loc=(0, 0, 0.0175), role="cloth", bevel=0.002, name="pouch_lid")
    tongue = lib.extrude([(-0.0175, 0.0), (0.0175, 0.0), (0.0165, -0.012), (0.008, -0.021), (0.0, -0.023), (-0.008, -0.021), (-0.0165, -0.012)],
                         0.0035, loc=(0, -0.0105, 0.019), role="cloth", bevel=0.001, smooth=True, resolution=2, name="pouch_flap")
    stud = lib.sphere(0.0034, loc=(0, -0.0128, -0.001), role="metal", segments=8, rings=6, name="pouch_stud")
    thong = lib.tube([V(0.0, -0.013, -0.001), V(0.004, -0.015, -0.008), V(0.002, -0.0145, -0.016)], [0.0022, 0.0019, 0.0013], role="trim", sides=6, resolution=2, name="pouch_tie")
    loop = lib.tube([V(-0.009, 0.0, 0.02), V(-0.009, 0.002, 0.03), V(-0.008, 0.006, 0.036)], 0.0026, role="cloth", sides=6, resolution=2, name="pouch_loop")
    loop2 = lib.tube([V(0.009, 0.0, 0.02), V(0.009, 0.002, 0.03), V(0.008, 0.006, 0.036)], 0.0026, role="cloth", sides=6, resolution=2, name="pouch_loop")
    for q in (box, lid, tongue, stud, thong, loop, loop2):
        put(q, pm)
        parts.append(q)

    # A small knife thrust through the belt on the left hip, handle up and
    # forward, the blade slanting back and down past the hip.
    knife = zframe(V(0.068, 0.012, 0.2), V(-0.2, -0.82, 0.54), xhint=(0, 0, 1))
    k = [lib.cylinder(0.006, 0.0065, 0.032, loc=(0, 0, 0.022), role="cloth", segments=8, name="knife_grip"),
         lib.sphere(0.0072, loc=(0, 0, 0.04), role="metal", segments=8, rings=6, name="knife_pommel"),
         lib.box((0.026, 0.009, 0.006), loc=(0, 0, 0.004), role="metal", bevel=0.002, name="knife_guard"),
         lib.extrude([(-0.008, 0.0), (0.0075, 0.0), (0.0065, -0.042), (-0.002, -0.07), (-0.008, -0.04)], 0.0034, role="metal", bevel=0.0012, name="knife_blade")]
    for p in k:
        put(p, knife)
    parts += k
    return parts


VEST_LIFT = 0.007
VEST_T = 0.009


def vest_open(z):
    """Half width of the vest's front opening at height z."""
    return 0.012 + (0.35 - z) * 0.2


def vest(src):
    def hem(a):
        return 0.235 + 0.02 * (0.65 * abs(math.sin(a * 4.5 + 0.5)) + 0.35 * abs(math.sin(a * 11.3 + 1.1)))

    neck_a, neck_b = J["back"], J["neck1"]
    neck_d = neck_b - neck_a

    def keep(c):
        a = math.atan2(c.x, -(c.y - 0.03))
        if not (hem(a) < c.z < 0.4):
            return False
        if abs(c.x) > 0.074 + max(0.0, 0.3 - c.z) * 0.6:
            return False
        # Over the shoulders, with a ragged neckline clear of the neck.
        if c.z > 0.32:
            t = min(1.0, max(0.0, (c - neck_a).dot(neck_d) / neck_d.length_squared))
            # Snug across the back of the neck, scooped wider at the front.
            clear = 0.047 - 0.009 * smoothstep(0.0, 0.03, c.y)
            if (c - neck_a - neck_d * t).length < clear + 0.003 * math.sin(a * 7):
                return False
        if c.y < 0.0 and abs(c.x) < vest_open(c.z):
            return False
        return True
    loops = []
    v = cut_shell(src, keep, VEST_LIFT, VEST_T, "cloth", relax=4, name="vest", rim="skin", loops=loops)
    parts = [v]
    # Rolled skin-role edges, so the garment outlines itself in Dark: a fat
    # collar round the back of the neck running down the front opening, and
    # a thinner roll round each armhole.
    loops.sort(key=len, reverse=True)
    main, holes = loops[0], [l for l in loops[1:] if len(l) > 12 and abs(sum(p.x for p, _ in l) / len(l)) > 0.05]
    k0 = min(range(len(main)), key=lambda k: main[k][0].z)
    main = main[k0:] + main[:k0]
    run, best = [], []
    for p, n in main:
        if p.z > 0.258:
            run.append(p + n * VEST_T * 0.5)
        else:
            best, run = max(best, run, key=len), []
    best = max(best, run, key=len)
    if len(best) > 4:
        pts = resample(best, 0.0135, smooth=4)
        radii = [0.0062 + 0.0018 * smoothstep(0.31, 0.345, q.z) for q in pts]
        parts.append(lib.tube(pts, radii, role="skin", sides=6, resolution=1, name="collar"))
    for hole in holes:
        pts = resample([p + n * VEST_T * 0.5 for p, n in hole], 0.0145, closed=True, smooth=3)
        parts.append(closed_tube(pts, 0.006, "skin", sides=6, resolution=1, name="armhole"))
    # Two scavenged leather patches, one on the chest and one on the back,
    # stitched on with rawhide.
    for name, test in (("patch_front", lambda c: c.y < 0 and 0.034 < c.x < 0.066 and 0.262 + 0.12 * (c.x - 0.034) < c.z < 0.298 + 0.1 * (c.x - 0.034)),
                       ("patch_back", lambda c: c.y > 0.03 and -0.046 < c.x < -0.006 and 0.27 - 0.1 * (c.x + 0.046) < c.z < 0.312 - 0.06 * (c.x + 0.046))):
        ploops = []
        lift = VEST_LIFT + VEST_T + 0.0005
        parts.append(cut_shell(src, test, lift, 0.0025, "skin", relax=2, name=name, loops=ploops))
        edge = max(ploops, key=len)
        ring = resample([p for p, _ in edge], 0.0105, closed=True, smooth=2)
        nrm = [min(edge, key=lambda e: (e[0] - q).length_squared)[1] for q in ring]
        # Stitched along one side only, the rest left to curl loose.
        top = max(range(len(ring)), key=lambda i: ring[i].z)
        for i, (q, n) in enumerate(zip(ring, nrm)):
            if (i - top) % len(ring) > len(ring) * 0.5:
                continue
            tan = (ring[(i + 1) % len(ring)] - ring[i - 1]).normalized()
            across = n.cross(tan).normalized()
            c = q + n * 0.0026
            parts.append(lib.tube([c - across * 0.0045, c + across * 0.0045], 0.0013, role="trim", sides=4, smooth=False, caps=False, name="stitch"))
    # Rawhide laces criss-crossing the open gap below the chest.
    co = np.array([p.co[:] for p in src.data.vertices])

    def edge(side, z):
        x = side * (vest_open(z) + 0.004)
        near = co[(np.abs(co[:, 0] - x) < 0.01) & (np.abs(co[:, 2] - z) < 0.008) & (co[:, 1] < 0.0)]
        y = float(near[:, 1].min()) if len(near) else -0.05
        return V(x, y - VEST_LIFT - VEST_T * 0.6, z)
    for (za, zb) in ((0.252, 0.278), (0.278, 0.252)):
        a, b = edge(-1, za), edge(1, zb)
        mid = (a + b) / 2 + V(0, -0.004, 0.0)
        parts.append(lib.tube([a, mid, b], 0.0032, role="trim", sides=6, resolution=3, name="lace"))
    a, b = edge(-1, 0.298), edge(1, 0.298)
    parts.append(lib.tube([a, (a + b) / 2 + V(0, -0.003, -0.003), b], 0.0032, role="trim", sides=6, resolution=3, name="lace"))
    return parts


# ---------------------------------------------------------------- build


def build():
    parts = []
    b = body()
    co = np.array([v.co[:] for v in b.data.vertices])
    parts += head_parts()
    for s in (-1, 1):
        parts += ear(s)
    parts += club()
    parts += fist()
    parts += claw_hand()
    parts += foot(J["r_ankle"], J["r_ankle"] + V(0.0, 0.012, -0.02), V(-0.105, 0.01, 0.02), V(-0.08, -1, 0))
    parts += foot(J["l_ankle"], V(0.095, -0.034, 0.019), V(0.101, -0.1, 0.019), V(0.12, -1, 0))
    parts += loincloth(co)
    parts += vest(b)
    parts.append(lib.decimate(b, 0.41))
    # A touch larger overall, so the little body holds its own on the board.
    for p in parts:
        put(p, Matrix.Scale(SCALE, 4))
    return parts
