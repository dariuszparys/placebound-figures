"""Fire: the fire elemental, serving either side. A living flame rising from a
glowing pool of fire on the plinth, ringed by brazier flames that lean out
over its rim. Out of the pool a few great flames twist up round a thin
glowing stalk into the waist, so the lower body is a flame tail, not a
column. Above it the torso is only a slim armature under tongues of flame
that carry its mass: they part round a glowing cavity in the chest where
the core burns, cracks of heat break out across the belly, and hot hearts
show through the flames on the collarbones, flanks and back. Collar
flames lick up beside the jaw of a heroic head with slit eyes, and the
crown of flame rises to a single forked tip like a candle, stepping down
to the temples and the shoulders, whose flames lean in toward it. The
right hand (-X) holds a burning fireball out and up, clear of the head;
the left forearm thrusts an open palm forward and its fingers burn away
into four flames sweeping back over the arm.

Every flame is a tongue with a lateral S-sway, a curled tip and, on the
big ones, forked sub-tips, so the outline flickers; hair and shoulder
flames grow from narrow roots buried in the surface. The hottest flames
carry a flame-shaped glowing heart; the accent colour burns on the crown,
the fireball, the shoulders, the hand, the brazier rim and the free tips
of the flames over the body.

The figure faces -Y; its right hand is on -X and its left on +X. Z is up and
the origin is the plinth top.
"""
import math
import os

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import lib

TAU = math.tau
DEBUG = bool(os.environ.get("FIRE_DEBUG"))
# Organic surfaces keep smooth normals; only real corners split.
FINISH = {"sharp_angle": 60}


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return Vector(a).lerp(Vector(b), t)


def interp(table, t):
    """Smooth (Catmull-Rom) interpolation through (t, value) pairs."""
    if t <= table[0][0]:
        return table[0][1]
    if t >= table[-1][0]:
        return table[-1][1]
    n = len(table)

    def slope(i):
        a, b = table[max(i - 1, 0)], table[min(i + 1, n - 1)]
        return (b[1] - a[1]) / (b[0] - a[0])

    for i in range(n - 1):
        (t0, v0), (t1, v1) = table[i], table[i + 1]
        if t <= t1:
            h = t1 - t0
            f = (t - t0) / h
            f2, f3 = f * f, f * f * f
            return ((2 * f3 - 3 * f2 + 1) * v0 + (f3 - 2 * f2 + f) * h * slope(i)
                    + (-2 * f3 + 3 * f2) * v1 + (f3 - f2) * h * slope(i + 1))
    return table[-1][1]


def curve(ctrl, samples):
    """Catmull-Rom curve through ctrl points, resampled evenly by arc length."""
    pts = [Vector(p) for p in ctrl]
    ext = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(24):
            t = k / 24
            dense.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    dense.append(pts[-1])
    acc = [0.0]
    for a, b in zip(dense, dense[1:]):
        acc.append(acc[-1] + (b - a).length)
    out, j = [], 0
    for i in range(samples):
        s = acc[-1] * i / (samples - 1)
        while j < len(acc) - 2 and acc[j + 1] < s:
            j += 1
        f = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-9)
        out.append(dense[j].lerp(dense[j + 1], f))
    return out


def frames(pts, ref=(0, -1, 0)):
    """Parallel-transport frames (tangent, normal, binormal) along a polyline;
    the first normal points as close to `ref` as it can."""
    n = len(pts)
    tang = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
    r = Vector(ref)
    nv = r - r.dot(tang[0]) * tang[0]
    if nv.length < 1e-6:
        nv = Vector((1, 0, 0)) - tang[0].x * tang[0]
    norms = [nv.normalized()]
    for i in range(1, n):
        m = norms[-1] - norms[-1].dot(tang[i]) * tang[i]
        norms.append(m.normalized())
    binorms = [t.cross(m) for t, m in zip(tang, norms)]
    return tang, norms, binorms


def mesh_object(name, verts, faces, role, face_roles=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    lib.set_role(obj, role)
    if face_roles:
        names = [role]
        for r in face_roles:
            if r not in names:
                names.append(r)
                me.materials.append(lib.mat(r))
        for p, r in zip(me.polygons, face_roles):
            p.material_index = names.index(r)
    return obj


def bands(*spec):
    """Role by position t along a flame: (t_from, role) pairs in order."""
    def fn(t):
        role = spec[0][1]
        for t0, r in spec:
            if t >= t0:
                role = r
        return role
    return fn


# The accent (trim) burns only on the flames that identify the piece: the
# crown, the fireball, the shoulders, the free licks and the brazier rim.
CROWN = bands((0, "body"), (0.08, "cloth"), (0.20, "trim"))
LICK = bands((0, "body"), (0.10, "cloth"), (0.26, "trim"))
BALLF = bands((0, "cloth"), (0.30, "trim"))
RIM = bands((0, "cloth"), (0.44, "trim"))
# Flames laid on the torso and the tail stay in the side's own colours.
LAIDR = bands((0, "body"), (0.16, "cloth"))
# ...except where a torso flame leaves the body: its free tip burns in trim.
LAIDT = bands((0, "body"), (0.16, "cloth"), (0.74, "trim"))
TAIL = bands((0, "cloth"), (0.55, "body"))
GLOW = bands((0, "eye"))
# Width along a flame tongue: a low belly and a long tail to a point.
FLAME = [(0.0, 0.80), (0.10, 1.0), (0.25, 0.90), (0.45, 0.60), (0.70, 0.30), (1.0, 0.0)]
# Hair and shoulder flames: a narrow root buried in the head or shoulder,
# so each one swells out of the surface as a flame, not a bulb.
CROWN_P = [(0.0, 0.25), (0.15, 0.70), (0.32, 1.0), (0.55, 0.72), (0.78, 0.32), (1.0, 0.0)]
# Laid along the body: a rounded foot that grows out of the surface.
LAID = [(0.0, 0.30), (0.12, 0.90), (0.25, 1.0), (0.45, 0.70), (0.70, 0.34), (1.0, 0.0)]
# Laid, with a free tip that keeps its width as it leaves the body.
LAIDF = [(0.0, 0.30), (0.12, 0.90), (0.30, 1.0), (0.55, 0.78), (0.80, 0.42), (1.0, 0.0)]
# The pecs: narrow low down beside the core, widest high on the chest.
PEC_P = [(0.0, 0.30), (0.20, 0.62), (0.40, 0.92), (0.58, 1.0), (0.78, 0.62), (1.0, 0.0)]
# The tail: narrow where it leaves the pool, widest at the waist.
TAILP = [(0.0, 0.50), (0.22, 0.85), (0.45, 1.0), (0.70, 0.60), (1.0, 0.0)]
# A forked sub-tip: rooted inside the parent, swelling as it leaves it.
FORK = [(0.0, 0.45), (0.20, 1.0), (0.42, 0.78), (0.72, 0.36), (1.0, 0.0)]
# The glowing core inside a flame, along its first part.
CORE_P = [(0.0, 0.80), (0.30, 1.0), (0.65, 0.62), (1.0, 0.0)]
# ...or floating higher up the flame, rounded at its foot.
CORE_FLOAT = [(0.0, 0.0), (0.12, 0.70), (0.35, 1.0), (0.68, 0.55), (1.0, 0.0)]
# A glowing seam: full where it leaves the core, thinning out.
SEAM = [(0.0, 1.0), (0.5, 0.75), (0.85, 0.35), (1.0, 0.0)]


def _sway(dense, binorms, W, sway, phase, curl, curl_dir):
    """Offsets a flame's centreline in its own plane: an S-sway that grows
    toward the tip, and a curl of the last fifth toward the sway side."""
    m = len(dense)
    if curl_dir is None:
        t = 0.8
        d = TAU * math.cos(TAU * t + phase) * t + math.sin(TAU * t + phase)
        curl_dir = 1.0 if d >= 0 else -1.0
    out = []
    for i, (p, B) in enumerate(zip(dense, binorms)):
        t = i / (m - 1)
        off = W * sway * math.sin(TAU * t + phase) * t
        off += curl_dir * W * curl * max(0.0, (t - 0.8) / 0.2) ** 2
        out.append(p + B * off)
    return out


def _tongue(geo, ctrl, W, thick, n, around, prof, twist, roles, ref, chevron, sway, phase, curl, curl_dir, fork,
            inner=None):
    verts, faces, froles = geo
    # With an inner core, each ring places vertices on the core's edges, so
    # the core is a smooth flame shape rather than a stripe of whole faces.
    core_faces = {6: (1, 4), 8: (1, 2, 5, 6), 12: (2, 3, 8, 9)}.get(around) if inner else None

    def core_w(t):
        r, w0, t1 = inner[:3]
        t0 = inner[3] if len(inner) > 3 else 0.0
        if not t0 <= t < t1:
            return 0.0
        return w0 * interp(CORE_FLOAT if t0 else CORE_P, (t - t0) / (t1 - t0))

    def angles(t):
        if not core_faces:
            return [TAU * j / around for j in range(around)]
        b = math.acos(min(max(core_w(t), 0.02), 0.98))
        front = {6: [0, b, math.pi - b], 8: [0, b, math.pi / 2, math.pi - b],
                 12: [0, b / 2, b, math.pi / 2, math.pi - b, math.pi - b / 2]}[around]
        return front + [a + math.pi for a in front]

    def role(t, j):
        if core_faces and j in core_faces and core_w(t) > 0.10:
            return inner[0]
        return roles(t)
    m = 64
    dense = curve(ctrl, m)
    tang, norms, binorms = frames(dense, ref)
    if sway or curl:
        dense = _sway(dense, binorms, W, sway, phase, curl, curl_dir)
        tang, norms, binorms = frames(dense, ref)

    def at(s):
        x = min(max(s, 0.0), 1.0) * (m - 1)
        i = min(int(x), m - 2)
        f = x - i
        return (dense[i].lerp(dense[i + 1], f), norms[i].lerp(norms[i + 1], f).normalized(),
                binorms[i].lerp(binorms[i + 1], f).normalized(), tang[i].lerp(tang[i + 1], f).normalized())

    o = len(verts)
    for i in range(n - 1):
        t = i / (n - 1)
        for th in angles(t):
            c, s = math.cos(th), math.sin(th)
            u = min(max(t + chevron * math.sin(math.pi * t) * (c * c - 0.5), 0.0), 0.995)
            p, N, B, _ = at(u)
            w = W * interp(prof, u) / 2
            h = max(w * thick, 0.0009)
            tw = twist(u) if twist else 0.0
            nn = N * math.cos(tw) + B * math.sin(tw)
            bb = B * math.cos(tw) - N * math.sin(tw)
            # A little thinner toward the edges, but rounded across the face,
            # so a broad flame does not crease down the middle like a leaf.
            verts.append(p + bb * (w * c) + nn * (h * s * (1 - 0.15 * c * c)))
    tip = len(verts)
    verts.append(dense[-1])
    base = len(verts)
    verts.append(dense[0] - tang[0] * 0.001)
    for i in range(n - 2):
        for j in range(around):
            a, b = o + i * around + j, o + i * around + (j + 1) % around
            faces.append((a, b, b + around, a + around))
            froles.append(role((i + 0.5) / (n - 1), j))
    last = o + (n - 2) * around
    for j in range(around):
        faces.append((last + j, last + (j + 1) % around, tip))
        froles.append(roles(1.0))
    for j in range(around):
        faces.append((base, o + (j + 1) % around, o + j))
        froles.append(role(0.0, j))

    # Forked sub-tips: (t where it leaves, length as a share of the rest of
    # the flame, side +1/-1 along the flame's width).
    if fork:
        acc = [0.0]
        for a, b in zip(dense, dense[1:]):
            acc.append(acc[-1] + (b - a).length)
        for k, (tf, lf, side) in enumerate(fork):
            p, N, B, T = at(tf)
            tw = twist(tf) if twist else 0.0
            nn = N * math.cos(tw) + B * math.sin(tw)
            bb = B * math.cos(tw) - N * math.sin(tw)
            L = acc[-1] * (1 - tf) * lf
            start, _, _, _ = at(tf - 0.07)
            sub = [start, p + (T * 0.22 + bb * side * 0.17) * L, p + (T * 0.58 + bb * side * 0.36) * L,
                   p + (T * 0.98 + bb * side * 0.40) * L]
            wf = W * interp(prof, tf) * 0.82
            _tongue(geo, sub, wf, thick, max(7, n // 2 + 1), max(6, around - 2), FORK, None,
                    lambda s, tf=tf: roles(tf + s * (1 - tf)), nn, chevron, 0.22, phase + 1.9 + k, 0.55, side, ())


def tongue(ctrl, W, thick=0.5, n=14, around=10, prof=FLAME, twist=None, roles=LICK, ref=(0, -1, 0),
           chevron=0.12, sway=0.35, phase=0.0, curl=0.4, curl_dir=None, fork=(), inner=None, name="tongue"):
    """A flame tongue along a curve through ctrl: a lens section W wide at its
    belly (thick times as deep) tapering to a sharp tip, turned by twist(t).
    The centreline sways in the flame's plane like an S and its tip curls;
    `fork` splits off sub-tips of unequal length near the tip. Each ring dips
    toward the base in the middle of the faces (chevron), so the tip colour
    has a rounded foot like a flame within the flame. `inner` (role, width
    share, t where it closes[, t where it opens]) paints a flame-shaped core
    down the middle of both faces."""
    geo = ([], [], [])
    _tongue(geo, ctrl, W, thick, n, around, prof, twist, roles, ref, chevron, sway, phase, curl, curl_dir, fork, inner)
    return mesh_object(name, geo[0], geo[1], "body", geo[2])


def kfac(stiff):
    """Visible radius of a lone metaball element as a share of its radius."""
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def along(d, up="Z"):
    """Euler that turns local +X onto direction d."""
    return Vector(d).to_track_quat("X", up).to_euler()


def ell(co, semi, rot=(0, 0, 0), stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": 1.0 / kfac(stiff), "type": "ELLIPSOID", "size": tuple(semi),
            "rot": tuple(rot), "stiffness": stiff, "negative": neg}


def cap(a, b, r, stiff=2.0, neg=False):
    a, b = Vector(a), Vector(b)
    d = b - a
    return {"co": tuple((a + b) / 2), "radius": r / kfac(stiff), "type": "CAPSULE", "size": (d.length / 2, 1, 1),
            "rot": tuple(along(d)), "stiffness": stiff, "negative": neg}


def ball(co, r, stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": r / kfac(stiff), "stiffness": stiff, "negative": neg}


def polar(r, a, z):
    """A point at radius r and angle a (0 = front, toward +X) at height z."""
    return V(math.sin(a) * r, -math.cos(a) * r, z)


# ---------------------------------------------------------------- surface


class Surface:
    """Ray casts onto a finished mesh, so flames can be laid on the body."""

    def __init__(self, obj):
        me = obj.data
        self.tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])

    def cast(self, origin, d):
        origin, d = Vector(origin), Vector(d).normalized()
        loc, nv, _, _ = self.tree.ray_cast(origin, d)
        if loc is None:
            return origin + d * 0.05, d
        return loc, nv

    def at(self, a, z, lift=0.0, axis=(0.0, 0.006)):
        """The body surface (pushed out by lift) at angle a round the trunk, height z."""
        loc, nv = self.cast(V(axis[0], axis[1], z), V(math.sin(a), -math.cos(a), 0))
        return loc + nv * lift

    def on_bone(self, a, b, t, toward, lift=0.0):
        """The surface round the bone a-b, t along it, on the side facing `toward`."""
        a, b = Vector(a), Vector(b)
        d = (b - a).normalized()
        u = Vector(toward) - Vector(toward).dot(d) * d
        loc, nv = self.cast(a.lerp(b, t), u.normalized())
        return loc + nv * lift


# ---------------------------------------------------------------- pose

# The whole figure sits this much lower than a plain humanoid would, so the
# crown of flame can rise well above the head inside the elemental's height.
DZ = 0.065

WAIST = V(0.0, 0.006, 0.330 - DZ)
CHEST = V(0.0, 0.010, 0.470 - DZ)
NECK = V(0.0, 0.014, 0.590 - DZ)
HEAD = V(0.0, -0.004, 0.648 - DZ)
HEAD_S = 1.15

R_SH = V(-0.128, 0.014, 0.540 - DZ)
R_EL = V(-0.258, 0.040, 0.482 - DZ)
R_WR = V(-0.295, -0.016, 0.585 - DZ)
# Left arm: elbow out to the side, forearm and palm thrust forward and out.
L_SH = V(0.128, 0.014, 0.540 - DZ)
L_EL = V(0.275, 0.030, 0.405)
L_WR = V(0.255, -0.115, 0.355)

BALL_R = 0.060
BALL = V(-0.305, -0.040, 0.625)
CORE = V(0.0, -0.060, 0.462 - DZ)
HOLLOW = V(0.0, -0.052, CORE.z + 0.008)


def wave(k, a=1.0):
    """A repeatable pseudo-random value in [-1, 1] for flame number k."""
    return math.sin(k * 12.9898 * a + 4.1) * 0.6 + math.sin(k * 3.7 + 1.3) * 0.4


# ---------------------------------------------------------------- body


def body():
    els = [
        # A thin glowing stalk rising out of the pool, flared where it
        # meets it; the tail flames twist round it.
        ell((0, 0.004, 0.030), (0.064, 0.060, 0.034)),
        cap((0, 0.004, 0.050), (0, 0.006, 0.225), 0.035),
        ell(WAIST, (0.050, 0.044, 0.050)),
        # Torso: only a slim armature of a chest over a narrow waist; the
        # flames laid over it carry the mass, so it reads as fire, not muscle.
        ell((0, 0.000, 0.392 - DZ), (0.060, 0.046, 0.060)),
        ell(CHEST, (0.080, 0.052, 0.070)),
        ell((0, 0.016, 0.538 - DZ), (0.110, 0.048, 0.030)),
        ell((0, 0.034, 0.496 - DZ), (0.066, 0.030, 0.056)),
        ell(NECK, (0.034, 0.032, 0.044)),
        # The core sits in a hollow in the breastbone.
        ell(HOLLOW, (0.034, 0.030, 0.056), neg=True),
    ]
    for s in (-1, 1):
        els += [
            ell((s * 0.046, 0.026, 0.570 - DZ), (0.034, 0.020, 0.018), rot=(0, -s * 0.45, 0)),
            ell((s * 0.056, 0.026, 0.452 - DZ), (0.020, 0.024, 0.036), rot=(0, -s * 0.25, 0)),
        ]
    for sh, el, wr in ((R_SH, R_EL, R_WR), (L_SH, L_EL, L_WR)):
        s = 1 if sh.x > 0 else -1
        up, fore = el - sh, wr - el
        els += [
            ell(sh + V(s * 0.004, 0.0, 0.006), (0.027, 0.032, 0.029), stiff=3),
            cap(sh, el, 0.022),
            ell(lerp(sh, el, 0.45), (up.length * 0.30, 0.025, 0.023), rot=along(up), stiff=3.5),
            ball(el, 0.021),
            cap(el, wr, 0.018),
            # Forearms swell toward the wrist, where they burn hottest.
            ell(lerp(el, wr, 0.62), (fore.length * 0.32, 0.024, 0.022), rot=along(fore), stiff=3.5),
            ball(wr, 0.019),
        ]
    obj = lib.metaball(els, role="body", resolution=0.005, name="body")
    obj = lib.decimate(obj, min(1.0, 3300 / max(lib.tris(obj), 1)))
    # The foot of the stalk is buried in the pool: drop what no one sees.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_center_median().z < 0.026], context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    # Where the stalk leaves the pool it glows, the heart of the fire.
    lib.paint(obj, "eye", lambda c: c.z < 0.20)
    # The walls of the chest cavity glow round the core: a fire inside.
    def in_hollow(c):
        d = c - HOLLOW
        return c.y < -0.025 and (d.x / 0.048) ** 2 + (d.y / 0.044) ** 2 + (d.z / 0.068) ** 2 < 1
    lib.paint(obj, "eye", in_hollow)
    return obj


def hp(x, y, z):
    """A point on the head, given at its unscaled size."""
    return HEAD + V(x, y, z) * HEAD_S


def hs(x, y, z):
    return (x * HEAD_S, y * HEAD_S, z * HEAD_S)


def head():
    """A head of flame: a heavy brow angled down in a scowl over slanted slit
    eyes, high sharp cheekbones, a pointed jaw and a glowing mouth."""
    els = [
        ell(hp(0, 0.004, 0.006), hs(0.038, 0.042, 0.044)),
        ell(hp(0, -0.020, -0.014), hs(0.032, 0.030, 0.030)),
        ell(hp(0, -0.026, -0.036), hs(0.021, 0.024, 0.020)),
        ell(hp(0, -0.036, -0.052), hs(0.010, 0.012, 0.012), stiff=3),
        # The mouth: a slot under the cheeks.
        ell(hp(0, -0.046, -0.031), hs(0.014, 0.008, 0.0035), stiff=3, neg=True),
    ]
    for s in (-1, 1):
        els += [
            # Brow: outer ends high, inner ends pressed down into a scowl.
            ell(hp(s * 0.019, -0.040, 0.014), hs(0.024, 0.012, 0.007), rot=(0, -s * 0.45, 0), stiff=3.5),
            ell(hp(s * 0.028, -0.030, -0.012), hs(0.015, 0.012, 0.008), rot=(0, s * 0.45, 0), stiff=3.5),
            ell(hp(s * 0.019, -0.045, 0.001), hs(0.0156, 0.008, 0.005), rot=(0, -s * 0.35, 0), stiff=3, neg=True),
        ]
    obj = lib.metaball(els, role="body", resolution=0.0034, name="head")
    obj = lib.decimate(obj, min(1.0, 750 / max(lib.tris(obj), 1)))
    parts = [obj]
    for s in (-1, 1):
        parts.append(lib.sphere(0.0095 * HEAD_S, loc=hp(s * 0.0195, -0.041, 0.0015), rot=(0, -s * 0.35, s * 0.35),
                                scale=(1.9, 0.45, 0.38), role="eye", segments=10, rings=6, name="eye"))
    parts.append(lib.sphere(0.012 * HEAD_S, loc=hp(0, -0.041, -0.031), scale=(1.15, 0.4, 0.3), role="eye",
                            segments=10, rings=6, name="mouth"))
    return parts


# ---------------------------------------------------------------- pool


def pool():
    """A shallow bowl whose surface is a disc of glowing fire."""
    prof = [(0.0, 0.0), (0.176, 0.0), (0.188, 0.010), (0.184, 0.026), (0.170, 0.038), (0.152, 0.042),
            (0.136, 0.038), (0.110, 0.036), (0.070, 0.035), (0.0, 0.035)]
    p = lib.lathe(prof, role="body", segments=40, name="pool")

    def ripple(v):
        r = math.hypot(v.x, v.y)
        a = math.atan2(v.x, -v.y)
        k = 1 + 0.05 * math.sin(7 * a + 1.0) * sstep(0.12, 0.18, r)
        v.x, v.y = v.x * k, v.y * k
        return v
    lib.deform(p, ripple)
    lib.paint(p, "eye", lambda c: math.hypot(c.x, c.y) < 0.14 and c.z > 0.02)
    return [p]


def pool_flames():
    """A ring of brazier flames leaning out over the rim, tall and short in
    turn, all swaying the same way round so the pool swirls."""
    parts = []
    n = 10
    for k in range(n):
        tall = k % 2 == 0
        a = k * TAU / n + 0.10 * wave(k)
        r0 = 0.146 + 0.006 * wave(k, 2)
        h = (0.168 if tall else 0.112) + 0.014 * wave(k, 3)
        out = 0.052 if tall else 0.040
        ctrl = [polar(r0 - 0.016, a, 0.016), polar(r0 + out * 0.35, a + 0.05, 0.020 + h * 0.32),
                polar(r0 + out * 0.85, a + 0.11, 0.020 + h * 0.68), polar(r0 + out * 0.80, a + 0.18, 0.020 + h)]
        parts.append(tongue(ctrl, 0.072 if tall else 0.070, thick=0.50, n=12, around=6, ref=polar(1, a, 0),
                            twist=lambda t, k=k: 0.5 * wave(k + 4) * t * t, sway=0.35, phase=1.3 * k, curl=0.45,
                            roles=RIM, inner=("eye", 0.60, 0.72, 0.15), fork=((0.52, 0.62, 1 if k % 4 == 0 else -1),) if tall else (),
                            name="pool_flame"))
    return parts


def tail_flames(surf):
    """Great flames twisting up out of the pool round the glowing stalk into
    the waist, of unequal length; the short ones flare free below the waist,
    the long ones lick up the belly and flanks."""
    parts = []
    spec = (  # start angle, turn as it rises, top, belly width, twist
        (0.60, 1.00, 0.305, 0.104, 0.75),
        (1.55, 0.80, 0.430, 0.112, 1.15),
        (2.80, 1.10, 0.360, 0.100, 0.60),
        (4.05, 0.90, 0.455, 0.114, 1.00),
        (5.60, 1.05, 0.260, 0.100, 0.85),
    )
    for k, (a0, spin, top, w, tw) in enumerate(spec):
        ctrl = []
        for f in (0.0, 0.2, 0.4, 0.6, 0.8):
            z = 0.030 + (top - 0.030) * f
            a = a0 + spin * f
            r = 0.062 + 0.004 * math.sin(math.pi * f)
            s = surf.at(a, z, lift=0.008)
            r = max(r, math.hypot(s.x, s.y - 0.006))
            ctrl.append(polar(r, a, z) + V(0, 0.006, 0))
        # The tip flares free, out and up.
        a = a0 + spin * 0.95
        ctrl.append(polar(math.hypot(ctrl[-1].x, ctrl[-1].y - 0.006) + 0.060, a, top + 0.040) + V(0, 0.006, 0))
        parts.append(tongue(ctrl, w, thick=0.52, n=18, around=12, ref=polar(1, a0, 0), prof=TAILP,
                            twist=lambda t, tw=tw: tw * t, sway=0.30, phase=0.9 * k, curl=0.45,
                            roles=TAIL, inner=("eye", 0.54, 0.58), name="tail_flame"))
    return parts


# ---------------------------------------------------------------- crown


def mane():
    """Flame hair: one great forked flame rising off the crown to the apex,
    with licks stepping down beside it like a candle flame, and more
    streaming off the back of the head."""
    parts = []
    H = HEAD
    top = 0.855 - H.z
    main = [hp(0, 0.010, -0.004), H + V(0, 0.022, 0.070), H + V(0.010, 0.042, 0.140), H + V(-0.010, 0.060, 0.205),
            H + V(0.004, 0.064, top)]
    parts.append(tongue(main, 0.140, thick=0.38, n=18, around=12, roles=CROWN, ref=(0, -1, 0), prof=CROWN_P,
                        twist=lambda t: 0.35 * t * t, chevron=0.14, sway=0.28, phase=0.6, curl=0.15,
                        fork=((0.56, 0.75, -1), (0.66, 0.55, 1)), inner=("eye", 0.55, 0.80, 0.24), name="mane_main"))
    licks = [  # base on the head (unscaled), then points relative to the head centre; belly width; fork side
        # Temples: tips near 0.79.
        ((-0.030, -0.012, 0.022), [V(-0.060, -0.004, 0.086), V(-0.080, 0.012, 0.150), V(-0.066, 0.028, 0.204)], 0.064, 1),
        ((0.030, -0.012, 0.022), [V(0.060, -0.004, 0.088), V(0.082, 0.014, 0.152), V(0.070, 0.030, 0.204)], 0.064, -1),
        # Sides: tips near 0.73, leaning in toward the crown.
        ((-0.034, 0.016, 0.000), [V(-0.074, 0.028, 0.052), V(-0.092, 0.044, 0.104), V(-0.085, 0.056, 0.146)], 0.058, 1),
        ((0.034, 0.016, 0.004), [V(0.076, 0.030, 0.054), V(0.094, 0.046, 0.106), V(0.087, 0.058, 0.146)], 0.058, -1),
        # Back of the head: low, streaming back, so they stay under the
        # crown even where the board camera looks down on them.
        ((-0.020, 0.034, -0.004), [V(-0.050, 0.074, 0.030), V(-0.066, 0.104, 0.060), V(-0.060, 0.122, 0.082)], 0.054, 0),
        ((0.020, 0.034, -0.002), [V(0.050, 0.076, 0.032), V(0.068, 0.106, 0.062), V(0.062, 0.124, 0.082)], 0.054, 0),
        # Forelock off the brow.
        ((0.000, -0.034, 0.024), [V(0.004, -0.040, 0.060), V(-0.008, -0.026, 0.094), V(0.004, -0.010, 0.120)], 0.040, 0),
    ]
    for k, (base, pts, w, fside) in enumerate(licks):
        # The root is buried a little inside the skull.
        root = hp(*base)
        root -= (root - H).normalized() * 0.010
        ctrl = [root] + [H + p for p in pts]
        parts.append(tongue(ctrl, w, thick=0.32, n=12, around=8, roles=LICK, ref=(0, 1, 0.3), prof=CROWN_P,
                            twist=lambda t, k=k: 0.5 * wave(k) * t * t, sway=0.35, phase=0.8 + 1.7 * k, curl=0.45,
                            fork=((0.55, 0.60, fside),) if fside else (), name="mane"))
    return parts


# ---------------------------------------------------------------- torso


def laid(surf, pts, w, tip=None, thick=0.48, n=13, around=8, roles=LAIDR, k=0, prof=LAID, sway=0.15, curl=0.3,
         chevron=0.12, turn=0.0, inner=None, fork=(), name="torso_flame"):
    """A tongue laid along the body through (angle, height, lift) points, its
    tip leaving the surface by `tip` when given, bending out as it goes and
    turning by `turn` so its face shows from the front."""
    ctrl = [surf.at(a, z, lift) for a, z, lift in pts]
    if tip is not None:
        tip = V(*tip)
        ctrl.append(ctrl[-1] + tip * 0.45 + V(0, 0, 0.35 * tip.z) * 0.2)
        ctrl.append(ctrl[-2] + tip)
    t0 = 1 - 1.6 / len(ctrl)
    return tongue(ctrl, w, thick=thick, n=n, around=around, ref=polar(1, pts[0][0], 0), prof=prof,
                  twist=lambda t: 0.5 * wave(k) * t * t + turn * sstep(t0, 1.0, t), roles=roles, sway=sway,
                  phase=1.1 * k, curl=curl, chevron=chevron, inner=inner, fork=fork, name=name)


def torso_flames(surf):
    """The torso is built of tongues licking upward: the pecs are two great
    flames framing the chest cavity and flicking off the shoulders; flames
    slant up the flanks to the armpits and up the back and spine, leaving
    the body near their tips, so the torso tapers to the waist and its
    outline is serrated with flame."""
    parts = []
    # A hot heart floating inside each flame, so fire burns through the torso.
    heart = ("eye", 0.45, 0.55, 0.15)
    for s in (-1, 1):
        o = V(s, 0, 0)
        # Pecs: part round the core from below, rise out across the chest and
        # stand off it as they go, leaving it in front of the shoulder.
        parts.append(laid(surf, [(s * 0.80, 0.335, 0.004), (s * 0.96, 0.388, 0.010), (s * 1.12, 0.432, 0.018),
                                 (s * 1.28, 0.468, 0.028)], 0.088, tip=(s * 0.026, -0.006, 0.085), k=1 + s, thick=0.55,
                          prof=PEC_P, roles=LAIDT, inner=("eye", 0.40, 0.78, 0.36), name="pec_flame"))
        # Flanks: rise on a slant from the waist to the armpit, and flick
        # out under the arm, so the torso tapers to the waist like a flame.
        lf = 1.0 + 0.25 * s
        parts.append(laid(surf, [(s * 1.05, 0.215, 0.004), (s * 1.28, 0.280, 0.010), (s * 1.45, 0.340, 0.018),
                                 (s * 1.55, 0.385, 0.028)], 0.095, tip=tuple((o * 0.045 + V(0, 0.000, 0.075)) * lf),
                          k=5 + s, thick=0.55, prof=LAIDF, roles=LAIDT, sway=0.25, turn=-s * 0.3, curl=0.6,
                          inner=heart, fork=((0.62, 0.55, 1),)))
        # Back.
        lb = 1.0 - 0.35 * s
        parts.append(laid(surf, [(s * 2.35, 0.255, 0.004), (s * 2.38, 0.330, 0.010), (s * 2.45, 0.405, 0.016),
                                 (s * 2.55, 0.465, 0.028)], 0.095, tip=tuple((o * 0.046 + V(0, 0.040, 0.085)) * lb),
                          k=7 + s, thick=0.55, prof=LAIDF, roles=LAIDT, sway=0.25, turn=-s * 0.7, curl=0.5, inner=heart))
    parts.append(laid(surf, [(math.pi, 0.270, 0.004), (math.pi, 0.365, 0.010), (math.pi, 0.450, 0.016),
                             (math.pi, 0.510, 0.030)], 0.090, tip=(0, 0.050, 0.100), k=11, thick=0.55, prof=LAIDF,
                      roles=LAIDT, sway=0.25, curl=0.5, inner=heart, name="spine_flame"))
    return parts


def core(surf):
    """The hot heart: a flame of pure glow standing in the hollow of the
    chest, its heat cracking out through seams down the belly."""
    ctrl = [CORE + V(0, 0.026, -0.052), CORE + V(0.002, 0.014, -0.012), CORE + V(-0.004, 0.010, 0.035),
            CORE + V(0.005, 0.006, 0.095)]
    parts = [tongue(ctrl, 0.088, thick=0.55, n=11, around=10, chevron=0.0, roles=GLOW, sway=0.12, phase=2.0,
                    curl=0.2, ref=(0, -1, 0), fork=((0.55, 0.55, 1),), name="core")]
    c = CORE.z
    for s in (-1, 1):
        # Cracks of heat breaking out from the rim of the chest cavity: a
        # jagged one across the ribs and one down the belly that forks.
        # The two sides break differently, so the cracks read as cracks,
        # not as a symmetric glyph.
        j = (s * 0.06, -s * 0.008)
        knee = (s * (0.66 + j[0]), c - 0.084 + j[1])
        parts += crack(surf, [(s * 0.30, c - 0.052), knee, (s * (0.56 - j[0]), c - 0.114 - j[1]),
                              (s * (0.98 - j[0]), c - 0.150)], (0.013, 0.011, 0.008))
        parts += crack(surf, [knee, (s * 1.00, c - (0.072 if s > 0 else 0.104))], (0.008,))
    return parts


SEG = [(0.0, 0.85), (0.5, 1.0), (1.0, 0.45)]


def crack(surf, pts, widths, lift=0.004):
    """A jagged glowing crack over the body: one straight lens per leg, so
    the turns stay sharp instead of rounding into a drip."""
    parts = []
    for (a0, z0), (a1, z1), w in zip(pts, pts[1:], widths):
        # Each leg runs a little past its end, under the next one's foot.
        a2, z2 = a1 + (a1 - a0) * 0.15, z1 + (z1 - z0) * 0.15
        parts.append(laid(surf, [(a0, z0, lift), ((a0 + a2) / 2, (z0 + z2) / 2, lift), (a2, z2, lift)], w, roles=GLOW,
                          prof=SEG, n=5, around=4, thick=0.5, sway=0.0, curl=0.0, chevron=0.0, name="seam"))
    return parts


def rising(root, s, h, out=0.028, back=0.040):
    """Control points of a flame licking up from root, leaning out to side s."""
    return [root, root + V(s * out * 0.45, back * 0.2, h * 0.36), root + V(s * out * 0.90, back * 0.6, h * 0.74),
            root + V(s * out, back, h)]


def shoulder_flames():
    """Big flames leaning out and up off each shoulder, stepping the crown's
    outline down to the arms."""
    parts = []
    for sh, s in ((R_SH, -1), (L_SH, 1)):
        # Rooted inside the shoulder; the tips lean in toward the crown.
        for k, (dx, dy, h, w, out) in enumerate(((0.026, 0.006, 0.152, 0.084, -0.012), (-0.006, 0.040, 0.130, 0.074, -0.022))):
            root = sh + V(s * dx, dy, 0.016)
            parts.append(tongue(rising(root, s, h, out=out, back=0.036), w, thick=0.32, n=11, around=8, ref=(0, -1, 0),
                                prof=CROWN_P, twist=lambda t: 0.4 * t * t, sway=0.35, phase=0.7 + 2.3 * k + s, curl=0.45,
                                roles=LICK, fork=((0.55, 0.62, -s if k == 0 else s),), name="shoulder_flame"))
        # A collar flame off the trapezius, licking up beside the jaw, so the
        # head rises out of fire rather than off a neck.
        root = V(s * 0.062, 0.000, 0.462)
        ctrl = [root, V(s * 0.066, -0.016, 0.500), V(s * 0.060, -0.020, 0.540), V(s * 0.052, -0.010, 0.580)]
        parts.append(tongue(ctrl, 0.046, thick=0.36, n=10, around=6, ref=(0, -1, 0), prof=CROWN_P,
                            twist=lambda t: 0.3 * t * t, sway=0.30, phase=1.7 + s, curl=0.5, curl_dir=-s,
                            roles=LICK, name="collar_flame"))
    return parts


# ---------------------------------------------------------------- arms


def bone_laid(surf, a, b, ts, angs, lifts, w, base, tip=None, k=0, roles=LAIDT, inner=None, name="arm_flame"):
    """A tongue laid round the limb a-b through (t along, angle round, lift)
    stations, angle 0 facing `base`; the tip leaves the limb by `tip`."""
    a, b = Vector(a), Vector(b)
    d = (b - a).normalized()
    u = Vector(base) - Vector(base).dot(d) * d
    u.normalize()
    v = d.cross(u)
    ctrl = [surf.on_bone(a, b, t, u * math.cos(g) + v * math.sin(g), lift) for t, g, lift in zip(ts, angs, lifts)]
    if tip is not None:
        ctrl.append(ctrl[-1] + V(*tip) * 0.5 + (ctrl[-1] - ctrl[-2]).normalized() * 0.012)
        ctrl.append(ctrl[-2] + V(*tip))
    out = u * math.cos(angs[0]) + v * math.sin(angs[0])
    return tongue(ctrl, w, thick=0.50, n=10, around=6, ref=out, twist=lambda t: 0.5 * wave(k) * t * t,
                  prof=LAIDF, roles=roles, sway=0.15, phase=1.3 * k, curl=0.35, inner=inner, name=name)


def arm_flames(surf):
    """Arms of flame: a pair of tongues, one over the front of the limb and
    one over its back, spiral along each part of each arm, so no bare limb
    shows. The raised forearm's flames run up toward the fireball; the
    thrust left forearm's stream back off the elbow."""
    parts = []
    front = (0, -1, 0)
    spin = (0.0, 0.35, 0.70, 1.05)
    lifts = (0.004, 0.006, 0.008, 0.012)
    heart = ("eye", 0.40, 0.60, 0.18)
    for k, (a, b, ts, base, g0, tip) in enumerate((
            # Right forearm, raised: from the elbow up toward the ball.
            (R_EL, R_WR, (0.04, 0.32, 0.60, 0.88), front, -0.4, (-0.010, 0.040, 0.050)),
            # Right upper arm: from the elbow up over the shoulder.
            (R_SH, R_EL, (0.96, 0.66, 0.36, 0.10), front, 0.5, (0.004, 0.040, 0.050)),
            # Left forearm, thrust forward: from the wrist back off the elbow.
            (L_EL, L_WR, (0.94, 0.66, 0.38, 0.08), (1, 0, 0.4), 1.2, (0.010, 0.060, 0.040)),
            # Left upper arm: from the elbow up over the shoulder.
            (L_SH, L_EL, (0.96, 0.66, 0.36, 0.10), front, -0.5, (-0.004, 0.040, 0.050)))):
        for j in range(2):
            g = g0 + math.pi * j
            sgn = 1 if j == 0 else -1
            parts.append(bone_laid(surf, a, b, ts, [g + sgn * sp for sp in spin], lifts, 0.070, base,
                                   tip=tip, k=2 * k + j, inner=heart if j == 0 else None))
    return parts


# ---------------------------------------------------------------- hands


FINGER = [(0.0, 1.0), (0.5, 0.88), (0.85, 0.6), (1.0, 0.0)]


def claw(pts, r, name="finger"):
    return tongue(pts, r * 2, thick=1.0, n=6, around=6, prof=FINGER, chevron=0.0, roles=bands((0, "body")),
                  ref=(0, 0, 1), sway=0.0, curl=0.0, name=name)


def hands():
    """The right hand cups the fireball; the left is an open palm thrust
    forward whose fingers dissolve into four flames streaming up and back."""
    parts = []
    # Right: palm up under the ball, fingers curling round its lower half.
    palm = BALL + V(0, 0.004, -BALL_R - 0.008)
    parts.append(lib.sphere(0.028, loc=palm, scale=(1.0, 1.1, 0.6), role="body", segments=10, rings=6, name="palm"))
    for k in range(4):
        a = -1.35 + k * 0.90
        base = palm + V(math.sin(a) * 0.016, -math.cos(a) * 0.016, 0.004)
        mid = BALL + V(math.sin(a) * (BALL_R + 0.006), -math.cos(a) * (BALL_R + 0.006), -BALL_R * 0.55)
        tip = BALL + V(math.sin(a) * (BALL_R + 0.002), -math.cos(a) * (BALL_R + 0.002), -BALL_R * 0.08)
        parts.append(claw([base, mid, tip], 0.0080))
    # Left: a flat palm thrust forward, facing the foe, at the end of the
    # forearm; its fingers burn away into four thin flames that fan up off
    # the fingertips and sweep back over the forearm.
    d = (L_WR - L_EL).normalized()
    hand = L_WR + d * 0.004 + V(0, 0, 0.006)
    parts.append(lib.sphere(0.022, loc=hand, scale=(1.25, 0.32, 1.05), rot=(-0.30, 0, 0.0), role="body",
                            segments=12, rings=8, name="hand"))
    for k, (lx, h, back, w) in enumerate(((-0.024, 0.150, 0.120, 0.032), (-0.008, 0.200, 0.140, 0.040),
                                          (0.008, 0.185, 0.135, 0.038), (0.024, 0.155, 0.125, 0.032))):
        root = hand + V(lx * 1.10, -0.002, 0.014 - abs(lx) * 0.25)
        fan = lx * 1.6
        ctrl = [root, root + V(fan * 0.4, -0.008, h * 0.22), root + V(fan * 0.9, back * 0.30, h * 0.58),
                root + V(fan * 1.2, back * 0.75, h * 0.88), root + V(fan * 1.1, back, h)]
        parts.append(tongue(ctrl, w, thick=0.32, n=11, around=6, ref=(0, -1, 0), prof=CROWN_P, sway=0.30,
                            phase=1.0 + 2.1 * k, curl=0.5, roles=bands((0, "body"), (0.10, "cloth"), (0.25, "trim")),
                            name="hand_flame"))
    return parts


# ---------------------------------------------------------------- fireball


def on_ball(az, el, rr):
    """A point on a shell of radius rr round the ball: azimuth from the front
    toward +X, elevation from the equator."""
    return BALL + V(math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)) * rr


def fireball():
    """A ball of fire: a glowing heart wrapped in tongues that cup its back,
    wind across its front and leap off its top into one tall plume sweeping
    back."""
    R = BALL_R
    parts = [lib.sphere(R, loc=BALL, role="eye", segments=16, rings=10, name="fireball")]
    # Cupping the back and sides, rising to free tips above the ball.
    for k, (a, height, w, lean) in enumerate(((1.7, 1.7, 0.058, 0.4), (3.0, 2.0, 0.064, 0.0), (4.4, 1.6, 0.056, -0.4))):
        ctrl = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            ang = a + (math.pi - a) * 0.30 * f
            r = R * (1.02 + 0.14 * math.sin(math.pi * min(f / 0.7, 1.0))) * (1 - 0.55 * max(0.0, f - 0.6) / 0.4)
            z = R * (-0.55 + (height + 0.55) * f)
            ctrl.append(BALL + V(math.sin(ang) * r - lean * R * f * f, -math.cos(ang) * r + 0.010 * f, z))
        parts.append(tongue(ctrl, w, thick=0.40, n=11, around=8, roles=BALLF, ref=polar(1, a, 0),
                            twist=lambda t, k=k: 0.5 * wave(k) * t * t, sway=0.3, phase=1.4 * k, curl=0.5,
                            name="ball_flame"))
    # Two flames winding up across the lower front and over the ball.
    for k, (a0, a1, w) in enumerate(((-1.3, 0.8, 0.056), (1.2, -0.5, 0.050))):
        ctrl = [on_ball(a0 + (a1 - a0) * f, -0.95 + 2.0 * f, R * (1.03 + 0.06 * f)) for f in (0.0, 0.3, 0.6, 0.85)]
        ctrl.append(on_ball(a1 + 0.25 * (a1 - a0), 1.35, R * 1.75))
        parts.append(tongue(ctrl, w, thick=0.36, n=12, around=8, roles=BALLF, ref=(ctrl[0] - BALL), prof=LAIDF,
                            twist=lambda t: 0.0, sway=0.25, phase=0.8 + 2 * k, curl=0.5, name="ball_wrap"))
    # The plume: off the top, rising and sweeping back and out.
    # Its tip stays well under the crown, so the figure keeps one apex.
    plume = [BALL + V(0.004, 0.004, R * 0.30), BALL + V(-0.006, 0.012, R * 0.95), BALL + V(-0.020, 0.030, R * 1.45),
             BALL + V(-0.028, 0.056, R * 1.85)]
    parts.append(tongue(plume, R * 1.30, thick=0.55, n=14, around=10, roles=BALLF, ref=(0, -1, 0),
                        twist=lambda t: 0.2 * t * t, sway=0.35, phase=0.3, curl=0.45,
                        fork=((0.50, 0.62, 1), (0.62, 0.45, -1)), name="ball_plume"))
    return parts


def build():
    groups = []
    groups.append(("pool", pool()))
    groups.append(("pool_flames", pool_flames()))
    b = body()
    surf = Surface(b)
    groups.append(("body", [b]))
    groups.append(("head", head()))
    groups.append(("core", core(surf)))
    groups.append(("tail", tail_flames(surf)))
    groups.append(("torso", torso_flames(surf)))
    groups.append(("shoulders", shoulder_flames()))
    groups.append(("arms", arm_flames(surf)))
    groups.append(("mane", mane()))
    groups.append(("hands", hands()))
    groups.append(("fireball", fireball()))
    parts = []
    for name, objs in groups:
        if DEBUG:
            area = {}
            for o in objs:
                for poly in o.data.polygons:
                    r = o.data.materials[poly.material_index].name.split(".")[0]
                    area[r] = area.get(r, 0.0) + poly.area
            print(f"FIRE {name}: {sum(lib.tris(o) for o in objs)} tris in {len(objs)} parts",
                  {r: round(v, 4) for r, v in area.items()})
        parts += objs
    return parts
