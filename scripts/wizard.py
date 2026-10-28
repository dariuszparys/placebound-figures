"""Wizard: the Light commander, an archmage. A tall bent hat with a wide floppy
brim and a star band, a long white beard over a heavy layered robe (an open
coat over an inner robe, a shoulder mantle), wide bell sleeves, a belt with
pouches and a scroll, a gnarled staff with a caged glowing crystal in the
right hand and a fireball conjured over the raised left palm.
The figure faces -Y; its right hand is on -X."""
import json
import math

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector
from mathutils.geometry import intersect_point_line

import lib

TAU = math.tau
TRIS = {}
# Cloth drapes bend more than 35 degrees between faces; keep those smooth and
# leave only real edges (rims, gem facets, stars) sharp.
FINISH = {"sharp_angle": 50}


# ---------------------------------------------------------------- helpers


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def bump(x, centre, width):
    return math.exp(-((x - centre) / width) ** 2)


def curve(keys, x):
    """Smooth cubic interpolation through (x, value) keys sorted by x; values
    may be numbers or tuples."""
    tup = isinstance(keys[0][1], tuple)
    xs = [k[0] for k in keys]
    vals = [k[1] if tup else (k[1],) for k in keys]
    n = len(keys)
    if x <= xs[0]:
        out = vals[0]
    elif x >= xs[-1]:
        out = vals[-1]
    else:
        i = max(j for j in range(n - 1) if xs[j] <= x)
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h

        def slope(j):
            a, b = max(j - 1, 0), min(j + 1, n - 1)
            return [(vb - va) / (xs[b] - xs[a]) for va, vb in zip(vals[a], vals[b])]

        m0, m1 = slope(i), slope(i + 1)
        t2, t3 = t * t, t * t * t
        h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
        out = tuple(h00 * p0 + h10 * h * s0 + h01 * p1 + h11 * h * s1
                    for p0, p1, s0, s1 in zip(vals[i], vals[i + 1], m0, m1))
    return tuple(out) if tup else out[0]


def ellipse_r(rx, ry, a):
    """Polar radius of an ellipse at angle a, measured from -Y toward +X."""
    s, c = math.sin(a), math.cos(a)
    return rx * ry / math.sqrt((ry * s) ** 2 + (rx * c) ** 2)


def polar(r, a, z):
    return (math.sin(a) * r, -math.cos(a) * r, z)


def surf(fn, us, vs, wrap=True, role="body", name="surf", tip0=False, tip1=False, closed=False, paint=None):
    """A grid surface through fn(u, v) for each u in us (around) and v in vs
    (along). A tip row collapses to the centre of its ring. paint(u, v), given
    a face's middle parameters, may return another role for that face."""
    bm = bmesh.new()
    rows = []
    last = len(vs) - 1
    for j, v in enumerate(vs):
        pts = [Vector(fn(u, v)) for u in us]
        if (j == 0 and tip0) or (j == last and tip1):
            rows.append([bm.verts.new(sum(pts, Vector()) / len(pts))])
        else:
            rows.append([bm.verts.new(p) for p in pts])
    n = len(us)
    seg = n if wrap else n - 1
    du = us[1] - us[0]
    face_roles = []
    for j, (a, b) in enumerate(zip(rows, rows[1:])):
        vm = (vs[j] + vs[j + 1]) / 2
        for i in range(seg):
            k = (i + 1) % n
            if len(a) == 1:
                bm.faces.new((a[0], b[k], b[i]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[k], b[0]))
            else:
                bm.faces.new((a[i], a[k], b[k], b[i]))
            face_roles.append(paint(us[i] + du / 2, vm) if paint else None)
    if closed:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = lib._from_bmesh(name, bm, role)
    if paint:
        names = [role]
        for r in face_roles:
            if r and r not in names:
                names.append(r)
                obj.data.materials.append(lib.mat(r))
        for poly, r in zip(obj.data.polygons, face_roles):
            poly.material_index = names.index(r) if r else 0
    return obj


def catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3)


class Path:
    """A Catmull-Rom curve through points, sampled by arc length."""

    def __init__(self, pts, samples=240):
        P = [Vector(p) for p in pts]
        ext = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
        n = len(P) - 1
        steps = max(4, samples // n)
        dense = []
        for i in range(n):
            for k in range(steps):
                dense.append(catmull(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], k / steps))
        dense.append(P[-1].copy())
        self.pts = dense
        L = [0.0]
        for a, b in zip(dense, dense[1:]):
            L.append(L[-1] + (b - a).length)
        self.L = L
        self.length = L[-1]

    def at(self, t):
        d = min(max(t, 0.0), 1.0) * self.length
        lo, hi = 0, len(self.L) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.L[mid] <= d:
                lo = mid
            else:
                hi = mid
        seg = self.L[hi] - self.L[lo]
        f = (d - self.L[lo]) / seg if seg > 1e-12 else 0.0
        return self.pts[lo].lerp(self.pts[hi], f)

    def tangent(self, t, eps=0.004):
        return (self.at(t + eps) - self.at(t - eps)).normalized()


def frames(path, vs, up=(0, 0, 1)):
    """Centres, tangents, normals and binormals along a path (parallel transport)."""
    C = [path.at(v) for v in vs]
    T = [path.tangent(v) for v in vs]
    upv = Vector(up)
    n0 = upv - T[0] * upv.dot(T[0])
    if n0.length < 1e-6:
        n0 = Vector((1, 0, 0)) - T[0] * T[0].x
    N = [n0.normalized()]
    for i in range(1, len(vs)):
        n = T[i - 1].rotation_difference(T[i]) @ N[-1]
        N.append((n - T[i] * n.dot(T[i])).normalized())
    B = [t.cross(n) for t, n in zip(T, N)]
    return C, T, N, B


def sweep(pts, vs, nu, radius, role="body", name="sweep", up=(0, 0, 1), warp=None,
          tip0=False, tip1=False, a0=0.0, paint=None):
    """A tube along a smooth path whose cross-section is radius(v, a) -> r or
    (rn, rb); warp(v, a, p, centre, direction) may move each point."""
    path = Path(pts)
    C, T, N, B = frames(path, vs, up)
    idx = {v: i for i, v in enumerate(vs)}
    us = [i / nu for i in range(nu)]

    def fn(u, v):
        i = idx[v]
        a = a0 + u * TAU
        r = radius(v, a)
        rn, rb = (r, r) if isinstance(r, (int, float)) else r
        d = N[i] * math.cos(a) + B[i] * math.sin(a)
        p = C[i] + N[i] * (math.cos(a) * rn) + B[i] * (math.sin(a) * rb)
        if warp:
            p = Vector(warp(v, a, p, C[i], d))
        return p

    return surf(fn, us, vs, wrap=True, role=role, name=name, tip0=tip0, tip1=tip1, closed=tip0 and tip1, paint=paint)


def strand(pts, radii, sides=8, rows=None, role="body", name="strand", up=(0, 0, 1), flat=1.0, point=False):
    """A light tube with rounded ends, its radius following `radii` spread
    evenly along the path: hair locks, fingers, tendrils. `point` ends it in
    a sharp tip instead: flames, brow and moustache ends."""
    path = Path(pts)
    n = len(radii)
    keys = [(i / (n - 1), r) for i, r in enumerate(radii)]
    L = max(path.length, 1e-6)
    c0 = min(0.3, radii[0] / L)
    c1 = 0.0 if point else min(0.3, radii[-1] / L)
    rows = rows or max(5, int(L / 0.014))
    body = [c0 + (1 - c0 - c1) * i / rows for i in range(rows + 1)]
    vs = [0.0, c0 * 0.35, c0 * 0.75] + body + ([] if point else [1 - c1 * 0.75, 1 - c1 * 0.35, 1.0])

    def radius(v, a):
        r = curve(keys, v)
        if v < c0:
            r *= math.sqrt(max(0.0, 1 - ((c0 - v) / c0) ** 2))
        elif c1 and v > 1 - c1:
            r *= math.sqrt(max(0.0, 1 - ((v - 1 + c1) / c1) ** 2))
        return (r, r * flat)

    return sweep(pts, vs, sides, radius, role=role, name=name, up=up, tip0=True, tip1=True)


def lock(pts, radii, flat=2.0, twist=0.0, sides=6, rows=6, role="metal", name="lock", up=(0, 0, 1)):
    """A flat, tapering lock of hair: a ribbon `flat` times wider than thick
    whose cross-section turns by `twist` radians from root to point."""
    path = Path(pts)
    n = len(radii)
    keys = [(i / (n - 1), r) for i, r in enumerate(radii)]
    c0 = min(0.3, radii[0] / max(path.length, 1e-6))
    vs = [0.0, c0 * 0.35, c0 * 0.75] + [c0 + (1 - c0) * i / rows for i in range(rows + 1)]

    def radius(v, a):
        r = curve(keys, v)
        if v < c0:
            r *= math.sqrt(max(0.0, 1 - ((c0 - v) / c0) ** 2))
        t = a - twist * v
        return r * flat / math.sqrt((math.cos(t) * flat) ** 2 + math.sin(t) ** 2)

    return sweep(pts, vs, sides, radius, role=role, name=name, up=up, tip0=True, tip1=True)


def spaced(n, ease=1.0):
    return [(i / n) ** ease for i in range(n + 1)]


def orient(obj, origin, normal, up):
    """Moves a part modelled facing -Y (in the XZ plane) onto a surface: its
    front turns to `normal` and its top toward `up`."""
    n = Vector(normal).normalized()
    y = -n
    z = Vector(up) - n * Vector(up).dot(n)
    z.normalize()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    obj.data.transform(m)
    obj.data.update()
    return obj


def star(r_out, r_in, n=5, turn=0.0):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = turn + math.pi * i / n
        pts.append((math.sin(a) * r, math.cos(a) * r))
    return pts


def transform(objs, matrix):
    for o in objs:
        o.data.transform(matrix)
        o.data.update()
    return objs


# ---------------------------------------------------------------- the pose

# Right arm (-X) grips the staff; left arm (+X) is raised with a palm-up hand.
R_SHOULDER = Vector((-0.118, 0.012, 0.755))
R_ELBOW = Vector((-0.228, 0.05, 0.615))
L_SHOULDER = Vector((0.118, 0.012, 0.755))
L_ELBOW = Vector((0.262, 0.055, 0.7))
L_WRIST = Vector((0.33, -0.045, 0.91))
STAFF_FOOT = Vector((-0.2, -0.13, 0.0))
STAFF_TOP = Vector((-0.335, -0.035, 1.235))
GRIP_Z = 0.74
HEAD = Vector((0.0, -0.004, 0.925))
NECK = Vector((0.0, 0.0, 0.84))
HEAD_TURN = Euler((-0.04, 0.03, 0.13))
HAT_LOC = (0.0, 0.002, 0.975)
HAT_ROT = (-0.09, 0.05, 0.0)


def staff_at(z):
    return STAFF_FOOT.lerp(STAFF_TOP, z / STAFF_TOP.z)


STAFF_DIR = (STAFF_TOP - STAFF_FOOT).normalized()
GRIP = staff_at(GRIP_Z)
R_WRIST = GRIP + Vector((0.01, 0.046, -0.016))


# ---------------------------------------------------------------- robe


def robe_gap(z):
    """Half-angle of the coat's front opening, where the inner robe shows."""
    if z >= 0.5:
        return 0.15 + 0.64 * smoothstep(0.5, 0.72, z)
    return 0.15 + 0.38 * smoothstep(0.5, 0.0, z) ** 0.85


ROBE_PROFILE = [  # z, (rx, ry)
    (0.0, (0.178, 0.146)),
    (0.03, (0.175, 0.143)),
    (0.10, (0.158, 0.131)),
    (0.25, (0.138, 0.115)),
    (0.40, (0.116, 0.097)),
    (0.47, (0.104, 0.087)),
    (0.50, (0.102, 0.085)),
    (0.55, (0.112, 0.088)),
    (0.62, (0.125, 0.092)),
    (0.70, (0.131, 0.093)),
    (0.77, (0.126, 0.087)),
    (0.80, (0.092, 0.07)),
    (0.83, (0.045, 0.04)),
]
ROBE_NU = 56
ROBE_NP = 5  # panel columns per half
ROBE_NT = 2  # trim columns per side
ROBE_TW = 0.10  # angular width of the trim band
HEM_TOP = 0.044


def robe_angle(u, z):
    s = u if u <= 0.5 else u - 1.0
    sp, st = ROBE_NP / ROBE_NU, ROBE_NT / ROBE_NU
    g = robe_gap(z)
    m = abs(s)
    if m <= sp:
        ang, region, f = m / sp * g, "panel", m / sp
    elif m <= sp + st:
        ang, region, f = g + (m - sp) / st * ROBE_TW, "trim", (m - sp) / st
    else:
        ang, region, f = g + ROBE_TW + (m - sp - st) / (0.5 - sp - st) * (math.pi - g - ROBE_TW), "coat", 0.0
    return (math.copysign(ang, s) if s else 0.0), region, f


def robe_fold(a, z):
    low = smoothstep(0.5, 0.0, z) ** 0.75
    amp = 0.125 * low * (0.8 - 0.2 * math.cos(a)) + 0.03 * bump(z, 0.6, 0.05)
    w = 0.62 * math.sin(7 * a + 0.5) + 0.38 * math.sin(11 * a + 1.7 + 4 * z)
    # Round crests and narrow valleys, like heavy cloth.
    fold = amp * (w + 0.35 * (abs(w) - 0.5))
    # Pleats gathered under the belt at the back, fading toward the knees.
    back = smoothstep(0.6, 1.4, abs(a))
    fold += 0.035 * back * bump(z, 0.42, 0.1) * math.sin(14 * a)
    return fold


def robe_radius(a, z, region="coat", f=0.0):
    rx, ry = curve(ROBE_PROFILE, z)
    r = ellipse_r(rx, ry, a)
    if region == "coat":
        r *= 1 + robe_fold(a, z)
        r *= 1 + 0.03 * bump(abs(a) - robe_gap(z) - ROBE_TW, 0.0, 0.16) * smoothstep(0.55, 0.45, z)
    elif region == "panel":
        r *= 1 - 0.03 * smoothstep(0.0, 0.6, 1 - f) * smoothstep(0.53, 0.45, z)
        r *= 1 + 0.035 * smoothstep(0.45, 0.05, z) * math.sin(f * math.pi * 2.5)
    else:
        r *= 1 + 0.045 * math.sin(f * math.pi) + 0.012 * smoothstep(0.5, 0.4, z)
    # Left knee forward under the robe.
    r += 0.013 * bump(a, 0.5, 0.45) * bump(z, 0.29, 0.13)
    if 0.006 < z < HEM_TOP:
        r *= 1.025
    r *= 1 + 0.03 * smoothstep(0.05, 0.0, z)
    return r


def hem_lift(a, z, region="coat"):
    """The hem lifts off the floor over each fold's crest."""
    if region != "coat" or z > 0.07:
        return 0.0
    crest = max(0.0, robe_fold(a, 0.0) / 0.125)
    return 0.012 * crest ** 1.5 * (1 - z / 0.07)


def robe():
    zs = [0.835, 0.77, 0.71, 0.63, 0.59, 0.555, 0.53, 0.475,
          0.445, 0.40, 0.35, 0.295, 0.24, 0.185, 0.135, 0.09, 0.06, HEM_TOP + 0.003, HEM_TOP - 0.002,
          0.025, 0.009, 0.004, 0.0]
    us = [i / ROBE_NU for i in range(ROBE_NU)]

    def fn(u, z):
        a, region, f = robe_angle(u, z)
        return polar(robe_radius(a, z, region, f), a, z + hem_lift(a, z, region))

    def paint(u, z):
        _, region, _ = robe_angle(u, z)
        if 0.005 < z < HEM_TOP:
            return "trim"
        if region == "panel" and z > 0.003:
            return "cloth"
        if region == "trim" and z < 0.78:
            return "trim"
        # An embroidered band down the middle of the back.
        if abs(u - 0.5) < 1 / ROBE_NU and 0.06 < z < 0.45:
            return "trim"
        return None

    return surf(fn, us, zs, role="body", name="robe", tip0=True, tip1=True, closed=True, paint=paint)


def ring_pts(r, n, cx=0.0, cy=0.0, a0=0.0):
    return [(cx + math.sin(a0 + TAU * i / n) * r, cy + math.cos(a0 + TAU * i / n) * r) for i in range(n)]


def glyph(splines, depth, role="trim", name="rune"):
    """Like lib.extrude without bevels, but from several outlines at once, so
    an outline inside another cuts a hole (and one inside a hole fills again)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = depth / 2
    for outline in splines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(outline) - 1)
        for pt, (u, v) in zip(sp.points, outline):
            pt.co = (u, v, 0, 1)
        sp.use_cyclic_u = True
    obj = lib.apply_modifiers(lib._curve_object(name, cu))
    bpy.data.curves.remove(cu)
    obj.name = name
    obj.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))
    return lib.set_role(obj, role)


def ringed_cross(ro=0.0118, ri=0.0082, w=0.0017):
    holes = []
    h = math.sqrt(ri * ri - w * w)
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        t0, t1 = math.atan2(h, w), math.atan2(w, h)
        arc = [(math.cos(t0 + (t1 - t0) * i / 3) * ri, math.sin(t0 + (t1 - t0) * i / 3) * ri) for i in range(4)]
        holes.append([(sx * w, sy * w)] + [(sx * x, sy * y) for x, y in arc])
    return [ring_pts(ro, 14)] + holes


def eye_triangle():
    top, bl, br = (0.0, 0.0145), (-0.0135, -0.0088), (0.0135, -0.0088)
    cy = (top[1] + bl[1] + br[1]) / 3
    inner = [(x * 0.6, cy + (y - cy) * 0.6) for x, y in (top, br, bl)]
    eye = [(math.cos(TAU * i / 8) * 0.0044, -0.0025 + math.sin(TAU * i / 8) * 0.0022 * (1 if i < 4 else 1)) for i in range(8)]
    return [[top, br, bl], inner, eye]


def sun_sign():
    return [ring_pts(0.0118, 14), ring_pts(0.0078, 14), ring_pts(0.0034, 7)]


# Arcane glyphs down the inner robe, top to bottom: the all-seeing eye, a star,
# a ringed cross and the sun.
GLYPHS = [eye_triangle, lambda: [star(0.0135, 0.0058)], ringed_cross, sun_sign]


def panel_runes():
    """A column of glyphs embroidered down the inner robe."""
    out = []
    for z, make in zip((0.345, 0.265, 0.185, 0.105), GLYPHS):
        r = robe_radius(0.0, z, "panel", 0.0)
        g = glyph(make(), 0.005, role="trim", name="rune")
        lib.place(g, (0, 0, 0), (0, 0, 0), (1.15, 1, 1.15))
        out.append(orient(g, (0.0, -r - 0.0022, z), (0, -1, 0.06), (0, 0, 1)))
    return out


def mantle():
    a0 = 0.45
    nu = 34
    us = [i / nu for i in range(nu + 1)]
    vs = [0.0, 0.18, 0.4, 0.62, 0.8, 0.9, 1.0]

    def edge_drop(a):
        return 0.03 * (0.5 + 0.5 * math.cos(7 * (a - math.pi))) ** 2

    def fn(u, v):
        a = a0 + u * (TAU - 2 * a0)
        z = curve([(0.0, 0.852), (0.3, 0.838), (0.6, 0.806), (1.0, 0.695)], v)
        rx = curve([(0.0, 0.05), (0.3, 0.114), (0.6, 0.158), (1.0, 0.186)], v)
        ry = curve([(0.0, 0.047), (0.3, 0.088), (0.6, 0.113), (1.0, 0.134)], v)
        side = abs(math.sin(a))
        z += 0.014 * side * smoothstep(0.15, 0.7, v)
        z -= edge_drop(a) * smoothstep(0.55, 1.0, v)
        r = ellipse_r(rx, ry, a)
        r *= 1 + 0.05 * math.sin(8 * a + 0.3) * smoothstep(0.35, 1.0, v) ** 2
        return polar(r, a, z)

    def paint(u, v):
        if v > 0.9 or ((u < 0.024 or u > 0.976) and v > 0.2):
            return "trim"
        return None

    obj = surf(fn, us, vs, wrap=False, role="cloth", name="mantle", paint=paint)
    return lib.solidify(obj, 0.008)


def belt():
    rx, ry = 0.102, 0.085
    prof = [(0.098, 0.482), (0.1125, 0.484), (0.1135, 0.502), (0.1125, 0.52), (0.098, 0.522), (0.098, 0.482)]
    parts = [lib.lathe(prof, role="trim", segments=30, scale=(1, ry / rx, 1), name="belt")]
    parts.append(lib.box((0.036, 0.012, 0.036), loc=(0.0, -0.094, 0.502), role="metal", bevel=0.005, segments=2, name="buckle"))
    # Pouches on the hips, each with a flap and a button.
    for a, scale in ((1.05, 1.0), (-1.35, 0.85)):
        n = Vector((math.sin(a), -math.cos(a), 0))
        r = ellipse_r(0.113, 0.094, a)
        base = Vector(polar(r + 0.012 * scale, a, 0.458))
        pouch = lib.sphere(0.03, scale=(0.95 * scale, 0.62 * scale, 1.05 * scale), role="skin", segments=10, rings=6, name="pouch")
        lib.deform(pouch, lambda p: Vector((p.x * (1 + 0.25 * smoothstep(0.02, -0.03, p.z)), p.y, p.z if p.z < 0.018 else 0.018 + (p.z - 0.018) * 0.3)))
        orient(pouch, base, n, (0, 0, 1))
        parts.append(pouch)
        flap = lib.extrude([(-0.026, 0.0), (0.026, 0.0), (0.022, -0.022), (0.0, -0.03), (-0.022, -0.022)], 0.006,
                           role="body", bevel=0.0015, smooth=False, name="flap")
        # Hinged at the top and folded down over the pouch's top front.
        lib.place(flap, (0, -0.008 * scale, 0.024 * scale), (-0.6, 0, 0), (scale, 1, scale))
        orient(flap, base, n, (0, 0, 1))
        parts.append(flap)
        btn = lib.sphere(0.0055, loc=(0, -0.0237 * scale, 0.008 * scale), role="metal", segments=6, rings=4, name="button")
        orient(btn, base, n, (0, 0, 1))
        parts.append(btn)
    return parts


def scroll():
    parts = []
    a = 2.1
    n = Vector((math.sin(a), -math.cos(a), 0))
    base = Vector(polar(0.122, a, 0.478))
    paper = lib.cylinder(0.015, 0.015, 0.13, role="skin", segments=14, bevel=0.003, name="scroll")
    ties = [lib.cylinder(0.0165, 0.0165, 0.007, loc=(0, 0, z), role="trim", segments=14, name="tie") for z in (-0.035, 0.035)]
    seal = lib.cylinder(0.008, 0.008, 0.004, loc=(0, -0.017, 0.035), rot=(math.pi / 2, 0, 0), role="trim", segments=10, name="seal")
    for p in [paper, *ties, seal]:
        lib.place(p, (0, -0.004, 0), (0.0, 0.55, 0.0))
        orient(p, base, n, (0, 0, 1))
        parts.append(p)
    return parts


# ---------------------------------------------------------------- arms


def fillet(a, e, b, rho, n=6):
    """Points along the corner a -> e -> b, rounded at e by an arc of radius
    rho, so a sleeve swept round it has room to bend without folding."""
    u = (e - a).normalized()
    f = (b - e).normalized()
    theta = u.angle(f)
    s = e - u * (rho * math.tan(theta / 2))
    nrm = (f - u * u.dot(f)).normalized()
    centre = s + nrm * rho
    return [centre + (-nrm * math.cos(theta * i / n) + u * math.sin(theta * i / n)) * rho for i in range(n + 1)]


def sleeve(pts, r0, r1, droop, side, name, fold_phase=1.0, recede=0.0, bunch=0.0, top=0.0, hug=0.0,
           squeeze=0.0, rise=0.4, sag=1.4, swing=0.2, clear=None, nu=24):
    """A wide bell sleeve hanging from the arm, lined with cloth. Its underside
    sags in a rounded belly whose lowest line curves into the mouth, so the
    hem has no corners. `top` and `hug` lay the side away from the hang on the
    arm; `squeeze` brings the walls of the bag together so it hangs as a
    curtain; `recede` sets the top of the mouth back toward the shoulder (a
    sleeve slid down a raised arm), and `bunch` gathers it in soft rings.
    The rows the mantle hides (v < 0.4) are left out."""
    vs = [0.0, 0.42, 0.48, 0.55, 0.62, 0.69, 0.76, 0.82, 0.88, 0.93, 0.965, 1.0]
    side = Vector(side).normalized()
    path = Path(pts)
    axis = path.tangent(1.0)

    def radius(v, a):
        r = r0 + (r1 - r0) * smoothstep(0.35, 1.0, v) ** 1.2
        # Broad folds running down the cloth (harmonics the ring can sample).
        r *= 1 + 0.075 * math.sin(4 * a + fold_phase) * smoothstep(0.45, 1.0, v) + 0.035 * math.sin(3 * a + 2.0 * fold_phase) * smoothstep(0.6, 1.0, v)
        # Rings of cloth gathered where the sleeve has slid down the arm.
        r *= 1 + bunch * max(0.0, math.sin((v - 0.45) / 0.42 * math.pi * 2.5)) * smoothstep(0.42, 0.55, v) * (1 - smoothstep(0.74, 0.84, v))
        # The hem flares a little, so its band stands proud of the cloth.
        r *= 1 + 0.06 * smoothstep(0.86, 0.93, v)
        return r

    tangents = {v: path.tangent(v) for v in vs}
    # Where the path bends, the direction to the bend's centre and its radius.
    bends = {}
    for v in vs:
        v0, v1 = max(0.0, v - 0.01), min(1.0, v + 0.01)
        kv = (path.tangent(v1) - path.tangent(v0)) / (path.length * (v1 - v0))
        bends[v] = (kv.normalized(), 1 / kv.length) if kv.length > 1e-6 else (None, 0.0)

    def warp(v, a, p, c, d):
        # The hang direction within this ring's plane, so k runs the full
        # range from the top of the arm (-1) to the bottom of the bag (1).
        t = tangents[v]
        h = (side - t * side.dot(t)).normalized()
        across = t.cross(h)
        k = max(-1.0, min(1.0, d.dot(h)))
        loose = smoothstep(0.42, 1.0, v)
        q = p - c
        # Hanging cloth falls flat: the front and back walls of the bag come
        # together, so it is a curtain, not a sack. The bottom of the bag
        # keeps its full round.
        qa = q.dot(across)
        q -= across * (qa * squeeze * loose * (1 - k * k))
        # The side away from the hang lies on the arm.
        if top:
            wt = hug * (0.5 - 0.5 * k) ** 2 * smoothstep(0.42, 0.6, v)
            q = q * (1 - wt) + q.normalized() * (top * wt)
        # On the inside of a bend the cloth cannot reach past the bend's
        # centre, or the rings would cross and fold into a fin.
        kdir, rho = bends[v]
        if kdir is not None:
            x = q.dot(kdir)
            lim = 0.8 * rho
            if x > 0.6 * lim:
                q -= kdir * (x - 0.6 * lim - 0.4 * lim * math.tanh((x - 0.6 * lim) / (0.4 * lim)))
        p = c + q
        # A low power spreads the drop over the whole lower half, so the
        # cloth hangs in a rounded belly rather than a keel; flat at the very
        # bottom, so the lowest line is a broad curve; it rises again into
        # the mouth.
        w = (0.5 + 0.5 * k) ** 1.3
        w = 1 - (1 - w) ** 2
        drop = droop * w * smoothstep(0.3, 0.85, v) ** sag * (1 - rise * smoothstep(0.7, 1.0, v) ** 1.3 * w)
        p.z -= drop
        # The hanging cloth swings away from the arm...
        p += d * (swing * drop)
        # ...and the top of the mouth recedes, ramped over several rows so no
        # row folds back over the one before; every row of the hem band moves
        # together, so the band keeps its width.
        rim = smoothstep(0.6, 0.9, v)
        p -= axis * (recede * (0.5 - 0.5 * k) ** 2 * rim)
        p += axis * (0.008 * math.sin(3 * a + fold_phase) * rim)
        if clear:
            # Cloth falling past the forearm drapes round it, not through it.
            a0, a1, rad = clear
            foot, f = intersect_point_line(p, a0, a1)
            dist = (p - a0.lerp(a1, max(0.0, min(1.0, f)))).length
            # Pushed square off the forearm by a smooth maximum of the
            # distance and the radius, so neighbouring rows move alike and
            # nothing creases where the push begins.
            p = p + (p - foot).normalized() * ((dist ** 4 + rad ** 4) ** 0.25 - dist)
        return p

    obj = sweep(pts, vs, nu, radius, role="body", name=name, warp=warp,
                paint=lambda u, v: "trim" if v > 0.94 else None)
    return [lined(obj, 0.007)]


def lined(obj, thickness):
    """Solidifies a cloth shell painted body and trim, giving its inner
    face the cloth role: a lining that shows inside the mouth."""
    obj.data.materials.append(lib.mat("cloth"))
    obj.data.materials.append(lib.mat("trim"))
    mod = obj.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0
    mod.material_offset = 2
    return lib.apply_modifiers(obj)


def undersleeve(pts, r0, r1, name):
    """The inner robe's snug sleeve: tapering to the wrist, gathered in soft
    wrinkles that spiral a little."""
    vs = [0.0, 0.06, 0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 0.94, 1.0]

    def radius(v, a):
        r = r0 + (r1 - r0) * v ** 0.8
        r *= 1 + 0.08 * bump(v, 0.3, 0.25) - 0.08 * smoothstep(0.75, 1.0, v)
        return r * (1 + 0.13 * max(0.0, math.sin(v * 14 + 1.6 * math.sin(a))) * (1 - 0.6 * v) + 0.035 * math.sin(3 * a))

    return sweep(pts, vs, 10, radius, role="cloth", name=name, tip0=True, tip1=True)


def arms():
    parts = []
    # Right sleeve: down the upper arm, then forward along the forearm.
    r_mouth = R_ELBOW.lerp(R_WRIST, 0.7)
    parts += sleeve([R_SHOULDER, *fillet(R_SHOULDER, R_ELBOW, r_mouth, 0.038), r_mouth],
                    0.047, 0.09, 0.09, (0, 0.2, -1), "sleeve_r", top=0.045, hug=0.6, squeeze=0.2)
    # Left sleeve has slid back to the elbow of the raised forearm and hangs.
    l_mouth = L_ELBOW.lerp(L_WRIST, 0.4)
    parts += sleeve([L_SHOULDER, *fillet(L_SHOULDER, L_ELBOW, l_mouth, 0.05), l_mouth],
                    0.047, 0.09, 0.22, (0.45, 0.3, -0.84), "sleeve_l", fold_phase=2.2,
                    recede=0.02, bunch=0.13, top=0.048, hug=0.9, squeeze=0.5, rise=0.3, sag=0.8, swing=0.1,
                    clear=(L_ELBOW, L_WRIST, 0.047))
    # Inner sleeves of the robe beneath, snug to the wrist.
    parts.append(undersleeve([R_ELBOW.lerp(R_WRIST, 0.2), R_ELBOW.lerp(R_WRIST, 0.7), R_WRIST], 0.031, 0.025, "undersleeve_r"))
    parts.append(undersleeve([L_ELBOW, L_ELBOW.lerp(L_WRIST, 0.5), L_WRIST], 0.036, 0.023, "undersleeve_l"))
    # The raised cuff tapers onto the wrist and closes there, so it shows no
    # pipe end round the hand.
    parts.append(lib.lathe([(0.0215, -0.009), (0.0272, -0.0075), (0.027, 0.004), (0.0235, 0.0095), (0.012, 0.0135), (0.0, 0.0148)],
                           loc=L_WRIST.lerp(L_ELBOW, 0.06), rot=(L_WRIST - L_ELBOW).to_track_quat("Z", "Y").to_euler(),
                           role="trim", segments=12, name="cuff_l"))
    parts += right_hand()
    parts += left_hand()
    return parts


def right_hand():
    """A fist around the staff: palm behind it, four fingers wrapped round the
    front, the thumb over the top."""
    parts = []
    zax = STAFF_DIR
    xax = Vector((0.15, 1.0, 0.0))
    xax = (xax - zax * xax.dot(zax)).normalized()  # toward the wrist side
    yax = zax.cross(xax)
    g = GRIP

    def at(theta, rad, h):
        return g + xax * (math.cos(theta) * rad) + yax * (math.sin(theta) * rad) + zax * h

    palm = lib.sphere(0.03, scale=(0.78, 1.0, 1.25), role="skin", segments=12, rings=8, name="palm_r")
    m = Matrix((xax, yax, zax)).transposed().to_4x4()
    m.translation = at(0.0, 0.026, -0.002)
    palm.data.transform(m)
    parts.append(palm)
    for k, h in enumerate((0.021, 0.007, -0.007, -0.02)):
        rad = 0.0245
        end = 4.6 - 0.15 * k
        pts = [at(0.75 + (end - 0.75) * i / 5, rad * (1.0 - 0.06 * (i == 5)), h - 0.002 * i) for i in range(6)]
        r = 0.0098 - 0.0007 * k
        parts.append(strand(pts, [r, r * 1.05, r, r * 0.95, r * 0.88], sides=6, rows=4, role="skin", name="finger_r"))
    thumb = [at(-0.4, 0.03, 0.022), at(-1.1, 0.026, 0.034), at(-1.9, 0.022, 0.034), at(-2.5, 0.02, 0.028)]
    parts.append(strand(thumb, [0.012, 0.011, 0.0095, 0.008], sides=6, rows=5, role="skin", name="thumb_r"))
    return parts


L_PALM = L_WRIST + Vector((0.004, -0.024, 0.026))
FIREBALL = L_PALM + Vector((0.008, -0.014, 0.1))


def left_hand():
    """Palm up and cupped under the fireball, fingers curling up around it."""
    parts = []
    fwd = Vector((0.1, -1.0, 0.25)).normalized()
    up = Vector((-0.15, 0.25, 1.0))
    up = (up - fwd * up.dot(fwd)).normalized()
    side = fwd.cross(up)  # toward +X (thumb side)
    c = L_PALM
    palm = lib.sphere(0.03, scale=(1.12, 1.25, 0.5), role="skin", segments=12, rings=8, name="palm_l")
    m = Matrix((side, fwd, up)).transposed().to_4x4()
    m.translation = c
    palm.data.transform(m)
    parts.append(palm)
    # The wrist, from inside the cuff to the heel of the palm.
    arm = (L_WRIST - L_ELBOW).normalized()
    parts.append(strand([L_WRIST - arm * 0.012, L_WRIST + arm * 0.004, c - fwd * 0.018], [0.0165, 0.0155, 0.0145],
                        sides=6, rows=2, role="skin", name="wrist_l"))
    for k, (dx, ln) in enumerate(((-0.021, 0.85), (-0.007, 1.0), (0.007, 0.97), (0.02, 0.82))):
        base = c + side * dx + fwd * 0.03 + up * 0.002
        spread = side * (dx * 0.35)
        pts = [base,
               base + fwd * 0.022 * ln + up * 0.008 + spread,
               base + fwd * 0.036 * ln + up * 0.026 + spread * 1.6,
               base + fwd * 0.038 * ln + up * 0.045 + spread * 2.0]
        r = 0.0095 if k in (1, 2) else 0.0085
        parts.append(strand(pts, [r, r, r * 0.92, r * 0.8], sides=6, rows=4, role="skin", name="finger_l"))
    thumb = [c + side * 0.026 - up * 0.002, c + side * 0.045 + fwd * 0.018 + up * 0.012, c + side * 0.05 + fwd * 0.03 + up * 0.035]
    parts.append(strand(thumb, [0.012, 0.0105, 0.0085], sides=6, rows=4, role="skin", name="thumb_l"))
    return parts


# ---------------------------------------------------------------- head


BEARD = [  # v: (cx, cy, cz, rx, d_front, d_back)
    (0.00, (0.0, -0.028, 0.918, 0.064, 0.048, 0.046)),
    (0.08, (0.0, -0.056, 0.868, 0.081, 0.048, 0.036)),
    (0.22, (0.0, -0.091, 0.800, 0.088, 0.045, 0.036)),
    (0.40, (0.004, -0.117, 0.715, 0.078, 0.039, 0.034)),
    (0.58, (0.011, -0.128, 0.635, 0.061, 0.033, 0.034)),
    (0.76, (0.02, -0.132, 0.56, 0.042, 0.027, 0.028)),
    (0.90, (0.03, -0.134, 0.50, 0.024, 0.019, 0.016)),
    (1.00, (0.044, -0.14, 0.452, 0.003, 0.003, 0.003)),
]
# Around the beard: a few columns over the back, which lies against the chest,
# and many over the front, where the locks and the grooves between them show.
BEARD_US = [i / 16 for i in range(8)] + [0.5 + i / 52 for i in range(26)]


def beard_point(v, a, out=0.0):
    cx, cy, cz, rx, df, db = curve(BEARD, v)
    # The outline ripples a little, as hair falls in waves.
    rx *= 1 + 0.07 * math.sin(v * 17 + 0.6) * smoothstep(0.1, 0.3, v)
    s, c = math.sin(a), math.cos(a)
    d = df if s < 0 else db
    grow = smoothstep(0.03, 0.25, v) * (1 - smoothstep(0.93, 1.0, v))
    if s < 0:
        # Five locks across the front with four grooves between them, evenly
        # spaced in x so they read face on, waving a little as they fall.
        amp = 0.22 + 0.14 * smoothstep(0.55, 0.92, v)
        g = 0.5 + 0.5 * math.cos(4 * math.pi * c + 0.55 * math.sin(5.5 * v) + 0.25)
        lk = 1 + amp * grow * (g ** 0.6 - 0.62)
    else:
        amp = 0.14 + 0.2 * smoothstep(0.55, 0.92, v)
        lk = 1 + 0.4 * amp * grow * (abs(math.cos(4.5 * a + 1.8 * v + 0.4)) - 0.62)
    return Vector((cx + (rx * lk + out) * c, cy + (d * lk + out) * s, cz))


def beard():
    vs = [0.0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.32, 0.39, 0.46, 0.53, 0.6, 0.67, 0.74, 0.81, 0.87, 0.92, 0.96, 1.0]
    return surf(lambda u, v: beard_point(v, u * TAU), BEARD_US, vs, role="metal", name="beard", tip0=True, tip1=True, closed=True)


def beard_tips():
    """Two locks parting from the beard's lower end, so it ends in three
    points instead of one."""
    parts = []
    for side, tip, lean in ((-1, (-0.006, -0.136, 0.448), 0.4), (1, (0.08, -0.132, 0.478), -0.5)):
        root = beard_point(0.6, -math.pi / 2 + side * 0.85, -0.006)
        mid = beard_point(0.83, -math.pi / 2 + side * 1.0, -0.002)
        mid.x += side * 0.006
        pts = [root, mid, Vector(tip).lerp(mid, 0.45) + Vector((side * 0.006, -0.002, 0)), Vector(tip)]
        parts.append(lock(pts, [0.012, 0.013, 0.009, 0.0045, 0.0008], flat=1.5, twist=lean, sides=6, rows=6,
                          role="metal", name="beard_tip", up=(0, -1, 0)))
    return parts


def beard_locks():
    """The moustache, lying on the beard and drooping into it."""
    parts = []
    for s in (-1, 1):
        def on(v, da, out):
            return beard_point(v, -math.pi / 2 - s * da, out)

        parts.append(strand([on(0.05, 0.2, 0.008), on(0.075, 0.85, 0.012), on(0.13, 1.2, 0.011), on(0.21, 1.36, 0.008), on(0.28, 1.3, 0.002)],
                            [0.011, 0.0145, 0.0125, 0.008, 0.002], sides=7, rows=9, role="metal", name="moustache", point=True))
    return parts


def head():
    parts = []
    parts.append(lib.sphere(0.07, loc=HEAD, scale=(0.92, 1.0, 1.06), role="skin", segments=20, rings=10, name="head"))
    for s in (-1, 1):
        parts.append(lib.sphere(0.02, loc=(s * 0.032, -0.056, 0.914), scale=(1.0, 0.8, 0.85), role="skin", segments=8, rings=5, name="cheek"))
        parts.append(lib.sphere(0.0075, loc=(s * 0.024, -0.064, 0.936), role="cloth", segments=8, rings=5, name="eye"))
        # Bushy brows jutting over the eyes, their ends swept up and out.
        parts.append(strand([(s * 0.008, -0.074, 0.953), (s * 0.036, -0.081, 0.957), (s * 0.067, -0.075, 0.96), (s * 0.099, -0.062, 0.966), (s * 0.128, -0.05, 0.973)],
                            [0.0135, 0.0185, 0.017, 0.012, 0.004], sides=6, rows=7, role="metal", name="brow", point=True, flat=1.3))
        parts.append(lib.sphere(0.0075, loc=(s * 0.011, -0.085, 0.897), role="skin", segments=6, rings=4, name="nostril"))
    parts.append(strand([(0, -0.062, 0.948), (0, -0.079, 0.926), (0, -0.09, 0.903)], [0.0085, 0.0095, 0.011], sides=8, rows=5, role="skin", name="nose"))
    parts.append(lib.sphere(0.0135, loc=(0, -0.091, 0.899), role="skin", segments=10, rings=6, name="nose_tip"))
    parts.append(beard())
    parts += beard_tips()
    parts += beard_locks()
    parts += hair()
    return parts


def mantle_r(a, z):
    """Roughly how far the mantle's outer surface lies from the axis at angle
    a and height z (for laying hair over it)."""
    best = None
    for i in range(41):
        v = i / 40
        zv = curve([(0.0, 0.852), (0.3, 0.838), (0.6, 0.806), (1.0, 0.695)], v) + 0.014 * abs(math.sin(a)) * smoothstep(0.15, 0.7, v)
        if best is None or abs(zv - z) < best[0]:
            best = (abs(zv - z), v)
    v = best[1]
    rx = curve([(0.0, 0.05), (0.3, 0.114), (0.6, 0.158), (1.0, 0.186)], v)
    ry = curve([(0.0, 0.047), (0.3, 0.088), (0.6, 0.113), (1.0, 0.134)], v)
    return ellipse_r(rx, ry, a) * (1 + 0.05 * abs(math.sin(8 * a + 0.3)) * smoothstep(0.35, 1.0, v) ** 2) + 0.006


def hair():
    """Long hair from under the brim: a short cap over the back of the head
    and thick locks that fall over the collar onto the mantle."""
    a0 = 1.1
    nu = 20
    us = [i / nu for i in range(nu + 1)]

    def fn(u, v):
        a = a0 + u * (TAU - 2 * a0)
        z = 0.975 - 0.085 * v
        r = ellipse_r(0.068 + 0.012 * v, 0.073 + 0.014 * v, a)
        # At the temples the hair is combed flat onto the head, so the cap
        # has no slab-like edge beside the face.
        k = math.sqrt(max(0.0, 1 - ((z - HEAD.z) / 0.0742) ** 2))
        on_head = ellipse_r(0.0644 * k, 0.07 * k, a) + 0.004
        r = on_head + (r - on_head) * smoothstep(0.0, 0.2, min(u, 1 - u))
        x, y, _ = polar(r, a, z)
        return (x, y + HEAD.y + 0.004, z)

    parts = [lib.solidify(surf(fn, us, [0.0, 0.5, 1.0], wrap=False, role="metal", name="hair"), 0.012)]
    # Irregular locks: uneven root angles, lengths and widths, each with its
    # own sway and twist, packed close enough to overlap into one mass.
    # (root angle, end height, width scale, flatness, sway, twist, lift)
    LOCKS = [
        (1.78, 0.79, 0.8, 1.9, 0.10, 0.5, 0.000),
        (2.18, 0.735, 1.15, 2.4, 0.06, -0.6, 0.004),
        (2.55, 0.775, 0.95, 2.1, -0.04, 0.8, 0.000),
        (2.86, 0.70, 1.2, 2.6, 0.05, -0.4, 0.006),
        (3.24, 0.755, 1.05, 2.2, -0.07, 0.7, 0.002),
        (3.6, 0.715, 1.2, 2.5, 0.03, -0.8, 0.005),
        (3.98, 0.785, 0.9, 2.0, -0.08, 0.5, 0.000),
        (4.42, 0.80, 0.8, 1.9, -0.11, -0.5, 0.002),
    ]
    for a, z_end, wscale, flat, sway, twist, lift in LOCKS:
        out = Vector((math.sin(a), -math.cos(a), 0))
        pts = []
        zs = [0.965, 0.91, 0.86, 0.86 + (z_end - 0.86) * 0.5, z_end]
        for i, z in enumerate(zs):
            extra = 0.012 if i == 1 else 0.0
            ang = a + sway * (0.975 - z) * 5
            if z < 0.9:
                r = max(ellipse_r(0.07, 0.075, ang) + 0.006 + extra, mantle_r(ang, z) + 0.004) + lift
            else:
                r = ellipse_r(0.07, 0.075, ang) + extra
            pts.append(Vector(polar(r, ang, z)) + Vector((0, HEAD.y + 0.004, 0)) * (1 if z > 0.9 else 0))
        rr = 0.0142 * wscale
        parts.append(lock(pts, [rr * 0.85, rr * 1.05, rr, rr * 0.82, rr * 0.52, rr * 0.16], flat=flat, twist=twist,
                          sides=6, rows=6, role="metal", name="hair_lock", up=tuple(out)))
    return parts


# ---------------------------------------------------------------- hat


HAT_PATH = [(0, 0, 0), (0, 0.005, 0.14), (0.004, 0.022, 0.26), (0.016, 0.058, 0.345), (0.042, 0.105, 0.398), (0.078, 0.148, 0.42), (0.108, 0.17, 0.41)]


def hat():
    parts = []
    vs = [0.0, 0.12, 0.29, 0.38, 0.46, 0.53, 0.6, 0.66, 0.72, 0.78, 0.83, 0.88, 0.94, 1.0]

    def radius(v, a):
        r = 0.091 * (1 - v) ** 0.92 + 0.0015
        r *= 1 + 0.07 * math.sin(v * math.pi * 6.5 + 0.4 * math.sin(2 * a)) * smoothstep(0.4, 0.6, v) * (1 - smoothstep(0.85, 1.0, v))
        return r

    parts.append(sweep(HAT_PATH, vs, 24, radius, role="cloth", name="hat_cone", up=(1, 0, 0), tip1=True))
    band = lib.lathe([(0.084, -0.004), (0.097, -0.003), (0.0985, 0.02), (0.095, 0.045), (0.082, 0.046), (0.084, -0.004)],
                     role="trim", segments=32, name="hat_band")
    parts.append(band)
    for a, size in ((0.0, 0.021), (0.85, 0.015), (-0.85, 0.015), (1.75, 0.013), (-1.75, 0.013)):
        p = Vector(polar(0.0975, a, 0.02))
        s = lib.extrude(star(size, size * 0.42), 0.004, role="metal", bevel=0.0, name="band_star")
        parts.append(orient(s, p, Vector((math.sin(a), -math.cos(a), 0.06)), (0, 0, 1)))
    parts.append(brim())
    for p in parts:
        lib.place(p, HAT_LOC, HAT_ROT)
    return parts


def brim():
    nu = 40
    us = [i / nu for i in range(nu)]
    vs = [0.0, 0.26, 0.55, 0.8, 0.925, 1.0]

    def fn(u, v):
        a = u * TAU
        r_out = ellipse_r(0.21, 0.195, a)
        r = 0.074 + (r_out - 0.074) * v
        z = 0.004 + 0.016 * v - 0.044 * v ** 2.6
        z += (0.013 * math.sin(3 * a + 0.8) + 0.006 * math.sin(5 * a + 2.0)) * v ** 2
        return polar(r, a, z)

    obj = surf(fn, us, vs, role="cloth", name="brim", paint=lambda u, v: "trim" if v > 0.93 else None)
    return lib.solidify(obj, 0.009)


# ---------------------------------------------------------------- staff


def staff():
    parts = []
    n = 11
    perp1 = STAFF_DIR.cross(Vector((0, 1, 0))).normalized()
    perp2 = STAFF_DIR.cross(perp1)
    pts, radii = [], []
    for i in range(n + 1):
        t = i / n
        p = STAFF_FOOT.lerp(STAFF_TOP, t)
        if 0 < i < n:
            p = p + perp1 * 0.007 * math.sin(i * 2.3) + perp2 * 0.006 * math.cos(i * 1.7)
        pts.append(p)
        radii.append(0.0155 + 0.0035 * math.sin(i * 2.9) ** 2 + 0.005 * smoothstep(0.82, 1.0, t))
    shaft = lib.tube(pts, radii, role="body", sides=10, resolution=3, name="staff")

    def grain(p):
        rel = p - STAFF_FOOT
        h = rel.dot(STAFF_DIR)
        radial = rel - STAFF_DIR * h
        if radial.length < 1e-6:
            return p
        ang = math.atan2(radial.dot(perp2), radial.dot(perp1))
        return p + radial.normalized() * 0.0028 * math.sin(3 * ang + h * 20)

    lib.deform(shaft, grain)
    parts.append(shaft)
    rot = STAFF_DIR.to_track_quat("Z", "Y").to_euler()
    # Burls and a broken-off twig make it gnarled.
    for h, ang, r in ((0.34, 0.6, 0.017), (0.98, 3.4, 0.015)):
        base = staff_at(h) + (perp1 * math.cos(ang) + perp2 * math.sin(ang)) * 0.011
        parts.append(lib.sphere(r, loc=base, scale=(1, 1, 1.6), rot=rot, role="body", segments=8, rings=5, name="burl"))
    # Two bands of the side's colour, below and above the hand, so the staff
    # still reads where its wood melts into a dark robe.
    for z in (0.42, 0.9):
        f = z / STAFF_TOP.z * n
        i = int(f)
        centre = pts[i].lerp(pts[i + 1], f - i)
        r = radii[i] + (radii[i + 1] - radii[i]) * (f - i) + 0.0028
        axis = (pts[i + 1] - pts[i]).normalized()
        parts.append(lib.lathe([(r - 0.001, -0.008), (r + 0.0035, -0.0055), (r + 0.0035, 0.0055), (r - 0.001, 0.008)],
                               loc=centre, rot=axis.to_track_quat("Z", "Y").to_euler(), role="trim", segments=10, name="staff_band"))
    twig0 = staff_at(0.5)
    out = (perp1 * math.cos(2.2) + perp2 * math.sin(2.2))
    parts.append(strand([twig0, twig0 + out * 0.03 + STAFF_DIR * 0.03, twig0 + out * 0.045 + STAFF_DIR * 0.07],
                        [0.009, 0.006, 0.004], sides=6, rows=4, role="body", name="twig"))
    # A ribbon wound under the head, its tails fluttering out.
    wrap = []
    for i in range(13):
        t = i / 12
        ang = t * math.pi * 3.2 + 0.4
        wrap.append(STAFF_TOP + STAFF_DIR * (-0.115 + 0.035 * t) + (perp1 * math.cos(ang) + perp2 * math.sin(ang)) * 0.0215)
    parts.append(strand(wrap, [0.0075] * 13, sides=6, rows=17, role="trim", name="ribbon", up=STAFF_DIR, flat=0.3))
    knot = wrap[-1]
    for dx, dz, sway in ((-0.03, -0.13, 0.012), (-0.05, -0.1, -0.01)):
        tail = [knot, knot + Vector((dx * 0.35, -0.006, dz * 0.3)), knot + Vector((dx * 0.7, sway, dz * 0.65)), knot + Vector((dx, -sway, dz))]
        parts.append(strand(tail, [0.0022, 0.0022, 0.002, 0.0015], sides=6, rows=6, role="trim", name="ribbon_tail", up=(0, 1, 0), flat=3.6, point=True))
    parts.append(lib.cylinder(0.019, 0.017, 0.045, loc=STAFF_FOOT + STAFF_DIR * 0.022, rot=rot, role="metal", segments=14, bevel=0.003, name="ferrule"))
    # Head: a cage of three curling tendrils round the crystal.
    top = STAFF_TOP
    for k in range(3):
        phi0 = k * TAU / 3 + 0.5
        pts = []
        for (rr, h, dphi) in ((0.013, -0.035, 0.0), (0.034, 0.01, 0.3), (0.042, 0.06, 0.6), (0.034, 0.105, 0.9), (0.014, 0.132, 1.15), (0.0, 0.122, 1.5)):
            phi = phi0 + dphi
            pts.append(top + perp1 * (math.cos(phi) * rr) + perp2 * (math.sin(phi) * rr) + STAFF_DIR * h)
        parts.append(strand(pts, [0.012, 0.0105, 0.0085, 0.0065, 0.005, 0.002], sides=6, rows=9, role="body", name="tendril", point=True))
    gem = lib.lathe([(0, -0.036), (0.025, 0.0), (0.021, 0.042), (0, 0.09)], role="eye", segments=6, name="crystal")
    lib.place(gem, top + STAFF_DIR * 0.028, rot)
    parts.append(gem)
    parts.append(lib.cylinder(0.022, 0.022, 0.012, loc=top - STAFF_DIR * 0.05, rot=rot, role="metal", segments=12, bevel=0.003, name="collar"))
    return parts


# ---------------------------------------------------------------- fireball


def fireball():
    """A ball of fire: a glowing core with flame tongues licking up round it to
    separate tips, a taller flame rising from the top, and embers."""
    up = Vector((0, 0, 1))
    c = FIREBALL
    parts = [lib.sphere(0.037, loc=c, role="eye", segments=12, rings=8, name="fire_core")]
    for phi, el, reach, h, t in ((0.3, 0.25, 0.05, 0.078, 0.018), (1.5, 0.05, 0.056, 0.058, 0.016), (2.6, 0.3, 0.05, 0.072, 0.017),
                                 (3.8, 0.1, 0.056, 0.052, 0.015), (5.0, 0.28, 0.05, 0.066, 0.017)):
        d0 = Vector((math.sin(phi) * math.cos(el), -math.cos(phi) * math.cos(el), math.sin(el)))
        side = up.cross(d0).normalized()
        pts = [c + d0 * 0.016,
               c + d0 * (reach * 0.82) + up * (h * 0.12) + side * 0.008,
               c + d0 * (reach * 0.92) + up * (h * 0.55) - side * 0.006,
               c + d0 * (reach * 0.8) + up * h + side * 0.007]
        parts.append(strand(pts, [t, t * 0.95, t * 0.55, 0.0012], sides=6, rows=5, role="eye", name="fire_tongue", point=True))
    parts.append(strand([c, c + Vector((0.008, -0.004, 0.05)), c + Vector((-0.007, 0.0, 0.092)), c + Vector((0.01, 0.004, 0.13))],
                        [0.03, 0.022, 0.011, 0.0012], sides=8, rows=6, role="eye", name="fire_tongue", point=True))
    for dx, dy, dz, r in ((0.075, -0.01, 0.055, 0.008), (-0.03, -0.035, 0.145, 0.007)):
        parts.append(lib.ico(r, loc=c + Vector((dx, dy, dz)), role="eye", subdiv=1, name="ember"))
    # A little larger than life, so it reads beside the brim on the board.
    return transform(parts, Matrix.Translation(c) @ Matrix.Scale(1.15, 4) @ Matrix.Translation(-c))


# ---------------------------------------------------------------- build


def build():
    parts = []

    def add(objs):
        for o in objs if isinstance(objs, list) else [objs]:
            key = o.name.split(".")[0]
            TRIS[key] = TRIS.get(key, 0) + lib.tris(o)
            parts.append(o)

    add(robe())
    add(panel_runes())
    add(mantle())
    add(belt())
    add(scroll())
    add(arms())
    # The head turns a little toward the conjured fireball.
    turn = Matrix.Translation(NECK) @ HEAD_TURN.to_matrix().to_4x4() @ Matrix.Translation(-NECK)
    add(transform(head() + hat(), turn))
    add(staff())
    add(fireball())
    print("PARTS " + json.dumps(dict(sorted(TRIS.items(), key=lambda kv: -kv[1]))) + f" total {sum(TRIS.values())}")
    return parts
