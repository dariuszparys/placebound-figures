"""Sorceress: the Dark commander, a regal witch-queen. A bold crescent-moon
crown, bladed and jewelled, whose horns rise high over her head; her head
pitched chin-down in a glare; long hair parted down the middle, falling in
clumps down her back and in two thin locks a side over her collarbones; a
stiff ribbed collar whose points frame her jaw; a fitted bodice with a
jewelled stomacher and a pointed basque; a trumpet gown, narrow to the knee,
flaring into a folded train embroidered with crescents; a wrapped staff
crowned by a crescent moon cradling a glowing orb in her right hand; and thin
lightning fanning from the claws of her raised left hand, with one long arc
leaping over her crown to the staff. The figure faces -Y; her right hand is
on -X."""
import json
import math
import random

import bmesh
from mathutils import Matrix, Vector

import lib

TAU = math.tau
TRIS = {}
# Tubes of six to eight sides (fingers, locks, horns) shade smooth; boxes and
# shell rims, at 90 degrees, stay crisp.
FINISH = {"sharp_angle": 62}


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    t = clamp((x - e0) / (e1 - e0))
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


def linear(keys, x):
    """Piecewise-linear interpolation through (x, value-tuple) keys: sharp points."""
    if x <= keys[0][0]:
        return keys[0][1]
    for (x0, v0), (x1, v1) in zip(keys, keys[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return tuple(a + (b - a) * t for a, b in zip(v0, v1))
    return keys[-1][1]


def ellipse_r(rx, ry, a):
    """Polar radius of an ellipse at angle a, measured from -Y toward +X."""
    s, c = math.sin(a), math.cos(a)
    return rx * ry / math.sqrt((ry * s) ** 2 + (rx * c) ** 2)


def polar(r, a, z, cy=0.0):
    return (math.sin(a) * r, -math.cos(a) * r + cy, z)


def frame(origin, z, front=(0, -1, 0)):
    """A matrix whose local z points along `z` and whose local -y leans toward
    `front`, placed at `origin`."""
    z = nrm(z)
    f = Vector(front)
    f = f - f.dot(z) * z
    if f.length < 1e-6:
        f = Vector((0, -1, 0)) - Vector((0, -1, 0)).dot(z) * z
        if f.length < 1e-6:
            f = Vector((1, 0, 0)) - Vector((1, 0, 0)).dot(z) * z
    y = -f.normalized()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def put(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def paint_faces(obj, role, pred):
    """Gives the faces for which pred(polygon) holds the role material."""
    m = lib.mat(role)
    names = [s.name for s in obj.data.materials]
    if role not in names:
        obj.data.materials.append(m)
        names.append(role)
    idx = names.index(role)
    for p in obj.data.polygons:
        if pred(p):
            p.material_index = idx
    return obj


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
            un = us[k] if k else (us[-1] + (us[-1] - us[-2]))
            face_roles.append(paint((us[i] + un) / 2, vm) if paint else None)
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


def torus_surf(fn, nu, nw, role="body", name="torus"):
    """A closed tube wrapped in both directions through fn(u, w), u and w in [0, 1)."""
    bm = bmesh.new()
    rows = [[bm.verts.new(Vector(fn(i / nu, j / nw))) for j in range(nw)] for i in range(nu)]
    for i in range(nu):
        a, b = rows[i], rows[(i + 1) % nu]
        for j in range(nw):
            k = (j + 1) % nw
            bm.faces.new((a[j], b[j], b[k], a[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._from_bmesh(name, bm, role)


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


def strand(pts, radii, sides=8, rows=None, role="body", name="strand", up=(0, 0, 1), flat=1.0, point=False, paint=None):
    """A tube with rounded ends whose radius follows `radii` spread evenly
    along the path: sleeves, hair locks, fingers. `point` ends it in a sharp
    tip instead: horns, flames, lock ends."""
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

    return sweep(pts, vs, sides, radius, role=role, name=name, up=up, tip0=True, tip1=True, paint=paint)


def cone_between(a, b, r0, r1, role="eye", sides=5, over=0.0, name="seg"):
    """A tapered rod from a to b, overrunning both ends by `over`."""
    a, b = Vector(a), Vector(b)
    d = b - a
    L = d.length
    o = lib.cylinder(r0, r1, L + 2 * over, role=role, segments=sides, name=name)
    return put(o, frame((a + b) / 2, d))


def jag_tube(pts, radii, sides=5, role="eye", name="bolt"):
    """One continuous tube through a zigzag of points with mitred corners,
    closed at the start and drawn to a point at the end: lightning."""
    pts = [Vector(p) for p in pts]
    n = len(pts)
    bm = bmesh.new()
    rings = []
    px = None
    for i, p in enumerate(pts):
        if i == 0:
            t, m = (pts[1] - p).normalized(), 1.0
        elif i == n - 1:
            t, m = (p - pts[i - 1]).normalized(), 1.0
        else:
            a, b = (p - pts[i - 1]).normalized(), (pts[i + 1] - p).normalized()
            t = (a + b).normalized() if (a + b).length > 1e-6 else a
            m = 1.0 / max(0.55, a.dot(t))
        x = t.orthogonal().normalized() if px is None else (px - t * px.dot(t)).normalized()
        y = t.cross(x)
        px = x
        if i == n - 1:
            rings.append([bm.verts.new(p)])
            continue
        r = radii[i] * m
        rings.append([bm.verts.new(p + (x * math.cos(k * TAU / sides) + y * math.sin(k * TAU / sides)) * r) for k in range(sides)])
    start = bm.verts.new(pts[0] - (pts[1] - pts[0]).normalized() * radii[0] * 0.5)
    for k in range(sides):
        bm.faces.new((start, rings[0][(k + 1) % sides], rings[0][k]))
    for a, b in zip(rings, rings[1:]):
        for k in range(sides):
            k1 = (k + 1) % sides
            if len(b) == 1:
                bm.faces.new((a[k], a[k1], b[0]))
            else:
                bm.faces.new((a[k], a[k1], b[k1], b[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._from_bmesh(name, bm, role)


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
    return put(obj, m)


# ---------------------------------------------------------------- the pose

HEAD = V(0.0, -0.004, 1.056)
HEAD_R = (0.064, 0.074, 0.081)
NECK0 = V(0.0, 0.006, 0.915)
NECK1 = V(0.0, 0.000, 1.010)
# The head lifts a little, chin up, about the top of the neck: haughty, and her
# face turned up toward the board camera above.
TILT_DEG = -3.0
TILT = (Matrix.Translation(NECK1) @ Matrix.Rotation(math.radians(TILT_DEG), 4, "X")
        @ Matrix.Translation(-NECK1))


def tilted(p):
    return TILT @ Vector(p)

R_SHO = V(-0.118, 0.008, 0.902)
L_SHO = V(0.118, 0.008, 0.902)
# Right hand grips the staff out at her side; left arm raised high over her head.
STAFF_FOOT = V(-0.232, -0.112, 0.0)
STAFF_TOP = V(-0.268, -0.084, 1.345)   # centre of the crescent


def staff_at(z):
    return STAFF_FOOT.lerp(STAFF_TOP, z / STAFF_TOP.z)


STAFF_DIR = (STAFF_TOP - STAFF_FOOT).normalized()
GRIP = staff_at(0.84)
R_WRI = GRIP + V(0.013, 0.046, -0.018)
R_ELB = V(-0.200, 0.058, 0.722)
# Raised well out to her left, clear of the crown, so the lightning stands
# apart from the horns from both board yaws.
L_ELB = V(0.300, -0.008, 1.052)
L_WRI = L_ELB + V(0.018, -0.024, 0.21).normalized() * 0.168


# ---------------------------------------------------------------- gown

SK_Z = [(0.0, 0.735), (0.08, 0.68), (0.16, 0.62), (0.25, 0.55), (0.4, 0.43), (0.55, 0.31), (0.7, 0.19),
        (0.82, 0.10), (0.9, 0.05), (0.95, 0.025), (1.0, 0.006)]
SK_ZT = [(0.0, 0.735), (0.08, 0.68), (0.16, 0.62), (0.25, 0.55), (0.4, 0.43), (0.55, 0.30), (0.68, 0.17),
         (0.78, 0.085), (0.86, 0.036), (0.93, 0.016), (1.0, 0.006)]
# A trumpet line: fitted and narrow through hips and thighs to the knee, then
# flaring hard to the hem and the train.
SK_FRONT = [(0.0, 0.058), (0.08, 0.071), (0.16, 0.082), (0.25, 0.088), (0.4, 0.089), (0.55, 0.093),
            (0.7, 0.120), (0.82, 0.160), (0.9, 0.184), (0.95, 0.194), (1.0, 0.198)]
SK_SIDE = [(0.0, 0.078), (0.08, 0.091), (0.16, 0.099), (0.25, 0.100), (0.4, 0.096), (0.55, 0.099),
           (0.7, 0.128), (0.82, 0.172), (0.9, 0.200), (0.95, 0.213), (1.0, 0.220)]
SK_BACK = [(0.0, 0.060), (0.08, 0.079), (0.16, 0.092), (0.25, 0.096), (0.4, 0.094), (0.55, 0.099),
           (0.7, 0.130), (0.82, 0.170), (0.9, 0.194), (0.95, 0.203), (1.0, 0.207)]
SK_TRAIN = [(0.0, 0.0), (0.45, 0.0), (0.58, 0.008), (0.70, 0.034), (0.79, 0.085), (0.87, 0.150),
            (0.93, 0.196), (1.0, 0.232)]
SK_NU = 60
SK_NP = 4   # panel columns per half
SK_NT = 2   # trim columns per side
SK_TW = 0.072
SK_HEM = 0.945


def sk_gap(v):
    """Half-angle of the gown's front opening, where the underskirt shows."""
    return 0.05 + 0.40 * smoothstep(0.1, 1.0, v) ** 1.1


def sk_base(a, v):
    c = math.cos(a)
    rf, rs, rb = curve(SK_FRONT, v), curve(SK_SIDE, v), curve(SK_BACK, v)
    r = rs + ((rf if c >= 0 else rb) - rs) * c * c
    tw = max(0.0, -c) ** 1.6
    r += curve(SK_TRAIN, v) * tw
    z = curve(SK_Z, v) * (1 - tw) + curve(SK_ZT, v) * tw
    return r, z, tw


def sk_wave(a, v):
    """Eight soft vertical folds round the gown, with a few smaller ones
    breaking them up toward the hem: rounded ridges, deeper troughs."""
    w = math.sin(8 * a + 0.6 + 0.8 * v) + 0.3 * math.sin(13 * a + 2.0 + 4 * v) * smoothstep(0.6, 1.0, v)
    return w + 0.25 * (abs(w) - 0.6)


def sk_angle(u, v):
    s = u if u <= 0.5 else u - 1.0
    sp, st = SK_NP / SK_NU, SK_NT / SK_NU
    g = sk_gap(v)
    m = abs(s)
    if m <= sp:
        ang, region, f = m / sp * g, "panel", m / sp
    elif m <= sp + st:
        ang, region, f = g + (m - sp) / st * SK_TW, "trim", (m - sp) / st
    else:
        ang, region, f = g + SK_TW + (m - sp - st) / (0.5 - sp - st) * (math.pi - g - SK_TW), "gown", 0.0
    return (math.copysign(ang, s) if s else 0.0), region, f


def skirt_point(u, v):
    vv = min(v, 1.0)
    a, region, f = sk_angle(u, vv)
    r, z, tw = sk_base(a, vv)
    g = sk_gap(vv)
    if region == "gown":
        edge = smoothstep(0.0, 0.3, abs(a) - g - SK_TW)
        # Barely a ripple over the fitted hips; deep folds below the knee,
        # deepest in the train.
        amp = (0.010 + 0.105 * smoothstep(0.42, 1.0, vv) ** 1.3 + 0.07 * tw * smoothstep(0.5, 1.0, vv)) * edge
        w = sk_wave(a, vv)
        r *= 1 + amp * w
        # The gown's edge stands proud of the opening.
        r *= 1 + 0.022 * bump(abs(a) - g - SK_TW, 0.0, 0.12) * smoothstep(0.05, 0.25, vv)
        # Folds lying on the floor lift off it between their contact points.
        z += 0.016 * max(0.0, w) * tw * smoothstep(0.82, 1.0, vv)
    elif region == "panel":
        r *= 1 - 0.04 * smoothstep(0.05, 0.3, vv)
        r *= 1 + 0.018 * math.sin(f * math.pi * 2.0 + 0.5) * smoothstep(0.3, 1.0, vv)
    else:
        r *= 1 + 0.01 * math.sin(f * math.pi) * smoothstep(0.05, 0.2, vv)
    # Left knee pushes the cloth forward a little.
    r += 0.010 * bump(a, 0.35, 0.4) * bump(vv, 0.55, 0.15)
    # The hem band stands out as a stiff border.
    r *= 1 + 0.014 * smoothstep(SK_HEM - 0.01, SK_HEM, vv)
    if v > 1.0:
        # Rolled under: the lip closing the hem on the floor.
        r *= 0.975
        z = 0.0
    return polar(r, a, z)


def skirt():
    vs = [0.0, 0.05, 0.12, 0.21, 0.32, 0.44, 0.54, 0.62, 0.69, 0.75, 0.80, 0.85,
          0.89, 0.92, SK_HEM - 0.004, SK_HEM + 0.004, 0.97, 0.99, 1.0, 1.02, 1.03]
    us = [i / SK_NU for i in range(SK_NU)]

    def paint(u, v):
        _, region, _ = sk_angle(u, min(v, 1.0))
        if SK_HEM < v < 1.0:
            return "trim"
        if region == "panel":
            return "cloth"
        # The gown's edge stands proud of the underskirt but is not piped:
        # trim is kept for what identifies her.
        return None

    return surf(skirt_point, us, vs, role="body", name="skirt", tip0=True, tip1=True, closed=True, paint=paint)


def skirt_r_at(a, z):
    """Radius of the gown's base form at angle a and height z (upper part)."""
    lo, hi = 0.0, 0.6
    for _ in range(24):
        mid = (lo + hi) / 2
        if sk_base(a, mid)[1] > z:
            lo = mid
        else:
            hi = mid
    return sk_base(a, (lo + hi) / 2)[0]


# ---------------------------------------------------------------- bodice

BD = [  # z, (rx, ry, cy)
    (0.600, (0.104, 0.092, 0.004)),
    (0.640, (0.098, 0.082, 0.004)),
    (0.680, (0.089, 0.071, 0.004)),
    (0.712, (0.083, 0.063, 0.004)),
    (0.740, (0.084, 0.063, 0.002)),
    (0.775, (0.092, 0.067, 0.000)),
    (0.810, (0.099, 0.071, -0.002)),
    (0.840, (0.103, 0.072, -0.002)),
    (0.870, (0.107, 0.068, 0.002)),
    (0.893, (0.116, 0.062, 0.006)),
    (0.912, (0.113, 0.056, 0.008)),
    (0.930, (0.090, 0.049, 0.008)),
    (0.946, (0.058, 0.040, 0.007)),
    (0.958, (0.036, 0.032, 0.005)),
]
BD_NU = 36
BD_NP = 3
BD_NT = 2
BD_TOP = 0.958
# A long inverted triangle down the front: bare skin above SKIN_Z, a jewelled
# stomacher below it, down to the point of the basque.
V_POINT = 0.628
SKIN_Z = 0.862


def bd_bottom(a):
    return 0.69 - 0.075 * max(0.0, math.cos(a)) ** 2.5


def vneck(z):
    """Half-angle of the front triangle at height z."""
    if z <= V_POINT:
        return 0.006
    x = 0.058 * ((z - V_POINT) / (BD_TOP - V_POINT)) ** 0.9
    rx = curve(BD, z)[0]
    return max(0.006, math.asin(min(0.98, x / rx)) * 1.05)


def vneck_tw(z):
    return 0.026


def bd_angle(u, z):
    s = u if u <= 0.5 else u - 1.0
    sp, st = BD_NP / BD_NU, BD_NT / BD_NU
    g, tw = vneck(z), vneck_tw(z)
    m = abs(s)
    if m <= sp:
        ang, region, f = m / sp * g, "skin", m / sp
    elif m <= sp + st:
        ang, region, f = g + (m - sp) / st * tw, "trim", (m - sp) / st
    else:
        ang, region, f = g + tw + (m - sp - st) / (0.5 - sp - st) * (math.pi - g - tw), "bodice", 0.0
    return (math.copysign(ang, s) if s else 0.0), region, f


def bd_z(v, a):
    if v <= 0.2:
        zb = bd_bottom(a)
        return zb + (0.712 - zb) * max(v, 0.0) / 0.2
    return 0.712 + (BD_TOP - 0.712) * (v - 0.2) / 0.8


def bd_radius(a, z, region="bodice", f=0.0):
    rx, ry, cy = curve(BD, z)
    r = ellipse_r(rx, ry, a)
    # Bust.
    r += 0.017 * bump(abs(a), 0.52, 0.34) * bump(z, 0.822, 0.033)
    # Shoulder blades.
    r += 0.006 * bump(abs(a), math.pi - 0.6, 0.35) * bump(z, 0.87, 0.04)
    if z < 0.725:
        # Cover the top of the gown, never inside it, but lie close on it so
        # the bodice flows into the skirt.
        need = skirt_r_at(a, z) + 0.005
        k = 0.012
        h = clamp(0.5 + 0.5 * (need - r) / k)
        r = r * (1 - h) + need * h + k * h * (1 - h)
    if region == "skin":
        # Bare skin sits back; the stomacher stands a little proud, quilted.
        if z > SKIN_Z:
            r *= 0.988
        else:
            r *= 1.005 + 0.003 * math.sin(f * math.pi)
    elif region == "trim":
        # A fine raised piping beside the stomacher; flush beside the skin.
        r *= 1 + (0.010 if z <= SKIN_Z else 0.003) * math.sin(f * math.pi)
    return r, cy


def bodice():
    zs = [0.735, 0.76, 0.785, 0.81, 0.835, 0.86, 0.88, 0.898, 0.914, 0.928, 0.94, 0.95, BD_TOP]
    vs = [-0.02, 0.0, 0.015, 0.05, 0.1, 0.15, 0.2] + [0.2 + 0.8 * (z - 0.712) / (BD_TOP - 0.712) for z in zs] + [1.01]
    us = [i / BD_NU for i in range(BD_NU)]

    def fn(u, v):
        vv = clamp(v, 0.0, 1.0)
        z = bd_z(vv, 0.0)
        a, region, f = bd_angle(u, z)
        z = bd_z(vv, a)
        r, cy = bd_radius(a, z, region, f)
        if v < 0:
            # The basque's lower edge turns under onto the gown.
            r = skirt_r_at(a, z) - 0.002
            z += 0.003
        elif v < 0.012:
            r *= 1.004
        return polar(r, a, z, cy)

    def paint(u, v):
        z = bd_z(clamp(v, 0.0, 1.0), 0.0)
        _, region, _ = bd_angle(u, z)
        if v < 0.0:
            return None
        if region == "skin":
            return "skin" if z > SKIN_Z else "cloth"
        if region == "trim":
            return "metal" if z <= SKIN_Z + 0.004 else None
        return None

    return surf(fn, us, vs, role="body", name="bodice", tip1=True, paint=paint)


def bodice_y(x, z):
    """The front surface of the bodice at (x, z): y and outward normal-ish."""
    lo, hi = 0.0, math.pi / 2
    for _ in range(30):
        mid = (lo + hi) / 2
        r, cy = bd_radius(mid, z)
        if r * math.sin(mid) < abs(x):
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    r, cy = bd_radius(a, z)
    return -r * math.cos(a) + cy, a


# ---------------------------------------------------------------- collar

COL_A0 = 1.20
COL_POINTS = (1.62, 2.04, 2.46)
COL_TOP = [  # |a|, (z, r): the rim rises to three points a side, framing the jaw
    (COL_A0, (0.986, 0.082)),
    (1.40, (1.040, 0.118)),
    (1.62, (1.092, 0.160)),
    (1.83, (1.050, 0.164)),
    (2.04, (1.140, 0.186)),
    (2.25, (1.062, 0.172)),
    (2.46, (1.090, 0.160)),
    (2.68, (1.012, 0.120)),
    (2.90, (0.982, 0.094)),
    (math.pi, (0.976, 0.090)),
]
COL_RIB = 0.022   # half-width of each raised rib, in radians


def col_abs(a):
    return abs(a if a <= math.pi else a - TAU)


def col_rib(m):
    """1 on a rib running from the neck up to a point, 0 between them."""
    return max(bump(m, p, COL_RIB * 0.8) for p in COL_POINTS)


def collar():
    # Columns fall exactly on each point and either side of its rib.
    angles = [COL_A0 + (TAU - 2 * COL_A0) * i / 30 for i in range(31)]
    keys = []
    for p in COL_POINTS:
        for q in (p, TAU - p):
            keys += [q - COL_RIB, q, q + COL_RIB]
    angles = sorted(set([round(x, 5) for x in angles if all(abs(x - k) > 0.035 for k in keys)] + keys))
    vs = [0.0, 0.22, 0.45, 0.66, 0.82, 0.93, 1.0]

    def fn(a, v):
        m = col_abs(a)
        zt, rt = linear(COL_TOP, m)
        z0 = 0.934
        r0 = ellipse_r(0.066, 0.052, a)
        r = r0 + (rt - r0) * v ** 1.5
        z = z0 + (zt - z0) * v
        # Stiff fluting between the ribs, the ribs standing proud, and a
        # rolled-out rim.
        r *= 1 + 0.025 * math.sin(m * 15) * v * smoothstep(1.3, 1.6, m)
        r += 0.007 * col_rib(m) * smoothstep(0.0, 0.25, v)
        r += 0.006 * smoothstep(0.85, 1.0, v)
        x, y, z = polar(r, a, z, 0.008 + 0.010 * v * v)
        # The side panels cup forward round the jaw.
        y -= 0.040 * v ** 1.6 * bump(m, 1.75, 0.45)
        return (x, y, z)

    def paint(a, v):
        m = col_abs(a)
        if a < COL_A0 + 0.03 or a > TAU - COL_A0 - 0.03:
            return "trim"
        if col_rib(m) > 0.5 and v > 0.05:
            return "trim"
        return None

    obj = surf(fn, angles, vs, wrap=False, role="body", name="collar", paint=paint)
    obj = lib.solidify(obj, 0.009)

    def inner(p):
        c = p.center
        radial = Vector((c.x, c.y - 0.01, 0))
        return p.normal.dot(radial) < -0.2 * radial.length

    names = [s.name for s in obj.data.materials]
    trim_idx = names.index("trim") if "trim" in names else -1
    return paint_faces(obj, "cloth", lambda p: p.material_index != trim_idx and inner(p))


# ---------------------------------------------------------------- arms


def sleeve(sho, elb, wri, side, name):
    inward = V(-side * 0.03, 0.0, 0.004)
    pts = [sho + inward, sho, sho.lerp(elb, 0.5) + V(side * 0.004, 0, 0), elb, elb.lerp(wri, 0.5), wri]
    radii = [0.036, 0.036, 0.029, 0.024, 0.025, 0.020]
    return strand(pts, radii, sides=10, rows=16, role="body", name=name)


def cuff(elb, wri, side, back):
    """A flared cuff at the wrist ending in a long point over the back of the hand."""
    d = nrm(wri - elb)
    m = frame(wri - d * 0.022, d, back)

    def point(v):
        a = math.atan2(v.x, -v.y)
        k = max(0.0, math.cos(a - math.pi)) ** 3
        v.z += 0.034 * k * clamp(v.z / 0.03)
        return v

    prof = [(0.021, -0.004), (0.024, 0.014), (0.028, 0.030)]
    o = lib.lathe(prof, role="trim", segments=12, name="cuff")
    lib.deform(o, point)
    o = lib.solidify(o, 0.003)
    return [put(o, m)]


def hanging_sleeve(elb, wri, side, drop, across, width, name):
    """A long pointed sleeve panel hanging open from the elbow, cupped round the
    arm at the top and falling flat to a point: body outside, cloth lining,
    edged in trim."""
    d = nrm(wri - elb)
    c = elb.lerp(wri, 0.06) + V(0, 0.010, 0)
    T = c + Vector(drop)
    down = nrm(T - c)
    ax = nrm(Vector(across) - down * Vector(across).dot(down))
    nml = nrm(ax.cross(down))
    if nml.y > 0:
        nml = -nml            # the panel's face looks forward
    nu, nv = 6, 8
    us = [i / nu for i in range(nu + 1)]
    vs = [j / nv for j in range(nv + 1)]
    W = [(0.0, 0.030), (0.25, 1.0), (0.55, 0.92), (0.8, 0.55), (1.0, 0.0)]

    def fn(u, v):
        x = 2 * u - 1
        w = width * curve(W, v) if v > 0 else 0.030
        p = c.lerp(T, v)
        p += ax * (x * w)
        # Cupped round the arm at the top, flattening as it falls; soft folds.
        cup = 0.030 * (1 - smoothstep(0.0, 0.45, v)) + 0.016 * (1 - v)
        p += nml * (cup * (1 - x * x) - 0.012)
        p += nml * 0.010 * math.sin(x * math.pi * 1.5 + 0.4) * smoothstep(0.2, 0.7, v)
        # The point swings out and back a little.
        p += ax * (0.018 * side * v * v) - nml * 0.012 * v * v
        return p

    def paint(u, v):
        if v > 0.86:
            return "trim"
        return None

    obj = surf(fn, us, vs, wrap=False, role="body", name=name, paint=paint)
    obj = lib.solidify(obj, 0.0045)
    names = [m.name for m in obj.data.materials]
    trim_idx = names.index("trim") if "trim" in names else -1
    return [paint_faces(obj, "cloth", lambda p: p.normal.dot(nml) > 0.3 and p.material_index != trim_idx)]


SHOULDER_W = [(0.0, 0.034), (0.30, 0.046), (0.60, 0.044), (0.82, 0.030), (1.0, 0.0015)]


def shoulder_cap(sho, elb, side):
    """An angular pauldron: a crested plate wrapped over the top of the upper
    arm, clear of the sleeve, whose lower end flares out and sweeps up into a
    point; edged in trim."""
    a = nrm(elb - sho)
    t0 = V(side * 0.75, 0.0, 0.66) if a.z < 0.3 else V(0, 0, 1)
    top = nrm(t0 - a * t0.dot(a))
    across = nrm(a.cross(top))
    s0, s1 = (-0.034, 0.066) if a.z < 0.3 else (0.004, 0.072)
    R = 0.045
    us = [-1 + 2 * i / 6 for i in range(7)]
    vs = [j / 10 for j in range(11)]

    def fn(u, v):
        s = s0 + (s1 - s0) * v
        w = curve(SHOULDER_W, v)
        b = u * w / R
        kick = smoothstep(0.66, 1.0, v)
        # A crest down the middle; the end lifts off the arm and turns up.
        r = R + 0.005 * (1 - abs(u)) ** 1.5 + 0.020 * kick ** 1.6
        d = top * math.cos(b) + across * math.sin(b)
        p = sho + a * s + d * r
        p.z += 0.024 * kick ** 2
        return p

    def paint(u, v):
        return "trim" if abs(u) > 0.7 or v > 0.9 else None

    obj = surf(fn, us, vs, wrap=False, role="body", name="shoulder_cap", paint=paint)
    return [lib.solidify(obj, 0.0045)]


def right_hand():
    """A slender fist round the staff: palm behind it, fingers wrapped round
    the front, the thumb over the top."""
    parts = []
    zax = STAFF_DIR
    xax = V(0.25, 1.0, 0.0)
    xax = (xax - zax * xax.dot(zax)).normalized()
    yax = zax.cross(xax)
    g = GRIP

    def at(theta, rad, h):
        return g + xax * (math.cos(theta) * rad) + yax * (math.sin(theta) * rad) + zax * h

    palm = lib.sphere(0.026, scale=(0.72, 0.95, 1.2), role="skin", segments=12, rings=8, name="palm_r")
    m = Matrix((xax, yax, zax)).transposed().to_4x4()
    m.translation = at(0.0, 0.022, -0.002)
    palm.data.transform(m)
    parts.append(palm)
    for k, h in enumerate((0.018, 0.006, -0.006, -0.017)):
        rad = 0.0195
        end = 4.6 - 0.15 * k
        pts = [at(0.75 + (end - 0.75) * i / 5, rad * (1.0 - 0.06 * (i == 5)), h - 0.002 * i) for i in range(6)]
        r = 0.0078 - 0.0006 * k
        parts.append(strand(pts, [r, r * 1.04, r, r * 0.95, r * 0.85], sides=6, rows=4, role="skin", name="finger_r"))
    thumb = [at(-0.4, 0.025, 0.020), at(-1.1, 0.022, 0.030), at(-1.9, 0.019, 0.030), at(-2.5, 0.017, 0.025)]
    parts.append(strand(thumb, [0.0095, 0.0088, 0.0076, 0.0064], sides=6, rows=5, role="skin", name="thumb_r"))
    return parts


def left_frame():
    up = nrm(L_WRI - L_ELB)
    n = V(0.05, -1.0, 0.25)
    n = (n - up * n.dot(up)).normalized()      # palm faces forward
    across = up.cross(n)                         # toward +X, away from the thumb
    if across.x < 0:
        across = -across
    return up, n, across


def left_hand():
    """Raised high, palm out, the fingers spread and hooked like claws."""
    parts = []
    up, n, across = left_frame()
    c = L_WRI + up * 0.030

    def wedge(v):
        # A flat palm, thickest at the heel and thinning toward the knuckles,
        # where a low ridge runs across the back of the hand.
        t = clamp(v.z / 0.03 * 0.5 + 0.5)
        v.y *= 1.25 - 0.45 * t
        if v.y > 0:
            v.y += 0.0035 * bump(v.z, 0.020, 0.008)
        return v

    palm = lib.sphere(0.027, role="skin", segments=12, rings=8, name="palm_l")
    palm.data.transform(Matrix.Diagonal((1.0, 0.30, 1.15, 1.0)))
    lib.deform(palm, wedge)
    m = Matrix((across, -n, up)).transposed().to_4x4()
    palm.data.transform(m)
    put(palm, Matrix.Translation(c))
    parts.append(palm)
    tips = []

    def bent(b):
        return lambda d: d * math.cos(math.radians(b)) + n * math.sin(math.radians(b))

    # Fingers close together and curled forward joint by joint into hooks:
    # a casting claw, not a wave.
    for k, (dx, ln, fan) in enumerate(((-0.016, 1.0, -0.20), (-0.0055, 1.1, -0.05), (0.0055, 1.05, 0.10), (0.016, 0.86, 0.26))):
        base = c + across * dx + up * 0.022
        d = nrm(up + across * fan)
        segs = [(0.022 * ln, 12), (0.017 * ln, 42), (0.014 * ln, 82)]
        pts = [base]
        for L, b in segs:
            pts.append(pts[-1] + bent(b)(d) * L)
        r = 0.0068 if k in (1, 2) else 0.0062
        parts.append(strand(pts, [r * 1.12, r, r * 0.92, r * 0.74], sides=6, rows=4, role="skin", name="finger_l"))
        # Long nails curling on round from the hooked tips: claws.
        e1, e2 = bent(100)(d), bent(125)(d)
        nail = [pts[-1] - bent(82)(d) * 0.004, pts[-1] + e1 * 0.007, pts[-1] + e1 * 0.009 + e2 * 0.006]
        parts.append(strand(nail, [r * 0.72, r * 0.45, 0.0007], sides=5, rows=3, role="metal", name="nail", point=True))
        tips.append((nail[-1], d))
    thumb = [c - across * 0.018 - up * 0.004, c - across * 0.034 + up * 0.010 + n * 0.008,
             c - across * 0.042 + up * 0.022 + n * 0.020, c - across * 0.040 + up * 0.028 + n * 0.033]
    parts.append(strand(thumb, [0.0085, 0.0078, 0.0066, 0.0050], sides=6, rows=4, role="skin", name="thumb_l"))
    tips.append((thumb[-1], nrm(thumb[-1] - thumb[-2])))
    return parts, tips


def arms():
    parts = []
    parts.append(sleeve(R_SHO, R_ELB, R_WRI, -1, "sleeve_r"))
    parts.append(sleeve(L_SHO, L_ELB, L_WRI, 1, "sleeve_l"))
    parts += cuff(R_ELB, R_WRI, -1, V(-0.4, -0.3, 0.6))
    # Slim dagger sleeves swung back off the arms, clear of the staff.
    parts += hanging_sleeve(L_ELB, L_WRI, 1, V(0.040, 0.070, -0.31), V(1.0, 0.30, 0.0), 0.042, "dagger_sleeve")
    parts += hanging_sleeve(R_ELB, R_WRI, -1, V(-0.040, 0.085, -0.28), V(1.0, -0.30, 0.0), 0.038, "dagger_sleeve")
    up, n, across = left_frame()
    parts += cuff(L_ELB, L_WRI, 1, n)
    for elb, wri, sho in ((L_ELB, L_WRI, L_SHO), (R_ELB, R_WRI, R_SHO)):
        d = nrm(nrm(wri - elb) + nrm(elb - sho))
        m = frame(elb + nrm(wri - elb) * 0.012, d)

        def band(u, w, m=m):
            a, b = u * TAU, w * TAU
            r = 0.0275 + 0.0035 * math.cos(b)
            return m @ V(math.sin(a) * r, -math.cos(a) * r, 0.0045 * math.sin(b))
        parts.append(torus_surf(band, 12, 4, role="trim", name="elbow_band"))
    parts += shoulder_cap(R_SHO, R_ELB, -1)
    parts += shoulder_cap(L_SHO, L_ELB, 1)
    parts += right_hand()
    return parts


# ---------------------------------------------------------------- head

HS = HEAD_R[0] / 0.055   # feature scale


def head_point(u, v):
    phi = u * TAU
    th = v * math.pi
    rx, ry, rz = HEAD_R
    s = math.sin(th)
    sp, cp = math.sin(phi), math.cos(phi)
    xn = s * sp
    z = math.cos(th) * rz
    zn = z / rz
    if cp > 0:
        # The face is a broad, gently rounded plane from brow to mouth, not a
        # wedge: the front of the skull squares off toward a superellipse.
        e = 1.0 / (1.0 + 0.38 * bump(zn, -0.12, 0.62))
        sp = math.copysign(abs(sp) ** e, sp)
        cp = cp ** e
    x = s * sp * rx
    y = -s * cp * ry
    low = smoothstep(0.05, -1.0, zn)
    # Jaw tapers to a pointed chin; the back of the skull stays full.
    x *= 1 - 0.40 * low ** 1.2
    if y < 0:
        y *= 1 - 0.05 * low
        # Chin forward, high cheekbones over hollow cheeks.
        y -= 0.009 * HS * bump(xn, 0, 0.22) * bump(zn, -0.90, 0.16)
        x *= 1 + 0.06 * bump(zn, -0.08, 0.20) * bump(abs(xn), 0.74, 0.28)
        y -= 0.004 * HS * bump(abs(xn), 0.58, 0.14) * bump(zn, -0.12, 0.10)
        y += 0.0045 * HS * bump(abs(xn), 0.52, 0.16) * bump(zn, -0.45, 0.13)
        # Eye sockets under a heavy brow ridge that dips toward the nose.
        for sx in (-1, 1):
            y += 0.0065 * HS * bump(xn, sx * 0.36, 0.18) * bump(zn, 0.04, 0.11)
        y -= 0.005 * HS * bump(xn, 0, 0.55) * bump(zn, 0.20 - 0.10 * bump(xn, 0, 0.25), 0.09)
        # A fine straight nose, its bridge rounded rather than a folded ridge.
        y -= 0.0095 * HS * bump(xn, 0, 0.17) * smoothstep(0.14, -0.22, zn) * smoothstep(-0.47, -0.30, zn)
        y += 0.0025 * HS * bump(abs(xn), 0.19, 0.07) * bump(zn, -0.38, 0.06)
        # A closed mouth: the line between the lips, and the lips either side.
        y -= 0.0016 * HS * bump(xn, 0, 0.2) * bump(zn, -0.515, 0.035)
        y -= 0.0018 * HS * bump(xn, 0, 0.2) * bump(zn, -0.605, 0.035)
    else:
        y *= 1 - 0.35 * low
    return (HEAD.x + x, HEAD.y + y, HEAD.z + z)


def face_point(x, z, out=0.0):
    """The point on the front of the (untilted) face at offsets x, z from the
    head's centre, pushed `out` along -Y."""
    th = math.acos(clamp(z / HEAD_R[2], -0.999, 0.999))
    lo, hi = -math.pi / 2, math.pi / 2
    for _ in range(30):
        mid = (lo + hi) / 2
        if head_point(mid / TAU, th / math.pi)[0] - HEAD.x < x:
            lo = mid
        else:
            hi = mid
    return V(*head_point(((lo + hi) / 2) / TAU, th / math.pi)) + V(0, -out, 0)


def head():
    parts = []
    # Columns crowd the face (u near 0), where the features need them.
    nu = 36
    us = sorted((math.copysign(abs(s) ** 1.30, s) / 2) % 1.0 for s in [-1 + 2 * i / nu for i in range(nu)])
    vs = [0.0, 0.07, 0.14, 0.21, 0.28, 0.34, 0.39, 0.43, 0.47, 0.51, 0.55, 0.59, 0.63, 0.67, 0.71, 0.75, 0.80, 0.86,
          0.93, 1.0]
    parts.append(surf(head_point, us, vs, role="skin", name="head", tip0=True, tip1=True, closed=True))
    k = HS
    for s in (-1, 1):
        # Large glowing almond eyes, tilted up at the outer corners: the one
        # mark on the face that reads from the board.
        ex = s * 0.0215 * k
        e = lib.sphere(0.0115 * k, scale=(1.24, 0.42, 0.42), role="eye", segments=10, rings=5, name="eye")
        put(e, Matrix.Translation(face_point(ex, 0.003 * k, -0.0014)) @ Matrix.Rotation(-s * 0.30, 4, "Y")
            @ Matrix.Rotation(s * 0.35, 4, "Z"))
        parts.append(e)
        # Winged kohl sweeping up and out from the outer corner.
        pts = [face_point(s * 0.0335 * k, 0.0052 * k, 0.0004), face_point(s * 0.0420 * k, 0.0100 * k, 0.0006),
               face_point(s * 0.0490 * k, 0.0160 * k, 0.0004)]
        parts.append(strand(pts, [0.0032, 0.0024, 0.0007], sides=4, rows=3, role="trim", name="kohl", point=True,
                            flat=0.5, up=(0, -1, 0)))
        # Straight, heavy brows rising to a high arch: arrogant, not a frown.
        pts = [face_point(s * 0.007 * k, 0.0150 * k, 0.0012), face_point(s * 0.022 * k, 0.0215 * k, 0.0014),
               face_point(s * 0.038 * k, 0.0245 * k, 0.0010)]
        parts.append(strand(pts, [0.0048, 0.0045, 0.0014], sides=5, rows=4, role="trim", name="brow", point=True,
                            flat=0.55, up=(0, -1, 0)))
    # Dark closed lips, level and cold, one corner lifted into a smirk.
    lips = [face_point(-0.0135 * k, -0.0400 * k), face_point(-0.006 * k, -0.0390 * k, 0.0004),
            face_point(0.004 * k, -0.0387 * k, 0.0004), face_point(0.0110 * k, -0.0384 * k),
            face_point(0.0150 * k, -0.0366 * k, -0.0004)]
    parts.append(strand(lips, [0.0010, 0.0024, 0.0024, 0.0016, 0.0007], sides=5, rows=6, role="trim", name="lips",
                        flat=0.6, up=(0, 0, 1)))
    neck = strand([NECK0, NECK0.lerp(NECK1, 0.5), NECK1], [0.027, 0.025, 0.024], sides=12, rows=5, role="skin", name="neck")
    parts.append(neck)
    return parts


# ---------------------------------------------------------------- hair


def wrap_pi(a):
    a = a % TAU
    return a - TAU if a > math.pi else a


def cap_xyz(phi, th, off=0.0, locks=True):
    """A point on the hair over the skull: full at the sides, swept back from
    a centre parting in shallow locks."""
    rx, ry, rz = HEAD_R
    s = math.sin(th)
    ux, uy, uz = s * math.sin(phi), -s * math.cos(phi), math.cos(th)
    k = 1.12 + 0.10 * smoothstep(0.6, 1.6, th) * abs(math.sin(phi)) ** 2
    kz = 1.09
    g = 1.0
    if locks:
        # Long shallow strands combed from a deep centre parting back and
        # down toward the nape, each strand a gentle swell.
        t = math.atan2(-uy, uz)
        side = abs(ux)
        w = abs(math.sin(4.0 * (t + 1.6 * side) + 0.3)) ** 0.7 - 0.62
        g += 0.05 * w * smoothstep(0.04, 0.24, side)
        g -= 0.08 * bump(ux, 0, 0.05) * smoothstep(0.1, 0.5, uz)
    x = ux * (rx * k * g + off)
    y = uy * (ry * k * g + off)
    z = uz * (rz * kz * g + off)
    return HEAD + V(x, y + 0.002, z)


def cap_th_max(phi):
    c = math.cos(phi)
    front = max(0.0, c)
    back = max(0.0, -c)
    return 1.72 - 0.80 * front ** 1.5 + 0.55 * back ** 2 + 0.10 * bump(wrap_pi(phi), 0, 0.12)


def hair_cap():
    nu = 36
    us = [i / nu for i in range(nu)]
    vs = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.90, 1.0, 1.04]

    def fn(u, v):
        phi = u * TAU
        if v > 1.0:
            # The edge rolls under, onto the scalp, so the hair has thickness.
            return cap_xyz(phi, cap_th_max(phi) + 0.02, -0.010, False)
        return cap_xyz(phi, v * cap_th_max(phi))

    return surf(fn, us, vs, role="cloth", name="hair_cap", tip0=True)


def back_clear(x, z):
    """How far back the hair must hang at (x, z) to clear her back and the
    low roll of the collar."""
    if z > 0.6:
        r, cy = bd_radius(math.pi, z)
        rx = curve(BD, z)[0] + 0.012
    else:
        r, cy = skirt_r_at(math.pi, z), 0.0
        rx = r + 0.03
    y = cy + r * math.sqrt(max(0.05, 1 - (x / rx) ** 2))
    if z > 0.90:
        y = max(y, 0.118 * math.sqrt(max(0.05, 1 - (x / 0.11) ** 2)))
    return y


HAIR_CLUMPS = [  # x at the nape, lowest z, width, sway phase, depth offset
    (0.000, 0.580, 0.022, 0.0, 0.004),
    (-0.034, 0.668, 0.020, 1.4, 0.002),
    (0.034, 0.622, 0.020, 2.3, 0.002),
    (-0.064, 0.760, 0.017, 0.7, 0.0),
    (0.064, 0.712, 0.017, 2.9, 0.0),
]


def hair_back():
    """The long hair down her back: one broad sheet at the nape that parts
    lower down into flat tapering locks of uneven length, swaying."""
    parts = []
    for x0, zend, w, ph, dy in HAIR_CLUMPS:
        d = math.asin(clamp(0.8 * x0 / 0.075, -0.95, 0.95))
        pts = [tilted(cap_xyz(math.pi - d * 0.8, 1.25, -0.010, False)),
               tilted(cap_xyz(math.pi - d, 1.85, -0.006, False))]
        y_prev = pts[-1].y
        zs = [0.955, 0.905, 0.85, 0.79, 0.73, 0.67]
        zs = [z for z in zs if z > zend + 0.04] + [zend]
        for i, z in enumerate(zs):
            t = (0.955 - z) / (0.955 - zend)
            # Together down to about z 0.85, then each lock goes its own way.
            sep = smoothstep(0.86, 0.70, z)
            x = x0 * (0.92 + 0.30 * sep) + 0.012 * math.sin(t * 5.0 + ph) * sep
            y = max(back_clear(x, z) + w * 0.30 + 0.004 + dy, y_prev - 0.010)
            # The ends kick out a little.
            y += 0.010 * t * t
            y_prev = y
            pts.append(V(x, y, z))
        n = len(pts)
        # Broad and overlapping at the nape, narrowing below the split and
        # drawn to a sharp point.
        body = [w * (1.32 - 0.45 * ((i + 1) / (n - 2)) ** 0.9) for i in range(n - 4)]
        radii = [w * 1.15, w * 1.32] + body + [w * 0.45, 0.0008]
        parts.append(strand(pts, radii, sides=8, rows=9, role="cloth", name="hair_back", point=True, flat=0.25,
                            up=(1, 0, 0)))
    return parts


def hair_locks():
    """Two thin locks a side. They part from the hair behind the ear, pass
    behind the jaw and inside the collar, and first show at the collarbone,
    falling in an S over the bodice: well clear of the cheeks."""
    parts = []
    for s in (-1, 1):
        pts = [tilted(cap_xyz(s * 2.00, 1.45, -0.008, False)), tilted(HEAD + V(s * 0.078, 0.034, -0.052)),
               V(s * 0.071, -0.010, 0.976), V(s * 0.074, -0.046, 0.950)]
        for z, x in ((0.915, 0.080), (0.880, 0.088), (0.845, 0.081), (0.812, 0.073), (0.782, 0.079)):
            y, _ = bodice_y(s * x, z)
            pts.append(V(s * x, y - 0.009, z))
        parts.append(strand(pts, [0.008, 0.010, 0.011, 0.011, 0.010, 0.0095, 0.008, 0.006, 0.0015], sides=6, rows=10,
                            role="cloth", name="lock_front", point=True, flat=0.6, up=(1, 0, 0)))
        pts = [tilted(cap_xyz(s * 2.25, 1.55, -0.008, False)), tilted(HEAD + V(s * 0.082, 0.046, -0.062)),
               V(s * 0.078, 0.002, 0.966), V(s * 0.090, -0.038, 0.938)]
        for z, x in ((0.902, 0.101), (0.870, 0.095), (0.842, 0.102)):
            y, _ = bodice_y(s * x, z)
            pts.append(V(s * x, y - 0.008, z))
        parts.append(strand(pts, [0.007, 0.0095, 0.0095, 0.009, 0.007, 0.0045, 0.0015], sides=6, rows=8,
                            role="cloth", name="lock_front", point=True, flat=0.6, up=(1, 0, 0)))
    return parts


# ---------------------------------------------------------------- crown


_CR = [(0.068, 0.016, 1.402), (0.120, 0.022, 1.356), (0.150, 0.030, 1.276), (0.147, 0.040, 1.192),
       (0.114, 0.056, 1.132), (0.062, 0.078, 1.106), (0.0, 0.088, 1.100)]
# The crescent: one arc round the back of the head whose horns sweep up past
# the temples and curl in over the crown.
CRESCENT = [V(-x, y, z) for x, y, z in _CR] + [V(x, y, z) for x, y, z in reversed(_CR[:-1])]


CROWN_ORB = V(0.0, 0.064, 1.222)


def circlet_th(phi):
    # At the hairline in front, well clear of the brows; low at the back, where
    # it meets the crescent.
    return 0.78 + 0.22 * (1 - math.cos(phi)) / 2


def crown():
    parts = []
    # Circlet: a band lying on the hair, low on the brow.
    nu, nw = 26, 5
    off = 0.0055

    def centre(phi):
        return cap_xyz(phi, circlet_th(phi), off, False)

    def fn(u, w):
        phi = u * TAU
        c = centre(phi)
        t = (centre(phi + 0.01) - centre(phi - 0.01)).normalized()
        rel = c - HEAD
        rx, ry, rz = HEAD_R
        n = V(rel.x / rx ** 2, rel.y / ry ** 2, rel.z / rz ** 2).normalized()
        b = n.cross(t).normalized()
        if b.z < 0:
            b = -b
        ang = w * TAU
        return c + n * (0.0032 * math.cos(ang) + 0.0012) + b * (0.0062 * math.sin(ang))

    parts.append(torus_surf(fn, nu, nw, role="trim", name="circlet"))
    front = centre(0.0)
    # A gem on the band over the parting: nothing hangs over the eyes.
    gem = lib.sphere(0.0062, scale=(1, 0.6, 1.35), role="eye", segments=8, rings=6, name="brow_gem")
    parts.append(orient(gem, front + V(0, -0.0050, 0.001), V(0, -1, 0.55), (0, 0, 1)))
    # The crescent: one arc round the back of the head whose horns sweep up
    # past the temples and curl in over the crown.
    path = Path(CRESCENT)
    vs = [i / 34 for i in range(35)]

    def radius(v, a):
        t = abs(v - 0.5) * 2          # 0 at the back, 1 at the tips
        w = 0.028 * (1 - t ** 2.4) ** 0.62 + 0.0008
        d = 0.0105 * (1 - t ** 3) ** 0.5 + 0.0008
        # A blade: thin front to back, its edges drawn to a ridge, with a
        # fuller engraved down the face of each horn.
        d *= 0.35 + 0.65 * abs(math.cos(a)) ** 0.5
        if math.cos(a) < -0.99:
            d *= 1 - 0.5 * smoothstep(0.22, 0.34, t) * (1 - smoothstep(0.80, 0.90, t))
        return (d, w)

    horn = sweep(CRESCENT, vs, 10, radius, role="trim", name="crescent", up=(0, 1, 0), tip0=True, tip1=True)
    parts.append(horn)
    # A gem at the root of each horn, where it rises past the temple.
    for v in (0.24, 0.76):
        p = path.at(v)
        t = path.tangent(v)
        nf = V(0, -1, 0) - t * t.y
        parts.append(orient(lib.sphere(0.0085, scale=(0.8, 0.45, 1.2), role="eye", segments=8, rings=6, name="crown_gem"),
                            p + nf.normalized() * 0.0085, nf, t if t.z > 0 else -t))
    # A moon-orb held in the hollow of the crescent, behind and above her head,
    # on a fine stem rising from the back of the arc: the crown echoes the staff.
    orb_c = CROWN_ORB
    parts.append(lib.sphere(0.030, loc=orb_c, role="eye", segments=12, rings=8, name="crown_orb"))
    root = path.at(0.5)
    parts.append(cone_between(root + V(0, -0.002, 0.004), orb_c + V(0, 0.004, -0.026), 0.0042, 0.0026, role="metal",
                              sides=5, name="orb_stem"))
    return parts


# ---------------------------------------------------------------- jewels


def small_crescent(R, role="trim", depth=0.003, n=(10, 8), name="crescent_charm"):
    return lib.extrude(crescent_outline(R, R * 0.8, R * 0.42, *n), depth, role=role, bevel=0.0, name=name)


def skirt_u(A, v):
    """The skirt parameter u at which the gown (not its front panel) lies at
    angle A, 0 < |A| <= pi."""
    sp, st = SK_NP / SK_NU, SK_NT / SK_NU
    g = sk_gap(v)
    m = sp + st + (abs(A) - g - SK_TW) / (math.pi - g - SK_TW) * (0.5 - sp - st)
    return m if A >= 0 else 1.0 - m


def train_embroidery():
    """A row of crescents embroidered round the train just above the hem band,
    each set in the hollow of a fold."""
    parts = []
    v = 0.905
    for A0 in (2.05, 2.62, math.pi, -2.62, -2.05):
        # Settle into the nearest trough between folds.
        best = min((A0 + 0.02 * i for i in range(-14, 15)),
                   key=lambda a: Vector(skirt_point(skirt_u(a if abs(a) <= math.pi else math.copysign(math.pi, a), v), v)).xy.length)
        A = best if abs(best) <= math.pi else math.copysign(math.pi, best)
        u = skirt_u(A, v)
        p = Vector(skirt_point(u, v))
        tu = Vector(skirt_point((u + 0.003) % 1.0, v)) - Vector(skirt_point((u - 0.003) % 1.0, v))
        tv = Vector(skirt_point(u, v - 0.006)) - Vector(skirt_point(u, v + 0.006))   # up the skirt
        nml = tu.cross(tv).normalized()
        if nml.dot(V(p.x, p.y, 0)) < 0:
            nml = -nml
        c = small_crescent(0.021, depth=0.004, n=(10, 7), name="embroidery")
        parts.append(orient(c, p + nml * 0.0006, nml, tv.normalized()))
    return parts


def jewels():
    parts = []
    # A choker at the throat with a crescent pendant hanging into the neckline.
    zc = 0.958

    def choker(u, w):
        a = u * TAU
        ang = w * TAU
        r = 0.0262 + 0.0026 * math.cos(ang)
        c = NECK0.lerp(NECK1, (zc - NECK0.z) / (NECK1.z - NECK0.z))
        return (c.x + math.sin(a) * r, c.y - math.cos(a) * r, zc + 0.0038 * math.sin(ang))

    parts.append(torus_surf(choker, 18, 5, role="trim", name="choker"))
    zp = 0.922
    yp = bd_radius(0.0, zp, "skin")[0] * -1 + curve(BD, zp)[2] - 0.0035
    pend = small_crescent(0.016, depth=0.004, name="pendant")
    put(pend, Matrix.Translation(V(0, yp, zp)) @ Matrix.Rotation(-0.12, 4, "X") @ Matrix.Rotation(math.pi, 4, "Y"))
    parts.append(pend)
    parts.append(lib.sphere(0.0058, loc=V(0, yp - 0.003, zp + 0.003), scale=(1, 0.7, 1), role="eye", segments=8, rings=5,
                            name="pendant_gem"))
    top = NECK0.lerp(NECK1, (zc - NECK0.z) / (NECK1.z - NECK0.z)) + V(0, -0.027, -0.004)
    parts.append(cone_between(top, V(0, yp - 0.001, zp + 0.012), 0.0014, 0.0014, role="metal", sides=4, name="pendant_link"))
    parts += train_embroidery()
    # One jewel on the stomacher (a row of them reads as jacket buttons).
    for z, s in ((0.800, 1.15),):
        y = -bd_radius(0.0, z, "skin")[0] + curve(BD, z)[2]
        parts.append(orient(lib.extrude([(0.0, 0.015), (0.010, 0.0), (0.0, -0.017), (-0.010, 0.0)], 0.004, role="cloth",
                                        bevel=0.0012, name="stomacher_setting", scale=(s, 1, s)), V(0, y - 0.001, z),
                            V(0, -1, 0.05), (0, 0, 1)))
        parts.append(orient(lib.sphere(0.0058 * s, scale=(1, 0.7, 1.3), role="metal", segments=8, rings=5, name="stomacher_gem"),
                            V(0, y - 0.0035, z), V(0, -1, 0.05), (0, 0, 1)))
    # A gem set in the point of the basque.
    zb = bd_bottom(0.0) + 0.012
    yb = -(skirt_r_at(0.0, zb) + 0.012)
    parts.append(orient(lib.extrude([(0.0, 0.016), (0.012, 0.0), (0.0, -0.020), (-0.012, 0.0)], 0.005, role="trim", bevel=0.0,
                                    name="basque_setting"), V(0, yb, zb), V(0, -1, -0.15), (0, 0, 1)))
    parts.append(orient(lib.sphere(0.0062, scale=(1, 0.6, 1.4), role="eye", segments=8, rings=5, name="basque_gem"),
                        V(0, yb - 0.003, zb), V(0, -1, -0.15), (0, 0, 1)))
    return parts


# ---------------------------------------------------------------- staff


def crescent_outline(R, r, d, n_out=26, n_in=20):
    yi = (R * R - r * r + d * d) / (2 * d)
    xi = math.sqrt(R * R - yi * yi)
    to = math.atan2(yi, xi)
    ti = math.atan2(yi - d, xi)
    pts = []
    for k in range(n_out + 1):
        t = to - (math.pi + 2 * to) * k / n_out
        pts.append((R * math.cos(t), R * math.sin(t)))
    for k in range(1, n_in):
        t = (math.pi - ti) + (math.pi + 2 * ti) * k / n_in
        pts.append((r * math.cos(t), d + r * math.sin(t)))
    return pts


def staff():
    parts = []
    rot = STAFF_DIR.to_track_quat("Z", "Y").to_euler()
    top_z = 1.262
    top = staff_at(top_z)
    L = (top - STAFF_FOOT).length
    shaft = lib.cylinder(0.0105, 0.0095, L, loc=STAFF_FOOT.lerp(top, 0.5), rot=rot, role="metal", segments=12, name="staff")
    parts.append(shaft)
    parts.append(lib.cylinder(0.016, 0.012, 0.05, loc=staff_at(0.025), rot=rot, role="metal", segments=12, bevel=0.002, name="ferrule"))
    # Grip binding and rings.
    parts.append(lib.cylinder(0.0128, 0.0128, 0.11, loc=staff_at(0.835), rot=rot, role="cloth", segments=12, name="grip"))
    for z in (0.775, 0.895, 0.40, 1.05):
        parts.append(lib.cylinder(0.0148, 0.0148, 0.011, loc=staff_at(z), rot=rot, role="trim", segments=12, name="ring"))
    # Head: a flared socket, a collar and the crescent cradling the orb.
    parts.append(lib.lathe([(0.0, 1.21), (0.011, 1.21), (0.013, 1.235), (0.019, 1.262), (0.024, 1.275), (0.0, 1.278)],
                           loc=(0, 0, 0), role="metal", segments=14, name="socket"))
    for p in parts[-1:]:
        lib.place(p, (0, 0, -1.21))
        put(p, frame(staff_at(1.21), STAFF_DIR))
    cres = lib.extrude(crescent_outline(0.074, 0.060, 0.030, 16, 11), 0.020, role="trim", bevel=0.0035, name="staff_crescent")

    def lens(v):
        v.y *= 0.65 + 0.35 * smoothstep(0.075, 0.03, math.hypot(v.x, v.z - 0.0))
        return v
    lib.deform(cres, lens)
    put(cres, Matrix.Translation(STAFF_TOP))
    parts.append(cres)
    # Four thorns curling out and up from the socket.
    side = nrm(STAFF_DIR.cross(V(0, 1, 0)))
    fwd = nrm(STAFF_DIR.cross(side))
    base = staff_at(1.25)
    for k in range(4):
        a = k * TAU / 4 + math.pi / 4
        o = side * math.cos(a) + fwd * math.sin(a)
        pts = [base + o * 0.012, base + o * 0.026 + STAFF_DIR * 0.004, base + o * 0.036 + STAFF_DIR * 0.020,
               base + o * 0.034 + STAFF_DIR * 0.036]
        parts.append(strand(pts, [0.0045, 0.0038, 0.0024, 0.0006], sides=4, rows=4, role="metal", name="thorn", point=True))
    parts.append(lib.sphere(0.026, loc=STAFF_TOP + V(0, 0, 0.004), role="eye", segments=12, rings=8, name="orb"))
    return parts


# ---------------------------------------------------------------- lightning


def bolt(start, direction, length, segs, r0, r1, rng, jag=0.45, name="bolt", side=None):
    """A jagged bolt: uneven segments kinking either side of a straight line,
    mostly in the plane of `side` (default: any)."""
    d = nrm(direction)
    p1 = nrm(Vector(side) - d * Vector(side).dot(d)) if side is not None else d.orthogonal().normalized()
    p2 = d.cross(p1)
    pts = [Vector(start)]
    sign = 1
    cuts = sorted(rng.uniform(0.0, 1.0) for _ in range(segs - 1))
    ts = [(i + 0.5 * (c - 0.5)) / segs for i, c in zip(range(1, segs), cuts)] + [1.0]
    for i, t in enumerate(ts, start=1):
        base = Vector(start) + d * length * t
        amp = length / segs * jag
        off = (p1 * (sign * amp * rng.uniform(0.7, 1.25)) + p2 * (amp * rng.uniform(-0.35, 0.35))) if i < segs else Vector()
        sign = -sign
        pts.append(base + off)
    radii = [r0 + (r1 - r0) * i / segs for i in range(segs + 1)]
    return [jag_tube(pts, radii, sides=5, name=name)], pts


TOP_Z = 1.442     # nothing may rise above this (commander tier tops out at 1.45)
MAX_X = 0.445


def fit_length(start, d, L, margin=0.008):
    """Shortens a bolt so it stays inside the figure's allowed bounds."""
    end = start + d * L
    if end.z > TOP_Z - margin and d.z > 1e-6:
        L = min(L, (TOP_Z - margin - start.z) / d.z)
    if end.x > MAX_X - margin and d.x > 1e-6:
        L = min(L, (MAX_X - margin - start.x) / d.x)
    return max(L, 0.03)


def jag_path(pts, step, amp, rng):
    """Breaks a smooth path into a lightning zigzag of roughly `step` long
    segments, swinging `amp` either side."""
    path = Path(pts)
    segs = max(3, int(path.length / step))
    out = [path.at(0.0)]
    sign = 1
    for i in range(1, segs):
        t = i / segs
        c = path.at(t)
        T = path.tangent(t)
        side = T.cross(V(0, 0, 1))
        if side.length < 1e-3:
            side = T.cross(V(1, 0, 0))
        side.normalize()
        lift = T.cross(side).normalized()
        out.append(c + side * (sign * amp * rng.uniform(0.6, 1.2)) + lift * (amp * rng.uniform(-0.5, 0.5)))
        sign = -sign
    out.append(path.at(1.0))
    return out


def board_xy(p, yaw):
    """Where a point lands in the board camera's picture (40 degrees above),
    with the figure turned by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = p.x * c - p.y * s, p.x * s + p.y * c
    return (x, p.z * math.cos(math.radians(40)) + y * math.sin(math.radians(40)))


def horn_gap(bolts):
    """The smallest gap in the board picture, at either yaw, between any bolt
    and the crescent crown's blade."""
    path = Path(CRESCENT)
    blade = []
    for i in range(161):
        t = i / 160
        w = 0.028 * max(0.0, 1 - abs(2 * t - 1) ** 2.4) ** 0.62
        blade.append((tilted(path.at(t)), w))
    best = 9.0
    for yaw in (0.5, -0.5):
        cres = [(board_xy(p, yaw), w) for p, w in blade]
        for pts in bolts:
            for a, b in zip(pts, pts[1:]):
                for j in range(5):
                    q = board_xy(a.lerp(b, j / 4), yaw)
                    for (cx, cy), w in cres:
                        best = min(best, math.hypot(q[0] - cx, q[1] - cy) - w)
    return best


GAP = {}


def lightning(tips):
    """Crackling lightning thrown from the claws out and away from the crown:
    one long bolt forward and out, one down and out, one out along her side,
    a short spark upward; and a long forked arc leaping from the thumb behind
    her horns to the staff."""
    rng = random.Random(5)
    parts = []
    up, n, across = left_frame()
    fans = [  # direction in world (x, y, z), length, segments, fork lean
        ((0.40, -0.05, 1.00), 0.075, 3, None),
        ((0.62, -0.18, 0.78), 0.200, 5, (0.1, -0.3, 0.8)),
        ((0.78, -0.62, 0.02), 0.160, 4, None),
        ((0.50, -0.32, -0.80), 0.220, 5, (0.7, -0.2, 0.0)),
    ]
    fan_pts = []
    for (tip, d), ((wx, wy, wz), L, segs, fork) in zip(tips[:4], fans):
        dd = nrm(V(wx, wy, wz))
        start = tip + dd * 0.002
        L = fit_length(start, dd, L)
        bp, pts = bolt(start, dd, L, segs, 0.0070, 0.0024, rng, jag=0.48, side=(0, 0, 1) if abs(dd.z) < 0.7 else (1, 0, 0))
        parts += bp
        fan_pts.append(pts)
        if fork:
            mid = pts[segs // 2]
            fd = nrm(dd + V(*fork))
            fp, fpts = bolt(mid, fd, fit_length(mid, fd, L * 0.42), 3, 0.0042, 0.0016, rng, jag=0.55)
            parts += fp
            fan_pts.append(fpts)
    GAP["horn"] = round(horn_gap(fan_pts), 3)
    # The long arc: from the thumb up behind the horn tips, across the back of
    # the crown and down onto the staff's crescent.
    thumb, _ = tips[4]
    orb = STAFF_TOP + V(0.0, 0.0, 0.032)
    guide = [thumb, thumb + V(-0.03, 0.05, 0.065), V(0.15, 0.125, 1.382), V(0.02, 0.150, 1.398),
             V(-0.13, 0.130, 1.396), V(-0.235, 0.040, 1.392), orb]
    pts = jag_path(guide, 0.042, 0.010, rng)
    pts = [V(p.x, p.y, min(p.z, TOP_Z - 0.006)) for p in pts]
    n_ = len(pts)
    radii = [0.0048 + 0.0012 * math.sin(math.pi * i / (n_ - 1)) for i in range(n_)]
    parts.append(jag_tube(pts, radii, sides=5, name="arc"))
    # It forks as it nears the staff.
    for i0, lean in ((n_ - 4, V(0.0, -0.4, -0.6)),):
        p0 = pts[i0]
        dd = nrm((pts[i0 + 1] - p0).normalized() + lean)
        fp, _ = bolt(p0, dd, fit_length(p0, dd, 0.075), 3, 0.0038, 0.0014, rng, jag=0.6)
        parts += fp
    return parts


# ---------------------------------------------------------------- build


def hip_sway(p):
    w = smoothstep(0.26, 0.56, p.z) * (1 - smoothstep(0.66, 0.84, p.z))
    p.x -= 0.021 * w
    return p


def build():
    parts = []

    def add(objs):
        for o in objs if isinstance(objs, list) else [objs]:
            key = o.name.split(".")[0]
            TRIS[key] = TRIS.get(key, 0) + lib.tris(o)
            parts.append(o)

    add(skirt())
    add(bodice())
    add(collar())
    add(arms())
    lh, tips = left_hand()
    add(lh)
    hd = head()
    neck = hd.pop()
    add([put(o, TILT) for o in hd])
    add(neck)
    add(put(hair_cap(), TILT))
    add(hair_back())
    add(hair_locks())
    add(jewels())
    add([put(o, TILT) for o in crown()])
    add(lightning(tips))
    # Her weight sits on her right leg: the hips swing toward the staff.
    for o in parts:
        lib.deform(o, hip_sway)
    add(staff())
    print("REPORT parts " + json.dumps(dict(sorted(TRIS.items(), key=lambda kv: -kv[1]))) + f" total {sum(TRIS.values())}"
          + f" horn_gap {GAP.get('horn')}")
    return parts
