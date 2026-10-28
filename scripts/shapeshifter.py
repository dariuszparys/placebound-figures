"""Shapeshifter: a Dark flyer that copies its opponent in each battle. A
changeling caught mid-transformation: a hovering column of shifting matter
that twists upward from curling tendrils to a pale, blank mask, wider in the
middle where the parts of other pieces on the board are pushing out of it,
none of them finished, each pouring out of a swelling in the mass, its root
torn open in a burning seam:

- low on its right, the Knight's sword: a cone of matter pouring out of the
  column whose forearm is hardening into a plated vambrace (the steel's
  ragged edge still burning) and which, where the hand should be, has
  become a crossguard and a long blade;
- high on its left, the Dragon's bat wing: finger bones fanning almost
  straight out of the mass from a stub of matter and a wrist claw, the
  membrane grown between them with a burning, scalloped trailing edge, the
  innermost panel running from the last finger down to the flank of the
  column, still tearing;
- out of its back, the Manticore's scorpion tail: a soft unbanded root, then
  ringed segments arching up over the right of the head to a venom bulb
  seated in the last ring and a stinger aimed forward;
- low on its left, the Unicorn's horn: three ridges twisting to the point,
  the grooves and the point glowing, set in its ring, the matter lapping
  over the ring.

The column itself will not hold still: it is a barley twist of unequal
plies, three strands of unequal size lying in its grooves (one forking and
rejoining, one ending in a melting stump), bubbles boiling up through it.
Below, the strands leave it as tapering tendrils that curl at different
heights round a central one, all hanging clear of the plinth. Above, its
face is a symmetric, expressionless mask with five glowing eyes, turned up
to the board camera so its brow is the top of the figure; the column ends
under it, and from behind the mask is a dome of matter that runs down the
back in two drips. Its lower right is swelling into a beast's snout,
nostrils and a split lip already lit, the skin running off the chin in
drips that turn back into matter.

Ember trim marks what identifies each borrowed part (the blade's fuller and
crossguard, the wing's trailing edge and claws, the tail's rings and sting,
the horn's spiral), the torn seams where each erupts, the rim of the mask
and the burning ends of the tendrils. The figure faces -Y; its right hand is
on -X, its left on +X."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

TAU = math.tau
DEBUG = True
# Organic surfaces keep smooth normals; only real corners (rims, blades) split.
FINISH = {"sharp_angle": 60}
# The whole figure is drawn at this scale about its middle, so it holds its
# own beside the other flyers on the board.
SCALE, PIVOT = 1.06, (0.0, 0.0, 0.68)
# and lifted, so its lowest tendrils hang clear above the plinth on the board.
LIFT = 0.07


def note(name, obj):
    if DEBUG:
        print(f"SHIFTER {name}: {lib.tris(obj)} tris")
    return obj


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def wrap(a):
    """An angle folded into -pi..pi."""
    return (a + math.pi) % TAU - math.pi


def interp(table, t):
    """Smooth (Hermite, Catmull-Rom slopes) interpolation through (t, value) pairs."""
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


def catmull(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def dense_curve(ctrl):
    pts = [Vector(p) for p in ctrl]
    ext = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    dense = []
    for i in range(1, len(ext) - 2):
        for k in range(24):
            dense.append(catmull(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], k / 24))
    dense.append(pts[-1])
    acc = [0.0]
    for a, b in zip(dense, dense[1:]):
        acc.append(acc[-1] + (b - a).length)
    return dense, acc


def curve_at(ctrl, shares):
    """Points on the Catmull-Rom curve through `ctrl` at the given arc length shares."""
    dense, acc = dense_curve(ctrl)
    out, j = [], 0
    for sh in shares:
        s = acc[-1] * clamp(sh)
        while j < len(acc) - 2 and acc[j + 1] < s:
            j += 1
        while j > 0 and acc[j] > s:
            j -= 1
        f = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-9)
        out.append(dense[j].lerp(dense[j + 1], f))
    return out


def curve(ctrl, samples):
    """Catmull-Rom curve through the control points, resampled evenly by arc length."""
    return curve_at(ctrl, [i / (samples - 1) for i in range(samples)])


def polylen(pts):
    return sum((Vector(b) - Vector(a)).length for a, b in zip(pts, pts[1:]))


def frame(origin, z, front=(0, -1, 0)):
    """A matrix whose local z points along `z` and whose local -y leans toward
    `front`, placed at `origin`."""
    z = nrm(z)
    f = Vector(front)
    f = f - f.dot(z) * z
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


def mesh(name, verts, faces, face_roles):
    """A mesh object whose faces take the given roles, normals pointing out."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    if isinstance(face_roles, str):
        face_roles = [face_roles] * len(faces)
    roles = []
    for r in face_roles:
        if r not in roles:
            roles.append(r)
    for r in roles:
        me.materials.append(lib.mat(r))
    me.polygons.foreach_set("material_index", [roles.index(r) for r in face_roles])
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def frames(P, ref=(0, -1, 0)):
    """Tangents and parallel-transported normals and binormals along points."""
    n = len(P)
    T = [nrm(P[min(i + 1, n - 1)] - P[max(i - 1, 0)]) for i in range(n)]
    r0 = Vector(ref)
    nn = r0 - r0.dot(T[0]) * T[0]
    if nn.length < 1e-6:
        nn = Vector((1, 0, 0)) - T[0].x * T[0]
    nn.normalize()
    N = []
    for i in range(n):
        nn = nn - nn.dot(T[i]) * T[i]
        nn.normalize()
        N.append(nn.copy())
    B = [T[i].cross(N[i]) for i in range(n)]
    return T, N, B


class Sweep:
    """A tube swept along a curve. `rad(t, a)` gives the radius at arc length
    share t and angle a round the tube (a = 0 along the reference normal), or
    an (n, b) offset in the ring's plane; `paint(t, a)` may give a face another
    role. `shares` places the rings at chosen arc length shares (denser where
    the radius changes fast); t then is the ring's share. The frames stay
    available for placing details along the tube."""

    def __init__(self, ctrl, rings, sides, rad, role="body", paint=None, ref=(0, -1, 0), caps=(0.6, 0.6),
                 name="sweep", shares=None):
        if shares is None:
            shares = [i / (rings - 1) for i in range(rings)]
        P = curve_at(ctrl, shares)
        n = len(P)
        T, N, B = frames(P, ref)
        self.P, self.T, self.N, self.B, self.rad, self.S = P, T, N, B, rad, list(shares)
        verts, faces, roles = [], [], []
        for i in range(n):
            for j in range(sides):
                verts.append(self.at(self.S[i], TAU * j / sides, i))
        for i in range(n - 1):
            tm = (self.S[i] + self.S[i + 1]) / 2
            for j in range(sides):
                jn = (j + 1) % sides
                faces.append((i * sides + j, i * sides + jn, (i + 1) * sides + jn, (i + 1) * sides + j))
                r = paint(tm, TAU * (j + 0.5) / sides) if paint else None
                roles.append(r or role)
        for end, (i, sgn) in enumerate(((0, -1), (n - 1, 1))):
            k = caps[end]
            if k is None:
                continue
            ring = [i * sides + j for j in range(sides)]
            ravg = sum((verts[v] - P[i]).length for v in ring) / sides
            verts.append(P[i] + T[i] * sgn * ravg * k)
            pole = len(verts) - 1
            for j in range(sides):
                jn = (j + 1) % sides
                faces.append((ring[j], ring[jn], pole))
                r = paint(float(end), TAU * (j + 0.5) / sides) if paint else None
                roles.append(r or role)
        self.obj = mesh(name, verts, faces, roles)

    def index(self, t):
        """The ring segment holding share t and the fraction along it."""
        S = self.S
        t = clamp(t)
        i = 0
        while i < len(S) - 2 and S[i + 1] < t:
            i += 1
        return i, clamp((t - S[i]) / max(S[i + 1] - S[i], 1e-9))

    def at(self, t, a, i=None, out=0.0):
        """The surface point at (t, a), pushed `out` along the radius."""
        if i is None:
            i0, u = self.index(t)
            p = self.P[i0].lerp(self.P[i0 + 1], u)
            N = self.N[i0].lerp(self.N[i0 + 1], u).normalized()
            B = self.B[i0].lerp(self.B[i0 + 1], u).normalized()
        else:
            p, N, B = self.P[i], self.N[i], self.B[i]
        r = self.rad(t, a)
        if isinstance(r, tuple):
            d = N * r[0] + B * r[1]
            return p + d + nrm(d) * out
        d = N * math.cos(a) + B * math.sin(a)
        return p + d * (r + out)

    def centre(self, t):
        i0, u = self.index(t)
        return self.P[i0].lerp(self.P[i0 + 1], u)

    def tangent(self, t):
        i0, u = self.index(t)
        return self.T[i0].lerp(self.T[i0 + 1], u).normalized()

    def angle_of(self, t, d):
        """The angle round the tube at share t that faces world direction d."""
        i0, u = self.index(t)
        N = self.N[i0].lerp(self.N[i0 + 1], u).normalized()
        B = self.B[i0].lerp(self.B[i0 + 1], u).normalized()
        return math.atan2(Vector(d).dot(B), Vector(d).dot(N))

    def normal(self, t, a, eps=0.004):
        p = self.at(t, a)
        pt = self.at(clamp(t + eps), a) - self.at(clamp(t - eps), a)
        pa = self.at(t, a + eps * 4) - self.at(t, a - eps * 4)
        n = pa.cross(pt)
        radial = p - self.at(t, a, out=-0.001)
        return n.normalized() if n.dot(radial) > 0 else -n.normalized()


def ring_tube(pts, rad, sides=6, role="trim", up=None, name="ring", closed=True):
    """A tube through a loop of points (already dense and smooth); `rad(k)`
    gives the radius at point k. `up` is the loop's plane normal. An open
    tube (closed=False) is capped at both ends."""
    n = len(pts)
    pts = [Vector(p) for p in pts]
    if up is None:
        c = sum(pts, Vector()) / n
        up = Vector()
        for a, b in zip(pts, pts[1:] + pts[:1]):
            up += (a - c).cross(b - c)
    up = nrm(up)
    verts, faces = [], []
    for k in range(n):
        if closed:
            T = nrm(pts[(k + 1) % n] - pts[k - 1])
        else:
            T = nrm(pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)])
        N = nrm(T.cross(up))
        B = T.cross(N)
        r = rad(k)
        for j in range(sides):
            a = TAU * j / sides
            verts.append(pts[k] + (N * math.cos(a) + B * math.sin(a)) * r)
    for k in range(n if closed else n - 1):
        kn = (k + 1) % n
        for j in range(sides):
            jn = (j + 1) % sides
            faces.append((k * sides + j, k * sides + jn, kn * sides + jn, kn * sides + j))
    if not closed:
        for k, sgn in ((0, -1), (n - 1, 1)):
            T = nrm(pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)])
            verts.append(pts[k] + T * sgn * rad(k) * 0.5)
            pole = len(verts) - 1
            for j in range(sides):
                faces.append((k * sides + j, k * sides + (j + 1) % sides, pole))
    return mesh(name, verts, faces, role)


def bump(t, a, t0, a0, wt, wa):
    """A soft hump in (t, a) space centred on (t0, a0)."""
    return math.exp(-((t - t0) / wt) ** 2 - (wrap(a - a0) / wa) ** 2)


def fang(base, tip, r, role="trim", sides=5, name="fang"):
    """A small cone from base to tip."""
    axis = nrm(tip - base)
    ref = V(1, 0, 0) if abs(axis.x) < 0.9 else V(0, 1, 0)
    e1 = nrm(ref - ref.dot(axis) * axis)
    e2 = axis.cross(e1)
    verts = [base + (e1 * math.cos(TAU * j / sides) + e2 * math.sin(TAU * j / sides)) * r for j in range(sides)]
    verts.append(tip)
    faces = [(j, (j + 1) % sides, sides) for j in range(sides)] + [tuple(reversed(range(sides)))]
    return mesh(name, verts, faces, role)


def drip(pts, r0, neck=0.42, bulb=1.15, role="body", tip_role=None, rings=14, sides=7, name="drip"):
    """Matter running down a surface and hanging off it as a teardrop on a
    thinning neck. `pts` run from the source to the bottom of the drop."""
    table = [(0.0, r0), (0.25, r0 * 0.85), (0.62, r0 * neck), (0.84, r0 * bulb), (0.95, r0 * 0.8), (1.0, 0.0008)]
    paint = (lambda t, a: tip_role if t > 0.7 else None) if tip_role else None
    return Sweep(pts, rings, sides, lambda t, a: interp(table, t), role=role, paint=paint, caps=(0.5, 0.3),
                 name=name).obj


# ---------------------------------------------------------------- the mass

# The spine of the column: it leans out to the left at the root, swings
# right through the middle and bends its crown back behind the mask.
SPINE = [(0.014, 0.022, 0.392), (0.020, 0.014, 0.462), (0.008, 0.006, 0.548), (-0.010, 0.002, 0.638),
         (-0.012, 0.006, 0.728), (-0.002, 0.016, 0.812), (0.000, 0.030, 0.884), (-0.004, 0.048, 0.938),
         (-0.007, 0.056, 0.950), (-0.010, 0.062, 0.957)]
# Radius along the spine: a narrow root, a wide middle where the borrowed
# parts push out, a waist under the mask and a crown that holds it.
RADIUS = [(0.00, 0.018), (0.05, 0.042), (0.13, 0.068), (0.25, 0.090), (0.36, 0.100), (0.46, 0.100),
          (0.56, 0.093), (0.66, 0.080), (0.75, 0.066), (0.83, 0.056), (0.90, 0.052), (0.95, 0.044),
          (1.00, 0.030)]
# The mask sits on the upper front of the column, its centre at this height.
MASK_Z = 0.925
MASK_TIP = -0.28
# Strands wind round the column at this rate (radians per unit t), about a
# turn from the crown to the root; the column's lobes twist with them.
TWIST = -11.0
N_STRANDS = 3
STRAND_A0 = -1.213


def strand_angle(a_mid, t):
    return a_mid + TWIST * (t - 0.5)


def gap_angle(t, want):
    """The angle between two strands at height t nearest to `want`, so a part
    can erupt there without a strand running through it."""
    step = TAU / N_STRANDS
    g = STRAND_A0 + TWIST * (t - 0.5) + step / 2
    k = round((want - g) / step)
    return wrap(g + k * step)


# Where foreign parts erupt: (t, angle, height, width in t, width in angle).
ERUPT = {
    "arm": (0.44, gap_angle(0.44, -1.6), 0.060, 0.08, 0.55),
    "wing": (0.71, gap_angle(0.71, 1.7), 0.055, 0.10, 0.50),
    "tail": (0.53, gap_angle(0.53, -2.6), 0.030, 0.07, 0.50),
    "horn": (0.33, gap_angle(0.33, 1.6), 0.022, 0.06, 0.45),
}
# The half sizes (in t and angle) of the torn seam round each eruption.
SEAMS = {"arm": (0.080, 0.56), "wing": (0.070, 0.52), "tail": (0.070, 0.50), "horn": (0.050, 0.40)}
# Bubbles boiling out of the column: (t, angle, height, width in t, width in angle).
BOILS = [(0.36, 0.55, 0.022, 0.035, 0.28), (0.52, -0.75, 0.020, 0.030, 0.26), (0.64, 2.75, 0.024, 0.04, 0.30),
         (0.20, -1.85, 0.020, 0.035, 0.30), (0.72, -0.95, 0.016, 0.03, 0.26), (0.24, 2.40, 0.020, 0.035, 0.30)]


def column_radius(t, a):
    R = interp(RADIUS, t)
    # Three lobes of matter bulge between the strands and twist with them,
    # with slower irregular lumps drifting across them.
    ph = a - (STRAND_A0 + TWIST * (t - 0.5))
    # A barley twist: fat rounded plies, narrow grooves where the strands lie.
    c = 0.5 + 0.5 * math.cos(3 * ph + math.pi)
    # The plies are unequal, one fat and one thin, and swell and shrink as
    # they climb, so the twist reads as matter, not wrapping.
    uneven = 1 + 0.38 * math.cos(ph - 0.9) + 0.18 * math.sin(2 * ph + 9 * t)
    lobe = 0.30 * uneven * (2 * c ** 0.55 - 1.15) * smoothstep(0.03, 0.16, t) * (1 - smoothstep(0.68, 0.80, t))
    drift = 0.045 * math.sin(5 * a - 1.7 * t + 0.5) + 0.035 * math.sin(2 * a + 3.0 * t * t + 1.3)
    drift += 0.035 * math.sin(7 * a + 23 * t + 0.8) * smoothstep(0.1, 0.3, t)
    r = R * (1 + lobe + drift * (1 - 0.7 * smoothstep(0.7, 0.8, t)))
    # Bubbles of matter boiling up through the surface.
    for t0, a0, h, wt, wa in BOILS:
        r += h * bump(t, a, t0, a0, wt, wa)
    for t0, a0, h, wt, wa in ERUPT.values():
        r += h * bump(t, a, t0, a0, wt, wa)
    return r


SPINE_RINGS = 56
SPINE_P = curve(SPINE, SPINE_RINGS)
SPINE_T, SPINE_N, SPINE_B = frames(SPINE_P)
SPINE_LEN = polylen(SPINE_P)
MASK_T = next(i / (SPINE_RINGS - 1) for i, p in enumerate(SPINE_P) if p.z >= MASK_Z)
# The column is a little wider across than front to back.
DEPTH = 0.92


def spine_at(t):
    f = clamp(t) * (SPINE_RINGS - 1)
    i = min(int(f), SPINE_RINGS - 2)
    u = f - i
    return (SPINE_P[i].lerp(SPINE_P[i + 1], u), SPINE_T[i].lerp(SPINE_T[i + 1], u).normalized(),
            SPINE_N[i].lerp(SPINE_N[i + 1], u).normalized(), SPINE_B[i].lerp(SPINE_B[i + 1], u).normalized())


def spine_near(p):
    """The spine share nearest to p, and p's angle round the column there."""
    best, bi = 1e9, 0
    for i, q in enumerate(SPINE_P):
        d = (p - q).length_squared
        if d < best:
            best, bi = d, i
    t = bi / (SPINE_RINGS - 1)
    P, T, N, B = spine_at(t)
    r = p - P
    return t, math.atan2(r.dot(B), r.dot(N) / DEPTH), r - r.dot(T) * T


def edge_clamped_depth(x, z):
    """The mask's depth at (x, z), taken at its edge for points outside it
    (the face's shape means nothing there)."""
    rho = outline_rho(x, z)
    if rho > 1:
        x, z = x / rho, z / rho
    return mask_depth(x, z)


def socket(t, a, r):
    """Sinks the column behind the mask, so the mask is bedded in a socket
    whose lip rises round its edge instead of poking through it."""
    P, _, N, B = spine_at(t)
    D = N * (DEPTH * math.cos(a)) + B * math.sin(a)
    q0 = MASK_INV @ P
    dq = MASK_INV.to_3x3() @ D
    if dq.y > -0.2:
        return r
    q = q0 + dq * r
    if outline_rho(q.x, q.z) > 1.3:
        return r
    rr = r
    for _ in range(3):
        yb = -edge_clamped_depth(q.x * 1.03, q.z * 1.03) + MASK_TH + 0.002
        rr = (yb - q0.y) / dq.y
        q = q0 + dq * rr
    w = 1 - smoothstep(0.98, 1.28, outline_rho(q.x, q.z))
    # Never carve through the spine: a point behind it would flip to the far
    # side of the column.
    return r + (min(r, max(rr, 0.45 * r)) - r) * w


CORE = None


def column():
    """The column, built at its final resolution (never decimated, so the
    socket round the mask stays smooth): sparse rings low down, dense ones
    through the mask and crown."""
    global CORE

    def rad(t, a):
        r = socket(t, a, column_radius(t, a))
        return (math.cos(a) * r * DEPTH, math.sin(a) * r)

    def turn(t):
        """How far the rings turn with the twist: fully where the plies are,
        easing to a stop under the mask so the socket gets plain rings."""
        if t < 0.66:
            u = t
        elif t < 0.80:
            u = t - (t - 0.66) ** 2 / 0.28
        else:
            u = 0.73
        return TWIST * (u - 0.5)

    def twisted(t, a):
        return rad(t, a + turn(t))

    # The mesh is drawn with its edges following the plies (so the grooves
    # do not facet), then queried in plain angles like every other part.
    shares = [0.62 * i / 26 for i in range(26)] + [0.62 + 0.38 * (1 - (1 - i / 22) ** 1.35) for i in range(23)]
    CORE = Sweep(SPINE, 0, 30, twisted, role="body", caps=(0.9, 0.9), name="column", shares=shares)
    CORE.rad = rad
    if DEBUG:
        print(f"SHIFTER mask t={MASK_T:.3f} at {tuple(round(c, 3) for c in MASK_M.translation)}")
    return [note("column", CORE.obj)]


# ---------------------------------------------------------------- strands


def strand_path(t0, a_mid, rs, dip, t_end=0.03, sink_end=False, k=16):
    """The ridge of a strand lying in its groove from height share t0 down to
    t_end. A strand that ends early sinks back into the mass."""
    ctrl = []
    for i in range(k + 1):
        s = i / k
        t = t0 + (t_end - t0) * s
        out = -1.2 * rs + (1.2 * rs + 0.25 * rs) * smoothstep(0.0, 0.16, s)
        out -= 0.008 * max(0.0, math.sin(s * math.pi * 2.0 + dip)) ** 2 * smoothstep(0.15, 0.3, s) * (1 - smoothstep(0.7, 0.9, s))
        if sink_end:
            out -= 1.6 * rs * smoothstep(0.62, 1.0, s)
        ctrl.append(CORE.at(t, strand_angle(a_mid, t), out=out))
    return ctrl


def tail_curl(a_mid, out_dir, reach, drop, curl):
    """A tendril leaving the root: it swings out to the side and back (+y,
    away from the board camera) and rolls up into a spiral in the upright
    plane along its reach, so the front cameras see it curl."""
    a1 = strand_angle(a_mid, 0.03)
    d0 = nrm(CORE.N[0] * math.cos(a1) + CORE.B[0] * math.sin(a1))
    o = nrm(V(*out_dir, 0))
    root = CORE.P[0]
    z = V(0, 0, 1)
    m = reach
    pts = [root + d0 * 0.040 + o * 0.008 - z * (0.014 + drop * 0.3),
           root + d0 * 0.018 + o * 0.030 * m - z * (0.035 + drop),
           root + o * 0.052 * m - z * (0.050 + drop)]
    c = root + o * (0.052 * m + curl) - z * (0.050 + drop - curl)
    for j in range(1, 11):
        f = j / 10
        th = -math.pi / 2 + f * 2.2 * math.pi
        r = curl * (1 - 0.62 * f)
        pts.append(c + (-o * math.cos(th + math.pi) + z * math.sin(th)) * r)
    return pts


STRANDS = [  # kind, t0, radius, dip phase, tail (direction, reach, drop, curl)
    ("full", 0.84, 0.044, 4.0, ((-1.0, 0.75), 1.30, 0.004, 0.028)),
    ("fork", 0.88, 0.030, 0.3, ((1.0, 0.75), 1.25, 0.020, 0.026)),
    ("stump", 0.60, 0.020, 2.1, None),
]
FORK = (0.58, 0.42)
# The strands are the column's own matter, in its grooves and where they
# leave it as tendrils, so the tendrils never read as a pair of legs; only
# their tips burn.
STRAND_ROLE = "body"
BAND_RINGS, TAIL_RINGS = 20, 22


def band_sweep(ctrl, split, rs, ph, lit_from=None, name="strand", stump=False):
    """A flat muscle band lying against the column, rounding out into a
    tendril past `split` (share of the path)."""
    if split < 1:
        shares = ([split * j / (BAND_RINGS - 1) for j in range(BAND_RINGS)]
                  + [split + (1 - split) * j / TAIL_RINGS for j in range(1, TAIL_RINGS + 1)])
    else:
        shares = [j / (BAND_RINGS - 1) for j in range(BAND_RINGS)]
    P = curve_at(ctrl, shares)
    T, N, B = frames(P)
    theta = []
    for k, p in enumerate(P):
        _, _, r = spine_near(p)
        r = r - r.dot(T[k]) * T[k]
        th = math.atan2(r.dot(B[k]), r.dot(N[k])) if r.length > 1e-5 else (theta[-1] if theta else 0.0)
        if theta:
            th = theta[-1] + wrap(th - theta[-1])
        theta.append(th)
    tail = 1 - split
    if stump:
        # Full almost to the end, then a melting knob that sinks away.
        table = [(0.0, rs * 0.55), (0.12, rs), (0.70, rs * 0.92), (0.86, rs * 1.15), (0.95, rs * 0.8), (1.0, rs * 0.3)]
    elif split < 1:
        table = [(0.0, rs * 0.55), (0.10, rs), (0.35, rs * 0.92), (0.7 * split, rs * 0.80), (split, rs * 0.40),
                 (split + 0.15 * tail, rs * 0.30), (split + 0.55 * tail, rs * 0.18), (1.0, 0.0024)]
    else:
        table = [(0.0, rs * 0.5), (0.15, rs), (0.85, rs), (1.0, rs * 0.5)]

    def rad(t, a):
        k = 0
        while k < len(shares) - 2 and shares[k + 1] < t:
            k += 1
        u = clamp((t - shares[k]) / max(shares[k + 1] - shares[k], 1e-9))
        th = theta[k] * (1 - u) + theta[k + 1] * u
        r = interp(table, t) * (1 + 0.05 * math.sin(t * 30 + ph) * (1 - t))
        flat = smoothstep(split - 0.05, split + 0.25 * (1 - split), t) if split < 1 else 0.0
        rx = r * (0.80 + 0.20 * flat)
        ry = r * (1.06 - 0.06 * flat)
        x, y = math.cos(a) * rx, math.sin(a) * ry
        return (x * math.cos(th) - y * math.sin(th), x * math.sin(th) + y * math.cos(th))

    def paint(t, a):
        if lit_from and t > lit_from:
            return "trim"
        return None
    s = Sweep(ctrl, 0, 8, rad, role=STRAND_ROLE, paint=paint, caps=(0.5, 0.4), name=name, shares=shares)
    return s


def strands():
    parts = []
    for i, (kind, t0, rs, dip, tail) in enumerate(STRANDS):
        a_mid = STRAND_A0 + i * TAU / N_STRANDS
        if kind == "stump":
            ctrl = strand_path(t0, a_mid, rs, dip, t_end=0.46, sink_end=True, k=8)
            s = band_sweep(ctrl, 1.0, rs, i * 1.7, lit_from=0.80, name="strand", stump=True)
            parts.append(note("stump", s.obj))
            # The stump's end is soft: matter hangs off it in a drop.
            end = s.centre(0.93)
            n = nrm(end - spine_at(0.5)[0])
            pts = [end + n * 0.006, end + n * 0.010 - V(0, 0, 0.012), end + n * 0.010 - V(0, 0, 0.026),
                   end + n * 0.008 - V(0, 0, 0.036)]
            parts.append(drip(pts, 0.0085, neck=0.42, bulb=1.2, role="cloth", tip_role="trim", rings=9, sides=7,
                              name="drip"))
            continue
        ctrl = strand_path(t0, a_mid, rs, dip, k=16)
        out_dir, reach, drop, curl = tail
        tail_pts = tail_curl(a_mid, out_dir, reach, drop, curl)
        split = polylen(ctrl) / polylen(ctrl + tail_pts)
        # The last 30% of every tendril burns.
        lit_from = 1 - 0.30 * (1 - split)
        s = band_sweep(ctrl + tail_pts, split, rs, i * 1.7, lit_from=lit_from)
        parts.append(note("strand", s.obj))
        if kind == "fork":
            parts += fork(a_mid, rs)
    parts += central_tendril()
    return parts


def fork(a_mid, rs):
    """Between FORK[0] and FORK[1] the strand splits into two thinner ones
    that bow apart round a lobe and rejoin."""
    parts = []
    ta, tb = FORK
    for sgn in (-1, 1):
        ctrl = []
        for i in range(9):
            s = i / 8
            t = ta + (tb - ta) * s
            bow = math.sin(math.pi * s)
            a = strand_angle(a_mid, t) + sgn * 0.42 * bow
            out = -0.6 * rs + 1.1 * rs * smoothstep(0.0, 0.3, s) * smoothstep(0.0, 0.3, 1 - s)
            ctrl.append(CORE.at(t, a, out=out))
        table = [(0.0, rs * 0.45), (0.25, rs * 0.62), (0.5, rs * 0.55), (0.75, rs * 0.62), (1.0, rs * 0.45)]
        s = Sweep(ctrl, 14, 6, lambda t, a, tb=table: interp(tb, t), role=STRAND_ROLE, caps=(0.5, 0.5), name="fork")
        parts.append(note("fork", s.obj))
    return parts


def central_tendril():
    """One longer tendril ends the column in a spiral that faces the viewer,
    its last third burning."""
    root = CORE.P[0] + V(0, 0, 0.012)
    R = 0.026
    pts = [root + V(*o) for o in ((0, 0, 0), (0.004, 0.000, -0.026), (-0.004, -0.004, -0.050), (-0.010, -0.006, -0.062))]
    c = pts[-1] + V(R, -0.002, 0.0)
    for j in range(1, 11):
        f = j / 10
        th = math.pi + f * 2.3 * math.pi
        r = R * (1 - 0.60 * f)
        pts.append(c + V(math.cos(th) * r, -0.003 * f, math.sin(th) * r))
    s = Sweep(pts, 36, 8, lambda t, a: 0.018 * (1 - t) ** 0.8 + 0.0024, role="body",
              paint=lambda t, a: "trim" if t > 0.70 else None, caps=(0.5, 0.5), name="tendril")
    return [note("tendril", s.obj)]


def seam(key, rt, ra, seed=0.0, cracks=1, n=22, sides=4):
    """A torn ember seam round an eruption on the column, with a crack
    running out of it across the matter."""
    t0, a0 = ERUPT[key][0], ERUPT[key][1]
    pts = []
    for k in range(n):
        psi = TAU * k / n
        jag = 1 + 0.10 * math.sin(5 * psi + seed) + 0.05 * math.sin(11 * psi + seed * 2)
        pts.append(CORE.at(t0 + rt * jag * math.cos(psi), a0 + ra * jag * math.sin(psi), out=0.0016))
    parts = [ring_tube(pts, lambda k: 0.0058 * (1 + 0.35 * math.sin(7 * TAU * k / n + seed)), sides=sides,
                       name="seam")]
    for c in range(cracks):
        psi = seed + TAU * c / max(cracks, 1)
        ctrl = []
        for i in range(5):
            s = 1 + 0.55 * i / 4
            wob = 0.18 * math.sin(i * 2.3 + c) * (i / 4)
            ctrl.append(CORE.at(t0 + rt * s * math.cos(psi + wob), a0 + ra * s * math.sin(psi + wob), out=0.0016))
        parts.append(Sweep(ctrl, 6, 4, lambda t, a: 0.0052 * (1 - t) + 0.0008, role="trim", caps=(0.5, 0.5),
                           name="crack").obj)
    return parts


# ---------------------------------------------------------------- the mask

MASK_W, MASK_TOP, MASK_BOT = 0.072, 0.088, 0.084
# The mask's shape is drawn at its old size and scaled up by this much.
MS = 1.2
MASK_TH = 0.011
# Eye centres on the mask (x across, z up), half length, half height, slant:
# two large eyes, an upright one on the brow and two small ones under them,
# all mirrored, so the face has no fixed identity.
EYES = [
    (-0.035, 0.012, 0.0245, 0.0115, -0.22),
    (0.035, 0.012, 0.0245, 0.0115, 0.22),
    (0.000, 0.054, 0.0095, 0.0160, 0.0),
    (-0.050, -0.016, 0.0120, 0.0064, -0.42),
    (0.050, -0.016, 0.0120, 0.0064, 0.42),
]


# A beast's snout swells out of the mask's lower left (the viewer's right),
# pushing its edge out; the rim breaks over it (phi range).
SNOUT = (0.028, -0.050)
SNOUT_PHI = 2.50
RIM_BREAK = (2.05, 2.98)


def mask_depth(x, z, snout=True):
    """How far the mask face stands out (toward local -y) at (x, z)."""
    u, w = x / MS, z / MS
    Rs = 0.084
    W0 = MASK_W / MS
    q = Rs * Rs - u * u - (0.75 * w) ** 2
    d = math.sqrt(max(q, 0.0)) - math.sqrt(Rs * Rs - W0 * W0) + 0.004
    d -= 4.5 * u * u
    # A long nose bridge and a brow, and nothing else: no mouth, no expression.
    d += 0.0050 * math.exp(-(u / 0.011) ** 2) * smoothstep(-0.045, -0.010, w) * (1 - smoothstep(0.024, 0.040, w))
    d += 0.0040 * math.exp(-((w - 0.030) / 0.009) ** 2) * (1 - smoothstep(0.032, 0.056, abs(u)))
    # High cheekbones under the eyes, the cheeks falling away below them.
    d += 0.0030 * math.exp(-((abs(u) - 0.033) / 0.013) ** 2 - ((w + 0.008) / 0.011) ** 2)
    d -= 0.0030 * smoothstep(0.02, 0.05, abs(u)) * smoothstep(-0.02, -0.05, w)
    d *= MS
    if snout:
        sx, sz = SNOUT
        # A muzzle: a broad swelling with a blunt, flattened front.
        g = math.exp(-((x - sx) / 0.026) ** 2 - ((z - sz) / 0.022) ** 2)
        d += 0.033 * g ** 0.7 * smoothstep(0.0, 0.25, g)
    for ex, ez, hw, hh, _ in EYES:
        d -= 0.0062 * math.exp(-((x - ex) / (hw * 1.25)) ** 2 - ((z - ez) / (hh * 1.9)) ** 2)
    return d


def mask_outline(phi):
    """The mask's edge in direction phi (0 = up, toward +x at pi/2)."""
    s, c = math.sin(phi), math.cos(phi)
    h = MASK_TOP if c > 0 else MASK_BOT
    # The chin narrows to a soft point; the snout pushes the edge out.
    w = MASK_W * (1 - 0.32 * max(0.0, -c) ** 2)
    k = 1 + 0.20 * math.exp(-(wrap(phi - SNOUT_PHI) / 0.36) ** 2)
    return w * s * k, h * c * k


def outline_rho(x, z):
    """Where (x, z) lies relative to the mask's edge: 0 at the centre, 1 on it."""
    u, v = mask_outline(math.atan2(x, z))
    return math.hypot(x, z) / max(math.hypot(u, v), 1e-6)


def mask_frame():
    P, T, N, _ = spine_at(MASK_T)
    origin = P + N * (interp(RADIUS, MASK_T) * DEPTH + 0.002)
    # Tipped a little toward the right shoulder and turned up to the board
    # camera, its brow leaning back over the crown: uncanny, not upright.
    return frame(origin, T, N) @ Matrix.Rotation(-0.10, 4, "Y") @ Matrix.Rotation(MASK_TIP, 4, "X")


MASK_M = mask_frame()
MASK_INV = MASK_M.inverted()


def mask_point(x, z, out=0.0):
    return MASK_M @ V(x, -mask_depth(x, z) - out, z)


def mask_normal(x, z):
    e = 0.001
    dx = mask_point(x + e, z) - mask_point(x - e, z)
    dz = mask_point(x, z + e) - mask_point(x, z - e)
    n = dz.cross(dx).normalized()
    front = MASK_M.to_3x3() @ V(0, -1, 0)
    return n if n.dot(front) > 0 else -n


def lens(hw, hh, slant, dome, role="eye", name="eye"):
    """A domed almond in local XZ bulging toward -y, with a flat back."""
    n = 14
    verts = []
    for scale, lift in ((1.0, 0.0), (0.6, dome * 0.75)):
        for k in range(n):
            s = TAU * k / n
            u = hw * math.cos(s) * scale
            v = hh * math.sin(s) * abs(math.sin(s)) ** 0.35 * scale
            verts.append((u * math.cos(slant) - v * math.sin(slant), -lift, u * math.sin(slant) + v * math.cos(slant)))
    verts.append((0, -dome, 0))
    verts.append((0, 0.0025, 0))
    faces = []
    for k in range(n):
        kn = (k + 1) % n
        faces.append((k, kn, n + kn, n + k))
        faces.append((n + k, n + kn, 2 * n))
        faces.append((kn, k, 2 * n + 1))
    return mesh(name, verts, faces, role)


def column_gap(p):
    """How far p stands outside the column surface (negative when inside)."""
    t, a, r = spine_near(p)
    return r.length - column_radius(t, a) * math.hypot(DEPTH * math.cos(a), math.sin(a))


MASK_RHOS = (0.0, 0.16, 0.31, 0.45, 0.57, 0.68, 0.79, 0.89, 1.0)
MASK_NA = 36
BACK_RHOS = (0.97, 0.84, 0.62, 0.34)
# How far the back of the mask domes back into the column.
MASK_DOME = 0.050


def mask():
    na = MASK_NA
    front = []
    for rho in MASK_RHOS:
        ring = []
        for j in range(na if rho else 1):
            u, v = mask_outline(TAU * j / na)
            x, z = u * rho, v * rho
            ring.append(MASK_M @ V(x, -mask_depth(x, z), z))
        front.append(ring)
    # The back is a rounded skull of matter, so where the crown falls away
    # behind the mask it shows the back of a head, not the back of a plate.
    back = []
    for rho in BACK_RHOS:
        ring = []
        for j in range(na):
            u, v = mask_outline(TAU * j / na)
            bulge = MASK_DOME * (1 - (rho / BACK_RHOS[0]) ** 2) ** 0.5
            edge = mask_depth(u * BACK_RHOS[0], v * BACK_RHOS[0], snout=False)
            ring.append(MASK_M @ V(u * rho, -edge + MASK_TH + bulge, v * rho))
        back.append(ring)
    verts = [p for ring in front for p in ring] + [p for ring in back for p in ring]
    base = [0]
    for ring in front:
        base.append(base[-1] + len(ring))
    nr = len(front) - 1
    faces = [(0, base[1] + j, base[1] + (j + 1) % na) for j in range(na)]
    for i in range(1, nr):
        for j in range(na):
            jn = (j + 1) % na
            faces.append((base[i] + j, base[i] + jn, base[i + 1] + jn, base[i + 1] + j))
    n_front = len(faces)
    rings = [base[nr]] + [base[nr + 1] + k * na for k in range(len(BACK_RHOS))]
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(na):
            jn = (j + 1) % na
            faces.append((r0 + j, r0 + jn, r1 + jn, r1 + j))
    u, v = mask_outline(0.0)
    verts.append(MASK_M @ V(0, -mask_depth(0, v * BACK_RHOS[0], snout=False) + MASK_TH + MASK_DOME, 0))
    centre = len(verts) - 1
    faces += [(rings[-1] + (j + 1) % na, rings[-1] + j, centre) for j in range(na)]
    # Only the face is skin: its edge and back are matter flowing over it.
    m = mesh("mask", verts, faces, ["skin"] * n_front + ["body"] * (len(faces) - n_front))
    parts = [note("mask", lib.bevel_edges(m, 0.0016, 1))]
    # A thin rolled ember rim round the edge, torn open where the snout
    # pushes through. Where the column falls away behind it (over the crown,
    # down the right flank) it sinks under the socket lip instead of standing
    # proud as an arc seen from behind.
    lo, hi = RIM_BREAK
    n = 30
    phis = [hi + (lo + TAU - hi) * k / (n - 1) for k in range(n)]
    rim = []
    for phi in phis:
        u, v = mask_outline(phi)
        rim.append(MASK_M @ V(u * 1.015, -mask_depth(u, v) + 0.0035, v * 1.015))
    gaps = [column_gap(p) for p in rim]
    sink = [max(smoothstep(0.55, 0.85, math.cos(phi)), 1 - smoothstep(0.003, 0.011, g)) for phi, g in zip(phis, gaps)]
    inward = MASK_M.to_3x3() @ V(0, 1, 0)
    rim = [p + inward * 0.0045 * sink[k] for k, p in enumerate(rim)]

    def rim_r(k):
        end = smoothstep(0.0, 3.0, min(k, n - 1 - k))
        return (0.0038 - 0.0024 * sink[k]) * (0.45 + 0.55 * end)
    parts.append(note("mask_rim", ring_tube(rim, rim_r, sides=5, up=MASK_M.to_3x3() @ V(0, -1, 0), name="mask_rim",
                                            closed=False)))
    parts += chin_drips()
    parts += snout_features()
    up = MASK_M.to_3x3() @ V(0, 0, 1)
    for ex, ez, hw, hh, slant in EYES:
        n = mask_normal(ex, ez)
        e = lens(hw, hh, slant, 0.0036)
        u = nrm(up - up.dot(n) * n)
        parts.append(put(e, frame(mask_point(ex, ez, out=-0.0008), u, n)))
    return parts


def chin_drips():
    """The skin has not set: it runs off the chin under the snout in three
    drips of unequal length, each rooted inside the mask, flowing over the
    edge, thinning to a curving neck and hanging a teardrop."""
    parts = []
    inward = MASK_M.to_3x3() @ V(0, 1, 0)
    side = MASK_M.to_3x3() @ V(1, 0, 0)
    down = V(0, 0, -1)
    for phi, ln, r0, sway in ((2.62, 0.050, 0.0085, 0.006), (2.98, 0.035, 0.0072, -0.004), (3.40, 0.020, 0.0060, 0.003)):
        u, v = mask_outline(phi)
        root = mask_point(u * 0.86, v * 0.86, out=-0.004)
        lip = mask_point(u * 1.0, v * 1.0, out=-0.0015)
        pts = [root, lip, lip + down * ln * 0.30 + side * sway * 0.4 + inward * 0.002,
               lip + down * ln * 0.60 + side * sway + inward * 0.003,
               lip + down * ln * 0.85 + side * sway * 0.7 + inward * 0.004, lip + down * ln + side * sway * 0.5 + inward * 0.004]
        table = [(0.0, r0), (0.22, r0 * 0.92), (0.45, r0 * 0.55), (0.66, r0 * 0.36), (0.84, r0 * 0.95),
                 (0.94, r0 * 0.78), (1.0, 0.0008)]
        parts.append(note("mask_drip", Sweep(pts, 12, 8, lambda t, a, tb=table: interp(tb, t), role="skin",
                                             paint=lambda t, a: "body" if t > 0.5 else None,
                                             caps=(0.5, 0.3), name="mask_drip").obj))
    return parts


def snout_features():
    """Two nostril pits and a short split upper lip, lit from within, on the
    front of the swelling snout."""
    parts = []
    sx, sz = SNOUT
    up = MASK_M.to_3x3() @ V(0, 0, 1)
    for sgn in (-1, 1):
        x, z = sx + sgn * 0.0085, sz + 0.002
        n = mask_normal(x, z)
        e = lens(0.0050, 0.0030, sgn * 0.55, 0.0008, role="trim", name="nostril")
        u = nrm(up - up.dot(n) * n)
        parts.append(put(e, frame(mask_point(x, z, out=-0.0012), u, n)))
    # The cleft runs down from between the nostrils and splits the lip.
    for sgn in (-1, 1):
        pts = [(sx, sz - 0.004), (sx + sgn * 0.002, sz - 0.011), (sx + sgn * 0.008, sz - 0.017),
               (sx + sgn * 0.016, sz - 0.019)]
        parts.append(Sweep([mask_point(x, z, out=-0.0004) for x, z in pts], 9, 5,
                           lambda t, a: 0.0022 * (1 - 0.45 * t), role="trim", caps=(0.5, 0.5), name="lip").obj)
    return parts


# ---------------------------------------------------------------- the Knight's sword


def blade(base, d, flat, L, w0, th, name="blade"):
    """A straight double-edged blade from base along d, its flats facing
    `flat`: bevelled edges, a raised flat and a lit fuller down the middle."""
    side = nrm(d.cross(flat))
    rows = [(0.0, 1.0), (0.12, 0.97), (0.40, 0.90), (0.66, 0.82), (0.82, 0.66), (0.92, 0.38)]
    verts, faces, roles = [], [], []
    for s, wf in rows:
        c = base + d * (s * L)
        w = w0 * wf
        ft = th * (1 - 0.4 * s)
        verts += [c + side * w, c + side * (0.24 * w) + flat * ft, c - side * (0.24 * w) + flat * ft, c - side * w,
                  c - side * (0.24 * w) - flat * ft, c + side * (0.24 * w) - flat * ft]
    verts.append(base + d * L)
    nr = len(rows)
    for i in range(nr - 1):
        for j in range(6):
            jn = (j + 1) % 6
            faces.append((i * 6 + j, i * 6 + jn, (i + 1) * 6 + jn, (i + 1) * 6 + j))
            fuller = j in (1, 4) and rows[i + 1][0] <= 0.82
            roles.append("trim" if fuller else "metal")
    for j in range(6):
        faces.append(((nr - 1) * 6 + j, (nr - 1) * 6 + (j + 1) % 6, len(verts) - 1))
        roles.append("metal")
    faces.append(tuple(reversed(range(6))))
    roles.append("metal")
    return mesh(name, verts, faces, roles)


def sword_arm():
    """The Knight's sword arm, half copied: an arm of column matter reaches
    out and down to an elbow, then rises; its forearm is hardening into a
    plated vambrace whose ragged, burning edge is still spreading up the
    arm, and where its hand should be it has already become a crossguard and
    a long blade raised high. Matter still drips off the elbow and runs over
    the steel's edge."""
    parts = []
    t0, a0 = ERUPT["arm"][0], ERUPT["arm"][1]
    # The arm keeps its reach: it starts where the surface lay before the
    # column swelled up round its root.
    A = CORE.at(t0, a0, out=-0.026)
    S0 = CORE.at(t0, a0, out=-0.060)
    E = A + V(-0.064, -0.014, -0.040)
    W = E + V(-0.036, -0.036, 0.058)
    d = nrm(V(-0.52, 0.30, 0.80))
    G = W + d * 0.012
    table = [(0.0, 0.060), (0.22, 0.046), (0.42, 0.031), (0.55, 0.029), (0.72, 0.026), (0.90, 0.022), (1.0, 0.019)]
    plate_from = 0.62

    def edge(a):
        """Where the steel has spread to: a ragged line, further up the arm
        on its outer side."""
        return plate_from + 0.050 * math.sin(3 * a + 1.0) + 0.025 * math.sin(5 * a + 0.3) - 0.04 * math.cos(a)
    arm = Sweep([S0, A, A.lerp(E, 0.55) + V(0, 0, -0.004), E, E.lerp(W, 0.5), W], 30, 14,
                lambda t, a: interp(table, t), role="body",
                paint=lambda t, a: "metal" if t > edge(a) else None, caps=(0.6, 0.5), name="arm")
    parts.append(note("arm", arm.obj))
    # The steel's growing edge burns: a trim line along it.
    rim = [arm.at(edge(TAU * j / 28) + 0.006, TAU * j / 28, out=0.0008) for j in range(28)]
    parts.append(note("vambrace_rim", ring_tube(rim, lambda k: 0.0036, sides=5, role="trim", name="lame")))
    # Two lames down to a flared cuff at the wrist.
    for t, k, role, r in ((0.80, 1.07, "metal", 0.0034), (0.90, 1.07, "metal", 0.0034), (0.985, 1.18, "metal", 0.0042)):
        c = arm.centre(t)
        rr = interp(table, t) * k
        T = arm.tangent(t)
        i0, _ = arm.index(t)
        N, B = arm.N[i0], arm.B[i0]
        pts = [c + (N * math.cos(TAU * j / 16) + B * math.sin(TAU * j / 16)) * rr for j in range(16)]
        parts.append(ring_tube(pts, lambda kk, r=r: r, sides=5, up=T, role=role, name="lame"))
    # Matter runs over the rim and hangs off the elbow.
    for t, ang, ln, r0 in ((None, 0.4, 0.024, 0.0068), (None, 2.2, 0.016, 0.0056), (0.47, None, 0.032, 0.0085)):
        if ang is None:
            ang = arm.angle_of(t, V(0, 0, -1))
        if t is None:
            t = edge(ang) + 0.012
        p = arm.at(t, ang, out=-0.003)
        n = nrm(p - arm.centre(t))
        g = V(0, 0, -1)
        pts = [p, p + n * 0.003 + g * ln * 0.35, p + n * 0.005 + g * ln * 0.7, p + n * 0.006 + g * ln]
        parts.append(note("drip", drip(pts, r0, neck=0.40, bulb=1.25, rings=9, sides=6, name="drip")))
    # The hilt: a short grip wrapped in trim, a crossguard whose quillons
    # curve up toward the blade, and a lit gem at the cross.
    view = V(0, -0.75, 0.62)
    flat = nrm(view - view.dot(d) * d)
    side = nrm(d.cross(flat))
    grip = Sweep([W - d * 0.006, W + d * 0.004, G], 4, 8, lambda t, a: 0.0105 - 0.002 * t, role="trim",
                 caps=(None, 0.4), name="grip")
    parts.append(grip.obj)
    q = [G + side * 0.050 + d * 0.016, G + side * 0.034 + d * 0.002, G, G - side * 0.034 + d * 0.002,
         G - side * 0.050 + d * 0.016]
    parts.append(note("guard", Sweep(q, 15, 7, lambda t, a: 0.0068 * (1 - 0.35 * abs(2 * t - 1)) + 0.0014,
                                     role="trim", caps=(0.6, 0.6), name="guard").obj))
    gem = lib.sphere(0.0062, segments=8, rings=5, scale=(1.0, 0.6, 1.0), role="eye", name="gem")
    parts.append(put(gem, frame(G - flat * 0.0060, d, flat)))
    parts.append(note("blade", blade(G + d * 0.004, d, flat, 0.285, 0.0135, 0.0042)))
    return parts


# ---------------------------------------------------------------- the Dragon's wing

def membrane(edge_a, edge_b, nu, nv, scallop, billow, back, th=0.0016, trim_rows=1, ragged=0.0, name="membrane"):
    """A membrane panel stretched between two bone curves (point lists from
    the wrist outward), its free trailing edge scalloped in toward the
    wrist and burning. `back` is the direction the membrane billows to."""
    ca = [Vector(p) for p in edge_a]
    cb = [Vector(p) for p in edge_b]

    def along(c, s):
        return curve_at(c, [s])[0]

    grid = []
    for i in range(nu + 1):
        s = i / nu
        row = []
        for j in range(nv + 1):
            v = j / nv
            tear = ragged * (0.5 + 0.5 * math.cos(TAU * 3.5 * v)) * math.sin(math.pi * v) ** 0.5
            sm = s * (1 - scallop * math.sin(math.pi * v) - tear)
            p = along(ca, sm).lerp(along(cb, sm), v)
            p = p + back * (billow * math.sin(math.pi * v) * math.sin(math.pi * min(1.0, sm * 1.1)))
            row.append(p)
        grid.append(row)
    # Normals of the sheet, to give it thickness.
    verts, faces, roles = [], [], []
    idx = {}
    for i in range(nu + 1):
        for j in range(nv + 1):
            p = grid[i][j]
            du = grid[min(i + 1, nu)][j] - grid[max(i - 1, 0)][j]
            dv = grid[i][min(j + 1, nv)] - grid[i][max(j - 1, 0)]
            n = nrm(du.cross(dv))
            if n.dot(back) < 0:
                n = -n
            idx[(i, j, 0)] = len(verts)
            verts.append(p - n * th)
            idx[(i, j, 1)] = len(verts)
            verts.append(p + n * th)
    for i in range(nu):
        role = "trim" if i >= nu - trim_rows else "cloth"
        for j in range(nv):
            for side in (0, 1):
                f = (idx[(i, j, side)], idx[(i + 1, j, side)], idx[(i + 1, j + 1, side)], idx[(i, j + 1, side)])
                faces.append(f if side == 0 else tuple(reversed(f)))
                roles.append(role)
    # Close the trailing edge.
    for j in range(nv):
        faces.append((idx[(nu, j, 0)], idx[(nu, j, 1)], idx[(nu, j + 1, 1)], idx[(nu, j + 1, 0)]))
        roles.append("trim")
    return mesh(name, verts, faces, roles)


def dragon_wing():
    """The Dragon's bat wing, still growing out of the left of the column: a
    knobbed arm of matter bent at the elbow, a hooked claw at the wrist, three
    long finger bones fanning up and out with the membrane grown between
    them, its scalloped trailing edge burning as on the Dragon. The panel
    nearest the body has not grown yet: only strands of matter reach there."""
    parts = []
    t0, a0 = ERUPT["wing"][0], ERUPT["wing"][1]
    root = CORE.at(t0, a0, out=-0.016 - 0.025)
    out = nrm(V(1.0, 0.10, -0.05))
    up = nrm(V(-0.15, 0.55, 1.0))
    up = nrm(up - up.dot(out) * out)
    fr = nrm(up.cross(out))
    if fr.y > 0:
        fr = -fr

    def W2(u, v, w=0.0):
        return root + out * u + up * v + fr * w

    # The arm is only a stub of matter pouring out of the swelling: the
    # fingers fan almost straight out of the body.
    elbow, wrist = W2(0.042, 0.004), W2(0.072, 0.056, -0.004)
    table = [(0.0, 0.040), (0.30, 0.028), (0.55, 0.021), (0.80, 0.0185), (1.0, 0.015)]
    bone = Sweep([root, root.lerp(elbow, 0.5) + up * 0.004, elbow, elbow.lerp(wrist, 0.5), wrist], 20, 9,
                 lambda t, a: interp(table, t), role="body", caps=(0.6, 0.8), name="wing_bone")
    parts.append(note("wing_bone", bone.obj))
    # Three fingers: knuckle and tip in the wing plane, the outer ones bowed.
    fingers = [(W2(0.122, 0.192), W2(0.104, 0.302)), (W2(0.198, 0.172), W2(0.276, 0.244)),
               (W2(0.232, 0.092), W2(0.318, 0.070))]
    curves = []
    for i, (k, tip) in enumerate(fingers):
        pts = [wrist, wrist.lerp(k, 0.5) + fr * 0.002, k, k.lerp(tip, 0.5) + fr * 0.003, tip]
        curves.append(pts)
        s = Sweep(pts, 13, 6, lambda t, a: 0.0095 * (1 - t) ** 0.7 + 0.0026 + 0.0020 * math.exp(-((t - 0.5) / 0.06) ** 2),
                  role="body", caps=(0.5, 0.5), name="finger")
        parts.append(note("finger", s.obj))
        # A little hooked claw on each fingertip.
        d = nrm(tip - k)
        parts.append(fang(tip - d * 0.002, tip + d * 0.018 - up * 0.006, 0.0040, sides=5, name="claw"))
    # The thumb: a hooked trim claw at the wrist, pointing forward and up.
    th = [wrist + fr * 0.006, wrist + fr * 0.012 + up * 0.012 - out * 0.004, wrist + fr * 0.010 + up * 0.026 - out * 0.012,
          wrist + fr * 0.004 + up * 0.032 - out * 0.020]
    parts.append(Sweep(th, 9, 6, lambda t, a: 0.0070 * (1 - t) + 0.0005, role="trim", caps=(0.5, 0.3),
                       name="thumb").obj)
    # The membrane between the fingers, billowing back.
    back = -fr
    parts.append(note("membrane", membrane(curves[0], curves[1], 8, 6, 0.17, 0.014, back, trim_rows=1)))
    parts.append(note("membrane", membrane(curves[1], curves[2], 8, 6, 0.17, 0.014, back, trim_rows=1)))
    # The inner panel runs from the last finger down to the flank of the
    # column, as on a bat, so the wing grows out of the body instead of being
    # held up on a stick; it is still tearing, its edge ragged.
    flank = [CORE.at(t0 + 0.01, a0 - 0.05, out=-0.004), CORE.at(t0 - 0.08, a0 + 0.08, out=-0.004),
             CORE.at(t0 - 0.15, a0 + 0.18, out=-0.004), CORE.at(t0 - 0.21, a0 + 0.26, out=-0.004)]
    parts.append(note("membrane", membrane(curves[2], flank, 8, 10, 0.30, 0.012, back, trim_rows=1, ragged=0.30)))
    # Matter sags off the underside of the arm.
    for t, ln, r0 in ((0.34, 1.0, 0.0072), (0.66, 0.7, 0.0058)):
        a = bone.angle_of(t, V(0, 0, -1))
        p = bone.at(t, a, out=-0.004)
        g = V(0, 0, -1)
        pts = [p, p + g * 0.012 * ln, p + g * 0.024 * ln + fr * 0.002, p + g * 0.036 * ln + fr * 0.003]
        parts.append(note("drip", drip(pts, r0, neck=0.40, bulb=1.25, rings=9, sides=6, name="drip")))
    return parts


# ---------------------------------------------------------------- the Manticore's tail


def tail_segment(r, L, name="tail_seg"):
    """One armoured scorpion segment along local z: a barrel pinched at both
    joints and a banded trim rim lapping over the next."""
    prof = [(0, -0.002), (r * 0.74, 0.0), (r * 0.98, L * 0.12), (r * 1.05, L * 0.42), (r * 0.98, L * 0.68),
            (r * 1.14, L * 0.80), (r * 1.14, L * 0.92), (r * 0.80, L * 0.99), (0, L)]
    seg = lib.lathe(prof, role="body", segments=9, name=name)
    lib.paint(seg, "trim", lambda c: c.z > L * 0.76)
    return seg


TAIL_CTRL = [(0.0, 0.0, 0.0), (-0.020, 0.052, 0.050), (-0.052, 0.074, 0.130), (-0.090, 0.050, 0.214),
             (-0.112, -0.004, 0.262), (-0.110, -0.056, 0.258), (-0.096, -0.086, 0.230)]


def scorpion_tail():
    """The Manticore's tail breaking out of the back: a soft, unbanded root of
    matter, then ringed segments arching up and over the right of the head,
    ending in a venom bulb and a stinger aimed forward, as on the Manticore."""
    parts = []
    t0, a0 = ERUPT["tail"][0], ERUPT["tail"][1]
    root = CORE.at(t0, a0, out=-0.022)
    path = curve([root + V(*c) for c in TAIL_CTRL], 120)
    L = polylen(path)
    # The first stretch is still matter: a soft, unsegmented root.
    soft = 0.22
    k_soft = int(soft * (len(path) - 1))
    root_tab = [(0.0, 0.046), (0.4, 0.040), (0.75, 0.036), (1.0, 0.031)]
    s = Sweep(path[:k_soft + 1:5] + [path[k_soft]], 12, 10, lambda t, a: interp(root_tab, t), role="body",
              caps=(0.6, None), name="tail_root")
    parts.append(note("tail_root", s.obj))
    n_seg = 7
    acc = [0.0]
    for a, b in zip(path, path[1:]):
        acc.append(acc[-1] + (b - a).length)

    def at(sh):
        x = L * sh
        for i in range(len(acc) - 1):
            if acc[i + 1] >= x:
                f = (x - acc[i]) / max(acc[i + 1] - acc[i], 1e-9)
                return path[i].lerp(path[i + 1], f)
        return path[-1]

    centre = sum(path, Vector()) / len(path)
    for k in range(n_seg):
        sa = soft - 0.02 + (1.0 - soft) * k / n_seg
        sb = soft - 0.02 + (1.0 - soft) * (k + 1) / n_seg
        a, b = at(sa), at(sb)
        d = b - a
        r = 0.028 - 0.011 * k / (n_seg - 1)
        seg = tail_segment(r, d.length * 1.12)
        dorsal = a.lerp(b, 0.5) - centre
        parts.append(put(seg, frame(a, d, -dorsal)))
    # The sting: a teardrop venom bulb flowing into a straight spike aimed
    # forward and a little down, with barbs round the root of the point.
    end = path[-1]
    # The bulb flows on from the curl and is seated deep in the last ring.
    tangent = nrm(path[-1] - path[-8])
    aim = nrm(nrm(V(0.25, -0.80, -0.55)) + tangent)
    prof = [(0, -0.004), (0.013, 0.0), (0.022, 0.008), (0.028, 0.018), (0.030, 0.028), (0.028, 0.040),
            (0.021, 0.052), (0.0135, 0.066), (0.0098, 0.090), (0.0068, 0.120), (0.0038, 0.150), (0.0016, 0.172),
            (0, 0.186)]
    sting = lib.lathe(prof, role="trim", segments=14, name="telson")
    base = end - tangent * 0.020
    parts.append(note("telson", put(sting, frame(base, aim, V(0, 0, 1)))))
    sx = nrm(aim.cross(V(0, 0, 1)))
    sy = sx.cross(aim)
    for i in range(4):
        phi = TAU * i / 4 + TAU / 8
        radial = sx * math.cos(phi) + sy * math.sin(phi)
        dd = nrm(aim * math.cos(0.5) + radial * math.sin(0.5))
        rr = base + aim * 0.056 + radial * 0.011
        parts.append(fang(rr, rr + dd * 0.036, 0.0062, sides=5, name="barb"))
    # Matter drips off the soft root.
    p = s.at(0.6, s.angle_of(0.6, V(0, 0, -1)), out=-0.003)
    pts = [p, p - V(0, 0, 0.014), p - V(0, 0, 0.028), p - V(0, 0, 0.040)]
    parts.append(drip(pts, 0.0080, neck=0.40, bulb=1.25, rings=9, sides=6, name="drip"))
    return parts


# ---------------------------------------------------------------- the Unicorn's horn


def unicorn_horn():
    """The Unicorn's spiral horn pushing out of the lower left, set in its
    metal ring, a lumpy lip of matter lapping over the ring."""
    parts = []
    t0, a0 = ERUPT["horn"][0], ERUPT["horn"][1]
    b = CORE.at(t0, a0, out=-0.010)
    d = nrm(V(0.70, -0.40, 0.65))
    L = 0.190
    pts = [b, b + d * L * 0.5 + V(0, 0, 0.004), b + d * L]

    # Three ridges twisting two and three quarter turns to the point, the
    # grooves between them and the point glowing, as on the Unicorn. The
    # rings turn with the twist, so the mesh's edges run along the ridges.
    turns = 2.75

    def rad(t, a):
        r = (0.0200 * (1 - t) ** 0.9 + 0.0010) * (1 + 0.26 * math.cos(3 * a) * (1 - 0.4 * t))
        phi = a + TAU * turns * t
        return (math.cos(phi) * r, math.sin(phi) * r)
    s = Sweep(pts, 30, 18, rad, role="trim", caps=(0.4, 0.4), name="horn",
              paint=lambda t, a: "eye" if (math.cos(3 * a) < -0.5 and 0.08 < t < 0.86) or t > 0.84 else None)
    parts.append(note("horn", s.obj))
    # The ring the horn is set in.
    c = s.centre(0.10)
    T = s.tangent(0.10)
    i0, _ = s.index(0.10)
    N, B = s.N[i0], s.B[i0]
    ring = [c + (N * math.cos(TAU * j / 14) + B * math.sin(TAU * j / 14)) * 0.020 for j in range(14)]
    parts.append(note("horn_ring", ring_tube(ring, lambda k: 0.0048, sides=6, up=T, role="metal", name="horn_ring")))
    # The matter has not let go of it: a lumpy lip laps up over the ring.
    c0 = s.centre(0.03)
    lip = [c0 + (N * math.cos(TAU * j / 14) + B * math.sin(TAU * j / 14)) * (0.024 + 0.002 * math.sin(TAU * 2 * j / 14))
           for j in range(14)]
    parts.append(ring_tube(lip, lambda k: 0.0068 * (1 + 0.22 * math.sin(TAU * 2 * k / 14 + 1)), sides=6, up=T,
                           role="body", name="horn_lip"))
    p = lip[min(range(14), key=lambda j: lip[j].z)]
    pts = [p, p - V(0, 0, 0.012), p - V(0, 0, 0.024), p - V(0, 0, 0.034)]
    parts.append(drip(pts, 0.0062, neck=0.40, bulb=1.25, rings=9, sides=6, name="drip"))
    return parts


# ---------------------------------------------------------------- the crown


def mask_back_y(x, z):
    """The local depth of the mask's domed back at (x, z)."""
    rho = min(outline_rho(x, z), BACK_RHOS[0])
    u, v = mask_outline(math.atan2(x, z))
    edge = mask_depth(u * BACK_RHOS[0], v * BACK_RHOS[0], snout=False)
    return -edge + MASK_TH + MASK_DOME * (1 - (rho / BACK_RHOS[0]) ** 2) ** 0.5


def crest():
    """The matter has not let go of the mask: two lopsided runs of it slide
    down off the top of its domed back, over the crown and down the back of
    the column, and hang there as drips."""
    parts = []
    for a0, ln, r0, x0 in ((2.80, 1.0, 0.015, 0.024), (-2.62, 0.74, 0.012, -0.030)):
        pts = []
        for z, sink in ((0.064, 2.4), (0.044, 0.75), (0.018, 0.35)):
            x = x0 * (0.7 + 0.3 * z / 0.064)
            pts.append(MASK_M @ V(x, mask_back_y(x, z) - r0 * sink, z))
        for k in range(5):
            s = k / 4
            t = 0.995 - 0.11 * ln * s
            a = a0 + 0.15 * math.sin(s * 2.5)
            pts.append(CORE.at(t, a, out=-r0 * 0.25 + 0.002 * s))
        n = CORE.normal(0.995 - 0.11 * ln, a0)
        last = pts[-1]
        pts += [last + n * 0.008 - V(0, 0, 0.018) * ln, last + n * 0.012 - V(0, 0, 0.040) * ln]
        parts.append(note("wisp", drip(pts, r0, neck=0.45, bulb=1.1, rings=14, sides=8, name="wisp")))
    return parts


# ---------------------------------------------------------------- figure


def build():
    parts = []
    parts += column()
    parts += strands()
    parts += mask()
    parts += seam("arm", *SEAMS["arm"], seed=0.4)
    parts += seam("wing", *SEAMS["wing"], seed=1.9)
    parts += seam("tail", *SEAMS["tail"], seed=3.1)
    parts += seam("horn", *SEAMS["horn"], seed=5.2, cracks=0, n=16)
    parts += sword_arm()
    parts += dragon_wing()
    parts += scorpion_tail()
    parts += unicorn_horn()
    parts += crest()
    m = Matrix.Translation(V(*PIVOT) + V(0, 0, LIFT)) @ Matrix.Scale(SCALE, 4) @ Matrix.Translation(-V(*PIVOT))
    for p in parts:
        put(p, m)
    if DEBUG:
        for key, (te, ae, *_rest) in ERUPT.items():
            print(f"SHIFTER erupt {key} t={te:.2f} a={ae:.2f}")
        groups = {}
        for p in parts:
            key = p.name.split(".")[0].replace("_mesh", "")
            groups[key] = groups.get(key, 0) + lib.tris(p)
        print("SHIFTER groups: " + ", ".join(f"{k}={v}" for k, v in sorted(groups.items(), key=lambda kv: -kv[1])))
        print(f"SHIFTER total before join: {sum(groups.values())}")
        lows = sorted(((min(v.co.z for v in p.data.vertices), p.name) for p in parts))[:4]
        highs = sorted(((max(v.co.z for v in p.data.vertices), p.name) for p in parts))[-3:]
        print("SHIFTER lowest: " + ", ".join(f"{n}={z:.3f}" for z, n in lows))
        print("SHIFTER highest: " + ", ".join(f"{n}={z:.3f}" for z, n in highs))
        xs = sorted(((min(v.co.x for v in p.data.vertices), p.name) for p in parts))[:2]
        xs2 = sorted(((max(v.co.x for v in p.data.vertices), p.name) for p in parts))[-2:]
        print("SHIFTER x extremes: " + ", ".join(f"{n}={z:.3f}" for z, n in xs + xs2))
    return parts
