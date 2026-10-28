"""Dragon: the Dark side's great flyer, a classic western dragon hovering
with its bat wings raised and swept back. A horned head with an open, fanged
jaw glowing with fire, a long S-curved neck, a spined back ridge, a scaled
body with plated belly, four clawed legs drawn up in flight and a long tail
curling round to a spade. The figure faces -Y; its right side is on -X.

Technique: neck, body and tail are one loft along a smooth spine with
superelliptic sections on rotation-minimising frames (deep chest, slim
tail), its rows graded coarser down the tail, sculpted with overlapping belly
plates (skin), rows of scales lapping toward the tail, staggered scales on the
flanks of the neck and tail, a dorsal ridge and hips that swell back into
the root of the tail to take the thighs; ember spine blades run its length,
two rows of lapping keeled scutes flank them, and a lens-section arrowhead
spade, lying nearly flat and open to the board camera, ends the tail as it
curls forward on the left. The head, raised snout-up on the S of the neck, is
two lofts, finely divided toward the snout tip: a wedge skull with brow
shelf, sunken eye sockets and flared cheeks and a heavy hinged jaw, joined at
the corners of the mouth by cheek webs; the floor of the mouth glows between
the gums, ember at the lips and white-hot in the throat, framed by long trim
fangs. Slit pupils wrap the outer face of each eye. Great ridged horns sweep
back, with a lesser pair at the cheeks. Each leg is one continuous loft from
a root sunk in the flank to the foot, swelling over muscles and joints, with
thick toes and hooked claws. Wing membranes are quad grids spanned
between the finger bones (Coons patches) with deep scallops, a billow and
shallow folds, given thickness with solidify, and edged with a thin trim band;
the finger tips jut past the membrane as trim spikes, trim knuckles mark the
finger joints and an ember thumb claw crowns each wrist.

Roles: body is the hide, bones and pupils, cloth the wing membranes, skin the
belly plates, gums and cheek webs, trim the horns, spines, spade, wing edges,
finger spikes and knuckles, thumb claws, elbow spurs, long fangs and the fire
at the lips, metal the claws and small teeth, eye the eyes and the fire deep
in the mouth and gullet."""
import math
import os

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

import lib

TAU = math.tau
FINISH = {"sharp_angle": 60}
DEBUG = bool(os.environ.get("DRAGON_DEBUG"))


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def sstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


def spow(a, e):
    return math.copysign(abs(a) ** e, a)


def frame(origin, z, front=(0, -1, 0)):
    """A matrix whose local z points along `z` and whose local -y leans toward
    `front`, placed at `origin`."""
    z = nrm(z)
    f = Vector(front)
    f = f - f.dot(z) * z
    if f.length < 1e-6:
        f = Vector((0, 0, 1)) - Vector((0, 0, 1)).dot(z) * z
    y = -f.normalized()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def put(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def recalc(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def mesh_object(name, verts, faces, roles="body"):
    """A mesh from verts and faces; `roles` is one role or one per face."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    recalc(obj)
    if isinstance(roles, str):
        return lib.set_role(obj, roles)
    used = list(dict.fromkeys(roles))
    me.materials.clear()
    for r in used:
        me.materials.append(lib.mat(r))
    idx = {r: i for i, r in enumerate(used)}
    for p, r in zip(me.polygons, roles):
        p.material_index = idx[r]
    return obj


def catmull(points, per=8):
    """Uniform Catmull-Rom through `points` (tuples of any length: position
    and any values to carry along, such as radii)."""
    P = [[float(x) for x in p] for p in points]
    ext = [[2 * a - b for a, b in zip(P[0], P[1])]] + P + [[2 * a - b for a, b in zip(P[-1], P[-2])]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(per):
            t = k / per
            t2, t3 = t * t, t * t * t
            out.append([0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
                        for a, b, c, d in zip(p0, p1, p2, p3)])
    out.append(P[-1][:])
    return out


def resample(path, n):
    """`n` rows evenly spaced by the arc length of their first three values."""
    d = [0.0]
    for a, b in zip(path, path[1:]):
        d.append(d[-1] + math.dist(a[:3], b[:3]))
    out, j = [], 0
    for i in range(n):
        s = d[-1] * i / (n - 1)
        while j < len(d) - 2 and d[j + 1] < s:
            j += 1
        t = (s - d[j]) / max(d[j + 1] - d[j], 1e-9)
        out.append([x + (y - x) * t for x, y in zip(path[j], path[j + 1])])
    return out


def smooth_path(points, step, per=10):
    rows = catmull(points, per)
    length = sum(math.dist(a[:3], b[:3]) for a, b in zip(rows, rows[1:]))
    return resample(rows, max(3, int(round(length / step)) + 1))


def graded_path(points, step_fn, per=12):
    """Rows along a smooth curve spaced step_fn(arc) apart: fine where the
    form is big, coarser down a slim tail."""
    rows = resample(catmull(points, per), 2000)
    d = [0.0]
    for a, b in zip(rows, rows[1:]):
        d.append(d[-1] + math.dist(a[:3], b[:3]))
    marks, s = [0.0], 0.0
    while s < d[-1]:
        s += step_fn(s)
        marks.append(min(s, d[-1]))
    # Stretch the marks so the last lands exactly on the end.
    marks = [m * d[-1] / marks[-1] for m in marks]
    out, j = [], 0
    for m in marks:
        while j < len(d) - 2 and d[j + 1] < m:
            j += 1
        t = (m - d[j]) / max(d[j + 1] - d[j], 1e-9)
        out.append([x + (y - x) * t for x, y in zip(rows[j], rows[j + 1])])
    return out


def transport(P, up0):
    """Tangents and rotation-minimising dorsal and lateral axes along P."""
    n = len(P)
    T = [nrm(P[min(i + 1, n - 1)] - P[max(i - 1, 0)]) for i in range(n)]
    u = Vector(up0)
    u = nrm(u - T[0] * u.dot(T[0]))
    U = [u]
    for i in range(1, n):
        axis = T[i - 1].cross(T[i])
        if axis.length > 1e-8:
            u = Matrix.Rotation(T[i - 1].angle(T[i]), 3, axis.normalized()) @ u
        u = nrm(u - T[i] * u.dot(T[i]))
        U.append(u)
    L = [t.cross(u) for t, u in zip(T, U)]
    return T, U, L


class Loft:
    """A tube along a path whose stations are (x, y, z, half width, dorsal
    height, ventral depth). Section angle a: 0 is lateral (+L), pi/2 dorsal."""

    def __init__(self, stations, up0):
        self.st = stations
        self.P = [Vector(s[:3]) for s in stations]
        self.T, self.U, self.L = transport(self.P, up0)
        self.arc = [0.0]
        for a, b in zip(self.P, self.P[1:]):
            self.arc.append(self.arc[-1] + (b - a).length)

    def surface(self, i, a, power=2.2):
        w, up, dn = self.st[i][3:6]
        e = 2.0 / power
        c, s = math.cos(a), math.sin(a)
        return self.P[i] + self.L[i] * (w * spow(c, e)) + self.U[i] * ((up if s >= 0 else dn) * spow(s, e))

    def at(self, arc):
        """Station index nearest to an arc length."""
        best = min(range(len(self.arc)), key=lambda i: abs(self.arc[i] - arc))
        return best

    def build(self, sides, sculpt=None, role_fn=None, power=2.2, caps=(0.5, 0.5), name="loft"):
        verts, faces, roles = [], [], []
        n = len(self.P)
        for i in range(n):
            for k in range(sides):
                a = TAU * k / sides
                q = self.surface(i, a, power)
                if sculpt:
                    q = sculpt(q, i, a)
                verts.append(q)
        for i in range(n - 1):
            for k in range(sides):
                k1 = (k + 1) % sides
                faces.append((i * sides + k, i * sides + k1, (i + 1) * sides + k1, (i + 1) * sides + k))
                roles.append(role_fn(i + 0.5, TAU * (k + 0.5) / sides) if role_fn else "body")
        for end, cap in ((0, caps[0]), (n - 1, caps[1])):
            if cap is None:
                continue
            base = end * sides
            ring = verts[base:base + sides]
            centre = sum(ring, Vector()) / sides
            rad = sum((v - centre).length for v in ring) / sides
            verts.append(centre + self.T[end] * (1 if end else -1) * rad * cap)
            ci = len(verts) - 1
            for k in range(sides):
                faces.append((base + k, base + (k + 1) % sides, ci))
                roles.append(role_fn(end, math.pi / 2) if role_fn else "body")
        return mesh_object(name, verts, faces, roles)


def fin(base, along, up, side, height, length, thick, hook=0.45, rows=5, role="trim", name="spine"):
    """A curved spike blade standing on `base` in the plane of `along` (the
    way it sweeps) and `up`: convex leading edge, concave trailing edge, lens
    section thinning to the tip."""
    verts, faces = [], []
    along, up, side = nrm(along), nrm(up), nrm(side)
    tipx = hook * height
    for r in range(rows):
        f = r / rows
        xl = -length / 2 + (length / 2 + tipx) * f ** 1.5
        xt = length / 2 + (tipx - length / 2) * f ** 0.75
        xm = xl + (xt - xl) * 0.42
        t = thick * (1 - f) ** 0.8
        z = height * f
        for x, y in ((xl, 0), (xm, -t), (xt, 0), (xm, t)):
            verts.append(base + along * x + up * z + side * y)
    verts.append(base + along * tipx + up * height)
    tip = len(verts) - 1
    for r in range(rows - 1):
        for k in range(4):
            a, b = r * 4 + k, r * 4 + (k + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    last = (rows - 1) * 4
    for k in range(4):
        faces.append((last + k, last + (k + 1) % 4, tip))
    faces.append((3, 2, 1, 0))
    return mesh_object(name, verts, faces, role)


def cone(base, direction, length, r, role="metal", segments=4, name="tooth"):
    o = lib.cylinder(r, r * 0.04, length, role=role, segments=segments, name=name)
    return put(o, frame(Vector(base) + nrm(direction) * length / 2, direction))


def taper(points, radii, sides=8, step=0.01, role="body", name="tube", rings=0.0, ring_amp=0.07, power=2.0):
    """A tube along a smooth curve through `points` whose radius follows
    `radii`, closing to a point where the radius runs out: horns, claws,
    bones, toes. `rings` spaces raised growth rings along it."""
    rows = smooth_path([tuple(p) + (r, r, r) for p, r in zip(points, radii)], step, per=8)
    for r in rows:
        r[3] = r[4] = r[5] = max(r[3], 0.0002)
    lo = Loft(rows, up0=nrm(Vector(rows[1][:3]) - Vector(rows[0][:3])).orthogonal())
    sculpt = None
    if rings:
        def sculpt(q, i, a):
            k = max(0.0, math.sin(lo.arc[i] / rings * TAU)) ** 3
            return lo.P[i] + (q - lo.P[i]) * (1 + ring_amp * k)
    end = 0.0 if radii[-1] < 0.002 else 0.5
    return lo.build(sides, sculpt=sculpt, role_fn=lambda i, a: role, power=power, caps=(0.5, end), name=name)


def claw(points, radii, role="metal", sides=5, name="claw"):
    return taper(points, radii, sides=sides, step=0.013, role=role, name=name)


def leaf(ctrl, thick, centre, m, per=3, rings=3, role="trim", name="leaf"):
    """A flat blade with a lens section: a closed smooth outline through
    `ctrl` (u, v), domed `thick` at `centre`, put in frame `m` (u along local
    x, v along local z, thickness along local y)."""
    n = len(ctrl)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = (Vector(ctrl[(i + k) % n]) for k in (-1, 0, 1, 2))
        for k in range(per):
            t = k / per
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    c = Vector(centre)
    N = len(out)
    verts = [V(p.x, 0, p.y) for p in out]
    faces = []
    for side in (-1, 1):
        prev = list(range(N))
        for r in range(1, rings):
            f = 1 - r / rings
            ring = []
            for p in out:
                q = c + (p - c) * f
                ring.append(len(verts))
                verts.append(V(q.x, side * thick * (1 - f * f), q.y))
            for k in range(N):
                k1 = (k + 1) % N
                faces.append((prev[k], prev[k1], ring[k1], ring[k]))
            prev = ring
        verts.append(V(c.x, side * thick, c.y))
        ci = len(verts) - 1
        for k in range(N):
            faces.append((prev[k], prev[(k + 1) % N], ci))
    return put(mesh_object(name, verts, faces, role), m)


def grid(fn, nu, nv, name, role="cloth", thick=0.0, us=None, vs=None, role_fn=None):
    """A quad grid over fn(u, v) for u, v in [0, 1], optionally solidified.
    `us` and `vs` place the grid lines (evenly spaced by default);
    `role_fn(i, j)` gives the role of the face in row i, column j."""
    us = us or [i / nu for i in range(nu + 1)]
    vs = vs or [j / nv for j in range(nv + 1)]
    nu, nv = len(us) - 1, len(vs) - 1
    verts, faces, roles = [], [], []
    for u in us:
        for v in vs:
            verts.append(fn(u, v))
    for i in range(nu):
        for j in range(nv):
            q = i * (nv + 1) + j
            faces.append((q, q + 1, q + nv + 2, q + nv + 1))
            roles.append(role_fn(i, j) if role_fn else role)
    obj = mesh_object(name, verts, faces, roles if role_fn else role)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(obj.data)
    bm.free()
    if thick:
        obj = lib.solidify(obj, thick)
    return obj


def polyline_at(path, t):
    """Point at fraction t of a polyline's length."""
    d = [0.0]
    for a, b in zip(path, path[1:]):
        d.append(d[-1] + (b - a).length)
    s = d[-1] * clamp(t)
    for i in range(len(path) - 1):
        if d[i + 1] >= s:
            f = (s - d[i]) / max(d[i + 1] - d[i], 1e-9)
            return path[i].lerp(path[i + 1], f)
    return path[-1].copy()


def curve(points, per=12):
    return [Vector(p[:3]) for p in catmull(points, per)]


# ---------------------------------------------------------------- body


# Spine from the back of the head to the tail tip: x, y, z, half width,
# dorsal height, ventral depth. The torso rears a little; the neck rises in an
# S to a head held forward; the tail trails back and curls to the left.
SPINE = [
    (0.000, -0.280, 1.000, 0.036, 0.036, 0.040),
    (0.000, -0.238, 0.958, 0.039, 0.038, 0.044),
    (0.000, -0.204, 0.892, 0.042, 0.040, 0.049),
    (0.000, -0.200, 0.812, 0.046, 0.042, 0.055),
    (0.000, -0.228, 0.726, 0.053, 0.047, 0.064),
    (0.000, -0.212, 0.652, 0.063, 0.054, 0.078),
    (0.000, -0.158, 0.604, 0.078, 0.064, 0.096),
    (0.000, -0.094, 0.575, 0.100, 0.080, 0.139),
    (0.000, -0.012, 0.547, 0.102, 0.080, 0.130),
    (0.000, 0.075, 0.527, 0.087, 0.070, 0.103),
    (0.000, 0.150, 0.511, 0.086, 0.068, 0.074),
    (0.000, 0.225, 0.491, 0.068, 0.058, 0.058),
    (0.012, 0.310, 0.457, 0.054, 0.047, 0.047),
    (0.050, 0.388, 0.422, 0.043, 0.038, 0.037),
    (0.114, 0.444, 0.396, 0.034, 0.030, 0.029),
    (0.192, 0.470, 0.382, 0.027, 0.024, 0.023),
    (0.272, 0.460, 0.382, 0.021, 0.019, 0.018),
    (0.342, 0.420, 0.390, 0.016, 0.014, 0.014),
    (0.390, 0.350, 0.414, 0.012, 0.011, 0.011),
    (0.402, 0.270, 0.452, 0.009, 0.008, 0.008),
    (0.380, 0.196, 0.466, 0.007, 0.006, 0.006),
]
BELLY_END = 1.08  # arc length where the belly plates give way to tail scales
PLATE_STEP = 3  # stations per belly plate
SIDES = 18  # around the body loft
HIP_Y = 0.150  # where the thighs join the flanks
HIP_SWELL = 0.026


def body():
    rows = graded_path(SPINE, lambda s: 0.013 + 0.008 * sstep(0.95, 1.55, s))
    sp = Loft(rows, up0=(0, 0.6, 0.8))
    n = len(rows)
    total = sp.arc[-1]
    tail_mid = BELLY_END
    # The hips, where the flank swells to meet the thighs.
    hip = sp.arc[min((i for i in range(n) if sp.arc[i] < 1.2), key=lambda i: abs(sp.P[i].y - HIP_Y))]

    def sculpt(q, i, a):
        s = sp.arc[i]
        sn = math.sin(a)
        radial = nrm(q - sp.P[i])
        # The hips swell to meet the thighs, running back into the root of
        # the tail as a reptile's thigh muscles do.
        wv0 = sstep(0.50, 0.80, -sn)
        sig = 0.05 if s < hip else 0.10
        q = q + radial * HIP_SWELL * abs(math.cos(a)) ** 1.5 * (1 - wv0) * math.exp(-((s - hip) / sig) ** 2)
        # Dorsal ridge under the spines.
        q = q + sp.U[i] * 0.007 * max(0.0, sn) ** 10 * sstep(0.0, 0.05, s) * (1 - sstep(total - 0.25, total, s))
        # Belly plates: each band rises toward its rear edge and laps the next.
        wv = sstep(0.50, 0.80, -sn) * sstep(0.02, 0.06, s) * (1 - sstep(tail_mid * 0.85, tail_mid, s))
        f = (i % PLATE_STEP) / (PLATE_STEP - 1)
        if wv > 0:
            depth = 0.011 * min(1.0, sp.st[i][5] / 0.06)
            q = q + radial * wv * (0.002 + depth * f)
        # Rows of scales over the back and flanks, lapping toward the tail.
        q = q + radial * (1 - wv) * 0.0024 * f * min(1.0, sp.st[i][3] / 0.05) * sstep(0.03, 0.08, s)
        # Staggered scales on the flanks of the neck and tail: alternate
        # columns rise at the rear edge of each row, offset row by row.
        col = round(a / TAU * SIDES)
        row = i // PLATE_STEP
        flank = abs(math.cos(a)) ** 1.2 * (1 - wv) * (1 - sstep(0.7, 1.0, sn))
        zone = (1 - math.exp(-((s - 0.60) / 0.16) ** 2)) * (1 - math.exp(-((s - hip) / 0.06) ** 2))
        if (col + row) % 2 == 0:
            q = q + radial * 0.0045 * flank * zone * f * min(1.0, sp.st[i][3] / 0.04)
        return q

    def role(i, a):
        s = sp.arc[min(int(i), n - 1)]
        if -math.sin(a) > 0.62 and 0.025 < s < tail_mid:
            return "skin"
        return "body"

    obj = sp.build(SIDES, sculpt=sculpt, role_fn=role, name="body")
    return obj, sp


def spines(sp):
    parts = []
    total = sp.arc[-1]
    s = 0.07
    while s < total - 0.03:
        i = sp.at(s)
        # Tallest over the withers, small up the neck and down the tail.
        h = 0.052 * (0.45 + 0.55 * math.exp(-((s - 0.52) / 0.22) ** 2)) * (1 - 0.55 * sstep(0.68, total, s))
        h = max(h, 0.010)
        base = sp.surface(i, math.pi / 2) - sp.U[i] * 0.006
        parts.append(fin(base, sp.T[i], sp.U[i], sp.L[i], h, h * 0.85, h * 0.16, hook=0.5, rows=3 if h > 0.026 else 2,
                         name="spine"))
        s += max(0.024, 0.7 * h) * (1 + 0.5 * sstep(1.0, 1.4, s))
    return parts


def scutes(sp):
    """Two rows of keeled scales flanking the spines over the back and the
    root of the tail: each laps the next toward the tail, its free rear edge
    lifted, so the rows read as continuous armour."""
    parts = []
    s = 0.32
    k = 0
    while s < 1.04:
        i = sp.at(s)
        size = 0.020 + 0.011 * min(1.0, sp.st[i][3] / 0.09)
        for side in (-1, 1):
            a = math.pi / 2 + side * 0.64
            p = sp.surface(i, a)
            n = nrm(p - sp.P[i])
            t = nrm(sp.T[i] - n * sp.T[i].dot(n))
            c = n.cross(t)
            sink = n * 0.0025
            verts = [p - t * size * 0.55 - sink, p + c * size * 0.46 - t * size * 0.05 - sink,
                     p + t * size * 0.55 + n * size * 0.16, p - c * size * 0.46 - t * size * 0.05 - sink,
                     p + t * size * 0.16 + n * size * 0.22]
            faces = [(0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0), (1, 2, 3)]
            parts.append(mesh_object("scute", verts, faces, "body"))
        s += size * 0.80
        k += 1
    return parts


def spade(sp):
    """A flat arrowhead spade at the tail tip, tilted toward the viewer."""
    i = len(sp.P) - 1
    T = sp.T[i]
    # Lay the blade nearly flat, open to the board camera, which looks down
    # from the front at 40 degrees: from the side it shows as a slim
    # arrowhead, not a fin.
    view = nrm(V(0.22, -0.30, 0.93))
    N = nrm(view - T * T.dot(view))
    Lt = N.cross(T)
    m = Matrix((T, N, Lt)).transposed().to_4x4()
    m.translation = sp.P[i] - T * 0.012
    k = 1.25
    half = [(0.072, 0.022), (0.038, 0.043), (0.008, 0.053), (-0.040, 0.042), (-0.052, 0.030), (-0.012, 0.026),
            (-0.002, 0.010)]
    out = [(0.130 * k, 0.0)] + [(u * k, v * k) for u, v in half] + [(u * k, -v * k) for u, v in reversed(half)]
    blade = leaf(out, 0.0080, (0.030 * k, 0.0), m, per=2, rings=2, name="spade")
    rib = taper([sp.P[i] - T * 0.014, sp.P[i] + T * 0.07, sp.P[i] + T * 0.140], [0.0090, 0.0072, 0.0015],
                sides=6, step=0.014, role="trim", name="spade_rib")
    return [blade, rib]


# ---------------------------------------------------------------- legs


def toes(foot, fwd, n, length, curl, r, splay=0.5, name="toe"):
    """`n` thick toes fanning from `foot` toward `fwd`, curled down, each
    ending in a hooked claw."""
    parts = []
    fwd = nrm(fwd)
    lat = nrm(fwd.cross(V(0, 0, 1)))
    down = nrm(fwd.cross(lat))
    if down.z > 0:
        down = -down
    for j in range(n):
        k = (j - (n - 1) / 2) / max(1, (n - 1) / 2)
        d = nrm(fwd + lat * k * splay)
        ln = length * (1 - 0.18 * abs(k))
        b = foot + lat * k * r * 1.3
        m = b + d * ln * 0.55 + down * ln * 0.12 * curl
        t = m + d * ln * 0.35 + down * ln * 0.40 * curl
        parts.append(taper([b, m, t], [r, r * 0.88, r * 0.72], sides=6, step=0.018, name=name))
        c1 = t + nrm(d * 0.5 + down) * r * 1.5
        c2 = c1 + nrm(down - d * 0.7) * r * 1.5
        parts.append(claw([t - d * r * 0.3, c1, c2], [r * 0.72, r * 0.45, r * 0.04]))
    return parts


def segment(ctrl, front=(0, -1, 0), sides=10, step=0.011, role="body", name="limb", power=2.0, caps=(0.6, 0.6)):
    """A limb or a mass of one: a smooth loft through rows of (x, y, z, half
    width, front depth, back depth); `front` aims the front."""
    rows = smooth_path(ctrl, step, per=8)
    lo = Loft(rows, up0=front)
    return lo.build(sides, role_fn=lambda i, a: role, power=power, caps=caps, name=name)


def legs():
    """Forelegs reach forward with the claws half open; hind legs trail
    back. Each leg is one continuous loft from the body to the toes, its
    sections swelling over the muscles and the joints (shoulder and biceps,
    elbow point, forearm; haunch, knee, calf, hock) and slimming between,
    with its root sunk into the flank."""
    parts = []
    for s in (-1, 1):
        def P(x, y, z, w, f, b):
            return (s * x, y, z, w, f, b)
        # Shoulder, biceps, elbow point, forearm swell, wrist, hand.
        parts.append(segment([P(0.022, -0.098, 0.556, 0.036, 0.046, 0.044), P(0.062, -0.108, 0.512, 0.042, 0.050, 0.046),
                              P(0.088, -0.100, 0.466, 0.032, 0.036, 0.036), P(0.100, -0.084, 0.434, 0.030, 0.024, 0.038),
                              P(0.104, -0.094, 0.411, 0.026, 0.029, 0.024), P(0.100, -0.122, 0.395, 0.017, 0.020, 0.018),
                              P(0.094, -0.146, 0.386, 0.016, 0.015, 0.015), P(0.091, -0.162, 0.380, 0.022, 0.013, 0.013),
                              P(0.090, -0.174, 0.377, 0.022, 0.012, 0.012)],
                             sides=10, step=0.015, caps=(0.3, 0.5), name="foreleg"))
        parts += toes(V(s * 0.090, -0.174, 0.377), V(0, -1, -0.4), 3, 0.054, 1.0, 0.0105, name="toe")
        # Haunch sunk into the flank, thigh, knee, calf swell behind the
        # shank, hock, metatarsus and foot.
        parts.append(segment([P(0.012, 0.150, 0.535, 0.042, 0.062, 0.060), P(0.072, 0.140, 0.500, 0.048, 0.064, 0.062),
                              P(0.096, 0.116, 0.458, 0.042, 0.050, 0.050), P(0.113, 0.092, 0.424, 0.030, 0.030, 0.031),
                              P(0.118, 0.110, 0.402, 0.027, 0.024, 0.033), P(0.118, 0.160, 0.386, 0.021, 0.018, 0.022),
                              P(0.116, 0.204, 0.375, 0.019, 0.018, 0.018), P(0.114, 0.230, 0.368, 0.021, 0.014, 0.015),
                              P(0.112, 0.254, 0.362, 0.023, 0.013, 0.013)],
                             sides=10, step=0.016, caps=(0.3, 0.5), name="hindleg"))
        parts += toes(V(s * 0.112, 0.256, 0.361), V(0, 1, -0.45), 3, 0.050, 1.0, 0.011, name="toe")
    return parts


# ---------------------------------------------------------------- head

# Head frame: origin at the neck joint behind the skull, -y forward, z up.
HEAD_POS = V(SPINE[0][:3])
HEAD_PITCH = -0.10
HEAD_SCALE = 1.10
HEAD_M = Matrix.Translation(HEAD_POS) @ Matrix.Rotation(HEAD_PITCH, 4, "X") @ Matrix.Scale(HEAD_SCALE, 4)
JAW_OPEN = 0.40
EYE = (0.037, -0.060, 0.021)
JAW_HINGE = V(0, 0.014, -0.022)
JAW_M = Matrix.Translation(JAW_HINGE) @ Matrix.Rotation(JAW_OPEN, 4, "X") @ Matrix.Translation(-JAW_HINGE)

# y, z of the axis, half width, dorsal height, ventral depth
SKULL = [
    (0.042, 0.008, 0.016, 0.016, 0.016),
    (0.032, 0.008, 0.035, 0.033, 0.030),
    (0.012, 0.006, 0.048, 0.044, 0.032),
    (-0.016, 0.004, 0.054, 0.046, 0.030),
    (-0.046, 0.001, 0.048, 0.040, 0.027),
    (-0.082, -0.004, 0.038, 0.031, 0.024),
    (-0.122, -0.008, 0.031, 0.024, 0.021),
    (-0.156, -0.010, 0.026, 0.019, 0.019),
    (-0.181, -0.011, 0.020, 0.015, 0.016),
    (-0.190, -0.011, 0.015, 0.011, 0.012),
    (-0.199, -0.011, 0.007, 0.006, 0.008),
]
JAW = [
    (0.030, -0.028, 0.030, 0.014, 0.024),
    (0.010, -0.032, 0.054, 0.018, 0.038),
    (-0.040, -0.036, 0.048, 0.015, 0.032),
    (-0.100, -0.040, 0.039, 0.014, 0.026),
    (-0.148, -0.042, 0.031, 0.013, 0.026),
    (-0.172, -0.043, 0.026, 0.012, 0.029),
    (-0.188, -0.042, 0.018, 0.009, 0.019),
    (-0.198, -0.041, 0.008, 0.005, 0.008),
]


EYE_R = 0.016
EYE_SCALE = (0.65, 1.40, 0.55)


def EYE_ROT(s):
    return (0.30, 0, s * 0.32)


def pupil(s, turn=-0.30, half_width=0.075, half_height=0.78, depth=0.07, rows=7):
    """A slit pupil wrapped over the outer face of the eye: built in the
    eye's unit-sphere space (angles across and up from the outward pole,
    radius about 1) and carried through the eye's own scale and rotation."""
    verts, faces = [], []
    for k in range(rows):
        v = -1 + 2 * k / (rows - 1)
        if k in (0, rows - 1):
            th = v * half_height
            verts.append(V(s * math.cos(th) * math.cos(turn), math.cos(th) * math.sin(turn), math.sin(th)))
            continue
        env = (1 - v * v) ** 0.6
        for j in range(4):
            phi = TAU * j / 4
            r = 1 + depth * env * math.cos(phi) * 0.5 + depth * 0.5 * env
            ty = turn + half_width * env * math.sin(phi)
            th = v * half_height
            verts.append(V(s * r * math.cos(th) * math.cos(ty), r * math.cos(th) * math.sin(ty), r * math.sin(th)))
    last = len(verts) - 1
    for j in range(4):
        faces.append((0, 1 + (j + 1) % 4, 1 + j))
    for k in range(rows - 3):
        a, b = 1 + 4 * k, 1 + 4 * (k + 1)
        for j in range(4):
            faces.append((a + j, a + (j + 1) % 4, b + (j + 1) % 4, b + j))
    b = 1 + 4 * (rows - 3)
    for j in range(4):
        faces.append((b + j, b + (j + 1) % 4, last))
    m = Matrix.LocRotScale(V(s * EYE[0], EYE[1], EYE[2]), Euler(EYE_ROT(s)), V(EYE_SCALE) * EYE_R)
    return put(mesh_object("pupil", verts, faces, "body"), m)


def table(rows, step=0.0105, fine=0.0055, tip=0.040):
    """Loft stations along local -y from (y, z, w, up, down) rows, closer
    together toward the tip so the snout rounds off."""
    pts = [(0.0, y, z, w, u, d) for y, z, w, u, d in rows]
    length = sum(math.dist(a[:3], b[:3]) for a, b in zip(pts, pts[1:]))
    return graded_path(pts, lambda s: step - (step - fine) * sstep(length - tip, length - 0.010, s), per=10)


def lookup(rows, y):
    """Interpolated (z, w, up, down) at y."""
    for a, b in zip(rows, rows[1:]):
        if b[0] <= y <= a[0]:
            t = (a[0] - y) / (a[0] - b[0])
            return [a[k] + (b[k] - a[k]) * t for k in range(1, 5)]
    return list(rows[0][1:] if y > rows[0][0] else rows[-1][1:])


def head():
    parts = []
    sk = Loft(table(SKULL), up0=(0, 0, 1))

    def skull_sculpt(q, i, a):
        y = sk.P[i].y
        sn = math.sin(a)
        # A flat brow narrowing to the crown, a ridge down the bridge of the
        # snout, a brow shelf over the eyes and flared cheeks below them.
        if sn > 0:
            q.x *= 1 - 0.20 * sn ** 1.5
        q = q + sk.U[i] * 0.004 * max(0.0, sn) ** 12 * sstep(-0.02, -0.08, y)
        brow = math.exp(-((y + 0.035) / 0.035) ** 2) * math.exp(-((a % math.pi - (0.95 if math.cos(a) > 0 else 2.19)) / 0.35) ** 2)
        q = q + V(math.copysign(0.004, q.x), 0, 0.005) * brow
        cheek = math.exp(-((y + 0.005) / 0.03) ** 2) * max(0.0, -sn + 0.3) * abs(math.cos(a))
        q.x += math.copysign(0.009 * cheek, q.x)
        # Eye sockets: a hollow under the brow that the baked AO darkens, so
        # the glowing eye sits in shadow.
        sock = math.exp(-((q.y - EYE[1]) / 0.024) ** 2 - ((q.z - EYE[2] + 0.002) / 0.013) ** 2) * sstep(0.015, 0.03, abs(q.x))
        q.x -= math.copysign(0.0065 * sock, q.x)
        return q

    def skull_role(i, a):
        y = sk.P[min(int(i), len(sk.P) - 1)].y
        return "skin" if -math.sin(a) > 0.55 and -0.185 < y < -0.004 else "body"

    parts.append(sk.build(28, sculpt=skull_sculpt, role_fn=skull_role, power=2.6, caps=(0.4, 0.3), name="skull"))
    jw = Loft(table(JAW), up0=(0, 0, 1))

    def jaw_role(i, a):
        # The floor of the mouth glows with the fire welling up the throat.
        y = jw.P[min(int(i), len(jw.P) - 1)].y
        if math.sin(a) > 0.74 and -0.160 < y < -0.020:
            return "trim" if y < -0.115 else "eye"
        return "skin" if math.sin(a) > 0.5 and -0.18 < y < 0.0 else "body"

    def jaw_sculpt(q, i, a):
        # A trough between the gums, where the fire pools.
        y = jw.P[i].y
        sn, cs = math.sin(a), abs(math.cos(a))
        if sn > 0:
            k = sstep(-0.004, -0.030, y) * sstep(-0.192, -0.160, y)
            q.z -= jw.st[i][4] * 0.45 * k * max(0.0, 1 - cs / 0.78) ** 1.5
        return q

    jaw = [jw.build(22, sculpt=jaw_sculpt, role_fn=jaw_role, power=2.4, caps=(0.4, 0.3), name="jaw")]
    # Fire in the gullet, deep in the throat behind the cheek webs.
    parts.append(lib.sphere(0.016, loc=(0, -0.005, -0.030), scale=(1.0, 1.3, 0.8), role="eye",
                            segments=8, rings=5, name="gullet"))
    # Teeth: upper row with long fangs, lower row hinged with the jaw.
    for s in (-1, 1):
        for y, ln, r in ((-0.186, 0.010, 0.0032), (-0.164, 0.042, 0.0080), (-0.140, 0.012, 0.0036),
                         (-0.116, 0.013, 0.0036), (-0.092, 0.012, 0.0034), (-0.070, 0.010, 0.0032)):
            z, w, up, dn = lookup(SKULL, y)
            base = V(s * w * 0.66, y, z - dn * 0.80)
            fang = ln > 0.03
            parts.append(cone(base, (0, 0.12, -1), ln, r, role="trim" if fang else "metal", segments=6 if fang else 4,
                              name="fang" if fang else "tooth"))
        for y, ln, r in ((-0.176, 0.034, 0.0062), (-0.150, 0.011, 0.0034), (-0.124, 0.012, 0.0034),
                         (-0.098, 0.011, 0.0032), (-0.074, 0.009, 0.0030)):
            z, w, up, dn = lookup(JAW, y)
            base = V(s * w * 0.64, y, z + up * 0.75)
            fang = ln > 0.03
            jaw.append(cone(base, (0, -0.10, 1), ln, r, role="trim" if fang else "metal", segments=6 if fang else 4,
                            name="fang" if fang else "tooth"))
        # Jaw spikes along the underside, swept back.
        for y, ln in ((-0.010, 0.030), (0.020, 0.024)):
            z, w, up, dn = lookup(JAW, y)
            jaw.append(claw([(s * w * 0.7, y, z - dn * 0.5), (s * w * 0.95, y + ln * 0.6, z - dn - ln * 0.35),
                             (s * w * 1.1, y + ln * 1.1, z - dn - ln * 0.45)], [0.0065, 0.004, 0.0003], role="body",
                            sides=4, name="jaw_spike"))
    parts += [put(p, JAW_M) for p in jaw]
    # Cheek webs close the corners of the open mouth.
    for s in (-1, 1):
        def srim(y):
            z, w, up, dn = lookup(SKULL, y)
            return V(s * w * 0.86, y, z - dn * 0.62)

        def jrim(y):
            z, w, up, dn = lookup(JAW, y)
            return JAW_M @ V(s * w * 0.86, y, z + up * 0.55)
        A, B, H = srim(-0.078), jrim(-0.040), srim(0.022)
        sag = nrm(H - A.lerp(B, 0.5)) * (A - B).length * 0.22

        def cheek(u, v):
            p = srim(0.022 - 0.100 * u).lerp(jrim(0.022 - 0.062 * u), v)
            return p + sag * math.sin(math.pi * v) * u * u
        parts.append(grid(cheek, 6, 4, "cheek_web", role="skin", thick=0.003))
    for s in (-1, 1):
        # Eyes under a heavy brow, slanted down toward the snout.
        parts.append(lib.sphere(0.016, loc=(s * EYE[0], EYE[1], EYE[2]), rot=EYE_ROT(s), scale=EYE_SCALE, role="eye",
                                segments=12, rings=7, name="eye"))
        parts.append(pupil(s))
        parts.append(taper([(s * 0.018, -0.100, 0.024), (s * 0.033, -0.074, 0.034), (s * 0.043, -0.046, 0.037),
                            (s * 0.044, -0.014, 0.036), (s * 0.036, 0.014, 0.032)],
                           [0.003, 0.0058, 0.007, 0.0062, 0.0035], sides=10, step=0.010, name="brow"))
        # Great horns sweeping back from the brow, ridged; a lesser pair at the cheeks.
        parts.append(taper([(s * 0.028, 0.000, 0.040), (s * 0.046, 0.046, 0.054), (s * 0.063, 0.098, 0.051),
                            (s * 0.075, 0.148, 0.043), (s * 0.084, 0.192, 0.038), (s * 0.088, 0.228, 0.046)],
                           [0.0195, 0.0168, 0.0132, 0.0095, 0.0055, 0.0008], sides=12, step=0.015, role="trim",
                           rings=0.026, name="horn"))
        parts.append(taper([(s * 0.048, 0.006, 0.006), (s * 0.064, 0.048, -0.002), (s * 0.075, 0.088, -0.004),
                            (s * 0.081, 0.118, 0.006)], [0.010, 0.0075, 0.0045, 0.0006], sides=9, step=0.016,
                           role="trim", name="cheek_horn"))
        # Nostrils flared out at the corners of the snout tip.
        parts.append(lib.sphere(0.0040, loc=(s * 0.012, -0.186, 0.002), rot=(0.2, 0, -s * 0.5), scale=(1.6, 1.1, 0.7),
                                role="body", segments=6, rings=4, name="nostril"))
    parts.append(taper([(0, -0.146, 0.012), (0, -0.154, 0.030), (0, -0.146, 0.043)], [0.0085, 0.0055, 0.0005],
                       sides=8, step=0.008, name="nose_horn"))
    return [put(p, HEAD_M) for p in parts]


# ---------------------------------------------------------------- wings


def wing(s):
    def P(x, y, z):
        return V(s * x, y, z)

    S, E, W = P(0.060, -0.070, 0.610), P(0.165, 0.100, 0.838), P(0.290, 0.022, 1.040)
    tips = [P(0.352, 0.025, 1.204),P(0.428, 0.195, 1.095), P(0.432, 0.365, 0.880), P(0.358, 0.485, 0.665)]
    H = P(0.070, 0.200, 0.545)
    bows = [P(0.004, 0.012, 0.010), P(0.012, 0.018, 0.016), P(0.016, 0.022, 0.012), P(0.012, 0.024, 0.0)]
    fingers = [curve([W, W.lerp(t, 0.5) + b, t], 10) for t, b in zip(tips, bows)]
    arm = curve([S, S.lerp(E, 0.5) + V(0, 0, 0.012), E, E.lerp(W, 0.5) + V(0, -0.010, 0.0), W], 6)
    out = nrm((tips[1] - W).cross(tips[3] - W))
    if out.x * s < 0:
        out = -out
    # The membrane stops short of the finger tips, which jut out as spikes.
    REACH = 0.93
    SPLIT = 0.86  # where the bone becomes the trim spike
    BAND = 0.011  # width of the trim band along the trailing edges
    mtips = [polyline_at(f, REACH) for f in fingers]
    flen = [sum((b - a).length for a, b in zip(f, f[1:])) for f in fingers]

    def banded(n, width):
        """n even steps to 1 - width, then one thin step to the edge."""
        return [(1 - width) * i / n for i in range(n + 1)] + [1.0]

    parts = []

    def fan(f0, f1, t0, t1, length, nu=6, nv=5):
        inward = nrm(W - t0.lerp(t1, 0.5))
        depth = (t1 - t0).length * 0.36

        def fn(u, v):
            a, b = polyline_at(f0, u * REACH), polyline_at(f1, u * REACH)
            p = a.lerp(b, v)
            p = p + inward * depth * math.sin(math.pi * v) * u ** 1.6
            p = p + out * 0.016 * math.sin(math.pi * v) * math.sin(math.pi * u * 0.9)
            # Shallow folds running out from the wrist between the fingers.
            p = p + out * 0.0045 * math.sin(math.pi * v * 2) * u ** 1.3
            return p
        return grid(fn, nu, nv, "membrane", thick=0.005, us=banded(nu, BAND / (length * REACH)),
                    role_fn=lambda i, j: "trim" if i == nu else "cloth")

    for k in range(3):
        parts.append(fan(fingers[k], fingers[k + 1], mtips[k], mtips[k + 1], (flen[k] + flen[k + 1]) / 2))
    # Inner membrane: from the arm and last finger back to the flank.
    trail_in = nrm((S.lerp(W, 0.5)) - H.lerp(mtips[3], 0.5))
    tdepth = (mtips[3] - H).length * 0.24

    def inner(u, v):
        # u from the body (0) to the last finger (1); v from the arm (0) to the trailing edge (1).
        a = polyline_at(arm, u)
        tr = H.lerp(mtips[3], u) + trail_in * tdepth * math.sin(math.pi * u)
        root = S.lerp(H, v)
        tipf = polyline_at(fingers[3], v * REACH)
        corners = (S * (1 - u) * (1 - v) + W * u * (1 - v) + H * (1 - u) * v + mtips[3] * u * v)
        p = a * (1 - v) + tr * v + root * (1 - u) + tipf * u - corners
        p = p + out * 0.020 * math.sin(math.pi * u) * math.sin(math.pi * v)
        return p + out * 0.005 * math.sin(math.pi * u * 3) * math.sin(math.pi * v) * v
    parts.append(grid(inner, 9, 6, "membrane", thick=0.005, vs=banded(6, BAND / (mtips[3] - H).length * 1.5),
                      role_fn=lambda i, j: "trim" if j == 6 else "cloth"))

    # Leading membrane in front of the elbow.
    def front(u, v):
        a = polyline_at(arm, u)
        lead = S.lerp(W, u) + nrm(E - S.lerp(W, 0.5)) * 0.018 * math.sin(math.pi * u)
        return a.lerp(lead, v)
    parts.append(grid(front, 6, 2, "membrane", thick=0.005))

    # Bones.
    # One arm bone from the shoulder through a knobbed elbow to the wrist.
    eu, ef = nrm(E - S), nrm(W - E)
    parts.append(taper([S - P(0.030, 0, 0.012), S, S.lerp(E, 0.5) + V(0, 0, 0.012), E - eu * 0.030, E,
                        E + ef * 0.030, E.lerp(W, 0.5) + V(0, -0.010, 0), W - ef * 0.02, W],
                       [0.034, 0.030, 0.0215, 0.0155, 0.0165, 0.0135, 0.0115, 0.0128, 0.0135],
                       sides=8, step=0.034, name="arm"))
    parts.append(lib.sphere(0.0145, loc=W, role="body", segments=8, rings=5, name="wrist"))
    for f in fingers:
        # Bone to past the membrane edge, then an ember spike; an ember knuckle between.
        pts = [polyline_at(f, t) for t in (0.0, 0.3, 0.55, 0.75, SPLIT)]
        parts.append(taper(pts, [0.0115, 0.0085, 0.0090, 0.0072, 0.0060], sides=6, step=0.05, name="finger"))
        pts = [polyline_at(f, t) for t in (SPLIT - 0.012, 0.93, 1.0)]
        parts.append(taper(pts, [0.0064, 0.0045, 0.0008], sides=6, step=0.03, role="trim", name="finger_tip"))
        parts.append(lib.sphere(0.0108, loc=polyline_at(f, 0.55), role="trim", segments=6, rings=3, name="knuckle"))
    # Thumb claw at the wrist and a spur on the elbow.
    parts.append(claw([W, W + P(0.006, -0.014, 0.030), W + P(0.010, -0.034, 0.048)], [0.0105, 0.0065, 0.0004],
                      role="trim", sides=6, name="thumb"))
    parts.append(claw([E, E + P(0.014, 0.034, -0.008), E + P(0.018, 0.060, -0.030)], [0.011, 0.006, 0.0004],
                      role="trim", sides=5, name="elbow_spur"))
    return parts


# ---------------------------------------------------------------- figure


def build():
    groups = {}
    b, sp = body()
    groups["body"] = [b]
    groups["spines"] = spines(sp) + scutes(sp)
    groups["spade"] = spade(sp)
    groups["legs"] = legs()
    groups["head"] = head()
    groups["wings"] = wing(-1) + wing(1)
    parts = []
    tally = []
    for k, v in groups.items():
        parts += v
        tally.append(f"{k}={sum(lib.tris(p) for p in v)}")
    if DEBUG:
        names = {}
        for p in parts:
            key = p.name.split(".")[0]
            names[key] = names.get(key, 0) + lib.tris(p)
        print("REPORTNAMES " + " ".join(f"{k}={v}" for k, v in sorted(names.items(), key=lambda kv: -kv[1])))
    print("REPORTPARTS " + " ".join(tally))
    return parts
