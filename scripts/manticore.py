"""Manticore: a Dark troop. A prowling lion with a great shaggy mane, a
snarling face with a hint of human menace, leathery bat wings half raised and
a segmented scorpion tail hooked over its haunches, holding up a stinger
bristling with the spikes it shoots. It faces -Y; its right side is on -X.

Technique notes: the torso and limbs are superelliptic sweeps (`sweep`,
`limb`) rather than lib.skin, for exact control of width, depth and the
chest-to-waist profile; the wing membranes are quad grids so they can sag
and scallop; the mane is one lathe hood carved by `mane_sculpt` into grooved
strands, with a fringe of clumps that follow the strands past its edge. FINISH
raises the sharp-edge angle so these low-poly organic forms shade smooth."""
import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

import lib

TAU = math.tau
# Hair, hide and the metaball face are low-poly but organic: only creases
# sharper than this angle shade hard.
FINISH = {"sharp_angle": 80}


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a)


def orient(obj, origin, axis, up=None):
    """Bakes a transform taking local +z along `axis` and the local origin to
    `origin`; with `up`, local +y turns toward it."""
    z = Vector(axis).normalized()
    if up is None:
        q = z.to_track_quat("Z", "Y")
        rot = q.to_matrix()
    else:
        y = Vector(up) - z * Vector(up).dot(z)
        y.normalize()
        x = y.cross(z)
        rot = Matrix((x, y, z)).transposed()
    m = Matrix.Translation(Vector(origin)) @ rot.to_4x4()
    obj.data.transform(m)
    obj.data.update()
    return obj


def catmull(points, per=8):
    pts = [Vector(p) for p in points]
    ext = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(per):
            t = s / per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(pts[-1])
    return out


def resample(path, n):
    """`n` points evenly spaced by arc length along a polyline."""
    d = [0.0]
    for a, b in zip(path, path[1:]):
        d.append(d[-1] + (b - a).length)
    total = d[-1]
    out, j = [], 0
    for i in range(n):
        s = total * i / (n - 1)
        while j < len(d) - 2 and d[j + 1] < s:
            j += 1
        t = (s - d[j]) / max(d[j + 1] - d[j], 1e-9)
        out.append(path[j].lerp(path[j + 1], t))
    return out


def at(path, s):
    """The point at fraction `s` of a polyline's length."""
    d = [0.0]
    for a, b in zip(path, path[1:]):
        d.append(d[-1] + (b - a).length)
    target = d[-1] * min(max(s, 0.0), 1.0)
    for i in range(len(path) - 1):
        if d[i + 1] >= target:
            f = (target - d[i]) / max(d[i + 1] - d[i], 1e-9)
            return path[i].lerp(path[i + 1], f)
    return path[-1]


def interp(values, t):
    t = min(max(t, 0.0), 1.0) * (len(values) - 1)
    i = min(int(t), len(values) - 2)
    f = t - i
    return values[i] * (1 - f) + values[i + 1] * f


def sweep(path, widths, ups, downs, ref=(0, 0, 1), sides=24, power=2.0, role="body", name="sweep", cap=0.6):
    """Superelliptic sections along `path`: `widths` across, `ups` toward the
    `ref` hint and `downs` away from it. Both ends close in a domed fan."""
    bm = bmesh.new()
    path = [Vector(p) for p in path]
    n = len(path)
    e = 2.0 / power
    rings, tangents = [], []
    for i, p in enumerate(path):
        t = (path[min(i + 1, n - 1)] - path[max(i - 1, 0)]).normalized()
        r = Vector(ref(i, p) if callable(ref) else ref)
        u = r - t * r.dot(t)
        if u.length < 1e-6:
            u = t.orthogonal()
        u.normalize()
        lat = t.cross(u)
        ring = []
        for k in range(sides):
            a = TAU * k / sides
            c, s = math.cos(a), math.sin(a)
            x = widths[i] * math.copysign(abs(c) ** e, c)
            h = (ups[i] if s >= 0 else downs[i]) * math.copysign(abs(s) ** e, s)
            ring.append(bm.verts.new(p + lat * x + u * h))
        rings.append(ring)
        tangents.append(t)
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(sides):
            k1 = (k + 1) % sides
            bm.faces.new((r0[k], r0[k1], r1[k1], r1[k]))
    for ring, t, sign in ((rings[0], tangents[0], -1), (rings[-1], tangents[-1], 1)):
        centre = sum((v.co for v in ring), Vector()) / len(ring)
        rad = sum((v.co - centre).length for v in ring) / len(ring)
        c = bm.verts.new(centre + t * sign * rad * cap)
        for k in range(sides):
            bm.faces.new((ring[k], ring[(k + 1) % sides], c))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._from_bmesh(name, bm, role)


def limb(ctrl, w, front, back, n=12, ref=(0, -1, 0), sides=12, role="body", name="limb", per=8, sub=0, power=2.0):
    """A limb swept through control points; w/front/back are given per control
    point and interpolated along it."""
    path = resample(catmull(ctrl, per), n)
    ws = [interp(w, i / (n - 1)) for i in range(n)]
    fs = [interp(front, i / (n - 1)) for i in range(n)]
    bs = [interp(back, i / (n - 1)) for i in range(n)]
    obj = sweep(path, ws, fs, bs, ref=ref, sides=sides, role=role, name=name, power=power)
    return lib.subdivide(obj, sub) if sub else obj


def symmetric(obj):
    """Makes `obj` exactly symmetric by mirroring its right (-x) half, so the
    polygonizer and decimation leave no lopsided dents."""
    lib.place(obj, scale=(-1, 1, 1))
    obj = lib.mirror(obj)
    return lib.place(obj, scale=(-1, 1, 1))


def mesh_object(name, verts, faces, role):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, role)


def spike(base, tip, r, role="trim", segments=8, name="spike"):
    """A straight conical spike from `base` to `tip`, slightly swollen."""
    base, tip = Vector(base), Vector(tip)
    d = tip - base
    L = d.length
    prof = [(0, -r * 0.4), (r, 0.0), (r * 0.82, L * 0.34), (r * 0.32, L * 0.78), (0, L)]
    s = lib.lathe(prof, role=role, segments=segments, name=name)
    return orient(s, base, d)


def claw(base, direction, length, r, role="metal", curl=0.5, name="claw", sides=6):
    """A hooked claw or fang, bending downward along its length."""
    base, d = Vector(base), Vector(direction).normalized()
    down = Vector((0, 0, -1))
    side = d.cross(down)
    if side.length < 1e-4:
        side = Vector((1, 0, 0))
    bend = side.cross(d).normalized()
    if bend.dot(down) < 0:
        bend = -bend
    pts = [base + d * length * f + bend * length * curl * f * f * 0.5 for f in (0, 0.5, 1.0)]
    return lib.tube(pts, [r, r * 0.6, r * 0.05], role=role, sides=sides, resolution=2, name=name)


def bump(centre, radius, amount, axis_fn):
    """A gaussian swelling pushed away from an axis point."""
    c = Vector(centre)

    def fn(v):
        d2 = (v - c).length_squared / (radius * radius)
        if d2 > 9:
            return v
        out = v - axis_fn(v)
        if out.length < 1e-9:
            return v
        return v + out.normalized() * amount * math.exp(-d2)
    return fn


def lock(root, d0, d1, length, width, face, thick=0.4, flat=0.35, n=7, sides=10, taper=0.6, tip=0.16,
         role="cloth", name="lock", soft=0):
    """A clump of hair: a sweep domed on its outer side (toward `face`) and
    nearly flat beneath, so clumps lie on each other like shingles; broadest
    near its root, thinning and rounding off to a soft tip, leaving `root`
    along `d0` and curling toward `d1`."""
    root = Vector(root)
    d0, d1 = Vector(d0).normalized(), Vector(d1).normalized()
    p1 = root + d0 * length * 0.55
    p2 = p1 + d1 * length * 0.5
    pts, ws, up, down = [], [], [], []
    for i in range(n):
        t = i / (n - 1)
        pts.append(root * (1 - t) ** 2 + p1 * (2 * (1 - t) * t) + p2 * (t * t))
        w = max(width * (1 - t) ** taper * (1 + 0.25 * math.sin(math.pi * t)), width * tip)
        ws.append(w)
        up.append(w * thick * (1 - 0.45 * t))
        down.append(w * thick * flat * (1 - 0.45 * t))
    obj = sweep(pts, ws, up, down, ref=face, sides=sides, role=role, name=name, cap=0.45)
    # `soft` relaxes the crease between the domed top and the flat underside,
    # so small clumps read as fur rather than chipped stone.
    return lib.smooth(obj, soft, 0.5) if soft else obj


# ---------------------------------------------------------------- torso

# Stations along the spine from rump to neck: y, centre z, half width, up, down.
TORSO = [
    (0.322, 0.318, 0.034, 0.030, 0.034),
    (0.298, 0.316, 0.066, 0.050, 0.066),
    (0.266, 0.310, 0.086, 0.062, 0.090),
    (0.218, 0.302, 0.094, 0.068, 0.104),
    (0.160, 0.294, 0.088, 0.064, 0.102),
    (0.100, 0.286, 0.074, 0.060, 0.086),
    (0.040, 0.280, 0.081, 0.062, 0.098),
    (-0.020, 0.272, 0.094, 0.068, 0.114),
    (-0.080, 0.266, 0.106, 0.078, 0.126),
    (-0.140, 0.265, 0.110, 0.09, 0.126),
    (-0.195, 0.270, 0.104, 0.092, 0.118),
    (-0.245, 0.284, 0.090, 0.086, 0.104),
    (-0.290, 0.304, 0.070, 0.072, 0.080),
    (-0.330, 0.326, 0.054, 0.058, 0.060),
    (-0.352, 0.334, 0.032, 0.034, 0.034),
]


def spine_z(y):
    for (y0, z0, *_), (y1, z1, *_) in zip(TORSO, TORSO[1:]):
        if y1 <= y <= y0:
            return z0 + (z1 - z0) * (y - y0) / (y1 - y0)
    return TORSO[0][1] if y > TORSO[0][0] else TORSO[-1][1]


def spine_axis(v):
    return Vector((0, v.y, spine_z(v.y)))


def smooth_stations(stations, n):
    """`n` stations resampled from `stations` along a Catmull-Rom spline through
    all their values, so the hide flows between them without kinks."""
    pts = [list(st) for st in stations]
    ext = [[2 * a - b for a, b in zip(pts[0], pts[1])]] + pts + [[2 * a - b for a, b in zip(pts[-1], pts[-2])]]
    segs = len(pts) - 1
    out = []
    for k in range(n):
        u = k / (n - 1) * segs
        i = min(int(u), segs - 1)
        t = u - i
        out.append(tuple(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (3 * b - a - 3 * c + d) * t ** 3)
                         for a, b, c, d in zip(ext[i], ext[i + 1], ext[i + 2], ext[i + 3])))
    return out


def torso():
    st = smooth_stations(TORSO, 30)
    path = [V(0, y, z) for y, z, *_ in st]
    obj = sweep(path, [q[2] for q in st], [q[3] for q in st], [q[4] for q in st],
                sides=30, power=2.2, name="torso", cap=0.5)
    # Shoulder blades riding high in the stalk, the haunches, the ribcage
    # behind the elbows, and a groove down the spine.
    shapes = []
    for s in (-1, 1):
        shapes += [
            bump((s * 0.064, -0.152, 0.35), 0.05, 0.04, spine_axis),
            bump((s * 0.08, 0.215, 0.30), 0.06, 0.014, spine_axis),
            bump((s * 0.1, -0.05, 0.22), 0.055, 0.009, spine_axis),
            bump((s * 0.1, -0.2, 0.27), 0.05, 0.012, spine_axis),
            # The back of the haunch, rounding off the notch over the thigh.
            bump((s * 0.07, 0.272, 0.285), 0.04, 0.012, spine_axis),
        ]

    def sculpt(v):
        for fn in shapes:
            v = fn(v)
        top = spine_z(v.y) + 0.045
        if v.z > top and abs(v.x) < 0.03 and -0.12 < v.y < 0.27:
            v.z -= 0.007 * (1 - abs(v.x) / 0.03) * min(1, (v.z - top) / 0.02)
        return v
    return lib.deform(obj, sculpt)


# ---------------------------------------------------------------- legs and paws


def paw(centre, claws=True, name="paw", scale=1.0):
    """A broad lion paw: a padded mass with four toes and hooked claws."""
    cx, cy, cz = centre
    k = scale
    parts = [lib.sphere(1.0, loc=(cx, cy + 0.006 * k, cz + 0.002), scale=(0.04 * k, 0.044 * k, 0.026 * k),
                        role="body", segments=12, rings=8, name=name)]
    for x in (-0.029, -0.0098, 0.0098, 0.029):
        y = cy - (0.03 + (0.007 if abs(x) < 0.015 else 0.0)) * k
        parts.append(lib.sphere(0.0165 * k, loc=(cx + x * k, y, cz - 0.004), scale=(1, 1.15, 0.85), role="body",
                                segments=7 if claws else 6, rings=4 if claws else 3, name="toe"))
        if claws:
            parts.append(claw((cx + x * k, y - 0.013 * k, cz - 0.002), (0, -1, -0.5), 0.022 * k, 0.005, name="paw_claw"))
    return parts


def legs():
    parts = []
    # Forelegs: the right reaches forward in the stalk, the left is planted.
    for side, reach in ((-1, 1.0), (1, 0.0)):
        sx = side
        ctrl = [
            (sx * 0.064, -0.150, 0.305),
            (sx * 0.084, -0.124, 0.208),
            (sx * 0.086, -0.108 - 0.03 * reach, 0.150),
            (sx * 0.080, -0.150 - 0.05 * reach, 0.090),
            (sx * 0.076, -0.195 - 0.06 * reach, 0.046),
            (sx * 0.074, -0.214 - 0.06 * reach, 0.030),
        ]
        # Heavy lion forearms: about a fifth thicker than a panther's.
        k = 1.22
        w = [v * k for v in (0.06, 0.057, 0.047, 0.041, 0.033, 0.034)]
        front = [v * k for v in (0.066, 0.058, 0.045, 0.041, 0.03, 0.032)]
        back = [v * k for v in (0.066, 0.063, 0.058, 0.038, 0.03, 0.032)]
        parts.append(limb(ctrl, w, front, back, n=13, sides=16, name="foreleg"))
        parts += paw((sx * 0.077, -0.245 - 0.06 * reach, 0.024), name="forepaw", scale=1.2)
    # Hind legs, coiled under the haunches.
    for side, reach in ((-1, 0.0), (1, 1.0)):
        sx = side
        ctrl = [
            (sx * 0.062, 0.200, 0.322),
            (sx * 0.088, 0.168, 0.250),
            (sx * 0.090, 0.108 - 0.01 * reach, 0.170),
            (sx * 0.084, 0.180, 0.118),
            (sx * 0.078, 0.262 + 0.01 * reach, 0.088),
            (sx * 0.074, 0.245, 0.045),
            (sx * 0.072, 0.232, 0.030),
        ]
        w = [0.058, 0.068, 0.044, 0.034, 0.024, 0.026, 0.028]
        front = [0.072, 0.078, 0.046, 0.034, 0.022, 0.024, 0.026]
        back = [0.072, 0.084, 0.048, 0.036, 0.032, 0.024, 0.026]
        parts.append(limb(ctrl, w, front, back, n=14, sides=16, name="hindleg"))
        parts += paw((sx * 0.073, 0.212, 0.022), claws=False, name="hindpaw")
    return parts


# ---------------------------------------------------------------- head

HEAD = V(0, -0.392, 0.342)
# Slightly raised, so the snarl faces the board camera rather than the floor.
HEAD_PITCH = -0.05
F = 1 / 0.67
# The face is built at lion size and blown up heroically; the mane keeps its
# own, smaller scale so the face stands out of the ruff.
FACE_SCALE = 1.3
HEAD_M = Matrix.LocRotScale(HEAD, Euler((HEAD_PITCH, 0, 0)), Vector((FACE_SCALE,) * 3))
MANE_M = Matrix.LocRotScale(HEAD + V(0, -0.006, 0.0), Euler((HEAD_PITCH * 0.5, 0, 0)), Vector((1.0, 1.0, 1.0)))


def head_xf(obj):
    obj.data.transform(HEAD_M)
    obj.data.update()
    return obj


def mane_xf(obj):
    obj.data.transform(MANE_M)
    obj.data.update()
    return obj


JAW_OPEN = 0.4
JAW_HINGE = V(0, 0.0, -0.022)


def jaw_xf(obj):
    lib.place(obj, -JAW_HINGE)
    lib.place(obj, JAW_HINGE, (JAW_OPEN, 0, 0))
    return obj


def head():
    """The face, in head space (forward is -y): a broad lion face with a long
    muzzle, a heavy brow knotted into a V-shaped scowl, a gaping snarl with
    sabre fangs and glowing, up-slanted eyes. Metaball radii are scaled by F
    because the surface sits at about two thirds of an element's radius."""
    parts = []
    skull = [
        dict(type="ELLIPSOID", co=(0, 0.02, 0.012), radius=0.046 * F, size=(1.0, 1.08, 0.9)),
        dict(type="ELLIPSOID", co=(0, -0.034, 0.02), radius=0.031 * F, size=(1.3, 0.85, 0.7)),
        # Brows: inner ends driven down over the nose bridge, outer ends up.
        dict(type="ELLIPSOID", co=(-0.019, -0.062, 0.024), radius=0.0135 * F, size=(1.6, 0.8, 0.62), rot=(0, 0.5, -0.3)),
        dict(type="ELLIPSOID", co=(0.019, -0.062, 0.024), radius=0.0135 * F, size=(1.6, 0.8, 0.62), rot=(0, -0.5, 0.3)),
        dict(co=(-0.036, -0.048, -0.006), radius=0.022 * F),
        dict(co=(0.036, -0.048, -0.006), radius=0.022 * F),
        # A long, deep muzzle with a broad bridge.
        dict(type="ELLIPSOID", co=(0, -0.086, -0.012), radius=0.026 * F, size=(1.25, 1.3, 0.85)),
        dict(type="ELLIPSOID", co=(0, -0.09, 0.01), radius=0.013 * F, size=(1.05, 3.0, 0.8)),
        dict(co=(-0.031, -0.068, -0.025), radius=0.016 * F),
        dict(co=(0.031, -0.068, -0.025), radius=0.016 * F),
        # Cheekbones filling out under the eyes, and the sockets above them.
        dict(type="ELLIPSOID", co=(-0.033, -0.066, -0.003), radius=0.011 * F, size=(1.2, 1.0, 0.8)),
        dict(type="ELLIPSOID", co=(0.033, -0.066, -0.003), radius=0.011 * F, size=(1.2, 1.0, 0.8)),
        dict(co=(-0.026, -0.071, 0.0135), radius=0.0105 * F, negative=True),
        dict(co=(0.026, -0.071, 0.0135), radius=0.0105 * F, negative=True),
    ]
    sk = lib.smooth(lib.decimate(lib.smooth(lib.metaball(skull, role="body", resolution=0.007, name="skull"), 3, 0.5), 0.52), 2, 0.4)
    parts.append(head_xf(symmetric(sk)))
    # Pale whisker pads and upper lip, a separate mass so the edge where they
    # meet the hide stays clean.
    pads = [
        dict(type="ELLIPSOID", co=(-0.0168, -0.103, -0.0215), radius=0.016 * F, size=(1.12, 0.8, 0.86), rot=(0.25, 0, 0)),
        dict(type="ELLIPSOID", co=(0.0168, -0.103, -0.0215), radius=0.016 * F, size=(1.12, 0.8, 0.86), rot=(0.25, 0, 0)),
        dict(type="ELLIPSOID", co=(0, -0.095, -0.027), radius=0.012 * F, size=(1.8, 1.0, 0.7)),
    ]
    parts.append(head_xf(lib.decimate(lib.smooth(lib.metaball(pads, role="skin", resolution=0.005, name="pads"), 2, 0.5), 0.6)))
    # The lower jaw: a broad U of a mandible as wide as the muzzle, with a
    # heavy chin, dropped wide open in a snarl; pale lips and mouth inside.
    jaw = [
        dict(type="ELLIPSOID", co=(-0.024, -0.056, -0.036), radius=0.018 * F, size=(0.8, 2.3, 0.75)),
        dict(type="ELLIPSOID", co=(0.024, -0.056, -0.036), radius=0.018 * F, size=(0.8, 2.3, 0.75)),
        dict(type="ELLIPSOID", co=(0, -0.06, -0.043), radius=0.018 * F, size=(1.2, 1.9, 0.5)),
        dict(type="ELLIPSOID", co=(0, -0.094, -0.04), radius=0.016 * F, size=(1.4, 0.9, 0.8)),
    ]
    jw = lib.decimate(lib.smooth(lib.metaball(jaw, role="body", resolution=0.007, name="jaw"), 2, 0.5), 0.6)
    parts.append(head_xf(jaw_xf(jw)))
    # A tongue on the floor of the dropped jaw, and the mouth cavity set deep
    # behind the teeth so the occlusion darkens it.
    parts.append(head_xf(jaw_xf(lib.sphere(1.0, loc=(0, -0.07, -0.029), scale=(0.017, 0.032, 0.008), role="skin",
                                           segments=10, rings=6, name="tongue"))))
    parts.append(head_xf(lib.sphere(1.0, loc=(0, -0.056, -0.036), scale=(0.023, 0.04, 0.017), role="skin",
                                    segments=10, rings=6, name="maw")))
    # A broad, flat nose pad, wider at the top.
    nose = lib.sphere(1.0, scale=(0.0165, 0.008, 0.009), role="cloth", segments=10, rings=6, name="nose")
    lib.deform(nose, lambda v: Vector((v.x * (1.0 + 30 * v.z), v.y, v.z)))
    parts.append(head_xf(lib.place(nose, (0, -0.113, 0.006), (0.55, 0, 0))))
    # Glowing eyes slanting up at their outer corners, set deep under the brow.
    for s in (-1, 1):
        parts.append(head_xf(lib.sphere(1.0, loc=(s * 0.025, -0.0695, 0.0128), rot=(0, -s * 0.38, s * 0.3),
                                        scale=(0.0098, 0.0042, 0.0045), role="eye", segments=10, rings=6, name="eye")))
    # Thick sabre fangs curving back, a row of smaller teeth between them, and
    # lower fangs rising from the dropped jaw.
    for s in (-1, 1):
        parts.append(head_xf(claw((s * 0.019, -0.1, -0.028), (0, -0.15, -1), 0.057, 0.0096, curl=0.3, sides=8, name="fang")))
        for x in (0.0045, 0.0105):
            parts.append(head_xf(spike((s * x, -0.109, -0.026), (s * x, -0.11, -0.039), 0.0034, role="metal", segments=6, name="tooth")))
        parts.append(head_xf(jaw_xf(claw((s * 0.019, -0.094, -0.031), (0, -0.1, 1), 0.028, 0.0072, curl=-0.2, name="lowfang"))))
    # A goatee of mane hair hanging from the chin: the man in the beast.
    for x, ln, w in ((0.0, 0.042, 0.034), (-0.017, 0.032, 0.024), (0.017, 0.032, 0.024)):
        parts.append(head_xf(jaw_xf(lock((x, -0.084, -0.046), (x * 4, -0.6, -1), (x * 2, -0.1, -1), ln, w, (0, -1, 0.3),
                                         thick=0.9, flat=0.8, n=5, sides=8, taper=0.4, tip=0.4, name="beard", soft=2))))
    return parts


MANE_STRANDS = 14
MANE_DEPTH = 0.20
# The widest ring sits just behind the face, so the mass frames it as a ruff
# and then falls away into the shoulders.
MANE_PROFILE = [(0.0, -0.006), (0.05, -0.014), (0.078, -0.014), (0.1, -0.007), (0.116, 0.006), (0.127, 0.024),
                (0.132, 0.05), (0.13, 0.074), (0.123, 0.097), (0.112, 0.12), (0.098, 0.141), (0.08, 0.16),
                (0.059, 0.176), (0.033, 0.19), (0.0, MANE_DEPTH)]
MANE_DROOP = 1.25
# Girth of the hood across: the face must fill nearly half the mane's width.
MANE_GIRTH = 0.96


def strand_coord(a, t):
    """Where a point of the hood lies across the strands: the strands fan out
    from the face, then sweep back and droop down the sides, wavering."""
    u = a - MANE_DROOP * t ** 1.3 * math.sin(a)
    return u + 0.09 * math.sin(3 * u + 0.7) + 0.05 * math.sin(7 * u + 2.0 + 5 * t)


def mane_sculpt(amp, lumps):
    """The deformation that carves the mane hood, in its lathe space (x right,
    -y up, z back from the face): rounded strands of uneven weight parted by
    deep grooves, irregular lumps of fur along the crown and the back, the
    back of the hood slumping over the withers and the mass swelling low into
    a bib over the chest."""
    n = MANE_STRANDS

    def fn(v):
        r = math.hypot(v.x, v.y)
        if r < 1e-6:
            return v
        a = math.atan2(v.x, -v.y)
        t = min(max(v.z / MANE_DEPTH, 0.0), 1.0)
        low = max(0.0, -math.cos(a))
        top = max(0.0, math.cos(a))
        side = abs(math.sin(a))
        u = strand_coord(a, t)
        k = int(round(u * n / TAU)) % n
        bump = abs(math.cos(n * u / 2)) ** 0.75
        reach = min(1.0, r / 0.07)
        g = 1 + 0.15 * (bump * amp[k] - 0.6) * reach * (1 - 0.3 * t)
        g *= 1 + 0.04 * math.sin(3 * a + 1.0) + 0.045 * math.sin(5 * a + 2.0 + 8 * v.z) + 0.22 * low ** 1.5
        for la, lt, lamp, wa, wt in lumps:
            da = (a - la + math.pi) % TAU - math.pi
            g += lamp * math.exp(-(da / wa) ** 2 - ((t - lt) / wt) ** 2) * reach
        rim = min(max((r - 0.08) / 0.03, 0.0), 1.0)
        z = v.z - 0.02 * bump * rim * (1 - t) ** 3 + 0.03 * bump * t ** 3 + 0.05 * low * t
        drop = (0.05 * low ** 1.5 + 0.02 * side * low) * math.sin(math.pi * min(1.0, t * 1.2))
        slump = 0.045 * top * t ** 1.6
        return Vector((v.x * g * (1 + 0.08 * side * t), v.y * g + drop + slump, z))
    return fn


def mane():
    """A great shaggy mane in its own space (forward -y, origin at the head):
    one sculpted mass of grooved fur strands with the rim round the face
    scalloped into tongues, soft clumps carrying the strands on past its
    edge over the shoulders, and stray tufts at the rim and the chest."""
    rng = random.Random(4)
    parts = []
    n = MANE_STRANDS
    amp = [rng.uniform(0.45, 1.6) for _ in range(n)]
    lumps = []
    for _ in range(9):
        lumps.append((rng.uniform(-2.3, 2.3), rng.uniform(0.25, 0.95), rng.uniform(-0.04, 0.085),
                      rng.uniform(0.22, 0.42), rng.uniform(0.1, 0.2)))
    sculpt = mane_sculpt(amp, lumps)
    fine = [(p.x, p.y) for p in resample(catmull([V(r, z, 0) for r, z in MANE_PROFILE], 4), 21)]
    fine[0], fine[-1] = MANE_PROFILE[0], MANE_PROFILE[-1]
    base = lib.lathe(fine, segments=52, scale=(MANE_GIRTH, MANE_GIRTH, 1), role="cloth", name="mane")
    lib.deform(base, sculpt)
    hood = [base]

    def radius(z):
        for (r0, z0), (r1, z1) in zip(MANE_PROFILE, MANE_PROFILE[1:]):
            if z0 <= z <= z1:
                return r0 + (r1 - r0) * (z - z0) / max(z1 - z0, 1e-9)
        return 0.0

    def point(u, t):
        z = t * MANE_DEPTH
        a = u
        for _ in range(12):
            a += 0.5 * (u - strand_coord(a, t))
        r = radius(z) * MANE_GIRTH
        return sculpt(V(r * math.sin(a), -r * math.cos(a), z)), a

    down = V(0, 1, 0)
    back = V(0, 0, 1)
    # Soft clumps hugging the hood past its back rim, every other strand.
    for i in range(0, n, 2):
        u = TAU * i / n
        p, a = point(u, 0.42)
        low = max(0.0, -math.cos(a))
        top = max(0.0, math.cos(a))
        if top > 0.5:
            continue
        q, _ = point(u, 0.55)
        radial = V(math.sin(a), -math.cos(a), 0)
        flow = (q - p).normalized()
        root = V(p.x * 0.9, p.y * 0.9, p.z)
        d0 = flow + radial * 0.05 + down * (0.15 + 0.4 * low)
        d1 = flow * 0.5 - radial * 0.1 + down * (0.8 + 0.3 * low)
        ln = (0.1 + 0.03 * low - 0.02 * top) * rng.uniform(0.88, 1.12)
        hood.append(lock(root, d0, d1, ln, 0.032 * amp[i] ** 0.5, radial, thick=0.75, flat=0.55, tip=0.45, taper=0.35,
                         n=6, sides=8, name="fringe", soft=2))
    # Stray tufts breaking the rim round the face, flicking out and forward.
    for u in (1.2, 2.0, -1.25, -2.05, math.pi + 0.1):
        u += rng.uniform(-0.12, 0.12)
        p, a = point(u, 0.1)
        radial = V(math.sin(a), -math.cos(a), 0)
        root = V(p.x * 0.88, p.y * 0.88, p.z)
        ln = 0.055 * rng.uniform(0.85, 1.15)
        hood.append(lock(root, radial * 0.8 + back * 0.4 + down * 0.3, radial * 0.3 + back + down * 0.6, ln, 0.03, radial,
                         thick=0.85, flat=0.7, tip=0.4, taper=0.45, n=5, sides=8, name="rimtuft", soft=2))
    # Sideburns sweeping forward from the ruff along the back of the jaw.
    for u in (2.05, -2.05, 2.5, -2.5):
        p, a = point(u, 0.02)
        radial = V(math.sin(a), -math.cos(a), 0)
        root = V(p.x * 0.78, p.y * 0.78, p.z + 0.012)
        hood.append(lock(root, -back + down * 0.6, -back * 0.3 + down - radial * 0.2, 0.05, 0.03, radial - back * 0.5,
                         thick=0.85, flat=0.7, tip=0.45, taper=0.45, n=5, sides=8, name="cheekruff", soft=2))
    # Shaggy tufts hanging from the bib over the chest.
    for x, t, ln in ((-0.04, 0.42, 0.06), (0.035, 0.5, 0.055), (0.0, 0.62, 0.065)):
        u = math.pi + x * 8
        p, a = point(u, t)
        radial = V(math.sin(a), -math.cos(a), 0)
        root = V(p.x * 0.9, p.y * 0.9, p.z)
        hood.append(lock(root, down + back * 0.5 + radial * 0.2, down * 0.6 + back, ln, 0.03, radial + back * 0.3,
                         thick=0.8, flat=0.65, tip=0.35, taper=0.5, n=5, sides=8, name="bibtuft", soft=2))
    for obj in hood:
        lib.place(obj, (0, 0.0, 0.0), (-math.pi / 2, 0, 0))
        parts.append(mane_xf(obj))
    return parts


# ---------------------------------------------------------------- tail

# A tight '?' hook over the haunches: the tail rises behind the rump, rolls
# forward over the top and holds the stinger up above the back, clear of the
# wings and the mane.
TAIL_CTRL = [(0, 0.262, 0.335), (0, 0.345, 0.385), (0, 0.40, 0.46), (0, 0.42, 0.545), (0, 0.395, 0.62),
             (0, 0.335, 0.668), (0, 0.26, 0.69), (0, 0.192, 0.694), (0, 0.142, 0.682)]
ARCH_CENTRE = V(0, 0.29, 0.52)
INFO = {}
# The stinger's line of attack: ahead and a little down at the enemy.
AIM = V(0, -1, -0.12).normalized()


def tail_segment(r, L, name="tail_seg"):
    """One armoured tail segment in its own frame (z along the tail, +y on the
    dorsal side): a barrel pinched at both joints, a low keel along its back
    and a raised banded rim lapping over the next segment."""
    prof = [(0, -0.002), (r * 0.74, 0.0), (r * 0.95, L * 0.08), (r * 1.04, L * 0.25), (r * 1.07, L * 0.48),
            (r * 1.03, L * 0.66), (r * 0.97, L * 0.74), (r * 1.08, L * 0.79), (r * 1.15, L * 0.9),
            (r * 0.84, L * 0.99), (0, L)]
    seg = lib.lathe(prof, role="body", segments=10, scale=(0.92, 1, 1), name=name)

    def keel(v):
        rr = math.hypot(v.x, v.y)
        if rr < 1e-6 or not 0.02 * L < v.z < 0.74 * L:
            return v
        c = max(0.0, v.y / rr)
        lift = r * 0.42 * c ** 4 * math.sin(math.pi * (v.z - 0.02 * L) / (0.72 * L)) ** 0.8
        return Vector((v.x * (1 - 0.2 * c ** 4), v.y + lift, v.z))
    lib.deform(seg, keel)
    lib.paint(seg, "trim", lambda c: c.z > L * 0.76)
    return seg


def tail():
    parts = []
    path = resample(catmull(TAIL_CTRL, 12), 160)
    n_seg = 9
    for k in range(n_seg):
        t0, t1 = k / n_seg, (k + 1) / n_seg
        a, b = at(path, t0), at(path, t1)
        d = b - a
        dorsal = a.lerp(b, 0.5) - ARCH_CENTRE
        # Tapering hard from the rump to the sting.
        r = 0.045 - 0.021 * k / (n_seg - 1)
        seg = tail_segment(r, d.length * 1.1)
        orient(seg, a, d, dorsal)
        parts.append(seg)
    # The sting: one aimed weapon, a teardrop venom bulb flowing into a long
    # straight spike, with a crown of short barbs round the root of the point
    # like an arrowhead; the back of the bulb stays smooth.
    end = path[-1]
    tang = (path[-1] - path[-8]).normalized()
    prof = [(0, -0.004), (0.02, 0.0), (0.036, 0.008), (0.047, 0.018), (0.053, 0.029), (0.055, 0.0385),
            (0.054, 0.05), (0.049, 0.063), (0.041, 0.077), (0.031, 0.091), (0.023, 0.106), (0.0175, 0.126),
            (0.0135, 0.155), (0.0095, 0.193), (0.0058, 0.232), (0.0026, 0.266), (0, 0.292)]
    sting = lib.lathe(prof, role="trim", segments=28, name="telson")
    base = end - tang * 0.025
    orient(sting, base, AIM)
    parts.append(sting)
    centre = base + AIM * 0.0385
    tip = base + AIM * 0.292
    INFO["bulb"] = tuple(round(c, 3) for c in centre)
    INFO["sting"] = tuple(round(c, 3) for c in tip)
    side_axis = AIM.cross(V(0, 0, 1)).normalized()
    up_axis = side_axis.cross(AIM).normalized()
    for i in range(6):
        phi = TAU * i / 6 + TAU / 12
        radial = side_axis * math.cos(phi) + up_axis * math.sin(phi)
        tilt = 0.42
        d = (AIM * math.cos(tilt) + radial * math.sin(tilt)).normalized()
        root = centre + AIM * 0.058 + radial * 0.016
        ln = 0.075 * (0.8 if radial.dot(up_axis) > 0.6 else 1.0)
        parts.append(spike(root, root + d * ln, 0.011, segments=7, name="spike"))
    return parts


# ---------------------------------------------------------------- wings


def fan_panel(f0, f1, hub, tip0, tip1, sag_dir, nu=5, nv=5, veins=(2, 4), name="membrane"):
    """Membrane between two fingers fanning from the wrist; its free edge
    between the tips is scalloped deep toward the wrist, and faint creases run
    out from the wrist through it."""
    inward = (hub - tip0.lerp(tip1, 0.5)).normalized()
    depth = (tip1 - tip0).length * 0.42
    verts, faces = [], []
    for i in range(nu + 1):
        t = i / nu
        a, b = f0(t), f1(t)
        for j in range(nv + 1):
            s = j / nv
            p = a.lerp(b, s)
            p += inward * depth * math.sin(math.pi * s) * t ** 2.2
            p += sag_dir * 0.016 * math.sin(math.pi * s) * math.sin(math.pi * 0.5 * min(t / 0.85, 1.0))
            if j in veins:
                p -= sag_dir * 0.0028 * math.sin(math.pi * min(1.0, t * 1.15))
            verts.append(p)
    for i in range(nu):
        for j in range(nv):
            q = i * (nv + 1) + j
            faces.append((q, q + 1, q + nv + 2, q + nv + 1))
    return lib.solidify(mesh_object(name, verts, faces, "skin"), 0.0045)


def inner_panel(arm, flank, tip, sag_dir, nu=10, nv=6, veins=(3, 5, 7), name="membrane_inner"):
    """Membrane from the arm back to the flank and the fifth finger, its
    trailing edge scalloped into two lobes, with faint creases running from
    the arm to the edge."""
    verts, faces = [], []
    for i in range(nu + 1):
        s = i / nu
        a = at(arm, s)
        p = flank.lerp(tip, s)
        b = p + (a - p).normalized() * (0.08 * abs(math.sin(2 * math.pi * s)) + 0.025 * math.sin(math.pi * s))
        for j in range(nv + 1):
            t = j / nv
            q = a.lerp(b, t) + sag_dir * 0.02 * math.sin(math.pi * t) * math.sin(math.pi * min(1.0, s * 1.2)) ** 0.7
            if i in veins:
                q -= sag_dir * 0.0028 * math.sin(math.pi * t)
            verts.append(q)
    for i in range(nu):
        for j in range(nv):
            q = i * (nv + 1) + j
            faces.append((q, q + 1, q + nv + 2, q + nv + 1))
    return lib.solidify(mesh_object(name, verts, faces, "skin"), 0.0045)


def lead_panel(arm, neck, sag_dir, nu=6, nv=2, name="membrane_lead"):
    """The narrow web ahead of the arm, from the neck to the wrist, so the arm
    is the edge of the wing rather than a bare bone."""
    verts, faces = [], []
    wrist = arm[-1]
    for i in range(nu + 1):
        s = i / nu
        a = at(arm, s)
        b = neck.lerp(wrist, s)
        b = b + (a - b) * 0.3 * math.sin(math.pi * s)
        for j in range(nv + 1):
            t = j / nv
            verts.append(a.lerp(b, t) + sag_dir * 0.006 * math.sin(math.pi * t) * math.sin(math.pi * s))
    for i in range(nu):
        for j in range(nv):
            q = i * (nv + 1) + j
            faces.append((q, q + 1, q + nv + 2, q + nv + 1))
    return lib.solidify(mesh_object(name, verts, faces, "skin"), 0.0045)


def board_dirs():
    """Directions toward the board camera for the Light and Dark rows (the
    board yaws pieces by +0.5 and -0.5 and looks from 40 degrees up)."""
    out = {}
    for row, yaw in (("light", 0.5), ("dark", -0.5)):
        az = -yaw
        el = math.radians(40)
        out[row] = V(math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))
    out["front"] = V(0, -math.cos(math.radians(8)), math.sin(math.radians(8)))
    return out


def wings():
    """Bat wings half raised: a short arm out to a wrist level with the back,
    long fingers fanning out and back, the leading fan tipped toward the board
    camera, the fifth finger and the inner membrane curving down to the hip
    with a two-lobed trailing edge."""
    parts = []
    cams = board_dirs()
    for side in (-1, 1):
        def P(x, y, z):
            return V(side * x, y, z)
        S, E, W = P(0.068, -0.135, 0.366), P(0.125, -0.128, 0.412), P(0.178, -0.104, 0.43)
        tips = [P(0.33, -0.13, 0.462), P(0.36, -0.025, 0.53), P(0.332, 0.085, 0.58), P(0.268, 0.15, 0.462)]
        flank = P(0.07, 0.125, 0.327)
        fingers = []
        for i, tip in enumerate(tips):
            bow = V(0, 0.008 + 0.004 * i, 0.012 - 0.004 * i)
            fingers.append(catmull([W, W.lerp(tip, 0.5) + bow, tip], 6))
        arm = catmull([S, E, W], 6)
        fan_n = (tips[0] - W).cross(tips[2] - W).normalized()
        if fan_n.z < 0:
            fan_n = -fan_n
        sag = -fan_n
        INFO[f"fan{side:+d}"] = {k: round(abs(fan_n.dot(c)), 2) for k, c in cams.items()}
        INFO[f"fan_n{side:+d}"] = tuple(round(c, 2) for c in fan_n)
        INFO[f"tips{side:+d}"] = [tuple(round(c, 3) for c in t) for t in tips]
        for i in range(3):
            f0 = (lambda s, f=fingers[i]: at(f, s))
            f1 = (lambda s, f=fingers[i + 1]: at(f, s))
            parts.append(fan_panel(f0, f1, W, tips[i], tips[i + 1], sag))
        parts.append(inner_panel(arm, flank, tips[3], sag))
        parts.append(lead_panel(arm, P(0.07, -0.2, 0.39), sag))
        # A fleshy ridge where the membrane grows from the back.
        parts.append(lib.tube([S + V(0, 0.01, -0.006), S.lerp(flank, 0.5) + P(0.002, 0, 0.0), flank + V(0, 0.01, -0.002)],
                              [0.015, 0.011, 0.007], role="body", sides=8, resolution=3, name="wing_root"))
        # Bones: humerus, forearm and fingers, knuckled joints and a thumb claw.
        parts.append(lib.tube([S, S.lerp(E, 0.5) + V(0, 0, 0.008), E], [0.03, 0.021, 0.017], role="body", sides=8, resolution=2, name="humerus"))
        parts.append(lib.tube([E, E.lerp(W, 0.5) + V(0, -0.004, 0.004), W], [0.018, 0.013, 0.014], role="body", sides=8, resolution=2, name="forearm"))
        parts.append(lib.sphere(0.017, loc=W, role="body", segments=8, rings=6, name="wrist"))
        parts.append(lib.sphere(0.019, loc=E, role="body", segments=8, rings=6, name="elbow"))
        for f in fingers:
            parts.append(lib.tube([f[0], f[len(f) // 2], f[-1]], [0.0095, 0.0065, 0.003], role="body", sides=6, resolution=2, name="finger"))
        parts.append(claw(W + V(0, -0.006, 0.006), V(side * 0.15, -1, 0.7), 0.04, 0.007, name="thumb"))
    return parts


# ---------------------------------------------------------------- build

# The whole beast sits a little back on its plinth so the big face stays
# inside the footprint, and is built a touch large then scaled down so it
# keeps to a troop's share of the board.
SHIFT = V(0, 0.03, 0)
FIG_SCALE = 0.9


def screen(p, cam):
    c = cam.normalized()
    right = (-c).cross(V(0, 0, 1)).normalized()
    up = right.cross(-c).normalized()
    return V(p.dot(right), p.dot(up))


def seg_dist(p, a, b):
    ab = b - a
    t = min(max((p - a).dot(ab) / max(ab.length_squared, 1e-12), 0.0), 1.0)
    return (p - (a + ab * t)).length


def build():
    parts = [torso()]
    parts += legs()
    parts += head()
    parts += mane()
    parts += tail()
    parts += wings()

    def x_extent(names):
        xs = [v.co.x for p in parts if p.name.split(".")[0] in names for v in p.data.vertices]
        return max(xs) - min(xs)
    INFO["face/mane"] = round(x_extent({"skull"}) / x_extent({"mane", "fringe"}), 3)
    # How close the sting comes to the wings on screen in each board row.
    bulb, sting = V(*INFO["bulb"]), V(*INFO["sting"])
    wing_names = {"membrane", "membrane_inner", "membrane_lead", "finger", "forearm", "wrist"}
    wing_pts = [v.co.copy() for p in parts if p.name.split(".")[0] in wing_names for v in p.data.vertices]
    for row, cam in board_dirs().items():
        if row == "front":
            continue
        a, b = screen(bulb, cam), screen(sting, cam)
        d, q = min((seg_dist(screen(q, cam), a, b), tuple(round(c, 3) for c in q)) for q in wing_pts)
        INFO[f"clear_{row}"] = (round(d, 3), q)
    INFO["wing_top"] = round(max(q.z for q in wing_pts) * FIG_SCALE, 3)
    totals = {}
    for p in parts:
        lib.place(p, SHIFT, scale=(FIG_SCALE,) * 3)
        key = p.name.split(".")[0]
        totals[key] = totals.get(key, 0) + lib.tris(p)
    print("REPORTPARTS " + " ".join(f"{k}={v}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))
    print("REPORTINFO " + repr(INFO))
    return parts
