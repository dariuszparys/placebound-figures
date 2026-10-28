"""Water: the water elemental, serving either side, grounded.

Its silhouette is one breaking wave in the spirit of Hokusai's Great Wave,
drawn in profile: a wall of water lofted from that profile along a crest line
that runs toward the front of the board, so both board cameras (half a radian
either side of the front) see the profile at the wall's front end, Light
from the face side and Dark from the back. The profile's back slope is
scooped, so the water thins as it rises into the crest, which pitches over
into a curl three quarters of a turn round, a sheet thinning from crest to
lip whose lip plunges down and back toward the face round an open barrel.

The wave is a peeling breaker: the full curl stands only over the front end;
toward the back the crest settles to about half height and the curl unrolls
into a cresting lip, and the crest line bows back behind the figure. The
front end is rounded, and cut back at the foot so the curl overhangs it.
Grooves stream up the back slope, fanning out across the foot and converging
on the curl, each its own depth. The outer curl and the crest are foam:
slender claws of foam spring from the curl, from just past the crest round
to the lip, growing as they go, each with two smaller ones branching off
it (those that would reach the head are left out, so the claws frame the
face), and spray flies off the lip. The wave rises out of a pool that swells up round its foot, with ripple
arms of light round the front of the pool and a swell that sweeps round the
front and climbs into the foot of the face.

A smooth humanoid upper body leans out of the face below the curl. Below the
ribs it is water: the face of the wave swells forward under the chest and
pours down into the pool, and the belly turns to water in a V. The left arm
hangs and its forearm sinks back into the face. The right forearm thins into
a lash of water flung out to its right in the upright plane both cameras see
square on: it dips, rises and curls over at its tip like a small breaking
wave, throwing two drops. A calm face with glowing eyes, turned after the
lash, and a fin of water swept back from the crown into a dorsal crest that
runs on back into the face of the wave, instead of hair.

Roles: water is cloth; foam (the outer curl and crest, claws, most spray) is
skin, the lightest value in both palettes; the humanoid is metal, which
reads as neither flesh (skin in Light) nor a dark lump (body in Dark); trim
draws the curl's rim, the three lowest claw tips, the leading drop, the
lash's crest, the fin's leading edge and the pool's ripples, never bands
round the body. The figure faces -Y; its right hand is on -X and its left on
+X. Z is up and the origin is the plinth top.
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

TAU = math.tau
DEBUG = bool(os.environ.get("WATER_DEBUG"))
FOAM = "skin"
FINISH = {"sharp_angle": 60, "ao_floor": 0.22, "ao_distance": 0.15}


def note(name, obj):
    if DEBUG:
        print(f"WATER {name}: {lib.tris(obj)} tris")
    return obj


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return Vector(a).lerp(Vector(b), t)


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


def kfac(stiff):
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def ell(co, semi, rot=(0, 0, 0), stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": 1.0 / kfac(stiff), "type": "ELLIPSOID", "size": tuple(semi),
            "rot": tuple(rot), "stiffness": stiff, "negative": neg}


def cap(a, b, r, stiff=2.0, neg=False):
    a, b = Vector(a), Vector(b)
    d = b - a
    rot = d.to_track_quat("X", "Z").to_euler()
    return {"co": tuple((a + b) / 2), "radius": r / kfac(stiff), "type": "CAPSULE", "size": (d.length / 2, 1, 1),
            "rot": tuple(rot), "stiffness": stiff, "negative": neg}


def ball(co, r, stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": r / kfac(stiff), "stiffness": stiff, "negative": neg}


def limb(a, b, r0, r1, n=4):
    out = []
    for k in range(n):
        p, q = lerp(a, b, k / n), lerp(a, b, (k + 1) / n)
        out.append(cap(p, q, r0 + (r1 - r0) * (k + 0.5) / n))
    return out


def to_budget(obj, target):
    n = lib.tris(obj)
    return lib.decimate(obj, target / n) if n > target else obj


def catmull(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def closed_curve(ctrl, samples, density=None):
    """Cyclic Catmull-Rom curve through 2D ctrl points, resampled by arc length
    weighted by density(c), where c is the fractional control index. Returns
    (points, control indices)."""
    pts = [Vector(p) for p in ctrl]
    n = len(pts)
    dense, idx = [], []
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        for k in range(24):
            dense.append(catmull(p0, p1, p2, p3, k / 24))
            idx.append(i + k / 24)
    dense.append(dense[0])
    idx.append(float(n))
    acc = [0.0]
    for k in range(len(dense) - 1):
        w = density(idx[k]) if density else 1.0
        acc.append(acc[-1] + (dense[k + 1] - dense[k]).length * w)
    out, cs, j = [], [], 0
    for i in range(samples):
        s = acc[-1] * i / samples
        while j < len(acc) - 2 and acc[j + 1] < s:
            j += 1
        f = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-12)
        out.append(dense[j].lerp(dense[j + 1], f))
        cs.append(idx[j] + (idx[j + 1] - idx[j]) * f)
    return out, cs


def open_curve(ctrl, samples):
    """Catmull-Rom curve through 3D ctrl points, resampled evenly by arc length."""
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
    out, j = [], 0
    for i in range(samples):
        s = acc[-1] * i / (samples - 1)
        while j < len(acc) - 2 and acc[j + 1] < s:
            j += 1
        f = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-9)
        out.append(dense[j].lerp(dense[j + 1], f))
    return out


def ribbon_sweep(pts, width, thick, side0, twist, role="cloth", name="ribbon", sides=12, tip=0.4):
    """A tube of elliptical section along pts: half `width(t)` along a side
    axis that starts as side0 and turns by twist(t), half `thick(t)` across.
    Ring j of the section sits at angle TAU * j / sides from the side axis."""
    n = len(pts)
    verts, faces = [], []
    side = nrm(side0)
    for i, p in enumerate(pts):
        t = i / (n - 1)
        T = nrm(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)])
        side = nrm(side - side.dot(T) * T)
        u = Matrix.Rotation(twist(t), 3, T) @ side
        v = T.cross(u)
        for j in range(sides):
            th = TAU * j / sides
            verts.append(p + u * (math.cos(th) * width(t)) + v * (math.sin(th) * thick(t)))
    for i in range(n - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    for k, sign in ((0, -1), (n - 1, 1)):
        ring = [k * sides + j for j in range(sides)]
        c = sum((Vector(verts[i]) for i in ring), Vector()) / sides
        T = nrm(pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)])
        verts.append(c + T * sign * (thick(k / (n - 1)) * tip))
        ci = len(verts) - 1
        faces += [(ring[j], ring[(j + 1) % sides], ci) for j in range(sides)]
    return mesh_object(name, verts, faces, role)


def paint_faces(obj, role, pick):
    """Gives the faces for which pick(polygon) holds the role material."""
    me = obj.data
    names = [s.name for s in me.materials]
    if role not in names:
        me.materials.append(lib.mat(role))
        names.append(role)
    ri = names.index(role)
    for p in me.polygons:
        if pick(p):
            p.material_index = ri
    return obj


def bead(p, d, r, role="trim", name="bead", tail=1.6):
    """A drop in flight at p: an egg, its fat end leading along d, drawn out
    behind by `tail`."""
    d = nrm(d)
    obj = lib.sphere(r, role=role, segments=8, rings=5, name=name)

    def shape(v):
        k = tail if v.x < 0 else 1.05
        return Vector((v.x * k, v.y * 0.9, v.z * 0.9))
    lib.deform(obj, shape)
    z = V(0, 0, 1) if abs(d.z) < 0.9 else V(0, 1, 0)
    y = nrm(z.cross(d))
    z = d.cross(y)
    m = Matrix((d, y, z)).transposed().to_4x4()
    m.translation = Vector(p)
    obj.data.transform(m)
    obj.data.update()
    return obj


# ---------------------------------------------------------------- layout

# The wave breaks toward BREAK, the figure's right; its crest line runs along
# ACROSS, toward the front of the board, so the board cameras, turned half a
# radian either way, look down the barrel from either side of its axis.
# (Turning the break toward the front helps the Dark camera but foreshortens
# the curl into an arch for the Light one.)
BETA = 0.0
BREAK = V(-math.cos(BETA), -math.sin(BETA), 0)
ACROSS = V(-BREAK.y, BREAK.x, 0)
PLINTH_R = 0.152

# The breaking wave's profile in (a, z), a along BREAK. Clockwise from the
# foot of the back slope: one long lean up the back to the crest, round the
# outside of the curl to the lip's tip, back in under the barrel's ceiling,
# down the face to the trough, and along the hidden base.
PROFILE = [
    (-0.130, -0.006),   # 0 back foot
    (-0.096, 0.030),
    (-0.062, 0.076),
    (-0.028, 0.136),
    (0.006, 0.210),
    (0.040, 0.298),     # 5
    (0.075, 0.398),
    (0.111, 0.503),
    (0.149, 0.608),
    (0.186, 0.698),
    (0.222, 0.768),     # 10
    (0.262, 0.812),     # crest
    (0.305, 0.826),
    (0.350, 0.816),
    (0.386, 0.790),
    (0.410, 0.750),     # 15
    (0.421, 0.700),
    (0.418, 0.648),
    (0.402, 0.598),
    (0.377, 0.558),
    (0.348, 0.532),     # 20
    (0.318, 0.518),     # lip tip, pointing down and back at the face
    (0.336, 0.546),     # inside of the lip
    (0.360, 0.573),
    (0.378, 0.610),
    (0.387, 0.650),     # 25
    (0.383, 0.691),
    (0.367, 0.727),
    (0.342, 0.751),
    (0.312, 0.761),     # barrel ceiling
    (0.284, 0.754),     # 30
    (0.262, 0.733),
    (0.247, 0.703),
    (0.236, 0.665),     # face, scooped a little back under the curl
    (0.222, 0.622),
    (0.206, 0.575),     # 35
    (0.190, 0.520),
    (0.177, 0.460),
    (0.168, 0.390),
    (0.163, 0.320),
    (0.162, 0.250),     # 40
    (0.165, 0.180),
    (0.172, 0.115),
    (0.183, 0.062),
    (0.198, 0.022),
    (0.215, -0.004),    # 45 trough
    (0.100, -0.010),
    (0.000, -0.012),    # base
    (-0.075, -0.010),
]
C_CREST, C_LIP, C_CEIL, C_FACE, C_TROUGH = 11, 21, 29, 33, 45
A_FOOT, A_TROUGH = PROFILE[0][0], PROFILE[C_TROUGH][0]


def density(c):
    if C_CREST - 3 <= c <= C_FACE + 1:
        return 2.0  # the curl and barrel get twice the samples
    if c >= C_TROUGH + 0.3:
        return 0.3  # the hidden base gets few
    return 1.0


N_PROF = 80
# The curl is drawn out along BREAK, so it reaches further over the figure.
CURL_WIDEN = 1.15
for _k in range(C_CREST + 1, C_FACE):
    _a, _z = PROFILE[_k]
    _f = min(1.0, (_k - C_CREST) / 3.0) * min(1.0, (C_FACE - _k) / 3.0)
    PROFILE[_k] = (_a + (_a - PROFILE[C_CREST][0]) * (CURL_WIDEN - 1) * _f, _z)
PROF_PTS, PROF_C = closed_curve(PROFILE, N_PROF, density)


def outline_normals(pts):
    """Outward unit normals of a clockwise outline in (a, z)."""
    n = len(pts)
    out = []
    for i in range(n):
        t = (pts[(i + 1) % n] - pts[i - 1]).normalized()
        out.append(Vector((-t.y, t.x)))
    return out


def half_thickness(pts, normals):
    """Half the water's thickness under each outline point, along its inward
    normal: how far the ends of the wave are rounded in there."""
    n = len(pts)
    out = []
    for i in range(n):
        p, d = pts[i], -normals[i]
        best = 1.0
        for j in range(n):
            if j in (i, (i - 1) % n):
                continue
            a, b = pts[j], pts[(j + 1) % n]
            e = b - a
            den = d.x * e.y - d.y * e.x
            if abs(den) < 1e-12:
                continue
            w = a - p
            t = (w.x * e.y - w.y * e.x) / den
            s_ = (w.x * d.y - w.y * d.x) / den
            if t > 1e-5 and 0.0 <= s_ <= 1.0:
                best = min(best, t)
        out.append(best / 2)
    # A sharp tip's ray runs up the length of its sheet; its neighbours know
    # how thin it really is.
    return [min(out[(i + k) % n] for k in range(-3, 4)) for i in range(n)]


PROF_N = outline_normals(PROF_PTS)
PROF_HT = half_thickness(PROF_PTS, PROF_N)
if DEBUG:
    curl = [PROF_HT[i] for i in range(N_PROF) if C_CREST <= PROF_C[i] <= C_FACE]
    print(f"WATER curl half thickness {min(curl):.4f}..{max(curl):.4f}")

# The crest line runs from W_BACK to W_FRONT, toward the front of the board:
# a peeling breaker. At the front end the curl has rolled fully over; toward
# the back it unrolls into a cresting swell that settles to about half height.
# Both ends are rounded, each point of the outline over its own share of the
# water's thickness, at most R_BACK / R_FRONT.
W_BACK, W_FRONT = -0.11, 0.15
H_DROP = 0.55
W_LEN = W_FRONT - W_BACK
R_BACK, R_FRONT = 0.045, 0.09
# The wall leans along its crest line toward the front (SLANT, the crest
# leading the foot), its front end is cut back at the foot (CUT, so the curl
# overhangs the figure), and the crest line bows back behind the figure (BOW).
# All three are linear in z at the ends, so the end caps stay flat.
SLANT = 0.06
CUT = 0.1
BOW = 0.035
Z_TOP = 0.83
# Where the base sits: its middle on the plinth, a little back and right.
BASE_AT = V(0.03, 0.02, 0.0)
A_MID = (A_FOOT + A_TROUGH) / 2
ORIGIN = BASE_AT - BREAK * A_MID - ACROSS * ((W_BACK + W_FRONT - CUT) / 2 - SLANT / 2)


def u_of(w):
    return (w - W_BACK) / W_LEN


def height_scale(w):
    """The crest stands full height at the front end and settles toward the
    back, so the wall rises as a swell into the curl."""
    return 1.0 - H_DROP * smoothstep(W_FRONT - 0.09, W_BACK, w) ** 0.8


def peel(w):
    """How far the curl has rolled over at w: fully over the front end,
    unrolling toward the back into a cresting lip."""
    return 1.0 - 0.85 * smoothstep(W_FRONT - 0.06, W_BACK + 0.03, w)


def bow(w):
    """How far the crest line bows back (+X) at w, behind the figure."""
    u = min(max(u_of(w), 0.0), 1.0)
    return BOW * math.sin(math.pi * u)


def scaled(p, w):
    s = height_scale(w)
    return Vector((A_MID + (p.x - A_MID) * (0.55 + 0.45 * s), (p.y + 0.012) * s - 0.012))


def to_world(w, a, z):
    zz = max(z, 0.0)
    shear = SLANT * (zz / Z_TOP - 0.5) - CUT * (1 - zz / Z_TOP) * u_of(w)
    return ORIGIN + ACROSS * (w + shear) + BREAK * (a - bow(w)) + V(0, 0, z)


def wave_coords(p):
    """(a, w, z) of a world point: the inverse of to_world."""
    q = Vector(p) - ORIGIN
    z = p.z
    zz = max(z, 0.0)
    ws = q.dot(ACROSS) - SLANT * (zz / Z_TOP - 0.5)
    k = CUT * (1 - zz / Z_TOP) / W_LEN
    w = (ws - k * W_BACK) / (1 - k)
    return q.dot(BREAK) + bow(w), w, z


def clamp_r(p):
    """Keeps the foot of everything on the plinth: low parts beyond r0 are
    eased in toward PLINTH_R."""
    r = math.hypot(p.x, p.y)
    r0 = 0.11
    wz = smoothstep(0.2, 0.04, p.z)
    if r <= r0 or wz <= 0:
        return p
    r1 = r0 + (PLINTH_R - r0) * math.tanh((r - r0) / (PLINTH_R - r0))
    k = 1 + (r1 / r - 1) * wz
    return Vector((p.x * k, p.y * k, p.z))


# Hokusai's striations: grooves stream up the back slope and round the curl
# to the lip. They fan out across the foot and converge toward the curl at
# the front end, each its own depth: (u at the foot, u at the crest, depth).
GROOVES = ((0.08, 0.42, 0.75), (0.25, 0.56, 1.3), (0.43, 0.68, 0.65), (0.6, 0.79, 1.15), (0.78, 0.9, 0.8))
GROOVE_AMP = 0.022


def groove(u, c):
    """+ on the ridges, - in the grooves."""
    f = smoothstep(0.5, C_CREST - 0.5, c) ** 1.2
    sig = 0.075 - 0.035 * f
    g = 0.0
    for u0, u1, amp in GROOVES:
        uk = u0 + (u1 - u0) * f
        g += amp * math.exp(-((u - uk) / sig) ** 2)
    return 0.3 - g


def groove_amp(u, c, ht):
    on = smoothstep(0.2, 1.6, c) * smoothstep(C_LIP + 0.2, C_LIP - 1.4, c)
    ends = smoothstep(0.0, 0.06, u) * smoothstep(1.0, 0.93, u)
    return min(GROOVE_AMP, 0.33 * ht) * on * ends


def is_foam(c):
    return C_CREST - 1.6 <= c <= C_LIP + 0.7


FOAM_LIFT = 0.003
CREST_PT = PROFILE[C_CREST]


def curl_weight(c):
    """How much of the curl's peeling the outline point at c takes part in."""
    return smoothstep(C_CREST - 1.0, C_CREST + 1.5, c) * smoothstep(C_FACE + 1.5, C_FACE - 1.0, c)


def half_thick_at(i, w):
    k = peel(w)
    return PROF_HT[i] * (1 - (1 - k) * curl_weight(PROF_C[i])) * height_scale(w)


# Each outline point's bevel radius at the back and front ends.
R_ENDS = [(min(0.8 * half_thick_at(i, W_BACK), R_BACK), min(0.8 * half_thick_at(i, W_FRONT), R_FRONT))
          for i in range(N_PROF)]


def bulge(w):
    """The back slope swells across the crest line, fullest in the middle."""
    u = min(max(u_of(w), 0.0), 1.0)
    return 0.014 * math.sin(math.pi * u) ** 0.8


# Below the ribs the body is the water: the face of the wave swells forward
# under the chest and pours down into the pool, so there is no gap under the
# chest for a limb to fill.
W_BODY = 0.04
SWELL = 0.05


def face_swell(w, z, c):
    on = smoothstep(C_FACE + 0.5, C_FACE + 2.5, c) * smoothstep(C_TROUGH + 1.5, C_TROUGH - 0.5, c)
    rz = 0.11 if z > 0.25 else 0.2
    return SWELL * on * math.exp(-((w - W_BODY) / 0.06) ** 2) * math.exp(-((z - 0.25) / rz) ** 2)


def point(i, u, th=0.0, side=0, lift=0.0):
    """World position of outline point i on the ring at (u, th, side): u runs
    0..1 along the crest line between the bevels; at the ends th turns the
    point round its bevel onto the flat cap."""
    rb, rf = R_ENDS[i]
    w0, w1 = W_BACK + rb, W_FRONT - rf
    r_end = rf if side > 0 else (rb if side < 0 else 0.0)
    w = w0 + (w1 - w0) * u + side * r_end * math.sin(th)
    c = PROF_C[i]
    k = peel(w)
    crest = scaled(Vector(CREST_PT), w)
    p = scaled(PROF_PTS[i], w)
    wgt = curl_weight(c)
    p = p.lerp(crest + (p - crest) * k, wgt)
    ht = PROF_HT[i] * (1 - (1 - k) * wgt) * height_scale(w)
    n = PROF_N[i]
    f = 1 - math.cos(th)
    q = p - n * (r_end * f)
    uu = u_of(w)
    d = groove_amp(uu, c, ht) * groove(uu, c) * (1 - f)
    if is_foam(c):
        d += FOAM_LIFT * smoothstep(C_CREST - 1.6, C_CREST - 0.6, c) * (1 - f)
    d += bulge(w) * smoothstep(0.2, 1.2, c) * smoothstep(C_CREST + 2.0, C_CREST - 1.0, c)
    d += face_swell(w, p.y, c) * (1 - f)
    q = q + n * (d + lift)
    return clamp_r(to_world(w, q.x, q.y))


def slices():
    """(u, theta, side) of each ring: a quarter round at the back end (side
    -1), steps along the crest line closing up toward the curl, a quarter
    round at the front end (side +1)."""
    out = []
    n_back, n_main, n_front = 5, 25, 7
    for k in range(n_back):
        out.append((0.0, (math.pi / 2) * (1 - k / n_back), -1))
    for k in range(n_main + 1):
        t = k / n_main
        out.append((1 - (1 - t) ** 1.25, 0.0, 0))
    for k in range(1, n_front + 1):
        out.append((1.0, (math.pi / 2) * k / n_front, 1))
    return out


def wave():
    rings = slices()
    verts, faces, roles = [], [], []
    for u, th, side in rings:
        verts += [point(i, u, th, side) for i in range(N_PROF)]
    for k in range(len(rings) - 1):
        th = (rings[k][1] + rings[k + 1][1]) / 2
        for i in range(N_PROF):
            a, b = k * N_PROF + i, k * N_PROF + (i + 1) % N_PROF
            faces.append((a, b, b + N_PROF, a + N_PROF))
            c0, c1 = PROF_C[i], PROF_C[(i + 1) % N_PROF]
            if c1 < c0:
                c1 += len(PROFILE)
            c = (c0 + c1) / 2
            roles.append(FOAM if is_foam(c) and th < 0.5 else "cloth")
    # Each end is cut flat inside its bevelled rim.
    last = (len(rings) - 1) * N_PROF
    faces.append(tuple(range(N_PROF)))
    faces.append(tuple(range(last, last + N_PROF)))
    roles += ["cloth", "cloth"]
    obj = mesh_object("wave", verts, faces, "cloth", roles)
    return [note("wave", obj)]


def rim_stroke():
    """One stroke of light along the outer edge of the curl's cut front end,
    from the crest round to the lip's tip: it draws the spiral."""
    th = math.radians(50)
    pts = []
    lo, hi = C_CREST - 1.2, C_LIP + 0.15
    for i in range(N_PROF):
        c = PROF_C[i]
        if lo <= c <= hi:
            pts.append((c, point(i, 1.0, th, 1, lift=0.0016)))
    pts.sort(key=lambda t: t[0])
    path = open_curve([p for _, p in pts], 32)

    def rad(t):
        return 0.0068 * smoothstep(0.0, 0.1, t) * (1 - 0.5 * smoothstep(0.8, 1.0, t)) + 0.0012
    return note("rim", ribbon_sweep(path, rad, rad, ACROSS, lambda t: 0.0, role="trim", name="rim", sides=6, tip=0.8))


def wave_at(w, c, lift=0.0):
    """World position, outward normal and profile tangent (toward higher c)
    on the wave's surface at w and control index c."""
    i = min(range(N_PROF), key=lambda k: abs(PROF_C[k] - c))
    rb, rf = R_ENDS[i]
    w0, w1 = W_BACK + rb, W_FRONT - rf
    u = min(max((w - w0) / (w1 - w0), 0.0), 1.0)
    p = point(i, u, lift=lift)
    q0, q1 = point(i - 1, u), point((i + 1) % N_PROF, u)
    t3 = nrm(q1 - q0)
    nn = PROF_N[i]
    n3 = nrm(BREAK * nn.x + V(0, 0, nn.y))
    return p, n3, t3


# ---------------------------------------------------------------- claws and spray


def claw_path(base, out, fwd, size, lift, turn, n):
    """A claw's path: it springs out of the surface, rolls along `fwd` and
    hooks round by `turn` back toward the surface."""
    out = nrm(out)
    fwd = nrm(Vector(fwd) - Vector(fwd).dot(out) * out)
    psi = math.atan(lift)
    pts = [base - (math.cos(psi) * fwd + math.sin(psi) * out) * 0.008]
    heads = []
    for k in range(n):
        s_ = k / (n - 1)
        h = psi - turn * max(0.0, (s_ - 0.2) / 0.8) ** 1.2
        heads.append(math.cos(h) * fwd + math.sin(h) * out)
        if k:
            pts.append(pts[-1] + heads[-1] * size / (n - 1) * (1.15 - 0.4 * s_))
    return pts, heads, out, fwd


def frame_euler(x, y):
    """Euler angles of the frame whose x axis is x and y axis is (nearly) y."""
    x = nrm(x)
    y = nrm(Vector(y) - Vector(y).dot(x) * x)
    z = x.cross(y)
    return Matrix((x, y, z)).transposed().to_euler()


def finger_elements(pts, r0, side, flat=1.5, tipk=0.8):
    """Metaball ellipsoids along pts: a soft finger of foam, flattened
    across `side` and tapering to its tip."""
    els = []
    n = len(pts)
    for i in range(n - 1):
        t = (i + 0.5) / (n - 1)
        a, b = pts[i], pts[i + 1]
        d = b - a
        r = r0 * (1 - tipk * t ** 0.9) + 0.0016
        els.append(ell((a + b) / 2, (d.length * 0.62 + r * 0.6, r * flat, r), rot=frame_euler(d, side), stiff=2.0))
    return els


def claw_cluster(base, out, fwd, size, r0, splay=1.0, lift=1.1, turn=1.25, tips=True, name="claw", budget=200,
                 flat=1.5):
    """A clawed finger of foam as Hokusai drew them: a soft hooked finger
    with two smaller ones branching off its back."""
    pts, heads, out_, fwd_ = claw_path(base, out, fwd, size, lift, turn, 7)
    pts = open_curve(pts, 7)
    side = nrm(out_.cross(fwd_))
    els = finger_elements(pts, r0, side, flat)
    ends = [pts[-1]]
    for k, sp, sz in ((2, 0.8 * splay, 0.42), (4, -0.6 * splay, 0.32)):
        h = nrm(pts[k + 1] - pts[k])
        up = nrm(out_ - out_.dot(h) * h)
        f = nrm(h + side * sp + up * 0.35)
        sub, _, _, _ = claw_path(pts[k], up, f, size * sz, 0.8, turn * 1.1, 5)
        sub = open_curve(sub, 4)
        els += finger_elements(sub, r0 * 0.65, side, flat)
        ends.append(sub[-1])
    obj = lib.metaball(els, role=FOAM, resolution=0.0028, name=name)
    obj = to_budget(obj, budget)
    if tips:
        lib.paint(obj, "trim", lambda c: (c - ends[0]).length < size * 0.16)
    return [obj]


# (w, control index, size, base radius, splay) of the claws of foam along
# the outer curl, from just past the crest round to the lip, growing as they
# go, spread over the front of the crest line where the curl has rolled over.
LIP_CLAWS = (
    (0.136, C_CREST + 1.2, 0.065, 0.0075, 1.0),
    (0.104, C_CREST + 1.8, 0.07, 0.0075, -1.0),
    (0.122, C_CREST + 2.8, 0.075, 0.008, -1.0),
    (0.092, C_CREST + 3.6, 0.08, 0.008, 1.0),
    (0.138, C_CREST + 4.4, 0.085, 0.0085, 1.0),
    (0.110, C_CREST + 5.2, 0.09, 0.009, -1.0),
    (0.126, C_CREST + 6.2, 0.095, 0.0092, 1.0),
    (0.098, C_CREST + 7.0, 0.1, 0.0095, -1.0),
    (0.140, C_CREST + 7.8, 0.105, 0.01, -1.0),
    (0.116, C_CREST + 8.6, 0.11, 0.0105, 1.0),
    (0.132, C_LIP - 0.3, 0.115, 0.0105, -1.0),
)
def shrink_from_head(base, out, fwd, size, lift, turn):
    """The claw's size, cut down until every point of it keeps CLEAR from the
    head, so the claws frame the face rather than grab it."""
    while size > 0.03:
        pts, _, _, _ = claw_path(base, out, fwd, size, lift, turn, 7)
        if min((p - HEAD_W).length for p in pts) > HEAD_R + CLEAR:
            break
        size *= 0.9
    return size


CLEAR = 0.04


def claws():
    plans = []
    for k, (w, c, size, r0, splay) in enumerate(LIP_CLAWS):
        pos, n3, t3 = wave_at(w, c)
        pos = pos - n3 * 0.006
        # Each springs out of the curl and reaches on round it the way the
        # wave rolls, hooking over at the tip; the lower ones reach down.
        f = (c - C_CREST) / (C_LIP - C_CREST)
        fwd = nrm(t3 + BREAK * 0.6 + V(0, 0, -0.3 * f))
        lift, turn = 1.25 - 0.25 * f, 1.15 + 0.25 * f
        size0, size = size, shrink_from_head(pos, n3, fwd, size, lift, turn)
        if size < 0.75 * size0:
            if DEBUG:
                print(f"WATER claw {k} dropped: would reach the head")
            continue
        plans.append((pos, n3, fwd, size, r0, splay, lift, turn))
    parts = []
    for j, (pos, n3, fwd, size, r0, splay, lift, turn) in enumerate(plans):
        # Light catches only the tips of the three lowest claws.
        parts += claw_cluster(pos, n3, fwd, size, r0, splay, lift=lift, turn=turn, tips=j >= len(plans) - 3)
    return [note("claws", o) for o in parts]


def spray():
    """Spray thrown off the lip and its claws, out and down the way the wave
    breaks: only the leading drop catches the light."""
    parts = []
    for k, (w, c, dist, up, side, r) in enumerate((
            (0.13, C_LIP - 2.0, 0.10, -0.2, 0.1, 0.011), (0.09, C_LIP - 1.5, 0.12, -0.35, -0.2, 0.010),
            (0.12, C_CREST + 3.0, 0.11, 0.25, 0.3, 0.009), (0.05, C_LIP - 2.0, 0.11, -0.5, -0.3, 0.0095),
            (0.11, C_LIP - 0.5, 0.15, -0.7, 0.4, 0.008), (0.07, C_CREST + 4.0, 0.14, 0.1, -0.4, 0.0075),
            (0.10, C_CREST + 1.5, 0.09, 0.6, 0.2, 0.0075), (0.13, C_LIP - 3.5, 0.08, 0.0, 0.5, 0.008))):
        pos, n3, t3 = wave_at(w, c)
        d = nrm(BREAK + V(0, 0, up) + ACROSS * side)
        q = pos + d * dist + V(0, 0, -0.6 * dist * dist)
        q.z = min(q.z, 0.84)
        q.x = max(q.x, -0.425)
        parts.append(bead(q, d, r, role="trim" if k < 1 else FOAM, name="spray"))
        q2 = q + d * 0.035 + V(0, 0, -0.01)
        q2.x = max(q2.x, -0.428)
        if k < 4:
            parts.append(bead(q2, d, r * 0.6, role="cloth", name="spray"))
    return parts


# ---------------------------------------------------------------- pool

POOL_R = 0.15
POOL = [(0.0, 0.026), (0.02, 0.026), (0.04, 0.0257), (0.06, 0.025), (0.075, 0.0238), (0.09, 0.0222), (0.103, 0.0205),
        (0.115, 0.018), (0.126, 0.0148), (0.135, 0.011), (0.142, 0.0072), (0.147, 0.0035), (POOL_R, 0.0)]
FILLET = 0.06
FILLET_R = 0.07


def base_distance(p):
    """How far p (on the plinth) is from the foot of the wave."""
    a, w, _ = wave_coords(Vector((p.x, p.y, 0.0)))
    da = max(A_FOOT - a, 0.0, a - A_TROUGH)
    dw = max(W_BACK - w, 0.0, w - W_FRONT)
    return math.hypot(da, dw)


def fillet(p):
    """The pool swells up round the foot of the wave, so the wave rises out
    of it rather than standing on it."""
    if p.z < 0.001:
        return p
    d = base_distance(p)
    r = math.hypot(p.x, p.y)
    lift = FILLET * smoothstep(FILLET_R, 0.0, d) ** 1.8 * smoothstep(0.15, 0.125, r)
    return V(p.x, p.y, p.z + lift)


def pool_z(r):
    for (r0, z0), (r1, z1) in zip(POOL, POOL[1:]):
        if r0 <= r <= r1:
            return z0 + (z1 - z0) * (r - r0) / (r1 - r0)
    return 0.0


SWIRL_C = V(0.02, 0.02, 0)


def swirl_pt(r, th, lift=0.0):
    p = SWIRL_C + V(math.sin(th) * r, -math.cos(th) * r, 0)
    q = fillet(V(p.x, p.y, pool_z(math.hypot(p.x, p.y)) + 0.0011))
    return V(q.x, q.y, q.z + lift)


def pool():
    """A pool of water over the plinth, swirling in toward the wave's foot:
    spiral ripples with light on their crests, a raised spiral swell that
    climbs into the foot, and small waves curling at its edge."""
    p = lib.lathe(POOL + [(0.0, 0.0)], role="cloth", segments=56, name="pool")
    lib.deform(p, fillet)
    parts = [note("pool", p)]
    # Ripple arms swirl round the front of the pool, where both board cameras
    # see them, each winding a little further in.
    for r0, r1, th0, th1 in ((0.142, 0.122, -1.55, 1.05), (0.127, 0.1, -1.05, 1.55), (0.112, 0.086, -1.85, 0.45)):
        pts = []
        for i in range(22):
            f = i / 21
            pts.append(swirl_pt(r0 + (r1 - r0) * f, th0 + (th1 - th0) * f, 0.0016))
        pts = open_curve(pts, 22)
        rip = ribbon_sweep(pts, lambda t: 0.002 + 0.0065 * math.sin(math.pi * t) ** 0.7,
                           lambda t: 0.0012 + 0.0016 * math.sin(math.pi * t) ** 0.7, V(0, 0, 1),
                           lambda t: math.pi / 2, role="trim", name="ripple", sides=6)
        parts.append(rip)
    # A raised swell sweeps round the front of the pool and climbs into the
    # foot of the wave's face, under the figure.
    pts = []
    for i in range(18):
        f = i / 17
        r = 0.135 - 0.02 * f
        pts.append(swirl_pt(r, 1.9 - 3.3 * f, 0.002 + 0.03 * f ** 2.5))
    pts = open_curve(pts, 18)
    ridge = ribbon_sweep(pts, lambda t: 0.006 + 0.008 * math.sin(math.pi * min(1.0, 0.15 + t)) ** 0.5,
                         lambda t: 0.003 + 0.005 * math.sin(math.pi * min(1.0, 0.15 + t)) ** 0.5, V(0, 0, 1),
                         lambda t: math.pi / 2, role="cloth", name="swell", sides=8)
    parts.append(ridge)
    return parts


# ---------------------------------------------------------------- body

# The humanoid is modelled facing -Y round x = 0, grown by BODY_S and leant
# forward out of the wave by LEAN about PIVOT, turned by TURN and set into
# the wave's face at (W_BODY, BODY_A). It is the colour of foam.
FIG = "metal"
BODY_S = 1.1
TURN = math.radians(-12)
LEAN = math.radians(8)
# It leans to its right, out over the lash it throws.
ROLL = math.radians(-8)
PIVOT = V(0, 0, 0.35)
BODY_A = 0.23
BODY_DROP = 0.13
HEAD = V(0.0, -0.022, 0.572)
SH_R = V(-0.09, 0.0, 0.474)
SH_L = V(0.088, 0.008, 0.48)
# The right arm flings a lash of water out to its right, forward and down.
JET = nrm(V(-0.45, -0.65, -0.3))


def body_matrix():
    at = to_world(W_BODY, BODY_A, 0.45) - V(0, 0, 0.45)
    return (Matrix.Translation((at.x, at.y, -BODY_DROP)) @ Matrix.Rotation(TURN + BETA, 4, "Z")
            @ Matrix.Translation(PIVOT) @ Matrix.Rotation(ROLL, 4, "Y") @ Matrix.Rotation(LEAN, 4, "X")
            @ Matrix.Scale(BODY_S, 4) @ Matrix.Translation(-PIVOT))


BODY_M = body_matrix()
BODY_INV = BODY_M.inverted()


def wave_pt(a, w, z):
    """A point given in the wave's own frame."""
    return to_world(w, a, z)


def local_of(p):
    """A world point in the humanoid's own (modelling) frame."""
    return BODY_INV @ Vector(p)


# The right arm: the upper arm out to its side and down, the forearm thrown
# out to its right, thinning at the wrist into the lash.
SH_R_W = BODY_M @ SH_R
EL_R_W = SH_R_W + nrm(V(-0.8, -0.3, -0.5)) * 0.08
FORE_R = nrm(V(-0.7, -0.6, 0.05))
WR_R_W = EL_R_W + FORE_R * 0.06
EL_R, WR_R = local_of(EL_R_W), local_of(WR_R_W)
# The left arm hangs, and its forearm sinks back into the swell of the face.
EL_L = SH_L + V(0.012, 0.006, -0.085)
LEFT_FORE = [EL_L, EL_L + V(0.008, 0.02, -0.035), EL_L + V(0.012, 0.045, -0.06)]
# The head turns to its right, after the lash.
HEAD_YAW = math.radians(-26)
NECK = HEAD + V(0, 0.012, -0.05)
HEAD_ROT = Matrix.Translation(NECK) @ Matrix.Rotation(HEAD_YAW, 4, "Z") @ Matrix.Translation(-NECK)
HEAD_M = BODY_M @ HEAD_ROT
HEAD_W = HEAD_M @ HEAD
HEAD_R = 0.055 * BODY_S


def torso():
    els = [
        ell((0.0, -0.002, 0.372), (0.05, 0.04, 0.045)),  # belly
        ell((0.0, 0.0, 0.43), (0.07, 0.05, 0.058)),  # ribcage
        ell((0.0, 0.006, 0.478), (0.086, 0.042, 0.03)),  # shoulder girdle
        ell((0.0, 0.004, 0.522), (0.026, 0.026, 0.045)),  # neck
        ell((0.0, 0.034, 0.45), (0.066, 0.042, 0.064)),  # upper back, sunk into the wave
        ell((0.0, 0.03, 0.36), (0.055, 0.045, 0.06)),  # the small of the back, flowing into the swell
    ]
    for s in (-1, 1):
        els += [
            ell((s * 0.034, -0.026, 0.452), (0.03, 0.016, 0.022), rot=(0, 0, s * 0.2)),  # pectoral
        ]
    for sh, el in ((SH_R, EL_R), (SH_L, EL_L)):
        s = 1 if sh.x > 0 else -1
        d = 0.026 if s < 0 else 0.024
        els += [ell(sh + V(s * 0.006, 0, 0.0), (d, d, d))]  # deltoid
        els += limb(sh, el, 0.022, 0.016)
        els += [ell(lerp(sh, el, 0.4) + V(0, -0.003, 0), (0.018, 0.019, 0.018))]  # upper arm swell
    els += limb(EL_R, WR_R, 0.018, 0.0085, n=5)  # right forearm, thinning into the lash
    els += [ell(lerp(EL_R, WR_R, 0.25), (0.018, 0.017, 0.017))]  # forearm swell
    els += limb(EL_L, LEFT_FORE[1], 0.018, 0.015, n=3)
    body = lib.metaball(els, role=FIG, resolution=0.005, name="torso")
    body = lib.smooth(body, 2, 0.4)
    body = to_budget(body, 1800)
    # The belly turns to water in a V, and so does the left forearm.
    lib.paint(body, "cloth", lambda c: c.z < 0.352 + 0.45 * abs(c.x) - 0.022 * max(0.0, math.cos(c.x * 80)) ** 3)
    body.data.transform(BODY_M)
    return note("torso", body)


def left_forearm():
    """The left forearm sinks into the face of the wave and dissolves."""
    path = open_curve([BODY_M @ p for p in LEFT_FORE], 10)
    obj = ribbon_sweep(path, lambda t: 0.018 + 0.012 * t, lambda t: 0.014 - 0.006 * t, V(0, 0, 1),
                       lambda t: 0.4 * t, role="cloth", name="left_forearm", sides=10)
    return note("left_forearm", obj)


# The lash in the wrist's frame, (out to its right, up), in the upright
# plane the board cameras see square on: it sweeps out and down, rises, and
# curls over at its tip like a breaking wavelet, a small echo of the wave.
LASH = [(-0.012, 0.004), (0.0, 0.0), (0.03, -0.012), (0.06, -0.022), (0.088, -0.022), (0.108, -0.01),
        (0.12, 0.008), (0.126, 0.03), (0.134, 0.048), (0.147, 0.056), (0.159, 0.05), (0.164, 0.036)]


def lash():
    """Water flung from the right forearm: the arm thins into a lash that
    curls over like a wavelet and throws two drops; light along the outside
    of its crest."""
    out = nrm(V(-1.0, -0.22, 0.0))
    up = V(0, 0, 1)
    # Scaled to keep inside the footprint.
    k = min(1.0, (0.443 + WR_R_W.x) / (0.172 * -out.x))
    ctrl = [WR_R_W + (out * x + up * z) * k for x, z in LASH]
    pts = open_curve(ctrl, 34)
    sides = 10

    def width(t):
        return 0.0055 + 0.009 * smoothstep(0.05, 0.55, t) * (1 - 0.4 * smoothstep(0.8, 1.0, t))

    def thick(t):
        return 0.0052 - 0.0022 * t

    side0 = nrm(up.cross(out))
    obj = ribbon_sweep(pts, width, thick, side0, lambda t: 0.0, role="cloth", name="lash", sides=sides, tip=1.0)
    n = len(pts)
    paint_faces(obj, "trim", lambda p: len(p.vertices) == 4 and min(p.vertices) // sides > n * 0.6
                and abs(min(p.vertices) % sides - 3 * sides // 4) <= 1)
    lib.paint(obj, FIG, lambda c: (c - WR_R_W).length < 0.014)
    parts = [note("lash", obj)]
    end, head_ = pts[-1], nrm(pts[-1] - pts[-4])
    for j, (fwd, r) in enumerate(((0.016, 0.006), (0.032, 0.0045))):
        q = end + head_ * fwd + V(0, 0, -0.004 * j)
        q.x = max(q.x, -0.445 + r * 1.6)
        parts.append(bead(q, head_, r, role="trim" if j == 0 else "cloth", name="lash_drop"))
    return parts


# The fin of water swept back over the skull instead of hair, in the head's
# median plane as (back, up) from HEAD: it rises from the crown into a swept
# dorsal crest behind the head and runs on back into the face of the wave.
FIN_LEAD = [(-0.006, 0.056), (0.018, 0.07), (0.04, 0.083), (0.058, 0.092), (0.072, 0.096)]


def skull_d(c):
    """Normalized distance of a head-local point from the skull."""
    H = HEAD
    d1 = ((c - (H + V(0, 0.008, 0.01))) * 1.0)
    d2 = (c - (H + V(0, 0.03, -0.004)))
    return min((d1.x / 0.044) ** 2 + (d1.y / 0.05) ** 2 + (d1.z / 0.05) ** 2,
               (d2.x / 0.043) ** 2 + (d2.y / 0.04) ** 2 + (d2.z / 0.045) ** 2) ** 0.5


def fin_trail():
    """Head-local points where the fin runs on back into the face of the
    wave: behind the head and a little higher."""
    end = HEAD_M @ (HEAD + V(0, 0.07, 0.08))
    a, w, z = wave_coords(end)
    best = None
    for c in [C_FACE + 0.25 * k for k in range(16)]:
        p, n3, _ = wave_at(w - 0.01, c)
        if best is None or abs(p.z - z + 0.05) < abs(best[0].z - z + 0.05):
            best = (p, n3)
    p, n3 = best
    tgt = p - n3 * 0.002
    mid = (end + tgt) / 2 + V(0, 0, 0.012)
    inv = HEAD_M.inverted()
    return [inv @ mid, inv @ tgt]


def fin():
    """The fin: flat lobes of water along its leading edge, each reaching
    down into the skull, then a stream on back into the wave; a stroke of
    light runs along the leading edge."""
    H = HEAD
    lead = [H + V(0, y, z) for y, z in FIN_LEAD]
    els = []
    for k, p in enumerate(lead):
        t = k / (len(lead) - 1)
        q = p + V(0, 0.012, -0.018 - 0.01 * t)
        d = (lead[min(k + 1, len(lead) - 1)] - lead[max(k - 1, 0)])
        els.append(ell(q, (0.022, 0.012 - 0.0055 * t, 0.024), rot=frame_euler(d, V(1, 0, 0)), stiff=2.2))
    trail = [lead[-1] + V(0, 0.0, -0.02)] + fin_trail()
    for a_, b_ in zip(trail, trail[1:]):
        els.append(cap(a_, b_, 0.0095))
    obj = lib.metaball(els, role="cloth", resolution=0.003, name="fin")
    obj = to_budget(obj, 500)
    path = open_curve([q + V(0, -0.003, 0.002) for q in lead[:-1]] + [lead[-1] + V(0, 0.008, -0.004)], 14)

    def rad(t):
        return 0.0024 * math.sin(math.pi * min(1.0, 0.1 + t)) ** 0.5 + 0.0008
    stroke = ribbon_sweep(path, rad, rad, V(1, 0, 0), lambda t: 0.0, role="trim", name="fin_edge", sides=6, tip=0.8)
    return [note("fin", obj), stroke]


def head():
    H = HEAD
    els = [
        ell(H + V(0, 0.008, 0.01), (0.044, 0.05, 0.05)),  # cranium
        ell(H + V(0, -0.012, -0.026), (0.032, 0.035, 0.03)),  # jaw
        ell(H + V(0, -0.033, -0.046), (0.014, 0.012, 0.012), stiff=3),  # chin
        ell(H + V(0, -0.036, 0.022), (0.025, 0.006, 0.005)),  # soft brow
        ell(H + V(0, -0.045, -0.01), (0.0048, 0.0072, 0.0096), rot=(-0.25, 0, 0), stiff=3.5),  # nose
        ell(H + V(0, 0.03, -0.004), (0.043, 0.04, 0.045)),  # back of the skull
    ]
    for s in (-1, 1):
        els += [
            ell(H + V(s * 0.026, -0.03, -0.016), (0.009, 0.007, 0.0063)),  # cheekbone
            ball(H + V(s * 0.0175, -0.048, 0.0), 0.006, stiff=3, neg=True),  # eye socket
        ]
    obj = lib.metaball(els, role=FIG, resolution=0.0032, name="head")
    obj = to_budget(obj, 1000)
    parts = [note("head", obj)] + fin()
    for s in (-1, 1):
        # Level, almond eyes, the outer corners a touch low: calm.
        parts.append(lib.sphere(0.0092, loc=H + V(s * 0.0175, -0.042, 0.0), rot=(0, s * 0.06, s * 0.12),
                                scale=(1.0, 0.45, 0.46), role="eye", segments=10, rings=6, name="eye"))
    for p in parts:
        p.data.transform(HEAD_M)
    return parts


def debug_points():
    m = BODY_M
    pts = {"head": m @ HEAD, "chin": m @ (HEAD + V(0, -0.034, -0.05)), "sh_r": m @ SH_R, "el_r": EL_R_W,
           "wr_r": WR_R_W, "sh_l": m @ SH_L, "el_l": m @ EL_L, "fore_l": m @ LEFT_FORE[2], "belly": m @ V(0, 0, 0.37),
           "back": m @ V(0, 0.06, 0.45), "lip_front": to_world(W_FRONT, *PROFILE[C_LIP])}
    for k, p in pts.items():
        a, w, z = wave_coords(p)
        print(f"WATER pt {k}: world ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) wave a={a:.3f} w={w:.3f} z={z:.3f}")


def build():
    only = os.environ.get("WATER_ONLY")
    parts = []
    if not only or "pool" in only:
        parts += pool()
    if not only or "wave" in only:
        parts += wave()
        parts.append(rim_stroke())
        parts += claws()
        parts += spray()
    if not only or "body" in only:
        parts.append(torso())
        parts.append(left_forearm())
        parts += head()
        parts += lash()
    if DEBUG:
        debug_points()
        acc = {}
        for p in parts:
            k = p.name.split(".")[0].split("_mesh")[0]
            acc[k] = acc.get(k, 0) + lib.tris(p)
        print("WATER parts:", sorted(acc.items(), key=lambda kv: -kv[1]))
        ext = []
        for p in parts:
            xs = [v.co.x for v in p.data.vertices]
            ys = [v.co.y for v in p.data.vertices]
            ext.append((min(xs), min(ys), p.name))
        print("WATER min x:", sorted(ext)[:3])
        print("WATER min y:", sorted(ext, key=lambda t: t[1])[:3])
        tops = [(max(v.co.z for v in p.data.vertices), p.name) for p in parts]
        print("WATER max z:", sorted(tops, reverse=True)[:3])
        print(f"WATER total: {sum(acc.values())} tris")
    return parts
