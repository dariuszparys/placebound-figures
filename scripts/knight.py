"""Knight: a Light foot knight in articulated plate, with a great helm and a
tall plume, a tabard and short cape, a kite shield on the left arm and a
longsword raised ready in the right. The figure faces -Y; its right hand is
on -X and its left on +X.

Plates are open shells (lathes and panels) given real thickness with
solidify, the most visible with a chamfered rim, stacked as overlapping lames
over a mail underbody. Each limb plate is built along its own bone in a local
frame."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

TAU = math.tau


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def frame(origin, z, front=(0, -1, 0)):
    """A matrix whose local z points along `z` and whose local -y leans toward
    `front`, placed at `origin`."""
    z = nrm(z)
    f = Vector(front)
    f = f - f.dot(z) * z
    if f.length < 1e-6:
        f = Vector((0, -1, 0)) - Vector((0, -1, 0)).dot(z) * z
    y = -f.normalized()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def put(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def mesh_object(name, bm, role):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, role)


def lathe_at(profile, angles, scale=(1, 1, 1), role="body", name="lathe"):
    """A closed lathe like lib.lathe, but with its columns at the given angles
    (radians from local -y toward +x), so a ridge can have a narrow column of
    its own on the crest."""
    bm = bmesh.new()
    rings = [[bm.verts.new((math.sin(a) * r * scale[0], -math.cos(a) * r * scale[1], z * scale[2])) for a in angles]
             for r, z in profile]
    n = len(angles)
    for a, b in zip(rings, rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_object(name, bm, role)


def ridge_angles(seg, half):
    """Evenly spaced lathe angles with the front column split in two `half`
    either side of the front, so one narrow face column runs down the crest."""
    out = []
    for k in range(seg):
        a = TAU * k / seg
        out += [-half, half] if k == 0 else [a if a <= math.pi else a - TAU]
    return sorted(out)


def shell(profile, t, m=None, role="body", seg=24, centre=0.0, sweep=TAU, bevel=0.0,
          fn=None, scale=(1, 1, 1), name="plate", paint=None, angles=None):
    """A lathed plate: `profile` of (radius, z) turned about local z, covering
    `sweep` radians centred on angle `centre` (0 = local front, pi/2 = +x),
    given thickness `t` and optionally a chamfered rim, then put in frame `m`.
    `paint` is a list of (role, where) applied in the local frame before
    placing, for metal lips and trim bands that cost no geometry. `angles`
    gives a closed lathe its own column angles instead of `seg` even ones."""
    start = centre - sweep / 2 if sweep < TAU - 1e-6 else 0.0
    if angles:
        o = lathe_at(profile, angles, scale=scale, role=role, name=name)
    else:
        o = lib.lathe(profile, role=role, segments=seg, start=start, sweep=sweep, scale=scale, name=name)
    if fn:
        lib.deform(o, fn)
    if t:
        o = lib.solidify(o, t)
    if bevel:
        o = lib.bevel_edges(o, bevel, 1)
    for prole, where in paint or ():
        lib.paint(o, prole, where)
    if m is not None:
        put(o, m)
    return o


def cap(r, th0, th1, n):
    """Profile of a spherical cap from polar angle th0 to th1 (radians)."""
    return [(r * math.sin(th0 + (th1 - th0) * i / n), r * math.cos(th0 + (th1 - th0) * i / n)) for i in range(n + 1)]


def roll(r, z, rr, n=6):
    """Closed circular profile: lathed, a rolled rim or ring."""
    return [(r + rr * math.cos(k * TAU / n), z + rr * math.sin(k * TAU / n)) for k in range(n + 1)]


def panel(shape, nu, nv, t, role="body", name="panel", border=None, border_role="trim"):
    """A thick plate: `shape(i, j)` maps grid indices (i across 0..nu, j down
    0..nv) to a point (u, v) of the flat outline in the XZ plane; the plate is
    `t` thick along y, its face toward -y. `border(i, j)` picks the grid cells
    (and with them the edge walls) that take `border_role`."""
    bm = bmesh.new()
    fr, bk = [], []
    tags = []
    for j in range(nv + 1):
        rf, rb = [], []
        for i in range(nu + 1):
            u, v = shape(i, j)
            rf.append(bm.verts.new((u, -t / 2, v)))
            rb.append(bm.verts.new((u, t / 2, v)))
        fr.append(rf)
        bk.append(rb)
    for j in range(nv):
        for i in range(nu):
            bm.faces.new((fr[j][i], fr[j + 1][i], fr[j + 1][i + 1], fr[j][i + 1]))
            bm.faces.new((bk[j][i], bk[j][i + 1], bk[j + 1][i + 1], bk[j + 1][i]))
            tags += [bool(border and border(i, j))] * 2
    loop = ([(0, i) for i in range(nu + 1)] + [(j, nu) for j in range(1, nv + 1)]
            + [(nv, i) for i in range(nu - 1, -1, -1)] + [(j, 0) for j in range(nv - 1, 0, -1)])
    for k in range(len(loop)):
        (a0, a1), (b0, b1) = loop[k], loop[(k + 1) % len(loop)]
        bm.faces.new((fr[a0][a1], fr[b0][b1], bk[b0][b1], bk[a0][a1]))
        tags.append(bool(border and border(min(a1, b1, nu - 1), min(a0, b0, nv - 1))))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    obj = mesh_object(name, bm, role)
    if any(tags):
        obj.data.materials.append(lib.mat(border_role))
        for poly, tag in zip(obj.data.polygons, tags):
            poly.material_index = 1 if tag else 0
    return obj


def stud(p, n, r=0.0045, role="metal", name="rivet"):
    """A small domed rivet head at `p` facing `n`."""
    o = lib.cylinder(r, r * 0.55, r * 0.9, role=role, segments=5, name=name)
    return put(o, frame(Vector(p) + nrm(n) * r * 0.25, n))


def on_lathe(m, r, z, a, sy=1.0):
    """World point and outward normal on a lathe surface in frame `m`."""
    p = Vector((math.sin(a) * r, -math.cos(a) * r * sy, z))
    n = Vector((math.sin(a), -math.cos(a) * sy, 0))
    return m @ p, (m.to_3x3() @ n).normalized()


def fan(w, h, name):
    """A fan plate for a joint: a leaf outline, rounded above and pointed
    below, as a thick panel with a narrow border of bright metal, curved to
    wrap the joint and creased down its middle. Its face is toward -y."""
    nu = nv = 4
    e = 0.16

    def inset(k, n):
        return 0.0 if k == 0 else 1.0 if k == n else e + (1 - 2 * e) * (k - 1) / (n - 2)

    def shape(i, j):
        tu, tv = inset(i, nu), inset(j, nv)
        if tv < 0.42:
            hw = w * (0.45 + 0.55 * math.sin(tv / 0.42 * math.pi / 2))
        else:
            hw = w * (1 - 0.9 * ((tv - 0.42) / 0.58) ** 1.5)
        top = 0.80 * h * (1 - 0.18 * (2 * tu - 1) ** 2)
        return hw * (2 * tu - 1), top - tv * (top + h)

    p = panel(shape, nu, nv, 0.004, role="body", name=name, border=lambda i, j: i in (0, nu - 1) or j in (0, nv - 1),
              border_role="metal")

    def wrap(v):
        v.y += 7.0 * v.x * v.x - 0.003 * max(0.0, 1 - abs(v.x) / w)
        return v
    return lib.deform(p, wrap)


# ---------------------------------------------------------------- pose

def about(pivot, rot):
    """A rotation `rot` (3x3) about the point `pivot`."""
    return Matrix.Translation(pivot) @ rot.to_4x4() @ Matrix.Translation(-Vector(pivot))


PELVIS = V(0.0, 0.006, 0.298)
WAIST = V(0.0, 0.008, 0.362)
# Left leg forward (shield side), right leg back and turned out.
L_HIP, L_KNEE, L_ANKLE = V(0.054, -0.006, 0.288), V(0.090, -0.064, 0.172), V(0.106, -0.052, 0.050)
R_HIP, R_KNEE, R_ANKLE = V(-0.054, 0.014, 0.288), V(-0.090, 0.030, 0.166), V(-0.110, 0.080, 0.050)
# The upper body twists the sword shoulder back and the shield forward, and
# leans in a little over the front knee; the head turns back toward the foe.
TWIST, LEAN, HEAD_TURN = -0.13, 0.09, 0.09
UPPER = about(V(0, 0.008, 0.330), Matrix.Rotation(TWIST, 3, "Z") @ Matrix.Rotation(LEAN, 3, "X"))
UP3 = UPPER.to_3x3()
FRONT_U = UP3 @ V(0, -1, 0)
CHEST = UPPER @ V(0.0, 0.010, 0.432)
NECK = UPPER @ V(0.0, 0.006, 0.505)
HEAD = UPPER @ about(V(0, 0.006, 0.505), Matrix.Rotation(HEAD_TURN, 3, "Z"))
# Sword arm raised in a high guard, shield arm across the body.
R_SHO, R_ELB, R_WRI = (UPPER @ V(-0.112, 0.010, 0.466), UPPER @ V(-0.180, 0.036, 0.384), UPPER @ V(-0.182, -0.048, 0.380))
L_SHO, L_ELB, L_WRI = (UPPER @ V(0.112, 0.010, 0.466), UPPER @ V(0.172, -0.020, 0.380), UPPER @ V(0.100, -0.052, 0.368))

SHIELD_YAW = 0.40 + TWIST
SWORD_DIR = nrm(UP3 @ nrm((-0.34, 0.05, 0.94)))
FIST = R_WRI + nrm(R_WRI - R_ELB) * 0.022 + V(0, 0, -0.002)
GUARD_TURN = 0.48


def underbody():
    """Mail beneath the plate, glimpsed behind the knees and inside the elbows.
    The torso is wholly plated, so only the hips, legs and arms get mail."""
    legs = [PELVIS, L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE]
    parts = [lib.skin([tuple(j) for j in legs], [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)],
                      [(0.046, 0.036), 0.036, 0.019, 0.013, 0.036, 0.019, 0.013], role="metal", subdiv=0, name="mail")]
    # The arm mail is a slim round sleeve wholly inside the plates, only
    # glimpsed in the crook of the elbow.
    for sho, elb, wri in ((R_SHO, R_ELB, R_WRI), (L_SHO, L_ELB, L_WRI)):
        parts.append(lib.tube([tuple(sho), tuple(elb), tuple(wri)], [0.026, 0.020, 0.017],
                              role="metal", sides=8, resolution=3, name="mail"))
    return parts


# ---------------------------------------------------------------- torso


def keel(depth, width, z0, z1, cy=0.0):
    """Pushes the front centre forward into a ridge, strongest between z0 and z1."""
    def fn(v):
        if v.y < cy:
            zt = clamp((v.z - z0) / (z1 - z0))
            k = math.sin(zt * math.pi) ** 0.7
            v.y -= depth * k * max(0.0, 1 - abs(v.x) / width) ** 1.6
        return v
    return fn


def torso():
    parts = []
    m = UPPER @ Matrix.Translation((0, 0.008, 0))
    sy = 0.8
    # Breastplate and backplate as one lathed shell with a keel down the front.
    prof = [(0.0805, 0.349), (0.087, 0.392), (0.094, 0.416), (0.098, 0.436),
            (0.099, 0.452), (0.095, 0.469), (0.087, 0.483), (0.075, 0.494), (0.062, 0.500)]
    kc = keel(0.024, 0.06, 0.34, 0.50, cy=-0.02)
    parts.append(shell(prof, 0.007, m, seg=24, scale=(1, sy, 1), fn=kc, name="cuirass"))

    def prof_r(z):
        for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
            if z0 <= z <= z1:
                return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
        return prof[-1][0]

    def on_cuirass(a, z, off=0.0035):
        r = prof_r(z) + off
        return m @ kc(Vector((math.sin(a) * r, -math.cos(a) * r * sy, z)))
    # Rolled edges in metal: the neckline under the helm, and the arm
    # openings sweeping from under each pauldron round to the armpit.
    parts.append(shell(roll(0.0835, 0.488, 0.0032, 4), 0, m, seg=24, scale=(1, sy, 1), fn=kc, role="metal", name="neck_roll"))
    for side in (-1, 1):
        pts = [on_cuirass(side * a, z) for a, z in ((1.08, 0.487), (0.88, 0.470), (0.83, 0.450), (0.93, 0.431),
                                                     (1.20, 0.419), (1.55, 0.415))]
        parts.append(lib.tube([tuple(q) for q in pts], 0.0038, role="metal", sides=6, resolution=2, name="arm_roll"))
    # Plackart: a second plate over the belly with an arched top and a rolled edge.
    kp = keel(0.026, 0.06, 0.34, 0.50, cy=-0.02)

    def plackart(v):
        a = math.atan2(v.x, -v.y)
        top = 0.405 - 0.035 * min(1.0, abs(a) / 1.3) ** 1.5
        t = clamp((v.z - 0.346) / (0.405 - 0.346))
        v.z = 0.346 + t * (top - 0.346)
        return kp(v)
    pp = [(0.089, 0.346), (0.088, 0.362), (0.091, 0.385), (0.097, 0.405)]
    parts.append(shell(pp, 0.006, m, seg=18, sweep=2.6, scale=(1, sy, 1), fn=plackart, name="plackart"))
    edge = [m @ plackart(Vector((math.sin(a) * 0.097, -math.cos(a) * 0.097 * sy, 0.405)))
            for a in (-1.3 + 2.6 * i / 8 for i in range(9))]
    parts.append(lib.tube([tuple(q) for q in edge], 0.0040, role="metal", sides=6, resolution=2, name="plackart_roll"))
    for a in (-0.75, 0.75):
        p, n = on_lathe(m, 0.096, 0.392, a, sy)
        parts.append(stud(p + n * 0.002, n, name="rivet"))
    # Gorget lame round the neck.
    gm = UPPER @ Matrix.Translation((0, 0.006, 0))
    parts.append(shell([(0.070, 0.484), (0.058, 0.506)], 0.005, gm, seg=24, scale=(1, 0.9, 1), name="gorget"))
    # Belt over the waist with a buckle; the fauld below it stays with the hips.
    belt = [(0.084, 0.330), (0.094, 0.332), (0.097, 0.341), (0.094, 0.350), (0.084, 0.353)]
    parts.append(shell(belt, 0, m, seg=24, scale=(1, sy, 1), role="cloth", name="belt"))
    bp, bn = on_lathe(m, 0.097, 0.3415, 0.0, sy)
    bk = lib.box((0.026, 0.008, 0.024), role="metal", bevel=0.003, name="buckle")
    parts.append(put(bk, frame(bp + bn * 0.003, UP3 @ V(0, 0, 1), -bn)))
    m = Matrix.Translation((0, 0.008, 0))
    # Fauld: three lames flaring over the hips, each lapping the one below
    # and turned out at its lower edge into a lip of bright metal.
    for i in range(3):
        zt = 0.336 - i * 0.026
        r0 = 0.084 + i * 0.006
        parts.append(shell([(r0, zt), (r0 + 0.010, zt - 0.029), (r0 + 0.0145, zt - 0.036)], 0.005, m, seg=15,
                           sweep=TAU * 0.75, scale=(1, sy, 1), name="fauld", paint=[("metal", lambda q, e=zt - 0.028: q.z < e)]))
    return parts


def tabard():
    """A cloth panel hanging from the belt over the fauld, edged in trim, with
    the Light side's sunburst."""
    top, bot, bd, nu, nv = 0.345, 0.178, 0.0085, 8, 10

    def shape(i, j):
        w = min(j / (nv - 1), 1.0)
        hw = 0.047 + 0.012 * w
        if i in (0, nu):
            u = hw * (1 if i else -1)
        else:
            u = (hw - bd) * (2 * (i - 1) / (nu - 2) - 1)
        v = top + (bot + bd - top) * w - 0.020 * (1 - abs(u) / hw)
        if j == nv:
            v -= bd
        elif j == nv - 1:
            pass
        else:
            v += 0.020 * (1 - abs(u) / hw) * (1 - w)
        return u, v

    p = panel(shape, nu, nv, 0.006, role="cloth", name="tabard",
              border=lambda i, j: i in (0, nu - 1) or j == nv - 1)

    def drape(v):
        s = clamp((top - v.z) / (top - bot))
        fwd = 0.5 + 0.5 * clamp(v.x / 0.06, -1, 1)
        v.y += -0.070 - 0.030 * s ** 0.6 - 0.030 * s * s * fwd + 1.5 * v.x * v.x + 0.006 * s * math.sin(v.x * 85 + 0.6)
        v.x += 0.006 * s * s
        return v
    lib.deform(p, drape)
    parts = [p]
    # Sunburst: four long rays and four short ones round a disc.
    star = []
    for k in range(16):
        a = k * TAU / 16
        r = (0.038 if k % 4 == 0 else 0.024 if k % 2 == 0 else 0.012)
        star.append((math.sin(a) * r, math.cos(a) * r))
    z = 0.278
    s = clamp((top - z) / (top - bot))
    y = -0.070 - 0.030 * s ** 0.6 - 0.030 * s * s * 0.5
    parts.append(lib.extrude(star, 0.006, loc=(0, y - 0.003, z), rot=(-0.21, 0, 0), role="trim", bevel=0.0, name="sunburst"))
    return parts


def cape():
    """A short cape from the shoulders to mid thigh, hanging in three deep
    pleats and swinging toward the sword side, pinned by a clasp at each
    shoulder."""
    prof = [(0.092, 0.478), (0.109, 0.450), (0.125, 0.405), (0.134, 0.350), (0.139, 0.290), (0.1415, 0.228), (0.1425, 0.212)]
    c = lib.lathe(prof, role="cloth", segments=30, start=math.pi * 0.60, sweep=math.pi * 0.80, name="cape")
    lib.paint(c, "trim", lambda p: p.z < 0.222)

    def fold(v):
        a = math.atan2(v.x, -v.y)
        back = math.cos(a - math.pi) ** 2
        # The top edge sags between the shoulder clasps.
        v.z -= 0.016 * back * clamp((v.z - 0.40) / 0.078)
        t = clamp((0.47 - v.z) / 0.26)
        # Three pleats: ridges at the centre back and either side of it, and
        # a shallow secondary fold turning each side edge.
        # The pleats drift a little toward the sword side as they fall.
        w = math.sin(a * 7 - math.pi / 2 + 0.45 * t)
        f = 1 + (0.02 + 0.28 * t ** 1.1) * w
        for ae in (0.665 * math.pi, 1.335 * math.pi):
            f += 0.12 * t * math.exp(-((a - ae) / 0.07) ** 2)
        v.x, v.y = v.x * f, v.y * f + 0.012
        # The ridges hang a little lower than the hollows.
        v.z -= 0.012 * t * t * w
        v.x -= 0.015 * t * t
        v.y += 0.015 * t * t
        return v
    lib.deform(c, fold)
    parts = [put(lib.solidify(c, 0.006), UPPER)]
    # Clasps where the cape is pinned to the backplate below the pauldrons.
    for side in (-1, 1):
        a = math.pi + side * 0.27 * math.pi
        p = fold(Vector((math.sin(a) * 0.101, -math.cos(a) * 0.101, 0.452)))
        n = nrm(Vector((math.sin(a), -math.cos(a), 0.0)) + V(0, 0, 0.5))
        disc = lib.cylinder(0.0105, 0.0085, 0.005, role="trim", segments=10, name="clasp")
        parts.append(put(disc, UPPER @ frame(p + n * 0.003, n)))
        parts.append(put(stud(V(0, 0, 0), V(0, 0, 1), r=0.0045), UPPER @ frame(p + n * 0.0065, n)))
    return parts


# ---------------------------------------------------------------- arms


PSEG = 18


def pauldron(side, sho):
    """A cop in two overlapping shells over the shoulder: a domed cap whose
    rolled trim edge laps a lower lame that flares out to a rolled rim of its
    own, and three spaulder lames below it lapping down the outside of the
    upper arm."""
    parts = []
    tilt = 0.50
    out = UP3 @ V(side * math.sin(tilt), 0, math.cos(tilt))
    m = frame(sho + UP3 @ V(side * 0.004, 0.0, 0.000), out, FRONT_U)
    m3 = m.to_3x3()
    c = math.pi / 2 if side > 0 else 3 * math.pi / 2
    R, sy, sz = 0.051, 1.06, 0.74
    inner = c + math.pi

    def saddle(v):
        # The cop rides high on the inner side, clear of the neck, and low
        # over the outside of the arm; a raised medial ridge runs front to
        # back over its crown; then it is squashed into an oval.
        rho = v.length
        t = math.atan2(math.hypot(v.x, v.y), v.z)
        a = math.atan2(v.x, -v.y)
        # ...and drops lower over the outside of the arm, a hood not a disc.
        t *= 1 - 0.42 * max(0.0, math.cos(a - inner)) ** 2 + 0.13 * max(0.0, math.cos(a - c)) ** 2
        v = Vector((math.sin(a) * math.sin(t), -math.cos(a) * math.sin(t), math.cos(t))) * rho
        if abs(v.x) < 0.015 and rho > 1e-6:
            v *= 1 + 0.0065 / rho * math.cos(v.x / 0.015 * math.pi / 2) ** 2
        return Vector((v.x, v.y * sy, v.z * sz))

    def on_cop(a, th, r):
        d = Vector((math.sin(a) * math.sin(th), -math.cos(a) * math.sin(th), math.cos(th)))
        return m @ saddle(d * r), (m3 @ Vector((d.x / sy, d.y / sy, d.z / sz))).normalized()

    # Lower lame: tucked under the cap at its top, wrapping down over the
    # deltoid like a hood rather than flaring into a brim.
    th0, th1, n = 0.60, 1.40, 3
    low = []
    for i in range(n + 1):
        th = th0 + (th1 - th0) * i / n
        r = R - 0.0025 + 0.0045 * (i / n) ** 1.5
        low.append((r * math.sin(th), r * math.cos(th)))
    rim_r, rim_z = low[-1][0] + 0.003, low[-1][1] - 0.008
    low.append((rim_r, rim_z))
    parts.append(shell(low, 0.006, m, seg=PSEG, fn=saddle, name="pauldron"))
    parts.append(shell(roll(rim_r, rim_z, 0.0048, 4), 0, m, seg=PSEG, fn=saddle, role="trim", name="pauldron_rim"))
    # Cap: the outer shell, its rolled edge in plain metal standing proud of
    # the lame, so only the outer rim carries the accent.
    Rc, thc = R + 0.004, 0.72
    parts.append(shell(cap(Rc, 0.0, thc, 3), 0.006, m, seg=PSEG, fn=saddle, name="pauldron"))
    parts.append(shell(roll(Rc * math.sin(thc), Rc * math.cos(thc), 0.0036, 3), 0, m, seg=PSEG, fn=saddle, role="metal",
                       name="pauldron_rim"))
    # Spaulder lames, each lapping the one below, round the outside of the arm.
    for i in range(3):
        z0 = rim_z - 0.012 - 0.017 * i
        r0 = rim_r - 0.008 - 0.002 * i
        prof = [(r0, z0), (r0 + 0.009, z0 - 0.021)]
        parts.append(shell(prof, 0.005, m, seg=12, centre=c, sweep=TAU * (0.60 - 0.05 * i), scale=(1, sy, sz),
                           name="spaulder"))
    # Rivets round the cap.
    for a, th, r in ((-0.95, 0.45, Rc + 0.003), (0.95, 0.45, Rc + 0.003)):
        p, nn = on_cop(c + a, th, r)
        parts.append(stud(p, nn, r=0.0046, name="rivet"))
    # Haute-piece: an upright flange standing on the inner edge of the cop,
    # curved round the neck, its top edge arched and turned out in metal.
    hc, hs = (math.pi / 2 if side > 0 else -math.pi / 2), 1.05

    def arch(v):
        a = math.atan2(v.x, -v.y)
        k = math.cos(clamp(abs(a - hc) / (hs / 2)) * math.pi / 2)
        v.z += (v.z - 0.484) / 0.026 * 0.009 * k
        return v
    parts.append(shell([(0.0895, 0.482), (0.0905, 0.502), (0.0945, 0.510)], 0.0045, UPPER, seg=8, centre=hc, sweep=hs,
                       fn=arch, name="haute_piece", paint=[("metal", lambda q: q.z > 0.507)]))
    return parts


def arm(side, sho, elb, wri):
    parts = []
    up = nrm(sho - elb)
    L1 = (sho - elb).length
    mu = frame(sho, up, FRONT_U)
    parts.append(shell([(0.031, -0.022), (0.030, -L1 * 0.62), (0.033, -L1 * 0.92)], 0.004, mu, seg=12, name="rerebrace"))
    # Couter: a cop on the point of the elbow with a fan on the outside.
    fore = nrm(elb - wri)
    point = nrm(nrm(elb - sho) - nrm(wri - elb))
    mc = frame(elb + point * 0.008, point, up)
    parts.append(shell(cap(0.030, 0.0, 1.25, 3), 0.005, mc, seg=12, name="couter"))
    lat = nrm(up.cross(fore))
    if lat.x * side < 0:
        lat = -lat
    fan_at = elb + lat * 0.030 + point * 0.004
    parts.append(put(fan(0.023, 0.027, "couter_fan"), frame(fan_at, nrm(up + fore * 0.2), lat)))
    parts.append(stud(fan_at + lat * 0.0052, lat, r=0.0045, name="rivet"))
    # Vambrace along the forearm and a flared gauntlet cuff.
    L2 = (elb - wri).length
    mf = frame(elb, fore, FRONT_U)
    parts.append(shell([(0.028, -0.012), (0.028, -0.040), (0.025, -L2 + 0.010)], 0.004, mf, seg=12, name="vambrace"))
    parts.append(shell([(0.024, -L2 - 0.004), (0.028, -L2 + 0.016), (0.036, -L2 + 0.032)], 0.004, mf, seg=12,
                       name="cuff"))
    return parts


def fist(centre, grip, along, fingers=True):
    """A plated fist closed round a grip running along `grip`; `along` is the
    forearm direction, so the knuckles face away from the wrist. The finger
    lames wrap the grip, each turned a little further round it than the one
    above, and the thumb lies over the first finger below the guard."""
    m = frame(centre, grip, along)
    parts = [put(lib.box((0.036, 0.040, 0.044), loc=(0, 0.004, 0), role="body", bevel=0.010, segments=2, name="hand"), m)]
    for k in range(3 if fingers else 0):
        z = 0.012 - k * 0.0125
        r = 0.0175 - 0.001 * k
        f = shell(roll(r, z, 0.0062 - 0.0004 * k, 4), 0, None, seg=7, centre=-0.30 + 0.32 * k, sweep=3.3, name="finger")
        parts.append(put(f, m))
    parts.append(put(lib.box((0.040, 0.012, 0.016), loc=(0.0, -0.006, 0.023), role="body", bevel=0.004, segments=1, name="knuckle"), m))
    if fingers:
        th = lib.box((0.010, 0.026, 0.009), role="body", bevel=0.003, segments=1, name="thumb")
        lib.place(th, (0.011, -0.012, 0.021), (0.0, 0.0, -0.55))
        parts.append(put(th, m))
    return parts


# ---------------------------------------------------------------- legs


def crest(depth, width):
    """A rounded ridge down the front (-y) of a lathed plate: a cos^2 bulge
    `width` either side of the centre line, so the crest catches one soft
    highlight instead of a hard crease between flat facets."""
    def fn(v):
        if v.y < 0 and abs(v.x) < width:
            v.y -= depth * math.cos(v.x / width * math.pi / 2) ** 2
        return v
    return fn


def leg(side, hip, knee, ankle, yaw):
    parts = []
    up = nrm(hip - knee)
    L1 = (hip - knee).length
    mt = frame(hip, up)
    # Cuisse wrapping the front and outside of the thigh, with a medial ridge,
    # narrowing to a waist above the knee; its lower edge is turned in metal.
    parts.append(shell([(0.048, -0.030), (0.048, -0.060), (0.043, -0.088), (0.037, -L1 + 0.030), (0.035, -L1 + 0.022)],
                       0.005, mt, seg=16, centre=side * 0.25, sweep=TAU * 0.72, fn=crest(0.005, 0.028), name="cuisse",
                       paint=[("metal", lambda q, e=-L1 + 0.028: q.z < e)]))
    # Leather straps round the back of the thigh, buckled on the outside.
    for z, r in ((-0.048, 0.0395), (-0.088, 0.0355)):
        parts.append(shell([(r, z + 0.006), (r, z - 0.006)], 0.003, mt, seg=8, centre=math.pi + side * 0.25,
                           sweep=2.3, role="cloth", name="strap"))
        p, n = on_lathe(mt, r + 0.002, z, math.pi + side * 1.25)
        parts.append(put(lib.box((0.010, 0.004, 0.014), role="metal", name="strap_buckle"),
                         frame(p, n, up)))
    # Tasset: a plate hung from the fauld over the top of the thigh, riveted.
    parts.append(shell([(0.058, 0.004), (0.061, -0.018), (0.066, -0.040)], 0.005, mt, seg=10,
                       centre=side * 0.55, sweep=1.7, name="tasset"))
    p, n = on_lathe(mt, 0.0625, -0.010, side * 0.55)
    parts.append(stud(p, n, r=0.0045, name="rivet"))
    # Poleyn: articulating lames above and below the knee, each turned out
    # at its lower edge into a bright lip, and over them a keeled cop that
    # sits on the lames and points forward, with a curved fan on the outside.
    shin = nrm(knee - ankle)
    L2 = (knee - ankle).length
    ms = frame(knee, shin)
    parts.append(shell([(0.0305, -L1 + 0.040), (0.0345, -L1 + 0.022), (0.0372, -L1 + 0.0175)], 0.0035, mt, seg=12,
                       sweep=KNEE_SWEEP, name="poleyn_lame",
                       paint=[("metal", lambda q, e=-L1 + 0.021: q.z < e)]))
    parts.append(shell([(0.0340, -0.010), (0.0362, -0.0245), (0.0390, -0.0305)], 0.0035, ms, seg=12,
                       sweep=KNEE_SWEEP, name="poleyn_lame",
                       paint=[("metal", lambda q: q.z < -0.026)]))
    front = nrm(nrm(knee - hip) - nrm(ankle - knee) + V(0, -0.6, 0))
    off = 0.010
    mk = frame(knee + front * off, front, up)
    rcx, rcy, Rl = 0.025, 0.021, 0.036

    def conform(v):
        # The cop's rim lies on the round of the lames; the dome rises from it
        # to a forward point, with a keel down its middle.
        s = clamp(math.hypot(v.x / rcx, v.y / rcy))
        dome = 1 - s * s
        base = math.sqrt(max(Rl * Rl - v.x * v.x, 0.0)) - off + 0.003
        kl = math.cos(min(abs(v.x) / 0.010, 1.0) * math.pi / 2) ** 2
        v.z = base + 0.0115 * (1 - s) ** 1.1 + 0.0055 * kl * dome ** 0.5
        return v
    disc = [(rcx * i / 4, 0.0) for i in range(5)]
    parts.append(shell(disc, 0.0035, mk, seg=12, fn=conform, scale=(1, rcy / rcx, 1), name="poleyn",
                       paint=[("metal", lambda q: math.hypot(q.x / rcx, q.y / rcy) > 0.84)]))
    lat = V(side, 0, 0)
    lat = nrm(lat - lat.dot(shin) * shin)
    fan_at = knee + lat * 0.040 + front * 0.004
    parts.append(put(fan(0.027, 0.033, "poleyn_fan"), frame(fan_at, shin, lat)))
    parts.append(stud(fan_at + lat * 0.0052, lat, r=0.0048, name="rivet"))
    # Greave with a calf swell and a rounded ridge down the shin, the crest
    # picked out in a narrow strip of bright metal.
    shin_crest = crest(0.004, 0.030)

    def calf(v):
        if v.y > 0:
            v.y += 0.010 * math.sin(clamp((-v.z - 0.020) / 0.070) * math.pi) * math.cos(min(abs(v.x) / 0.04, 1.0) * math.pi / 2) ** 2
        # The crest rises out of the greave below the knee lame, not into it.
        y0 = v.y
        v = shin_crest(v)
        v.y = y0 + (v.y - y0) * clamp((-v.z - 0.026) / 0.020)
        return v
    parts.append(shell([(0.0295, -0.014), (0.0325, -0.030), (0.0350, -0.050), (0.0300, -0.078), (0.0215, -L2 + 0.022),
                        (0.0220, -L2 + 0.010), (0.0280, -L2 + 0.002)],
                       0.004, ms, fn=calf, name="greave", angles=ridge_angles(16, 0.05),
                       paint=[("metal", lambda q: q.z < -L2 + 0.009),
                              ("metal", lambda q: abs(q.x) < 0.0025 and q.y < -0.02)]))
    parts += sabaton(ankle, yaw)
    return parts


KNEE_SWEEP = 3.6


def sabaton(ankle, yaw):
    """An armoured foot: a plated sole and heel, four uniform lames over the
    instep following the top of the foot, each lapping the one toward the toe
    with a bright lip, and a pointed toe cap."""
    parts = []
    bev = 0.005
    # Raised by the bevel width: a bevelled extrusion grows past its outline,
    # and the sole must sit on the plinth, not in it.
    m = Matrix.Translation((ankle.x, ankle.y - 0.006, bev)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(-math.pi / 2, 4, "Z")
    side_prof = [(-0.034, 0.0), (-0.038, 0.020), (-0.030, 0.042), (-0.010, 0.050), (0.020, 0.040),
                 (0.048, 0.025), (0.068, 0.011), (0.072, 0.0)]
    hw = 0.021
    core = lib.extrude(side_prof, 2 * hw, role="body", bevel=bev, name="foot")

    def narrow(x):
        return 1 - 0.38 * clamp((x - 0.02) / 0.052)

    def taper(v):
        v.y *= narrow(v.x)
        return v
    lib.deform(core, taper)
    parts.append(put(core, m))
    top_line = side_prof[3:]

    def top(x):
        for (x0, z0), (x1, z1) in zip(top_line, top_line[1:]):
            if x0 <= x <= x1:
                return z0 + (z1 - z0) * (x - x0) / (x1 - x0)
        return top_line[0][1]

    def section(x, d):
        # The core's cross-section at x (flat top, rounded corners of the
        # bevel radius, upright sides) pushed out by d.
        t, w, rr = top(x), hw - bev, bev + d
        low = max(t - 0.016, 0.006)
        half = [(w + rr, low), (w + rr, t)] + [(w + rr * math.cos(a), t + rr * math.sin(a)) for a in (0.5, 1.07)]
        half += [(0.0, t + rr + 0.0009)]
        pts = [(-y, z) for y, z in half] + [(y, z) for y, z in half[-2::-1]]
        s = narrow(x)
        return [(x, y * s, z) for y, z in pts]

    for i in range(4):
        x0 = 0.004 + 0.012 * i
        x1 = x0 + 0.016
        rows = [section(x0, 0.0012), section(x1 - 0.0035, 0.0012 + 0.0036 * (12.5 / 16)), section(x1, 0.0048)]
        bm = bmesh.new()
        vs = [[bm.verts.new(p) for p in row] for row in rows]
        for a, b in zip(vs, vs[1:]):
            for k in range(len(a) - 1):
                bm.faces.new((a[k], a[k + 1], b[k + 1], b[k]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        lame = lib.solidify(mesh_object("sabaton", bm, "body"), 0.0025)
        lib.paint(lame, "metal", lambda c, e=x1 - 0.0035: c.x > e)
        parts.append(put(lame, m))
    return parts


# ---------------------------------------------------------------- head


HELM = [(0.000, 0.498), (0.054, 0.498), (0.060, 0.503), (0.063, 0.515), (0.064, 0.545), (0.064, 0.585),
        (0.062, 0.603), (0.057, 0.615), (0.046, 0.623), (0.024, 0.627), (0.000, 0.628)]
HELM_SY = 1.08


def helm_r(z):
    for (r0, z0), (r1, z1) in zip(HELM, HELM[1:]):
        if z0 <= z <= z1 and z1 > z0:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return HELM[-1][0] if z > HELM[-1][1] else HELM[0][0]


def helm_band(z0, z1, off, n=4):
    return [(helm_r(z0 + (z1 - z0) * i / n) + off, z0 + (z1 - z0) * i / n) for i in range(n + 1)]


def helm():
    parts = []
    k = keel(0.010, 0.040, 0.50, 0.64)
    core = lib.lathe(HELM, role="body", segments=28, scale=(1, HELM_SY, 1), name="helm")
    parts.append(lib.deform(core, k))
    # The sight: a glowing band recessed between the brow band and the face plate.
    eye = lib.lathe(helm_band(0.556, 0.574, 0.0008, 2), role="eye", segments=20, start=-1.25, sweep=2.5, scale=(1, HELM_SY, 1), name="sight")
    parts.append(lib.deform(eye, k))
    parts.append(shell(helm_band(0.505, 0.556, 0.0035, 2), 0.004, seg=28, centre=0, sweep=3.6, scale=(1, HELM_SY, 1), fn=k,
                       name="faceplate"))
    parts.append(shell(helm_band(0.574, 0.600, 0.0035, 2), 0.004, seg=24, scale=(1, HELM_SY, 1), fn=k, role="trim", name="browband"))
    # The cross: a raised strip down the face, splitting the sight in two.
    parts.append(shell(helm_band(0.507, 0.600, 0.0075, 6), 0.004, seg=4, centre=0, sweep=0.30, scale=(1, HELM_SY, 1), fn=k, role="trim", name="cross"))
    # Breaths on the right cheek: three short pierced slots.
    for i in range(3):
        z = 0.517 + 0.0105 * i
        a = -0.64
        r = helm_r(z) + 0.0045
        p = k(Vector((math.sin(a) * r, -math.cos(a) * r * HELM_SY, z)))
        slot = lib.box((0.016, 0.0034, 0.004), role="metal", name="breath")
        parts.append(put(slot, frame(p, (math.sin(a), -math.cos(a) * HELM_SY, 0), (0, 0, 1))))
    # Rivets along the brow band.
    for a in (-1.15, -0.62, 0.62, 1.15):
        z = 0.587
        r = helm_r(z) + 0.0055
        p = k(Vector((math.sin(a) * r, -math.cos(a) * r * HELM_SY, z)))
        parts.append(stud(p, (math.sin(a), -math.cos(a), 0), r=0.004, name="rivet"))
    # A plume holder on the crown and a tall plume sweeping back.
    parts.append(lib.cylinder(0.013, 0.009, 0.016, loc=(0, 0.012, 0.632), role="metal", segments=10, name="holder"))
    for i in range(5):
        s = (i - 2) / 2
        x = s * 0.012
        lift = 1 - 0.22 * abs(s)
        hi = 0.640 + 0.066 * lift
        pts = [(x * 0.5, 0.012, 0.636), (x * 1.5, 0.026, 0.646 + 0.048 * lift), (x * 2.3, 0.085, hi),
               (x * 3.2, 0.150, hi - 0.032), (x * 4.0, 0.186, hi - 0.100)]
        rr = [r * (1 - 0.15 * abs(s)) for r in (0.008, 0.019, 0.022, 0.015, 0.003)]
        parts.append(lib.tube(pts, rr, role="trim", sides=6, resolution=2, name="plume"))
    return parts


# ---------------------------------------------------------------- arms gear


def shield():
    """A curved kite shield: a metal backing showing as a rim round a trim
    field, a raised sunburst and boss, studs on the rim and straps behind."""
    W, T, B = 0.092, 0.105, 0.175

    def hw(w):
        if w < 0.14:
            return W * (0.90 + 0.10 * math.sin(w / 0.14 * math.pi / 2))
        return W * ((1 - w) / 0.86) ** 0.78

    def outline(inset, nu=8, nv=11):
        def shape(i, j):
            s, w = i / nu, j / nv
            h = max(hw(w) - inset, 0.0012)
            u = h * (2 * s - 1)
            v = (T - inset) - w * (T + B - inset * 2.2)
            v += 0.014 * (1 - (2 * s - 1) ** 2) * (1 - w) ** 3
            return u, v
        return shape

    def bend(v):
        v.y += 2.4 * v.x * v.x
        return v

    back = lib.bevel_edges(panel(outline(0.0, 6), 6, 11, 0.012, role="metal", name="shield"), 0.003, 1)
    field = panel(outline(0.011, 6), 6, 11, 0.008, role="trim", name="shield_field")
    lib.place(field, (0, -0.006, 0))
    star = []
    for k in range(16):
        a = k * TAU / 16
        r = (0.050 if k % 4 == 0 else 0.030 if k % 2 == 0 else 0.015)
        star.append((math.sin(a) * r, math.cos(a) * r))
    emblem = lib.extrude(star, 0.010, loc=(0, -0.011, 0.012), role="body", bevel=0.002, name="shield_emblem")
    boss = lib.cylinder(0.014, 0.011, 0.012, loc=(0, -0.016, 0.012), rot=(math.pi / 2, 0, 0), role="body", segments=12, bevel=0.002, name="shield_boss")
    parts = [back, field, emblem, boss]
    for w in (0.05, 0.3, 0.55, 0.8):
        for sgn in (-1, 1):
            parts.append(stud((sgn * (hw(w) - 0.0055), -0.007, T - w * (T + B)), (0, -1, 0), r=0.0042, name="rivet"))
    parts.append(stud((0, -0.007, T + 0.008), (0, -1, 0), r=0.0042, name="rivet"))
    # Enarmes on the back.
    for v in (0.045, -0.005):
        parts.append(lib.box((0.075, 0.010, 0.014), loc=(0, 0.011, v), role="cloth", bevel=0.003, segments=1, name="enarme"))
    for p in parts:
        lib.deform(p, bend)
    mid = (L_ELB + L_WRI) / 2
    face = V(math.sin(SHIELD_YAW), -math.cos(SHIELD_YAW), 0)
    m = Matrix.Translation(mid + face * 0.036 + V(0, 0, -0.020)) @ Matrix.Rotation(SHIELD_YAW, 4, "Z")
    for p in parts:
        put(p, m)
    return parts


def sword():
    """A longsword with a diamond blade and a fuller in trim, a down-curved
    guard and a wheel pommel, raised in the right fist."""
    parts = []
    L, w0 = 0.310, 0.0155
    bm = bmesh.new()
    rings = []
    n = 10
    for i in range(n + 1):
        t = i / n
        z = 0.040 + L * t
        w = w0 * (1 - 0.22 * t) if t < 0.82 else w0 * 0.78 * (1 - (t - 0.82) / 0.18) ** 0.9
        th = 0.0045 * (1 - 0.3 * t)
        g = 0.5 if t < 0.72 else 1.0
        sec = [(-w, 0), (-0.5 * w, -th), (-0.22 * w, -th), (-0.15 * w, -th * g), (0.15 * w, -th * g), (0.22 * w, -th), (0.5 * w, -th),
               (w, 0), (0.5 * w, th), (0.22 * w, th), (0.15 * w, th * g), (-0.15 * w, th * g), (-0.22 * w, th), (-0.5 * w, th)]
        if i == n:
            rings.append([bm.verts.new((0, 0, z))])
        else:
            rings.append([bm.verts.new((x, y, z)) for x, y in sec])
    for a, b in zip(rings, rings[1:]):
        k = len(a)
        for j in range(k):
            jn = (j + 1) % k
            if len(b) == 1:
                bm.faces.new((a[j], a[jn], b[0]))
            else:
                bm.faces.new((a[j], a[jn], b[jn], b[j]))
    bm.faces.new(rings[0][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    blade = mesh_object("blade", bm, "metal")
    lib.paint(blade, "trim", lambda c: abs(c.x) < 0.0012 and c.z < 0.040 + L * 0.70)
    parts.append(blade)
    # Guard with quillon knobs, leather grip and a wheel pommel.
    parts.append(lib.tube([(-0.050, 0, 0.045), (-0.025, 0, 0.035), (0, 0, 0.032), (0.025, 0, 0.035), (0.050, 0, 0.045)],
                          [0.0065, 0.0055, 0.0085, 0.0055, 0.0065], role="trim", sides=8, resolution=3, name="guard"))
    for x in (-0.052, 0.052):
        parts.append(lib.sphere(0.0088, loc=(x, 0, 0.047), role="trim", segments=8, rings=4, name="quillon"))
    parts.append(lib.cylinder(0.009, 0.010, 0.068, loc=(0, 0, -0.004), role="cloth", segments=10, name="grip"))
    parts.append(lib.cylinder(0.018, 0.018, 0.012, loc=(0, 0, -0.046), rot=(math.pi / 2, 0, 0), role="trim", segments=14, bevel=0.003, name="pommel"))
    parts.append(lib.cylinder(0.008, 0.008, 0.016, loc=(0, 0, -0.046), rot=(math.pi / 2, 0, 0), role="metal", segments=8, name="pommel_gem"))
    # The guard is turned about the blade so the inner quillon points toward
    # the viewer and clears the sword-side shoulder.
    m = frame(FIST, SWORD_DIR, (0, -1, 0)) @ Matrix.Rotation(-GUARD_TURN, 4, "Z")
    for p in parts:
        put(p, m)
    return parts


# ---------------------------------------------------------------- figure


FINISH = {"ao_distance": 0.05}


def build():
    parts = []
    parts += underbody()
    parts += torso()
    parts += tabard()
    parts += cape()
    for side, (sho, elb, wri) in ((-1, (R_SHO, R_ELB, R_WRI)), (1, (L_SHO, L_ELB, L_WRI))):
        parts += pauldron(side, sho)
        parts += arm(side, sho, elb, wri)
    parts += fist(FIST, SWORD_DIR, nrm(R_WRI - R_ELB))
    parts += fist(L_WRI + nrm(L_WRI - L_ELB) * 0.02, nrm(L_ELB - L_SHO), nrm(L_WRI - L_ELB), fingers=False)
    parts += leg(1, L_HIP, L_KNEE, L_ANKLE, yaw=0.25)
    parts += leg(-1, R_HIP, R_KNEE, R_ANKLE, yaw=-0.65)
    parts += [put(p, HEAD) for p in helm()]
    parts += shield()
    parts += sword()
    return parts
