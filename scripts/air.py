"""Air: the elemental that serves either side, a flyer whose attack is the air
elemental arrow. A living whirlwind: a tornado funnel wound from overlapping
wraps of wind, a bright helix on each lip, rises from a point above the
plinth, swells into the chest and winds on up into a dome that becomes a
thick second twist of wind for a neck; banks of cumulus wheel round its
shoulders and pour on down round the funnel, and a cloud crown of tight
billows frames a scowling face with glowing eyes. Each arm is a sideways
mini-funnel of two wraps with a lit lip, ending in a vortex hand of whirling
blades round a glowing eye, the right raised, the left thrust at the foe;
wraps of wind peel off the funnel and streamers torn from it spiral round
the whole figure.
The figure faces -Y; its right hand is on -X and its left on +X."""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import lib

TAU = math.tau
DEBUG = bool(os.environ.get("AIR_DEBUG"))
# Decimated clouds keep smooth normals; only real corners split.
FINISH = {"sharp_angle": 70}


def note(name, obj):
    if DEBUG:
        print(f"AIR {name}: {lib.tris(obj)} tris")
    return obj


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def frac(u):
    return u - math.floor(u)


def interp(table, t):
    """Smooth (C1 Catmull-Rom) interpolation through (t, value) pairs."""
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
    nr = r - r.dot(tang[0]) * tang[0]
    if nr.length < 1e-6:
        nr = Vector((1, 0, 0)) - tang[0].x * tang[0]
    norms = [nr.normalized()]
    for i in range(1, n):
        m = norms[-1] - norms[-1].dot(tang[i]) * tang[i]
        norms.append(m.normalized())
    binorms = [t.cross(m) for t, m in zip(tang, norms)]
    return tang, norms, binorms


def frame(origin, z, front=(0, -1, 0)):
    """A matrix whose local z points along `z` and whose local -y leans toward
    `front`, placed at `origin`."""
    z = Vector(z).normalized()
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


def sweep(pts, radius, around, role="body", name="sweep", ref=(0, -1, 0), cap_start=True, tip=True,
          roles=None, twist=None):
    """A closed tube along pts whose radius is radius(t, theta, c); t runs 0..1
    along the path, theta round it from the first frame normal, and c is the
    column (0..1) of the vertex. twist(t) turns each ring so the columns follow
    a helix, which keeps spiral creases and colour bands on mesh edges.
    roles(t, c) may give each quad its own role. `tip` closes the end in a
    point a little beyond the last ring."""
    tang, norms, binorms = frames(pts, ref)
    verts, faces, froles = [], [], []
    n = len(pts)
    for i, p in enumerate(pts):
        t = i / (n - 1)
        tw = twist(t) if twist else 0.0
        for j in range(around):
            c = j / around
            th = TAU * c + tw
            r = radius(t, th, c)
            verts.append(p + r * (math.cos(th) * norms[i] + math.sin(th) * binorms[i]))
    for i in range(n - 1):
        for j in range(around):
            a, b = i * around + j, i * around + (j + 1) % around
            faces.append((a, b, b + around, a + around))
            froles.append(roles((i + 0.5) / (n - 1), (j + 0.5) / around) if roles else role)
    if cap_start:
        verts.append(pts[0] - tang[0] * 0.002)
        c = len(verts) - 1
        faces += [(c, (j + 1) % around, j) for j in range(around)]
        froles += [roles(0.0, 0.5) if roles else role] * around
    if tip:
        step = (pts[-1] - pts[-2]).length
        verts.append(pts[-1] + tang[-1] * step * 0.8)
        c = len(verts) - 1
        base = (n - 1) * around
        faces += [(base + j, base + (j + 1) % around, c) for j in range(around)]
        froles += [roles(1.0, 0.5) if roles else role] * around
    return mesh_object(name, verts, faces, role, froles if roles else None)


def ribbon(pts, width, thick, normal, role="trim", name="ribbon", roles=None):
    """A thin strip along pts with a lens section. width(t) is its full width,
    normal(i) the direction its flat face looks; roles(t) may colour it along
    its length."""
    tang = frames(pts)[0]
    verts, faces, froles = [], [], []
    n = len(pts)
    sec = [(-1, 0), (-0.55, 1), (0.55, 1), (1, 0), (0.55, -1), (-0.55, -1)]
    for i, p in enumerate(pts):
        t = i / (n - 1)
        nv = Vector(normal(i))
        nv = (nv - nv.dot(tang[i]) * tang[i]).normalized()
        wv = tang[i].cross(nv).normalized()
        w = max(width(t), 0.0005) / 2
        th = thick * min(1.0, 0.35 + w / 0.006)
        for u, v in sec:
            verts.append(p + wv * (u * w) + nv * (v * th / 2))
    k = len(sec)
    for i in range(n - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            faces.append((a, b, b + k, a + k))
            froles.append(roles((i + 0.5) / (n - 1)) if roles else role)
    faces.append(tuple(range(k - 1, -1, -1)))
    faces.append(tuple((n - 1) * k + j for j in range(k)))
    froles += [roles(0.0) if roles else role, roles(1.0) if roles else role]
    return mesh_object(name, verts, faces, role, froles if roles else None)


def kfac(stiff):
    """Visible radius of a lone metaball element as a share of its radius."""
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def ball(co, r, stiff=2.0):
    return {"co": tuple(co), "radius": r / kfac(stiff), "stiffness": stiff}


def to_budget(obj, target):
    n = lib.tris(obj)
    if n <= target:
        return obj
    return lib.decimate(obj, target / n)


def sculpt(obj, fns):
    """Pushes each vertex along its normal by the sum of fns(co, normal)."""
    me = obj.data
    me.update()
    moves = []
    for v in me.vertices:
        p, nv = v.co.copy(), v.normal.copy()
        moves.append(p + nv * sum(f(p, nv) for f in fns))
    for v, p in zip(me.vertices, moves):
        v.co = p
    me.update()
    return obj


def seg_dist(p, a, b):
    """Distance from 2D point p to segment a-b, and the position along it."""
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    s = 0.0 if L2 == 0 else clamp(((p[0] - ax) * dx + (p[1] - az) * dz) / L2)
    return math.hypot(p[0] - (ax + s * dx), p[1] - (az + s * dz)), s


def line_push(line, width, height, facing=(0, -1, 0), fade=True):
    """A ridge (height > 0) or crease (height < 0) drawn as a polyline in the
    x-z plane, pushed into or out of the surface where it faces `facing`."""
    fv = Vector(facing)

    def fn(p, n):
        w = smoothstep(0.1, 0.55, n.dot(fv))
        if w <= 0:
            return 0.0
        best, where = 1e9, 0.0
        total = len(line) - 1
        for k, (a, b) in enumerate(zip(line, line[1:])):
            d, s = seg_dist((p.x, p.z), a, b)
            if d < best:
                best, where = d, (k + s) / total
        end = smoothstep(0.0, 0.15, where) * smoothstep(1.0, 0.85, where) if fade else 1.0
        return height * w * end * math.exp(-(best / width) ** 2)

    return fn


def bump(center, semi, height, facing=None):
    c, s = Vector(center), Vector(semi)

    def fn(p, n):
        d = p - c
        q = (d.x / s.x) ** 2 + (d.y / s.y) ** 2 + (d.z / s.z) ** 2
        if q > 9:
            return 0.0
        w = 1.0 if facing is None else smoothstep(0.0, 0.5, n.dot(Vector(facing)))
        return height * w * math.exp(-q)

    return fn


def shingle(u, lip=0.80):
    """One wrap of wind across its width: it swells slowly out to a rounded
    lip, then drops steeply under the next wrap."""
    u = frac(u)
    if u < lip:
        return 0.5 - 0.5 * math.cos(math.pi * u / lip)
    return 1.0 - smoothstep(0.0, 1.0, (u - lip) / (1 - lip))


# ---------------------------------------------------------------- pose

# The funnel sways in a soft S from its tip up to a wide mouth under the chest.
# Its tip sits a little forward, so the cone leans back and shows its length
# to the board camera, and its top winds on up into a dome under the head.
AXIS = [V(0.040, -0.030, 0.170), V(0.030, -0.015, 0.280), V(0.006, 0.020, 0.430), V(-0.014, 0.010, 0.590),
        V(-0.008, 0.008, 0.755), V(0.000, 0.010, 0.905), V(0.000, -0.008, 0.965)]
FUNNEL_R = [(0.0, 0.007), (0.10, 0.013), (0.18, 0.024), (0.30, 0.046), (0.42, 0.080), (0.55, 0.122),
            (0.68, 0.160), (0.80, 0.192), (0.86, 0.196), (0.92, 0.165), (0.96, 0.115), (1.0, 0.060)]
STARTS = 4
FUNNEL_ROWS, FUNNEL_AROUND = 36, 40

HEAD = V(0.0, -0.040, 1.082)
HEAD_R = 0.096
HEAD_TILT = -0.16  # radians about X; negative lifts the face

# Right arm (-X) flung up with its vortex open to the sky; left arm (+X)
# thrust forward, its vortex aimed at the foe.
R_ARM = [V(-0.200, 0.035, 0.900), V(-0.295, 0.050, 0.945), V(-0.355, 0.030, 1.010), V(-0.368, -0.010, 1.065)]
L_ARM = [V(0.200, 0.020, 0.890), V(0.285, -0.030, 0.895), V(0.335, -0.110, 0.890), V(0.355, -0.200, 0.885)]


# ---------------------------------------------------------------- funnel


def funnel_twist(t):
    return TAU * 2.0 * t


def squash(t, th):
    """The mouth is flattened front to back, so the chest stays shallow and
    nothing bulks up behind the head."""
    return 1 - 0.24 * smoothstep(0.55, 0.95, t) * math.cos(th) ** 2


def funnel_radius(t, c):
    # The wraps fade out entirely at the top, so the dome closes without shards.
    amp = 0.10 + 0.30 * smoothstep(0.0, 0.35, t) - 0.40 * smoothstep(0.76, 1.0, t)
    th = TAU * c + funnel_twist(t)
    u = frac(c * STARTS)
    # A shallow second ripple on the swell up to each lip: wind striation.
    lip = (smoothstep(0.30, 0.50, u) * (1 - smoothstep(0.74, 0.80, u)) * smoothstep(0.10, 0.25, t)
           * (1 - smoothstep(0.84, 0.95, t)))
    ripple = 0.04 * math.sin(u * TAU * 3) * lip
    return interp(FUNNEL_R, t) * squash(t, th) * (1 + amp * (shingle(c * STARTS, 0.78) - 0.5) + ripple)


FUNNEL_PTS = curve(AXIS, FUNNEL_ROWS)
FUNNEL_FRAMES = frames(FUNNEL_PTS)


def funnel_point(t, c, out=0.0):
    """A point on (or `out` beyond) the funnel surface at height t and column
    c, with the outward direction there."""
    f = t * (FUNNEL_ROWS - 1)
    i = min(int(f), FUNNEL_ROWS - 2)
    w = f - i
    p = FUNNEL_PTS[i].lerp(FUNNEL_PTS[i + 1], w)
    n = FUNNEL_FRAMES[1][i].lerp(FUNNEL_FRAMES[1][i + 1], w).normalized()
    b = FUNNEL_FRAMES[2][i].lerp(FUNNEL_FRAMES[2][i + 1], w).normalized()
    th = TAU * c + funnel_twist(t)
    d = math.cos(th) * n + math.sin(th) * b
    return p + d * (funnel_radius(t, c) + out), d


def funnel():
    """The tornado: four wraps of wind wound round a swaying axis, each
    swelling to a lip that overhangs the next, light catching each lip."""
    def roles(t, c):
        u = frac(c * STARTS)
        # Above this the wraps fade into the dome; colour bands would taper
        # into slivers there, so they end cleanly.
        if t > 0.91:
            return "body"
        if 0.58 < u < 0.78 and t > 0.12:
            return "trim"
        return "cloth" if (u > 0.80 or u < 0.08) else "body"

    return note("funnel", sweep(FUNNEL_PTS, lambda t, th, c: funnel_radius(t, c), FUNNEL_AROUND, role="cloth",
                                name="funnel", twist=funnel_twist, tip=True, roles=roles))


def wind_sheet(k, ta, tb, width, name, peel=0.055):
    """A wrap of wind peeling off the funnel along its lip and flying free."""
    n = 20
    c0 = (k + 0.70) / STARTS
    pts, outs = [], []
    for i in range(n):
        s = i / (n - 1)
        t = ta + (tb - ta) * s
        p, d = funnel_point(t, c0, 0.004 + peel * smoothstep(0.35, 1.0, s) ** 1.4)
        pts.append(p + V(0, 0, 0.018 * smoothstep(0.5, 1.0, s)))
        outs.append(d)

    def wid(t):
        return width * smoothstep(0.0, 0.3, t) * (1 - t) ** 0.55

    return note(name, ribbon(pts, wid, 0.006, lambda i: outs[i], role="trim", name=name))


# ---------------------------------------------------------------- clouds


def crest(side):
    """One shoulder: a crest of cumulus swept round the funnel in the spin
    direction, its big billows biggest by the neck and dwindling out to the
    arm, small ones riding on top so the outline scallops. The two crests
    are turned half a turn from each other, so from above they wheel."""
    inner = V(-0.130, -0.030, 0.893)
    outer = V(-0.255, 0.050, 0.858)
    bow = V(-0.020, -0.030, 0.012)
    big = (0.048, 0.045, 0.040, 0.033)
    small = (0.030, 0.028, 0.026)
    pts = []
    for i, r in enumerate(big):
        f = i / (len(big) - 1)
        pts.append((inner.lerp(outer, f) + bow * math.sin(math.pi * f), r))
    for i, r in enumerate(small):
        f = (i + 0.5) / (len(big) - 1)
        p = inner.lerp(outer, f) + bow * math.sin(math.pi * f)
        pts.append((p + V(0.0, 0.012, 0.030 - 0.004 * i), r))
    # One billow low at the back of the crest hides the funnel's shoulder.
    if side < 0:
        pts.append((V(-0.175, 0.085, 0.848), 0.034))
    e = []
    for p, r in pts:
        if side > 0:
            # Half a turn about the funnel's axis.
            p = V(-p.x, 2 * CLOUD_Y - p.y, p.z)
        e.append(ball(tuple(p), r, stiff=2.7))
    # Small billows bulge off the big ones along the front and the outer
    # edge, each fused to its parent, so the outline scallops where the board
    # camera looks; none rise past the chin. They are placed per side (not
    # turned), so both shoulders scallop toward the camera.
    if side < 0:
        front = V(-0.090, -0.075, 0.905)
        groups = [((front, 0.040), [(0.026, (-0.15, -1.0, 0.25)), (0.024, (0.55, -0.9, 0.15)),
                                    (0.022, (-0.4, -0.5, 0.9))]),
                  ((pts[0][0], big[0]), [(0.028, (-0.55, -1.0, 0.0))]),
                  ((pts[1][0], big[1]), [(0.028, (-0.5, -1.0, 0.1)), (0.025, (-0.8, -0.2, 0.75))]),
                  ((pts[2][0], big[2]), [(0.026, (-1.0, -0.5, 0.2)), (0.022, (-0.7, 0.2, 0.8))])]
        e.append(ball(tuple(front), 0.040, stiff=2.7))
    else:
        # The turned crest leaves the front of this shoulder bare, so a small
        # bank of cloud rolls forward off its outer end.
        b1, b2, b3 = V(0.215, -0.065, 0.875), V(0.165, -0.085, 0.890), V(0.190, -0.040, 0.905)
        e.append(ball((0.175, 0.095, 0.848), 0.034, stiff=2.7))  # the turned crest's back billow
        e.append(ball(tuple(b1), 0.038, stiff=2.7))
        e.append(ball(tuple(b2), 0.034, stiff=2.7))
        e.append(ball(tuple(b3), 0.034, stiff=2.7))
        groups = [((b1, 0.038), [(0.025, (0.6, -1.0, 0.2)), (0.022, (1.0, -0.2, 0.5))]),
                  ((b2, 0.034), [(0.024, (-0.3, -1.0, 0.3)), (0.020, (-0.9, -0.4, 0.4))]),
                  ((b3, 0.034), [(0.022, (-0.4, -0.3, 1.0))])]
    for (c, rb), nubs in groups:
        for r, d in nubs:
            p = c + V(d).normalized() * (0.80 * (rb + r))
            if p.z + r > 0.955:
                p.z = 0.955 - r
            e.append(ball(tuple(p), r, stiff=2.7))
    return e


CLOUD_Y = 0.010


def clouds():
    """Two crests of storm cloud on the shoulders of the funnel. Nothing sits
    behind the neck, so the head stands clear of them."""
    e = crest(-1) + crest(1)
    c = lib.metaball(e, role="body", resolution=0.007, name="cloud")
    return note("cloud", to_budget(c, 4700))


def neck():
    """A short twist of wind, a second small vortex the head rides on."""
    pts = curve([V(0.0, -0.008, 0.950), V(0.0, -0.016, 0.990), V(0.0, -0.030, 1.030)], 11)
    prof = [(0.0, 0.064), (0.5, 0.058), (1.0, 0.052)]

    def radius(t, th, c):
        return interp(prof, t) * (1 + 0.24 * (shingle(c * 2, 0.78) - 0.5))

    def roles(t, c):
        u = frac(c * 2)
        return "cloth" if (u > 0.82 or u < 0.06) else "body"

    # Fourteen sides: seven columns for each of its two wraps, so the lips
    # stay smooth.
    return note("neck", sweep(pts, radius, 14, role="body", name="neck", twist=lambda t: TAU * 0.7 * t,
                              roles=roles, ref=(0, -1, 0)))


def around_funnel(a, z, out):
    """A point `out` beyond the funnel's mean surface at angle a (from the
    front toward +X) and height z."""
    i = min(range(FUNNEL_ROWS), key=lambda k: abs(FUNNEL_PTS[k].z - z))
    t = i / (FUNNEL_ROWS - 1)
    c = FUNNEL_PTS[i]
    r = interp(FUNNEL_R, t) * squash(t, a) + out
    return V(c.x + math.sin(a) * r, c.y - math.cos(a) * r, z)


def cloud_tails():
    """Each shoulder pours on down round the funnel as a chain of puffs half
    sunk in it, wound the way the wraps run and thinning to a wisp."""
    parts = []
    for a0, z0 in ((-1.78, 0.850), (1.36, 0.850)):
        e = []
        n = 16
        for j in range(n):
            s = j / (n - 1)
            a = a0 - 1.05 * TAU * s
            z = z0 - 0.38 * s ** 0.85
            # Softer, overlapping puffs fuse into one ridge that pours, not beads.
            r = (0.030 * (1 - s) ** 0.9 + 0.018) * (1.10 if j % 3 == 0 else 0.95)
            e.append(ball(tuple(around_funnel(a, z, 0.10 * r)), r, stiff=2.0))
        c = lib.metaball(e, role="body", resolution=0.008, name="cloud_tail")
        parts.append(note("tail", to_budget(c, 440)))
    return parts


# ---------------------------------------------------------------- head


def head():
    """A stern face pressed out of the cloud: a heavy scowling brow over deep
    sockets, a strong nose and cheekbones, a hard down-turned mouth."""
    o = lib.sphere(1.0, segments=44, rings=32, role="body", name="head")

    def shape(v):
        jaw = 1.0 - 0.10 * smoothstep(0.1, -1.0, v.z)
        v.x *= 0.94 * jaw
        v.y *= 0.92 if v.y > 0 else 0.80
        v.z *= 1.06
        return v

    lib.deform(o, shape)
    # Soft swells under the cloud crown, and two over the forehead; the crown
    # itself is the billows of head_cloud(). Smoothed once so nothing spikes.
    sculpt(o, [
        bump((-0.15, 0.05, 0.92), (0.26, 0.26, 0.22), 0.16),
        bump((0.45, 0.12, 0.70), (0.24, 0.26, 0.22), 0.14),
        bump((-0.60, 0.18, 0.55), (0.24, 0.26, 0.22), 0.14),
        bump((-0.32, -0.55, 0.72), (0.22, 0.26, 0.18), 0.12),
        bump((0.34, -0.50, 0.76), (0.22, 0.26, 0.18), 0.10),
    ])
    lib.smooth(o, iterations=1, factor=0.3)
    front = (0, -1, 0)
    brow = [(-0.72, 0.38), (-0.42, 0.28), (-0.10, 0.10), (0.10, 0.10), (0.42, 0.28), (0.72, 0.38)]
    sculpt(o, [
        line_push(brow, 0.20, 0.24, front),
        bump((-0.62, -0.7, 0.36), (0.14, 0.4, 0.10), 0.07, front),
        bump((0.62, -0.7, 0.36), (0.14, 0.4, 0.10), 0.07, front),
        bump((-0.26, -0.8, 0.22), (0.13, 0.4, 0.09), 0.06, front),
        bump((0.26, -0.8, 0.22), (0.13, 0.4, 0.09), 0.06, front),
        bump((-0.36, -0.8, 0.02), (0.26, 0.5, 0.12), -0.24, front),
        bump((0.36, -0.8, 0.02), (0.26, 0.5, 0.12), -0.24, front),
        # Frown furrows between the brows.
        line_push([(-0.075, 0.30), (-0.040, 0.12)], 0.030, -0.05, front),
        line_push([(0.075, 0.30), (0.040, 0.12)], 0.030, -0.05, front),
        line_push([(0.0, 0.05), (0.0, -0.22)], 0.12, 0.12, front, fade=False),
        bump((0.0, -0.9, -0.25), (0.13, 0.4, 0.08), 0.12, front),
        bump((-0.11, -0.9, -0.28), (0.07, 0.3, 0.05), 0.05, front),
        bump((0.11, -0.9, -0.28), (0.07, 0.3, 0.05), 0.05, front),
        bump((-0.48, -0.7, -0.12), (0.18, 0.4, 0.12), 0.08, front),
        bump((0.48, -0.7, -0.12), (0.18, 0.4, 0.12), 0.08, front),
        line_push([(-0.34, -0.58), (-0.16, -0.49), (0.16, -0.49), (0.34, -0.58)], 0.095, -0.14, front),
        bump((0.0, -0.85, -0.58), (0.20, 0.3, 0.06), 0.04, front),
        bump((0.0, -0.8, -0.76), (0.24, 0.4, 0.13), 0.10, front),
    ])
    parts = [o]
    # Glowing eyes set under the brow, slanted down toward the nose.
    me = o.data
    bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    for s in (-1, 1):
        hit = bvh.ray_cast(V(s * 0.37, -2.0, 0.03), V(0, 1, 0))
        loc = hit[0] if hit[0] is not None else V(s * 0.37, -0.8, 0.03)
        eye = lib.sphere(1.0, segments=10, rings=6, role="eye", name="eye")
        lib.place(eye, scale=(0.30, 0.09, 0.15))
        lib.place(eye, rot=(0.0, s * -0.34, 0.0))
        lib.place(eye, loc=loc + V(0, 0.022, 0))
        parts.append(eye)
    for p in parts:
        put(p, HEAD_M)
    return parts + [head_cloud()]


HEAD_M = Matrix.Translation(HEAD) @ Matrix.Rotation(HEAD_TILT, 4, "X") @ Matrix.Scale(HEAD_R, 4)

# Billows of the cloud crown in head space (unit radius, face toward -y):
# the crown, the back and the sides break into tight cumulus lobes, and two
# small ones trail off the lower back where the spin drags them.
HEAD_BILLOWS = [
    # Crown: a big billow over the brow and two smaller ones beside it, the
    # +x one swept back by the spin, so the top outline is irregular.
    ((-0.06, 0.18, 0.84), 0.44), ((-0.48, 0.10, 0.72), 0.30), ((0.40, 0.22, 0.78), 0.34),
    # The back, its billows dragged round toward +x.
    ((0.30, 0.66, 0.58), 0.40), ((-0.34, 0.66, 0.50), 0.34), ((0.04, 0.90, 0.16), 0.40),
    ((0.74, 0.50, 0.30), 0.30), ((-0.72, 0.56, 0.20), 0.26), ((0.66, 0.70, -0.08), 0.30),
    ((-0.56, 0.74, -0.12), 0.27), ((0.30, 0.90, -0.30), 0.30), ((-0.30, 0.88, -0.36), 0.27),
    ((0.50, 1.04, -0.48), 0.23), ((-0.24, 1.08, -0.58), 0.20),
]


def head_cloud():
    """The cranium as a storm cloud: billows half sunk in the head so its
    outline breaks into distinct lobes with creases between them."""
    e = [ball(tuple(HEAD_M @ V(p)), r * HEAD_R, stiff=2.7) for p, r in HEAD_BILLOWS]
    c = lib.metaball(e, role="body", resolution=0.006, name="head_cloud")
    return note("head_cloud", to_budget(c, 950))


# ---------------------------------------------------------------- arms


ARM_PROF = [(0.0, 0.050), (0.20, 0.047), (0.45, 0.036), (0.70, 0.031), (0.88, 0.026), (1.0, 0.028)]
ARM_TWIST = 0.8


def wind_arm(ctrl, name):
    """An arm of wind: a sideways mini-funnel of two overlapping wraps, each
    swelling to a lit lip, tapering to the wrist, so a helix of light runs
    down the arm to the vortex hand."""
    pts = curve(ctrl, 28)

    def radius(t, th, c):
        # The wraps open out of the shoulder cloud, so the buried root is round.
        amp = 0.55 * smoothstep(0.0, 0.16, t)
        return interp(ARM_PROF, t) * (1 + amp * (shingle(c * 2, 0.78) - 0.5))

    def roles(t, c):
        u = frac(c * 2)
        if 0.62 < u < 0.78 and 0.20 < t < 0.85:
            return "trim"
        return "cloth" if (u > 0.80 or u < 0.08) else "body"

    # Twelve sides: six columns for each wrap.
    # The wrist closes in a point inside the vortex hand's glowing core.
    return note(name, sweep(pts, radius, 12, role="body", name=name, twist=lambda t: TAU * ARM_TWIST * t, tip=True,
                            roles=roles))


def vortex_hand(wrist, direction, spin=1, name="hand"):
    """A hand of wind: thin blades of air whirling straight out of a glowing
    eye at the wrist, like fingers caught in a spin."""
    m = frame(wrist, direction)
    core = lib.sphere(0.024, loc=(0, 0, 0.022), segments=14, rings=8, role="eye", name=name + "_core")
    parts = [core]
    n = 13
    blades = 5
    for k in range(blades):
        pts, outs = [], []
        for i in range(n):
            s = i / (n - 1)
            a = k * TAU / blades + spin * (0.2 + 2.1 * s)
            r = 0.022 + 0.058 * s
            h = 0.020 + 0.046 * s ** 1.3
            pts.append(V(r * math.cos(a), r * math.sin(a), h))
            outs.append(V(math.cos(a) * 0.55, math.sin(a) * 0.55, 0.85))

        def wid(t):
            return 0.034 * (1 - t) ** 0.8 * (0.45 + 0.55 * smoothstep(-0.1, 0.3, t))

        parts.append(ribbon(pts, wid, 0.006, lambda i, outs=outs: outs[i], role="body", name=name + "_blade",
                            roles=lambda t: "trim" if t > 0.5 else "body"))
    return [put(p, m) for p in parts]


# ---------------------------------------------------------------- streamers


def streamer(t0, t1, c0, turns, peel, width, name, n=40, lift=0.0):
    """A band of wind torn off the funnel: it leaves the surface on a lip at
    height t0, winds `turns` times round the figure while it peels outward by
    `peel`, and tapers away."""
    dc = turns - 2.0 * (t1 - t0)
    pts, outs = [], []
    for k in range(n):
        s = k / (n - 1)
        t = t0 + (t1 - t0) * s
        p, d = funnel_point(t, c0 + dc * s, 0.004 + peel * smoothstep(0.0, 0.8, s) ** 1.2)
        pts.append(p + V(0, 0, lift * smoothstep(0.3, 1.0, s)))
        outs.append(d)

    def wid(t):
        return width * smoothstep(0.0, 0.18, t) ** 0.8 * (1 - t) ** 0.5 * (0.7 + 0.3 * t)

    return note(name, ribbon(pts, wid, 0.007, lambda i: outs[i], role="trim", name=name))


# ---------------------------------------------------------------- figure


def build():
    cloud = clouds()
    parts = [funnel(), cloud]
    parts += [wind_sheet(0, 0.26, 0.48, 0.034, "sheet"), wind_sheet(2, 0.40, 0.64, 0.038, "sheet"),
              wind_sheet(1, 0.55, 0.78, 0.040, "sheet")]
    parts += cloud_tails()
    parts.append(neck())
    parts += head()
    parts.append(wind_arm(R_ARM, "arm_r"))
    parts.append(wind_arm(L_ARM, "arm_l"))
    parts += vortex_hand(R_ARM[-1], R_ARM[-1] - R_ARM[-2], spin=-1, name="hand_r")
    parts += vortex_hand(L_ARM[-1], L_ARM[-1] - L_ARM[-2], spin=1, name="hand_l")
    parts.append(streamer(0.26, 0.70, 0.30, 1.2, 0.110, 0.038, "streamer_low"))
    parts.append(streamer(0.52, 0.86, 0.80, 0.75, 0.140, 0.034, "streamer_high", lift=0.02))
    parts.append(streamer(0.06, 0.32, 0.55, 1.4, 0.030, 0.016, "streamer_tip", n=30))
    if DEBUG:
        print(f"AIR total parts {len(parts)}: {sum(lib.tris(p) for p in parts)} tris")
    return parts
