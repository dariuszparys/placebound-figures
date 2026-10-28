"""Phoenix: a Light flyer, a majestic firebird hovering with its wings raised
high and fanned like flames. Every flight feather, crest plume and tail
streamer ends in a flame-tongue tip (trim) that licks upward. Hooked raptor
beak, glowing eyes, a collar of flame hackles, tucked talons. Feathered, never
scaly, so it cannot be mistaken for the Dragon.

The figure faces -Y; its right wing is on -X, its left on +X. Z is up and the
origin is the plinth top; the bird hovers above it.
"""
import math
import os

import bmesh
import bpy
from mathutils import Vector

import lib

DEBUG = bool(os.environ.get("PHOENIX_DEBUG"))

# ---------------------------------------------------------------- helpers


def sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def bands(*spec):
    """Role by position along a feather: (t_from, role) pairs in order. The
    lateral position s (-1 one edge, 0 the rachis, 1 the other) is ignored."""
    def at(t, s=0.0):
        role = spec[0][1]
        for t0, r in spec:
            if t >= t0:
                role = r
        return role
    return at


FLIGHT = bands((0, "body"), (0.36, "cloth"), (0.54, "trim"))
PLAIN = bands((0, "body"))


def crest_w(t):
    """Crest plume outline: rooted slim in the crown (no bulb), widest before
    the middle, then drawn out into a flame tongue."""
    return (0.22 + 0.78 * math.sin(math.pi * t ** 1.05) ** 1.3) * (1 - t ** 3)


def flame_w(t):
    """Flight feather outline: a narrow quill, a full vane, then a long thin
    flame tongue over the last third."""
    return math.sin(math.pi * t ** 0.55) ** 1.5


def STREAMER_W(t):
    """Tail plume outline: a slim quill fanning out of the covert ring,
    swelling into a flame head, then a tongue."""
    base = 0.3 + 0.7 * sstep(0.04, 0.5, t) ** 1.2
    head = 0.8 + 0.2 * math.exp(-((t - 0.6) / 0.18) ** 2)
    return base * head * math.sin(math.pi * t ** 0.42) ** 0.6


def STEM(t):
    """Plume thickness: a flat vane from the root, a little fuller at the head."""
    return 0.18 - 0.06 * sstep(0.3, 0.6, t)


def _object(name, bm, face_roles):
    roles = []
    for r in face_roles:
        if r not in roles:
            roles.append(r)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    for r in roles:
        me.materials.append(lib.mat(r))
    me.polygons.foreach_set("material_index", [roles.index(r) for r in face_roles])
    me.update()
    return obj


def ribbon(pts, front, wl, wr, ridge, back, roles=FLIGHT, k=1, split=False, name="feather"):
    """A closed feather shell along a centreline: a lens cross-section with a
    raised rachis on the face (`front`) and a lower one behind, thin vane
    edges, pointed at both ends. `front` is one hint vector or one per point;
    wl/wr/ridge/back give the half widths and rachis heights per point. With
    split=True each cell is two triangles whose roles come from roles(t, s),
    so colour boundaries can lick across the vane."""
    pts = [Vector(p) for p in pts]
    n = len(pts)
    if isinstance(front, Vector) or isinstance(front[0], (int, float)):
        fronts = [Vector(front)] * n
    else:
        fronts = [Vector(f) for f in front]
    bm = bmesh.new()
    rows = [[bm.verts.new(pts[0])]]
    for i in range(1, n - 1):
        p = pts[i]
        tan = (pts[i + 1] - pts[i - 1]).normalized()
        f = fronts[i] - tan * fronts[i].dot(tan)
        f.normalize()
        side = tan.cross(f)
        ring = []
        for j in range(-k, k + 1):
            s = j / k
            w = wr[i] if s > 0 else wl[i]
            ring.append(bm.verts.new(p + side * (s * w) + f * (ridge[i] * (1 - abs(s)) ** 0.7)))
        for j in range(k - 1, -k, -1):
            s = j / k
            w = wr[i] if s > 0 else wl[i]
            ring.append(bm.verts.new(p + side * (s * w) - f * (back[i] * (1 - abs(s)) ** 0.7)))
        rows.append(ring)
    rows.append([bm.verts.new(pts[-1])])
    # Lateral position of each ring vertex: -1 at one edge, 0 on the rachis.
    lat = [j / k for j in range(-k, k + 1)] + [j / k for j in range(k - 1, -k, -1)]
    face_roles = []
    for i in range(n - 1):
        a, b = rows[i], rows[i + 1]
        t0, t1 = i / (n - 1), (i + 1) / (n - 1)
        role = None if split else roles((i + 0.5) / (n - 1))
        if len(a) == 1:
            m = len(b)
            for j in range(m):
                bm.faces.new((a[0], b[(j + 1) % m], b[j]))
                face_roles.append(roles((t0 + 2 * t1) / 3, (lat[j] + lat[(j + 1) % m]) / 3) if split else role)
        elif len(b) == 1:
            m = len(a)
            for j in range(m):
                bm.faces.new((a[j], a[(j + 1) % m], b[0]))
                face_roles.append(roles((2 * t0 + t1) / 3, (lat[j] + lat[(j + 1) % m]) / 3) if split else role)
        elif split:
            # Two triangles per cell, cut on the diagonal that runs from the
            # inner corner of the lower ring to the outer corner of the upper
            # one, so a role boundary drawn by (t, s) is a pointed tongue.
            m = len(a)
            for j in range(m):
                j2 = (j + 1) % m
                if abs(lat[j]) <= abs(lat[j2]):
                    tri = [((a[j], a[j2], b[j2]), (2 * t0 + t1) / 3, (lat[j] + 2 * lat[j2]) / 3),
                           ((a[j], b[j2], b[j]), (t0 + 2 * t1) / 3, (2 * lat[j] + lat[j2]) / 3)]
                else:
                    tri = [((a[j], a[j2], b[j]), (2 * t0 + t1) / 3, (2 * lat[j] + lat[j2]) / 3),
                           ((a[j2], b[j2], b[j]), (t0 + 2 * t1) / 3, (lat[j] + 2 * lat[j2]) / 3)]
                for vs, tc, sc in tri:
                    bm.faces.new(vs)
                    face_roles.append(roles(tc, sc))
        else:
            m = len(a)
            for j in range(m):
                bm.faces.new((a[j], a[(j + 1) % m], b[(j + 1) % m], b[j]))
                face_roles.append(role)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _object(name, bm, face_roles)


def slab(top, bottom, roles, cyclic=False, name="slab"):
    """A closed shell between two grids of points, top[i][j] and bottom[i][j]
    (i across, j along); roles[i][j] names each cell's role."""
    ni, nj = len(top), len(top[0])
    bm = bmesh.new()
    T = [[bm.verts.new(p) for p in row] for row in top]
    B = [[bm.verts.new(p) for p in row] for row in bottom]
    fr = []
    span = range(ni) if cyclic else range(ni - 1)
    for i in span:
        i2 = (i + 1) % ni
        for j in range(nj - 1):
            r = roles[i][j]
            bm.faces.new((T[i][j], T[i2][j], T[i2][j + 1], T[i][j + 1]))
            bm.faces.new((B[i][j], B[i][j + 1], B[i2][j + 1], B[i2][j]))
            fr += [r, r]
        for j, jr in ((0, 0), (nj - 1, nj - 2)):
            bm.faces.new((T[i][j], B[i][j], B[i2][j], T[i2][j]))
            fr.append(roles[i][jr])
    if not cyclic:
        for i, ir in ((0, 0), (ni - 1, ni - 2)):
            for j in range(nj - 1):
                bm.faces.new((T[i][j], T[i][j + 1], B[i][j + 1], B[i][j]))
                fr.append(roles[ir][j])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _object(name, bm, fr)


def tips(s, count):
    """0 between feather tips, 1 at each tip's centre, along a row s in [0, 1]."""
    return math.sin(math.pi * ((s * count) % 1.0)) ** 2


def path2d(length, n, heading, sub=8):
    """n points along a curve of the given length starting at the origin along
    +v; heading(t) in radians turns it toward +u. Returns (points, headings)."""
    pts, hs = [(0.0, 0.0)], [heading(0.0)]
    u = v = 0.0
    steps = (n - 1) * sub
    ds = length / steps
    for s in range(steps):
        h = heading((s + 0.5) / steps)
        u += math.sin(h) * ds
        v += math.cos(h) * ds
        if (s + 1) % sub == 0:
            pts.append((u, v))
            hs.append(heading((s + 1) / steps))
    return pts, hs


def feather(origin, along, across, length, width, n=12, curl=lambda t: 0.0, lift=lambda t: 0.0,
            prof=flame_w, asym=0.0, ridge=0.13, back=0.07, roll=False, roles=FLIGHT, k=1, split=False,
            frame=False, twist=None, name="feather"):
    """A feather or plume grown from `origin` in the plane of `along` (its
    initial direction) and `across` (where positive curl turns it). With
    roll=False the vane lies in that plane, so the curl is a flat flame
    flick; with roll=True the vane spans the plane's normal, so the curl rolls
    it like a ribbon. lift(t) bends it out of its plane, toward its face;
    twist(t) turns the vane about its own centre line by that angle."""
    o, d, q = Vector(origin), Vector(along).normalized(), Vector(across)
    q = (q - d * q.dot(d)).normalized()
    nrm = d.cross(q)
    path, hs = path2d(length, n, curl)
    pts, fronts = [], []
    for i, ((u, v), h) in enumerate(zip(path, hs)):
        t = i / (n - 1)
        perp = q * math.cos(h) - d * math.sin(h)
        face = perp if roll else nrm
        if twist:
            a = twist(t)
            face = face * math.cos(a) + ((-nrm) if roll else perp) * math.sin(a)
        pts.append(o + d * v + q * u + face * (lift(t) * length))
        fronts.append(face)
    w = [width * prof(i / (n - 1)) for i in range(n)]
    wl = [x * (1 - asym) for x in w]
    wr = [x * (1 + asym) for x in w]
    rf = ridge if callable(ridge) else (lambda t: ridge)
    bf = back if callable(back) else (lambda t: back)
    rd = [x * rf(i / (n - 1)) + 0.0012 for i, x in enumerate(w)]
    bk = [x * bf(i / (n - 1)) + 0.0008 for i, x in enumerate(w)]
    obj = ribbon(pts, fronts, wl, wr, rd, bk, roles=roles, k=k, split=split, name=name)
    if not frame:
        return obj

    def at(t):
        """Point, tangent, width direction and half width at t along it."""
        f = t * (n - 1)
        i = min(int(f), n - 2)
        g = f - i
        p = pts[i].lerp(pts[i + 1], g)
        tan = (pts[min(i + 2, n - 1)] - pts[max(i - 1, 0)]).normalized()
        fr = fronts[i] - tan * fronts[i].dot(tan)
        side = tan.cross(fr.normalized())
        return p, tan, side, width * prof(t)
    return obj, at


def TONGUE_W(t):
    """A flame tongue: a full base tapering to a needle tip."""
    return math.sin(math.pi * (0.25 + 0.75 * t) ** 0.8) ** 1.2 * (1 - t) ** 0.25


def tongues(at, t, forks, roles=None, name="tongue"):
    """Flame tongues forking off a plume at t: (side, spread, length, width,
    flick) per tongue. Each leaves the vane at its edge (side -1 or 1), splays
    away from the parent in the vane's plane and flicks further out, so the
    plume ends in two or three licks of flame."""
    p, tan, side_dir, hw = at(t)
    out = []
    for side, spread, length, width, flick in forks:
        d = tan * math.cos(spread) + side_dir * (side * math.sin(spread))
        q = -tan * math.sin(spread) * side + side_dir * math.cos(spread)
        q = q * side
        o = p + side_dir * (side * hw * 0.45) - tan * (length * 0.12)
        out.append(feather(o, d, q, length, width, n=7, curl=lambda x, f=flick: f * sstep(0.1, 1.0, x),
                           prof=TONGUE_W, ridge=0.18, back=0.12, roles=roles or bands((0, "trim")), name=name))
    return out


def join_mirror(parts, name):
    """Joins parts that sit on +X and mirrors them onto -X."""
    obj = lib.join(parts, name)
    return lib.mirror(obj, "X", bisect=False)


# ---------------------------------------------------------------- the wing
#
# Each wing is laid out flat in its own plane: a outward from the shoulder,
# b up, c toward the front. wing_point() bends that plane into the world,
# sweeping the outer wing back and leaning the upper wing back, so the fan is
# cupped rather than a flat board.

SHOULDER = Vector((0.066, 0.03, 0.655))


def wing_point(co):
    a, b, c = co
    x, y, z = a, -c, b
    t = 0.10 + 0.34 * max(b, 0.0)
    y, z = y * math.cos(t) + z * math.sin(t), -y * math.sin(t) + z * math.cos(t)
    w = 0.12 + 0.40 * max(a, 0.0)
    x, y = x * math.cos(w) - y * math.sin(w), x * math.sin(w) + y * math.cos(w)
    return SHOULDER + Vector((x, y, z))


ARM = [(0.0, 0.0), (0.112, 0.056), (0.19, 0.162), (0.238, 0.282)]

# Flight feathers: (u along the arm, direction in degrees, length, width),
# from the outermost primary at the hand to the innermost tertial.
FLIGHT_FEATHERS = [
    (1.00, 92, 0.255, 0.040), (0.955, 81, 0.300, 0.047), (0.91, 70, 0.300, 0.050),
    (0.86, 60, 0.282, 0.051), (0.81, 50, 0.266, 0.051), (0.76, 41, 0.248, 0.050),
    (0.71, 33, 0.232, 0.048),
    (0.645, 25, 0.214, 0.047), (0.585, 17, 0.206, 0.046), (0.525, 9, 0.196, 0.046),
    (0.465, 1, 0.188, 0.045), (0.405, -7, 0.176, 0.044), (0.345, -16, 0.164, 0.043),
    (0.285, -25, 0.150, 0.042), (0.22, -35, 0.134, 0.040), (0.15, -45, 0.116, 0.037),
    (0.085, -55, 0.098, 0.033),
]


def arm_at(u):
    seg = [math.dist(p, q) for p, q in zip(ARM, ARM[1:])]
    s = u * sum(seg)
    for (p, q), L in zip(zip(ARM, ARM[1:]), seg):
        if s <= L or q == ARM[-1]:
            f = min(s / L, 1.0)
            return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
        s -= L
    return ARM[-1]


def flight_at(u):
    """Direction (radians) and length of the flight feathers at u."""
    tab = sorted(FLIGHT_FEATHERS)
    if u <= tab[0][0]:
        return math.radians(tab[0][1] - (tab[0][0] - u) * 160), tab[0][2]
    for (u0, t0, l0, _), (u1, t1, l1, _) in zip(tab, tab[1:]):
        if u <= u1:
            f = (u - u0) / (u1 - u0)
            return math.radians(t0 + (t1 - t0) * f), l0 + (l1 - l0) * f
    return math.radians(tab[-1][1]), tab[-1][2]


def wing_feather(u, theta_deg, length, width, c0, bend, n, flick=0.0, roles=FLIGHT, split=False, name="flight"):
    a0, b0 = arm_at(u)
    th = math.radians(theta_deg)
    # Flames rise: tips pointing outward or down turn up the most, while the
    # topmost primaries flick outward.
    curl_amt = 0.26 + 0.5 * (1 - math.sin(th)) / 2 - 0.62 * sstep(55, 95, theta_deg)
    f = feather(
        (a0, b0, c0), (math.cos(th), math.sin(th), 0), (-math.sin(th), math.cos(th), 0), length, width, n=n,
        curl=lambda t: curl_amt * sstep(0.2, 1.0, t) ** 1.3 + flick * math.sin(math.tau * (t - 0.5)) * sstep(0.45, 0.7, t),
        prof=flame_w, asym=0.14, ridge=0.09, back=0.06, roles=roles, split=split, name=name,
    )
    # Bend the vane back out of the wing plane, more toward the tip.
    lib.deform(f, lambda co: Vector((co.x, co.y, co.z + bend * ((co.x - a0) ** 2 + (co.y - b0) ** 2) / length ** 2)))
    return lib.deform(f, wing_point)


def wing_tier(count, u0, u1, reach, c_in, thick, depth, roles, facing=1, per=4, rows=3, name="tier"):
    """A shingled row of coverts: a shell growing from the arm along the fan
    directions, `reach(u, flight_length)` long, whose free edge is cut into
    `count` pointed feather tips with grooves between them. c_in(r) is the
    depth of its inner face (the side resting on the feathers below)."""
    top, bot, rl = [], [], []
    ni = count * per + 1
    for i in range(ni):
        s = i / (ni - 1)
        u = u0 + (u1 - u0) * s
        a0, b0 = arm_at(u)
        th, lf = flight_at(u)
        L = reach(u, lf) * (0.35 + 0.65 * (1 - sstep(0.88, 1.0, s)))
        tip = tips(s, count)
        rmax = 1 - depth * (1 - tip ** 0.6)
        t_row, b_row, r_row = [], [], []
        for j in range(rows + 1):
            r = rmax * j / rows
            a = a0 + math.cos(th) * r * L
            b = b0 + math.sin(th) * r * L
            ci = c_in(r)
            h = thick * (1 - 0.8 * r ** 1.5) * (0.55 + 0.45 * tip) + 0.0006
            t_row.append(wing_point((a, b, ci + facing * h)))
            b_row.append(wing_point((a, b, ci)))
            r_row.append(roles((j + 0.5) / rows * rmax))
        top.append(t_row)
        bot.append(b_row)
        rl.append(r_row)
    return slab(top, bot, rl, name=name)


def wing():
    parts = []
    add = parts.append
    fl = FLIGHT_FEATHERS
    for i, (u, th, L, W) in enumerate(fl):
        c0 = -0.002 + 0.0032 * (i % 3)
        L *= 1.0 if i % 2 == 0 else 0.9
        # The flame front is ragged: neighbouring feathers catch fire at very
        # different heights, and each burn line is a tongue licking down the
        # rachis, so the wing reads as fire rather than a dipped stripe.
        # (The fifth and sixth swap, so the sixth, tucked between its
        # neighbours, burns late and shows no stray fleck of flame.)
        fire = 0.36 + 0.26 * (((4 if i == 5 else 6 if i == 4 else i) * 5) % 7) / 6

        def roles(t, s, fire=fire):
            return "trim" if t > fire + 0.15 * (abs(s) - 0.5) else "body"
        add(wing_feather(u, th, L, W, c0, -0.024, 16, flick=0.6 if i % 2 else -0.45, roles=roles, split=True))
    # Greater coverts over the flight feather roots, then the lesser coverts
    # that clothe the arm; under-wing coverts behind. The greater row stops
    # short of the hand and shortens toward it, so a few broad coverts, not a
    # fern of blades, cover the primaries' roots.
    add(wing_tier(13, 0.03, 0.85, lambda u, lf: lf * (0.42 - 0.15 * sstep(0.6, 1.0, u)), lambda r: 0.0125 - 0.004 * r,
                  0.0045, 0.2, PLAIN, per=3, rows=2, name="greater"))
    add(wing_tier(11, 0.0, 0.95, lambda u, lf: (lf * 0.24 + 0.02) * (1 - 0.35 * sstep(0.6, 1.0, u)),
                  lambda r: 0.02 - 0.003 * r, 0.008, 0.3, PLAIN, per=3, rows=2, name="lesser"))
    add(wing_tier(6, 0.05, 1.0, lambda u, lf: lf * 0.38, lambda r: -0.0075 - 0.005 * r, 0.0045, 0.4,
                  PLAIN, facing=-1, per=3, rows=1, name="under"))
    # The alula: one slim flame licking forward and up off the wrist.
    a0, b0 = arm_at(0.69)
    t = math.radians(116)
    f = feather((a0, b0, 0.02), (math.cos(t), math.sin(t), 0), (-math.sin(t), math.cos(t), 0), 0.095, 0.012, n=9,
                curl=lambda x: -0.5 * sstep(0.3, 1.0, x), prof=flame_w, ridge=0.1, back=0.08,
                roles=bands((0, "body"), (0.55, "trim")), name="alula")
    add(lib.deform(f, wing_point))
    # The arm: a slim bone from shoulder to hand, tucked behind the leading
    # edge of the coverts (set back along the fan) so feathers, not a bare
    # forearm, make the wing's edge.
    us = [0.0, 0.08, 0.18, 0.3, 0.45, 0.6, 0.75, 0.88, 1.0]
    radii = [0.026, 0.017, 0.011, 0.009, 0.0085, 0.0085, 0.0075, 0.0055, 0.003]
    pts = []
    for u, r in zip(us, radii):
        a, b = arm_at(u)
        th, _ = flight_at(u)
        k = sstep(0.0, 0.2, u) * r * 1.1
        pts.append(wing_point((a + math.cos(th) * k, b + math.sin(th) * k, 0.009)))
    add(lib.tube(pts, radii, role="body", sides=10, resolution=2, name="arm"))
    # Marginal coverts scallop the back of the leading edge.
    add(wing_tier(8, 0.0, 0.8, lambda u, lf: lf * 0.13 + 0.018, lambda r: -0.0025 - 0.004 * r, 0.007, 0.5,
                  PLAIN, facing=-1, per=3, rows=1, name="marginal"))
    return join_mirror(parts, "wings")


# ---------------------------------------------------------------- the bird

SPINE = [  # (y, z, radius) down the body's centre line, tail to head
    (0.112, 0.452, 0.030),
    (0.070, 0.480, 0.050),
    (0.030, 0.528, 0.066),
    (-0.010, 0.592, 0.077),
    (-0.030, 0.655, 0.068),
    (-0.040, 0.712, 0.054),
    (-0.047, 0.765, 0.045),
    (-0.056, 0.812, 0.043),
]
# The head grows out of the same skeleton, so the neck flows into it: the
# occiput, the crown over the eyes, the face and the cere.
HEAD = [(-0.054, 0.846, 0.038), (-0.074, 0.861, 0.037), (-0.096, 0.858, 0.027), (-0.109, 0.856, 0.017)]
# The head is modelled at life size, then scaled up about the top of the neck
# to heroic proportions, so it reads at board scale.
HEAD_PIVOT = Vector((0.0, -0.056, 0.825))
HEAD_SCALE = 1.3


def heroic(co):
    return HEAD_PIVOT + (Vector(co) - HEAD_PIVOT) * HEAD_SCALE


def heroic_z(z):
    return HEAD_PIVOT.z + (z - HEAD_PIVOT.z) * HEAD_SCALE


def spine_y(z):
    pts = sorted((p[1], p[0]) for p in SPINE)
    if z <= pts[0][0]:
        return pts[0][1]
    for (z0, y0), (z1, y1) in zip(pts, pts[1:]):
        if z <= z1:
            return y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    return pts[-1][1]


def body():
    chain = SPINE + [(*heroic((0, y, z)).yz, r * HEAD_SCALE) for y, z, r in HEAD]
    joints = [(0.0, y, z) for y, z, _ in chain] + [(0.068, 0.032, 0.655), (-0.068, 0.032, 0.655)]
    n = len(chain)
    edges = [(i, i + 1) for i in range(n - 1)] + [(4, n), (4, n + 1)]
    radii = [r for _, _, r in chain] + [0.032, 0.032]
    b = lib.skin(joints, edges, radii, role="body", name="body")

    def shape(co):
        x, y, z = co
        # A deep keeled breast and a slimmer back.
        breast = math.exp(-((z - 0.6) / 0.075) ** 2)
        if y < spine_y(z):
            y = spine_y(z) + (y - spine_y(z)) * (1 + 0.4 * breast)
        x *= 0.95
        # A narrow raptor head with a flat crown.
        head = sstep(heroic_z(0.82), heroic_z(0.85), z)
        x *= 1 - 0.14 * head
        crown = heroic_z(0.886)
        if z > crown:
            z = crown + (z - crown) * (1 - 0.45 * head)
        return Vector((x, y, z))
    return lib.deform(b, shape)


def surface(obj, z, az, centre=None):
    """Where a horizontal ray at height z and azimuth az (0 = front, -Y;
    positive toward +X) meets `obj`, aimed at the body's centre line or at
    the vertical axis through `centre` (x, y)."""
    out = Vector((math.sin(az), -math.cos(az), 0.0))
    c = Vector((0.0, spine_y(z), z)) if centre is None else Vector((centre[0], centre[1], z))
    hit, loc, nrm, _ = obj.ray_cast(c + out * 0.3, -out)
    if not hit:
        raise RuntimeError(f"ray at z={z:.3f} az={az:.2f} missed {obj.name}")
    return loc, nrm


def body_tier(obj, z_top, z_bot, az0, az1, count, lift, thick, roles, per=4, rows=2, depth=0.5, flare=0.0, front=1.0,
              centre=None, name="tier"):
    """A shingled row of feathers wrapped on the body between two heights,
    its lower edge cut into pointed tips. Each row lifts off the surface
    toward its tips, so the row above overlaps the next."""
    full = abs(az1 - az0 - math.tau) < 1e-6
    ni = count * per + (0 if full else 1)
    top, bot, rl = [], [], []
    for i in range(ni):
        s = i / (count * per)
        az = az0 + (az1 - az0) * s
        tip = tips(s, count)
        rmax = 1 - depth * (1 - tip ** 0.6)
        # `front` < 1 shortens the row toward the front of the body.
        reach = front + (1 - front) * (1 - math.cos(az)) / 2
        t_row, b_row, r_row = [], [], []
        for j in range(rows + 1):
            r = rmax * j / rows
            z = z_top - (z_top - z_bot) * r * reach
            p, n = surface(obj, z, az, centre)
            off = 0.0004 + lift * r ** 1.3
            fl = flare * reach * r * r
            p = p + Vector((0, 0, -fl * 0.5)) + n * fl
            # Flush with the body along the top edge, thickest just below it.
            h = thick * (0.05 + 0.95 * sstep(0.0, 0.4, r)) * (1 - 0.65 * r) * (0.6 + 0.4 * tip)
            b_row.append(p + n * off)
            t_row.append(p + n * (off + h))
            r_row.append(roles((j + 0.5) / rows * rmax))
        top.append(t_row)
        bot.append(b_row)
        rl.append(r_row)
    return slab(top, bot, rl, cyclic=full, name=name)


def plumage(b):
    """Breast, back and neck feathered in overlapping rows, and a flaring
    collar of flame hackles round the neck."""
    bpy.context.view_layer.update()
    parts = []
    front = (-1.62, 1.62)
    tiers = [(0.705, 0.64, 9), (0.66, 0.585, 10), (0.605, 0.53, 9), (0.55, 0.49, 7)]
    for k, (zt, zb, count) in enumerate(tiers):
        sh = 0.5 / count * (k % 2)
        # The lowest row glows at its tips: embers falling from the breast.
        roles = bands((0, "body"), (0.68, "cloth"), (0.85, "trim")) if k == len(tiers) - 1 else bands((0, "body"), (0.82, "cloth"))
        parts.append(body_tier(b, zt, zb, front[0] + sh * 0.3, front[1] + sh * 0.3, count, 0.004, 0.0035,
                               roles, per=3, name="breast"))
    for zt, zb, count, span in [(0.70, 0.62, 6, 1.58), (0.64, 0.56, 6, 1.58), (0.58, 0.49, 6, 1.5)]:
        parts.append(body_tier(b, zt, zb, math.pi - span, math.pi + span, count, 0.004, 0.0035, PLAIN, per=3, name="mantle"))
    # Upper tail coverts shingle the rump, so the tail grows from feathers.
    parts.append(body_tier(b, 0.515, 0.462, math.pi - 1.45, math.pi + 1.45, 8, 0.004, 0.004,
                           bands((0, "body"), (0.7, "cloth")), per=3, depth=0.5, name="coverts"))
    # The collar: long hackles that flare out over the shoulders, burning
    # from halfway down, so the fire starts at the core and spreads outward.
    # Its top edge starts up on the cheeks and nape, so the head flows into
    # the hackles instead of sitting in a ring.
    parts.append(body_tier(b, 0.848, 0.715, 0.0, math.tau, 13, 0.006, 0.005,
                           bands((0, "body"), (0.55, "trim")), rows=3, depth=0.75, flare=0.05,
                           front=0.75, name="collar"))
    # Cheek and nape feathers, and a short beard under the jaw, overlap the
    # collar's top edge with their tips, so no ring circles the neck.
    parts.append(body_tier(b, 0.884, 0.83, 0.8, math.tau - 0.8, 10, 0.003, 0.003, PLAIN, per=3, rows=1, depth=0.3,
                           name="nape"))
    parts.append(body_tier(b, 0.853, 0.828, -0.95, 0.95, 5, 0.003, 0.003, PLAIN, per=3, rows=1, depth=0.3,
                           name="beard"))
    return parts


def loft(points, sections, segs=14, role="body", keel=1.0, name="loft"):
    """A closed body swept along `points` (in a plane x = const), with an
    oval section (half width, height above, depth below) at each point; a
    section of zero width ends it in a point. keel < 1 sharpens the lower
    edge, like the cutting edge of a beak."""
    pts = [Vector(p) for p in points]
    side = Vector((1.0, 0.0, 0.0))
    bm = bmesh.new()
    rings = []
    for i, (p, (w, top, bot)) in enumerate(zip(pts, sections)):
        if w <= 1e-6:
            rings.append([bm.verts.new(p)])
            continue
        tan = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        up = tan.cross(side).normalized()
        ring = []
        for k in range(segs):
            a = math.tau * k / segs
            c = math.cos(a)
            v = top * c if c >= 0 else -bot * abs(c) ** keel
            ring.append(bm.verts.new(p + side * (w * math.sin(a)) + up * v))
        rings.append(ring)
    if len(rings[0]) > 1:
        rings.insert(0, [bm.verts.new(sum((v.co for v in rings[0]), Vector()) / len(rings[0]))])
    if len(rings[-1]) > 1:
        rings.append([bm.verts.new(sum((v.co for v in rings[-1]), Vector()) / len(rings[-1]))])
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1:
            for k in range(segs):
                bm.faces.new((a[0], b[(k + 1) % segs], b[k]))
        elif len(b) == 1:
            for k in range(segs):
                bm.faces.new((a[k], a[(k + 1) % segs], b[0]))
        else:
            for k in range(segs):
                bm.faces.new((a[k], a[(k + 1) % segs], b[(k + 1) % segs], b[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _object(name, bm, [role] * len(bm.faces))


def head():
    parts = []
    add = parts.append
    # The beak: a short, deep raptor bill, high at the cere, its culmen
    # arching at once into a sharp hook over a short lower mandible.
    add(loft([(0, -0.104, 0.862), (0, -0.1165, 0.8615), (0, -0.1265, 0.856), (0, -0.1328, 0.846), (0, -0.134, 0.835),
              (0, -0.1305, 0.8265)],
             [(0.0108, 0.017, 0.0125), (0.0094, 0.0142, 0.0098), (0.0079, 0.0108, 0.0072), (0.0058, 0.0076, 0.005),
              (0.0036, 0.0046, 0.003), (0, 0, 0)],
             segs=16, role="skin", keel=0.6, name="beak"))
    add(loft([(0, -0.103, 0.846), (0, -0.1135, 0.843), (0, -0.1225, 0.840), (0, -0.128, 0.8395)],
             [(0.0088, 0.005, 0.0072), (0.0072, 0.004, 0.0055), (0.0048, 0.0028, 0.0035), (0, 0, 0)],
             segs=14, role="skin", name="jaw"))
    # A fleshy cere saddles the top of the beak's root.
    add(lib.tube([(-0.0095, -0.1035, 0.867), (0, -0.1055, 0.8765), (0.0095, -0.1035, 0.867)], 0.0026, role="body",
                 sides=6, resolution=3, name="cere"))
    for s in (-1, 1):
        # Sunken glowing eyes set into the skull under a brow shelf that runs
        # on toward the cere and melts into the crown at both ends.
        add(lib.sphere(0.0105, loc=(s * 0.021, -0.087, 0.864), rot=(0, 0, s * -0.45), scale=(0.45, 1.25, 0.55),
                       role="eye", segments=12, rings=8, name="eye"))
        add(lib.tube([(s * 0.014, -0.06, 0.881), (s * 0.0215, -0.079, 0.8785), (s * 0.02, -0.096, 0.8725),
                      (s * 0.011, -0.108, 0.868)], [0.0, 0.0062, 0.0055, 0.0], role="body", sides=8,
                     resolution=3, name="brow"))
    for o in parts:
        lib.deform(o, heroic)
    return parts


def crest():
    parts = []
    # Flame plumes from the crown: flat vanes facing sideways (so the side
    # view shows them), thick enough to read from the front, swept hard back.
    # They rise as a ridge from forehead to nape, the outer ones fanned wide,
    # so the front shows a low, broad crown of flame, not an upright bouquet.
    specs = [  # (angle, length, half width, y of the root)
        (-0.7, 0.13, 0.024, -0.042), (-0.35, 0.165, 0.028, -0.068), (0.0, 0.195, 0.031, -0.08),
        (0.35, 0.165, 0.028, -0.062), (0.7, 0.13, 0.024, -0.036),
    ]
    stem = lambda t: 0.12 + 0.18 * sstep(0.0, 0.4, t)  # noqa: E731
    for i, (ang, L, W, y0) in enumerate(specs):
        o = heroic((math.sin(ang) * 0.014, y0, 0.864))
        up = Vector((math.sin(ang) * 0.9, 0.6, 1.0)).normalized()
        across = Vector((math.sin(ang) * 1.4, 1.0, 0.0))

        def roles(t, s, i=i):
            return "trim" if t > 0.18 + 0.1 * abs(s) + 0.04 * (i % 2) else "body"
        f, at = feather(o, up, across, L, W, n=20,
                        curl=lambda t: 1.45 * sstep(0.1, 1.0, t) ** 1.1 - 0.1 * math.sin(math.pi * t),
                        prof=crest_w, ridge=stem, back=stem, k=2, roles=roles, split=True, frame=True, name="crest")
        parts.append(f)
        # Each plume splits into licks of flame near its end.
        parts += tongues(at, 0.62, [(-1, 0.42, L * 0.36, W * 0.55, 0.5), (1, 0.38, L * 0.3, W * 0.5, 0.5)],
                         name="crest_tongue")
    return parts


def plume_fire(fire):
    """Body below a burn line that licks down the rachis, fire above it."""
    def roles(t, s):
        return "trim" if t > fire + 0.15 * (abs(s) - 0.5) else "body"
    return roles


# Tail streamers, laid out in the world frame (after the body's lean), so
# they can be aimed at what the board camera sees: (direction, curl plane,
# length, half width, curl, where the curl starts, twist, side tongue t).
# Four burst down and out from the rump like an inverted crown of fire, each
# licking outward, back and up at its end just above the hover floor; the
# middle one trails down and back and scrolls up behind, the long tail the
# side view shows. No two are mirror images.
STREAMERS = [
    ((0.0, 0.4, -1.0), (0.15, 1.0, 0.3), 0.46, 0.05, 2.6, 0.35, 0.9, 0.5),
    ((0.5, 0.12, -1.0), (1.0, 0.15, 0.25), 0.45, 0.046, 1.85, 0.3, -0.7, 0.46),
    ((-0.56, 0.12, -1.0), (-1.0, 0.15, 0.25), 0.43, 0.046, 1.75, 0.3, 0.7, 0.55),
    ((1.05, 0.08, -1.0), (1.0, 0.15, 0.5), 0.38, 0.04, 1.5, 0.3, 0.6, 0.5),
    ((-1.0, 0.08, -1.0), (-1.0, 0.15, 0.5), 0.40, 0.04, 1.3, 0.3, -0.6, 0.42),
]


def tail():
    parts = []
    root = Vector((0.0, 0.105, 0.452))
    # A short fan of flame feathers at the rump (in the body's frame),
    # draping down over the streamer roots rather than jutting back.
    fan = []
    for spread in (-0.75, -0.4, 0.0, 0.4, 0.75):
        d = Vector((math.sin(spread) * 0.9, 0.55, -0.8)).normalized()
        fan.append(feather(root, d, (0, 0, 1), 0.17 - abs(spread) * 0.03, 0.05, n=11,
                           curl=lambda t: 0.5 * sstep(0.3, 1.0, t), prof=flame_w, roll=True,
                           ridge=0.12, back=0.08, roles=plume_fire(0.5), split=True, name="tailfan"))
    for f in fan:
        lib.deform(f, lean)
    parts += fan
    # Long streamers fanning down and out of the covert ring. Each twists
    # along its length, so from behind it shows edges and curls, and
    # flickers with tongues along its edges as well as at its end.
    for i, (d, q, L, W, curl, c0, tw, ts) in enumerate(STREAMERS):
        o = lean(root + Vector((0.008 * d[0], -0.01, 0.0)))
        f, at = feather(o, d, q, L, W, n=24,
                        curl=lambda t, c=curl, c0=c0: c * sstep(c0, 1.0, t) ** 1.5,
                        twist=lambda t, a=tw: a * math.sin(math.pi * min(t / 0.85, 1.0)),
                        prof=STREAMER_W, ridge=STEM, back=STEM, roles=plume_fire(0.3), split=True,
                        frame=True, name="streamer")
        parts.append(f)
        side = 1 if i % 2 else -1
        parts += tongues(at, ts, [(side, 0.4, L * 0.16, W * 0.42, 0.45)], name="streamer_lick")
        parts += tongues(at, ts + 0.16, [(-side, 0.4, L * 0.14, W * 0.4, 0.45)], name="streamer_lick")
        parts += tongues(at, 0.74, [(-1, 0.5, L * 0.2, W * 0.5, 0.6), (1, 0.45, L * 0.17, W * 0.45, 0.6)],
                         name="streamer_tongue")
    return parts


def legs():
    """Feathered thighs with the talons drawn up and clenched under the breast."""
    parts = []
    for s in (-1, 1):
        x = s * 0.04
        thigh = lib.sphere(0.028, loc=(x, 0.042, 0.49), scale=(0.85, 1.2, 0.9), role="body", segments=8, rings=5, name="thigh")
        parts.append(thigh)
        bpy.context.view_layer.update()
        parts.append(body_tier(thigh, 0.508, 0.47, 0.0, math.tau, 6, 0.004, 0.003, bands((0, "body"), (0.8, "cloth")),
                               per=2, centre=(x, 0.042), name="trousers"))
        knee = (x * 1.05, 0.03, 0.478)
        ankle = (x * 1.0, 0.0, 0.482)
        parts.append(lib.tube([knee, ankle], [0.0095, 0.0085], role="skin", sides=6, resolution=1, name="tarsus"))
        # Clenched talons drawn up tight under the breast.
        for ang, ln in ((-0.5, 0.9), (0.0, 1.0), (0.5, 0.9)):
            dx = math.sin(ang) * 0.01
            pts = [ankle, (ankle[0] + dx * 1.3, ankle[1] - 0.018 * ln, ankle[2] - 0.008),
                   (ankle[0] + dx * 1.2, ankle[1] - 0.011 * ln, ankle[2] - 0.02)]
            parts.append(lib.tube(pts, [0.0055, 0.0042, 0.001], role="skin", sides=6, resolution=2, name="talon"))
    return parts


# The body leans forward about the wing roots, so the bird reads as rising
# into flight rather than perching upright; the wings keep their raised V.
PIVOT = Vector((0.0, 0.03, 0.655))
LEAN = math.radians(16)


def lean(co):
    y, z = co.y - PIVOT.y, co.z - PIVOT.z
    c, s = math.cos(LEAN), math.sin(LEAN)
    return Vector((co.x, PIVOT.y + y * c - z * s, PIVOT.z + y * s + z * c))


def build():
    b = body()
    groups = {
        "wings": [wing()],
        "plumage": plumage(b),
        "head": head(),
        "crest": crest(),
        "tail": tail(),
        "legs": legs(),
        "body": [b],
    }
    parts = []
    for name, objs in groups.items():
        if DEBUG:
            print(f"PART {name}: {sum(lib.tris(o) for o in objs)} tris in {len(objs)} objects")
        # The wings keep their raised V; the tail leans its own fan and lays
        # its streamers out in the world frame.
        if name not in ("wings", "tail"):
            for o in objs:
                lib.deform(o, lean)
        parts += objs
    return parts
