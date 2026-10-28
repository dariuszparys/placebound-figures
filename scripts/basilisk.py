"""Basilisk: the Dark serpent king. A long scaled serpent body lies coiled on
the plinth on six short clawed lizard legs; its front rears up like a cobra,
spreading a hood behind a crowned head whose heavy brows shade the glowing,
deadly eyes. The figure faces -Y; its right side is on -X."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

# Smooth across the limbs and the hood's rim; the scale steps,
# keels and horn spikes stay crisp.
FINISH = {"sharp_angle": 48}

UP = Vector((0, 0, 1))
FWD = Vector((0, -1, 0))
ROLE = {r: i for i, r in enumerate(lib.ROLES)}


# ------------------------------------------------------------ curve helpers


def spow(a, e):
    return math.copysign(abs(a) ** e, a)


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def catmull(points, per=24):
    """Dense centripetal Catmull-Rom polyline through `points`, and the index
    of each control point in it."""
    P = [Vector(p) for p in points]
    ext = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out, idx = [], []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + max((p1 - p0).length, 1e-6) ** 0.5
        t2 = t1 + max((p2 - p1).length, 1e-6) ** 0.5
        t3 = t2 + max((p3 - p2).length, 1e-6) ** 0.5
        idx.append(len(out))
        for k in range(per):
            t = t1 + (t2 - t1) * k / per
            a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0))
            a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1))
            a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2))
            b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0))
            b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1))
            out.append(b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1)))
    idx.append(len(out))
    out.append(P[-1].copy())
    return out, idx


def arclen(pts):
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + (b - a).length)
    return s


def sample(pts, cum, s):
    """Point on the polyline at arc length s."""
    s = min(max(s, 0.0), cum[-1])
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= s:
            lo = mid
        else:
            hi = mid
    seg = cum[hi] - cum[lo]
    t = (s - cum[lo]) / seg if seg > 0 else 0
    return pts[lo].lerp(pts[hi], t)


def interp(keys, s):
    """Piecewise-linear value at s from sorted (s, value) keys."""
    if s <= keys[0][0]:
        return keys[0][1]
    for (s0, v0), (s1, v1) in zip(keys, keys[1:]):
        if s <= s1:
            t = (s - s0) / (s1 - s0) if s1 > s0 else 0
            t = t * t * (3 - 2 * t) * 0.5 + t * 0.5
            return v0 + (v1 - v0) * t
    return keys[-1][1]


# ------------------------------------------------------------ the loft


def loft(name, centers, frames, sections, sides, vert_fn=None, role_fn=None, caps=(True, True), face_fn=None):
    """A tube through `centers` whose cross-section at each station is a
    superellipse in the station's frame (L lateral, V ventral). Each section
    is a dict: w (half width), hu (dorsal half height), hd (ventral half
    height), el/evu/evd exponents (below 1 squares the shape off), cup (pushes
    the lateral edges ventrally). vert_fn(j, c, s, lat, ven) may move a vertex;
    role_fn(j, c, s) names a face's role from its station and angle;
    face_fn(j, k) may instead return [(corners, role), ...], corners indexing
    the quad (0: j,k  1: j,k+1  2: j+1,k+1  3: j+1,k), to split it."""
    bm = bmesh.new()
    rings = []
    for j, (C, (L, V), sec) in enumerate(zip(centers, frames, sections)):
        ring = []
        for k in range(sides):
            th = math.tau * k / sides
            c, s = math.cos(th), math.sin(th)
            lat = sec["w"] * spow(c, sec.get("el", 1.0))
            if s > 0:
                ven = sec["hd"] * spow(s, sec.get("evd", 1.0))
            else:
                ven = sec["hu"] * spow(s, sec.get("evu", 1.0))
            ven += sec.get("cup", 0.0) * c * c
            if vert_fn:
                lat, ven = vert_fn(j, c, s, lat, ven)
            ring.append(bm.verts.new(C + L * lat + V * ven))
        rings.append(ring)
    for j in range(len(rings) - 1):
        for k in range(sides):
            quad = (rings[j][k], rings[j][(k + 1) % sides], rings[j + 1][(k + 1) % sides], rings[j + 1][k])
            split = face_fn(j, k) if face_fn else None
            if split:
                for corners, r in split:
                    bm.faces.new([quad[i] for i in corners]).material_index = ROLE[r]
                continue
            f = bm.faces.new(quad)
            if role_fn:
                th = math.tau * (k + 0.5) / sides
                f.material_index = ROLE[role_fn(j + 0.5, math.cos(th), math.sin(th))]
    for end, ring in ((0, rings[0]), (1, rings[-1])):
        if not caps[end]:
            continue
        centre = bm.verts.new(sum((v.co for v in ring), Vector()) / len(ring))
        for k in range(sides):
            f = bm.faces.new((ring[k], ring[(k + 1) % sides], centre))
            if role_fn:
                f.material_index = ROLE[role_fn(0 if end == 0 else len(rings) - 1, 1.0, 0.0)]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    for r in lib.ROLES:
        me.materials.append(lib.mat(r))
    if not role_fn:
        for p in me.polygons:
            p.material_index = 0
    return obj


def straight_frames(n):
    """Frames for a loft along local -Y: lateral +X, ventral -Z."""
    return [(Vector((1, 0, 0)), Vector((0, 0, -1)))] * n


# ------------------------------------------------------------ the body


# Spine from the tail tip (curled up at the front left) round the right side
# and the back, through the hips and chest, then rearing up into the head.
# (point, half width)
SPINE = [
    ((0.252, -0.118, 0.048), 0.0035),
    ((0.236, -0.190, 0.030), 0.0160),
    ((0.150, -0.234, 0.022), 0.0275),
    ((0.040, -0.252, 0.027), 0.0363),
    ((-0.100, -0.240, 0.032), 0.0440),
    ((-0.207, -0.166, 0.038), 0.0506),
    ((-0.250, -0.036, 0.046), 0.0561),
    ((-0.238, 0.094, 0.055), 0.0616),
    ((-0.180, 0.192, 0.064), 0.0671),
    ((-0.082, 0.246, 0.071), 0.0715),
    ((0.030, 0.252, 0.074), 0.0759),
    ((0.082, 0.214, 0.068), 0.0759),
    ((0.050, 0.148, 0.062), 0.0770),   # 12 hips
    ((0.028, 0.060, 0.064), 0.0800),   # 13 middle
    ((0.012, -0.030, 0.068), 0.0800),  # 14 chest
    ((0.002, -0.105, 0.095), 0.0820),
    ((0.000, -0.145, 0.175), 0.0780),
    ((0.000, -0.136, 0.275), 0.0720),
    ((0.000, -0.102, 0.385), 0.0670),
    ((0.000, -0.062, 0.490), 0.0640),
    ((0.000, -0.048, 0.575), 0.0610),
    ((0.000, -0.064, 0.640), 0.0560),
    ((0.000, -0.100, 0.676), 0.0510),
]
HIPS, MIDDLE, CHEST = 12, 13, 14
HOOD_Z = (0.225, 0.672)
HOOD_W = 0.24
HOOD_SHAPE = [(0.0, 0.0), (0.12, 0.2), (0.3, 0.55), (0.55, 0.8), (0.74, 1.0), (0.86, 0.95), (0.93, 0.7), (0.975, 0.3), (1.0, 0.0)]
SCUTE = 0.03
SIDES = 24
KEELS = {0: 1.0, 2: 0.75, -2: 0.75, 4: 0.45, -4: 0.45}
DIAMOND = [0, 1, 2, 3, 4, 3, 2, 1]


def hood_weight(z):
    q = (z - HOOD_Z[0]) / (HOOD_Z[1] - HOOD_Z[0])
    if q <= 0 or q >= 1:
        return 0.0
    return interp(HOOD_SHAPE, q)


def body():
    pts, idx = catmull([p for p, _ in SPINE])
    cum = arclen(pts)
    keys = [(cum[i], w) for i, (_, w) in zip(idx, SPINE)]
    total = cum[-1]
    s_chest = cum[idx[CHEST]]
    s_neck = cum[idx[CHEST + 1]]

    # Two stations to a scale row: its raised trailing edge and, just short of
    # the next row, its tucked-in leading edge, so every row steps crisply.
    # Rows shrink toward the tail tip; up the reared neck they become the
    # broad belly scutes.
    stations, phase, plates, rows = [], [], [], []
    s = 0.0
    s_scutes = s_neck - 0.05
    s_hood = next(cum[i] for i in range(len(pts)) if cum[i] > s_neck and pts[i].z > HOOD_Z[0] + 0.03)
    while True:
        w = interp(keys, s)
        P = SCUTE if s >= s_scutes else min(max(w * 0.66, 0.021), 0.036)
        if s + P > total - 0.008:
            break
        i = len(plates)
        plates.append(s)
        stations.append(s)
        phase.append(0.0)
        rows.append(i)
        if s >= s_hood:
            # A third station mid-scute keeps the hood's outline round.
            stations.append(s + P * 0.5)
            phase.append(0.5)
            rows.append(i)
        stations.append(s + P - 0.0028)
        phase.append(1.0 - 0.0028 / P)
        rows.append(i + 1)
        s += P
    stations += [total - 0.006, total]
    phase += [0.5, 0.5]
    rows += [len(plates), len(plates)]
    n = len(stations)

    centers = [sample(pts, cum, x) for x in stations]
    frames = []
    for j in range(n):
        # Tangents from a fixed span on the curve, so closely paired stations
        # get the same frame and the hood does not ripple.
        t = (sample(pts, cum, stations[j] + 0.008) - sample(pts, cum, stations[j] - 0.008)).normalized()
        horiz = math.sqrt(t.x * t.x + t.y * t.y)
        if stations[j] > s_chest - 0.02:
            want = (Vector((0, 0, -1)) * horiz + FWD * abs(t.z)).normalized()
        else:
            want = Vector((0, 0, -1))
        V = (want - t * want.dot(t)).normalized()
        L = t.cross(V).normalized()
        frames.append((L, L.cross(t).normalized()))

    secs, hoods = [], []
    for j, x in enumerate(stations):
        w = interp(keys, x)
        neck = x > s_chest
        hw = hood_weight(centers[j].z) if neck else 0.0
        hoods.append(hw)
        sec = {"w": w, "hu": w * 0.86, "hd": w * 0.72, "el": 0.92, "evu": 1.0, "evd": 0.6}
        if neck:
            sec.update(hd=w * 0.8, evd=0.75)
        if hw > 0:
            sec["w"] = w + (HOOD_W - w) * hw
            sec["hu"] = sec["hu"] + (0.03 - sec["hu"]) * hw
            sec["hd"] = sec["hd"] + (0.031 - sec["hd"]) * hw
            sec["evd"] = sec["evd"] + (1.0 - sec["evd"]) * hw
            sec["cup"] = 0.045 * hw
        secs.append(sec)

    def vert(j, c, s, lat, ven):
        x = stations[j]
        u = phase[j]
        sec = secs[j]
        if x < s_scutes:
            # Scale rows: the broad belly scutes step out at their trailing
            # edges, the back only faintly, and a few keels ridge the back
            # like a crocodile's scutes.
            k = round((math.atan2(s, c) + math.pi / 2) / (math.tau / SIDES))
            belly = smoothstep(0.25, 0.65, s)
            grow = 1.0 + (0.012 + 0.024 * belly) * (1.0 - u) + 0.15 * KEELS.get(k, 0.0) * (1.0 - u) ** 0.7
            lat *= grow
            ven *= grow
        elif s > 0:
            # Belly scutes up the neck and the middle of the hood's front.
            strip = 1.0 - smoothstep(0.25, 0.6, abs(lat) / sec["w"])
            ven += 0.0045 * (1.0 - u) * smoothstep(0.25, 0.7, s) * strip
        if s < -0.7 and x >= s_scutes:
            ven -= 0.1 * sec["w"] * ((-s - 0.7) / 0.3) ** 2 * (1 - hoods[j] * 0.7)
        return lat, ven

    def role(j, c, s):
        j = min(int(j), n - 1)
        x = stations[j]
        hw = hoods[j]
        lat = abs(c) * secs[j]["w"]
        if hw > 0.3 and abs(c) > 0.93:
            return "trim"
        if x > s_chest:
            if s > 0.15:
                if abs(c) < 0.45:
                    return "cloth"
                # The pale hood panel beside the scutes is cut in
                # panel_edge(); only the slim edge faces fall through here.
                return "body"
            return "body"
        if s > 0.42:
            return "cloth"
        # The diamond-back pattern is cut in diamond(); everything else on
        # the back is the main hide.
        return "body"

    mid = SIDES * 3 // 4  # the vertex on the dorsal midline

    def half(i, j):
        return round(DIAMOND[i % len(DIAMOND)] * min(1.0, secs[j]["w"] / 0.03))

    # The pale hood panel (three face columns a side) starts at its outer
    # column and takes in one more column per long face band above, each
    # step cut along the quad's diagonal: its lower edge is a straight
    # chevron following the hood's flare, not a hem across the neck.
    longs = [j for j in range(n - 1) if stations[j] > s_chest and stations[j + 1] - stations[j] > 0.005]
    ordinal = {}
    count = 0
    for j in range(n - 1):
        ordinal[j] = count
        if j in longs:
            count += 1
    b0 = next(ordinal[j] for j in longs if hoods[j] > 0.12)

    def panel_edge(j, k):
        th = math.tau * (k + 0.5) / SIDES
        c, s = math.cos(th), math.sin(th)
        if not (s > 0.15 and abs(c) >= 0.45):
            return None
        u = (k + 0.5) % (SIDES / 2)
        col = int(round(min(u, SIDES / 2 - u) - 1.5))
        T = b0 + col
        o = ordinal[j]
        if j in longs and o == T:
            if (k + 0.5) < SIDES / 4:
                return [((0, 1, 2), "body"), ((0, 2, 3), "skin")]
            return [((0, 1, 3), "body"), ((1, 2, 3), "skin")]
        return [((0, 1, 2, 3), "skin" if o > T else "body")]

    def diamond(j, k):
        """The diamond-back pattern, one face wider or narrower per scale row,
        its edges cut along the quads' diagonals into straight lines: cloth
        inside, a pale skin rim, body outside."""
        x = stations[j]
        th = math.tau * (k + 0.5) / SIDES
        if x > s_chest:
            return panel_edge(j, k)
        if x < 0.1 or math.sin(th) > 0.42:
            return None
        # A scale row's long face band runs from its raised trailing edge
        # (phase 0) to just short of the next row; across it the pattern
        # widens or narrows by one face. The short step band holds steady.
        i = rows[j]
        h0 = half(i, j)
        h1 = half(i + 1, j) if phase[j] == 0.0 and rows[j + 1] == i + 1 else h0
        if h0 == 0 and h1 == 0:
            return None

        def off(kk):
            d = abs(kk - mid) % SIDES
            return min(d, SIDES - d)

        corner = [(off(k), 0.0), (off(k + 1), 0.0), (off(k + 1), 1.0), (off(k), 1.0)]

        def classify(pts):
            o = sum(p[0] for p in pts) / len(pts)
            t = sum(p[1] for p in pts) / len(pts)
            b = h0 + (h1 - h0) * t
            return "cloth" if o < b - 1 else "skin" if o < b else "body"

        lo = min(corner[0][0], corner[1][0])
        m = min(h0, h1)
        if abs(h1 - h0) == 1 and lo in (m, m - 1):
            edge = (lambda o, t: o - (h0 + (h1 - h0) * t)) if lo == m else (lambda o, t: o - (h0 + (h1 - h0) * t - 1))
            on = [abs(edge(*corner[q])) < 1e-6 for q in range(4)]
            tris = ((0, 1, 2), (0, 2, 3)) if on[0] and on[2] else ((0, 1, 3), (1, 2, 3))
            return [(tri, classify([corner[q] for q in tri])) for tri in tris]
        return [((0, 1, 2, 3), classify(corner))]

    obj = loft("body", centers, frames, secs, SIDES, vert, role, face_fn=diamond)
    lib.deform(obj, lambda v: Vector((v.x, v.y, max(v.z, 0.0))))
    spine = {"rows": rows, "stations": stations, "centers": centers, "frames": frames, "secs": secs, "plates": plates, "s_chest": s_chest, "keys": keys}
    return obj, pts, idx, spine


def dorsal_spines(spine):
    """Small horn spines down the middle of the back, every other scale row
    from the shoulders to near the tail tip, raking back."""
    parts = []
    st, plates = spine["stations"], spine["plates"]
    for i, a in enumerate(plates):
        if a < 0.12 or i % 2:
            continue
        j = st.index(a)
        if a > spine["s_chest"] and spine["centers"][j].z > 0.3:
            continue
        C, (L, V) = spine["centers"][j], spine["frames"][j]
        w = interp(spine["keys"], a)
        T = V.cross(L).normalized()
        h = 0.034 * (w / 0.075) ** 0.8
        base = C - V * (spine["secs"][j]["hu"] * 1.13)
        d = (-V * 1.0 - T * 0.55).normalized()
        spike = lib.cylinder(0.0105 * (w / 0.075) ** 0.8, 0.0, h, role="metal", segments=6, name="dorsal")
        spike.data.transform(Matrix.Translation((0, 0, h * 0.42)))
        spike.data.transform(d.to_track_quat("Z", "Y").to_matrix().to_4x4())
        lib.place(spike, base)
        parts.append(spike)
    return parts


def decal(target, strokes, width, lift=0.001, thickness=0.003, role="trim", name="decal", side=1):
    """Raised bands laid on `target` as seen from behind (side 1, +Y) or the
    front (side -1): each stroke is a list of (x, z) points, projected onto
    the surface along Y."""
    from mathutils.bvhtree import BVHTree
    me = target.data
    tree = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
    bm = bmesh.new()
    for stroke, closed in strokes:
        pts = [Vector((x, 0, z)) for x, z in stroke]
        ring = []
        m = len(pts)
        for i, p in enumerate(pts):
            a, b = pts[(i - 1) % m] if (closed or i > 0) else p, pts[(i + 1) % m] if (closed or i < m - 1) else p
            t = (b - a).normalized()
            perp = Vector((t.z, 0, -t.x))
            pair = []
            for o in (-width, width):
                q = p + perp * o
                hit, nrm, _, _ = tree.ray_cast(Vector((q.x, side * 1.0, q.z)), Vector((0, -side, 0)))
                if hit is None:
                    raise ValueError(f"{name}: ({q.x:.3f}, {q.z:.3f}) misses the surface")
                if nrm.y * side < 0:
                    nrm = -nrm
                pair.append(bm.verts.new(hit + nrm * lift))
            ring.append(pair)
        for i in range(m if closed else m - 1):
            a, b = ring[i], ring[(i + 1) % m]
            bm.faces.new((a[0], b[0], b[1], a[1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    lib.set_role(obj, role)
    return lib.solidify(obj, thickness)


def hood_marks(body_obj):
    """The basilisk's gaze turned behind it too: one great slit-pupilled eye
    marked on the back of the hood, where a cobra wears its spectacle."""
    cz = 0.5
    almond = []
    for i in range(24):
        a = math.tau * i / 24
        x = math.cos(a) * 0.085
        z = math.sin(a) * 0.04 * (1 - (x / 0.085) ** 2) ** 0.35
        almond.append((x, cz + z))
    slit = [(0.0, cz - 0.03), (0.003, cz), (0.0, cz + 0.03)]
    marks = decal(body_obj, [(almond, True)], 0.0055, name="hood_eye")
    pupil = decal(body_obj, [(slit, False)], 0.0075, name="hood_pupil")
    return [marks, pupil]


# ------------------------------------------------------------ legs


def limb(ctrl, radii, bands, sides=12, step=0.016, name="leg"):
    """A loft through the control points with the radius following `radii`
    (one per point). Over the arc-length fractions in `bands` (start, end,
    count) the limb is clad in overlapping scale cuffs, each swelling toward
    its lower edge and stepping in crisply at the next."""
    pts, idx = catmull(ctrl, per=10)
    cum = arclen(pts)
    total = cum[-1]
    keys = [(cum[i], r) for i, r in zip(idx, radii)]
    stations, swell = [], []
    a, b, count = bands
    a, b = a * total, b * total
    s = 0.0
    while s < a:
        stations.append(s)
        swell.append(0.0)
        s += step
    span = (b - a) / count
    for i in range(count):
        s0 = a + i * span
        stations += [s0, s0 + span - 0.0022]
        swell += [0.0, 1.0]
    s = b
    while s < total - 0.004:
        stations.append(s)
        swell.append(0.0)
        s += step
    stations.append(total)
    swell.append(0.0)
    centers = [sample(pts, cum, x) for x in stations]
    # Parallel-transported frames, so the section does not twist.
    frames = []
    t0 = (sample(pts, cum, 0.004) - centers[0]).normalized()
    n = t0.cross(UP if abs(t0.z) < 0.9 else Vector((1, 0, 0))).normalized()
    prev = t0
    for x in stations:
        t = (sample(pts, cum, x + 0.004) - sample(pts, cum, x - 0.004)).normalized()
        n = prev.rotation_difference(t) @ n
        n = (n - t * n.dot(t)).normalized()
        frames.append((n, t.cross(n).normalized()))
        prev = t
    secs = []
    for x, k in zip(stations, swell):
        r = interp(keys, x) * (1.0 + 0.06 * k)
        secs.append({"w": r, "hu": r, "hd": r})
    return loft(name, centers, frames, secs, sides)


def leg(C, f, side, w, reach, splay=1.0, hind=False):
    """A short sprawling lizard leg off the body at C (forward f, side +1
    left / -1 right): a heavy scaled upper limb thrust out sideways, an
    angular elbow (a knee on the hind legs), a lean forearm and a three-toed
    clawed foot."""
    f = Vector((f.x, f.y, 0)).normalized()
    l = UP.cross(f).normalized() * side
    base = Vector((C.x, C.y, 0))
    r0 = w * 0.42
    sh = base + l * r0 + UP * (C.z + 0.004)
    upper = base + l * (r0 + 0.046 * splay) + UP * (C.z + 0.022) + f * (reach * 0.3)
    elbow = base + l * (r0 + 0.074 * splay) + UP * (C.z + 0.03) + f * (reach * 0.55 + (0.012 if hind else -0.006))
    fore = base + l * (r0 + 0.085 * splay) + UP * 0.062 + f * (reach * 0.8)
    wrist = base + l * (r0 + 0.09 * splay) + UP * 0.03 + f * reach
    foot = wrist + l * 0.014 + f * 0.012
    foot.z = 0.015
    thigh = 1.12 if hind else 1.0
    parts = [limb([sh, upper, elbow, fore, wrist, foot], [0.052 * thigh, 0.044 * thigh, 0.027, 0.024, 0.019, 0.018], (0.22, 0.56, 3))]
    parts.append(lib.sphere(0.024, loc=foot, scale=(1.0, 1.0, 0.55), role="body", segments=8, rings=5, name="foot"))
    aim = (f * 1.1 + l * 0.7).normalized()
    for ang, ln in ((-0.62, 0.038), (0.0, 0.05), (0.62, 0.04)):
        d = Matrix.Rotation(ang, 3, "Z") @ aim
        a = foot + d * 0.006
        b = foot + d * (ln * 0.55)
        e = foot + d * ln
        a.z, b.z, e.z = 0.015, 0.014, 0.009
        parts.append(lib.tube([a, b, e], [0.0098, 0.0078, 0.006], role="body", sides=6, resolution=1, name="toe"))
        tip = e + d * 0.024
        tip.z = 0.0
        claw = lib.cylinder(0.0064, 0.0, 0.028, role="metal", segments=6, name="claw")
        claw.data.transform(Matrix.Translation((0, 0, 0.012)))
        claw.data.transform((tip - e).normalized().to_track_quat("Z", "Y").to_matrix().to_4x4())
        lib.place(claw, e)
        parts.append(claw)
    return parts


# ------------------------------------------------------------ head

HEAD_AT = Vector((0, -0.088, 0.654))
HEAD_PITCH = 0.4
HEAD_SCALE = 1.4
HL = 0.178
# A viper's spade: broad venom-gland cheeks narrowing in straight lines to a
# low, flat, squared-off snout.
H_W = [(0.0, 0.006), (0.03, 0.04), (0.08, 0.06), (0.15, 0.074), (0.28, 0.072), (0.42, 0.06), (0.56, 0.048), (0.7, 0.038), (0.84, 0.028), (0.93, 0.024), (0.975, 0.02), (1.0, 0.009)]
H_HU = [(0.0, 0.006), (0.03, 0.03), (0.08, 0.037), (0.15, 0.042), (0.3, 0.043), (0.45, 0.038), (0.6, 0.032), (0.75, 0.024), (0.88, 0.021), (0.95, 0.018), (0.985, 0.014), (1.0, 0.006)]
H_HD = [(0.0, 0.006), (0.03, 0.028), (0.08, 0.034), (0.15, 0.034), (0.3, 0.029), (0.45, 0.024), (0.6, 0.021), (0.75, 0.019), (0.88, 0.018), (0.95, 0.016), (0.985, 0.012), (1.0, 0.005)]
H_Z = [(0.0, 0.0), (0.3, 0.004), (0.6, 0.002), (0.85, -0.002), (1.0, -0.004)]
H_EL, H_EVU = 0.66, 0.5
EYE_AT = Vector((0.035, -0.098, 0.009))
EYE_R = 0.037
EYE_SCALE = (0.65, 1.15, 0.8)
EYE_YAW = 0.25


def brow(x):
    """A heavy supraocular plate over the eye on side x: a bevelled slab from
    the crown band forward over the eye, tilted down at its outer edge into an
    awning and stepped in three scales; its front dips toward the snout in an
    angry V and runs on as the sharp canthus ridge to the nostril."""
    path = [Vector((x * px, py, pz)) for px, py, pz in ((0.03, -0.05, 0.043), (0.047, -0.07, 0.046), (0.0515, -0.095, 0.04), (0.047, -0.117, 0.033), (0.038, -0.134, 0.027), (0.026, -0.151, 0.024), (0.02, -0.165, 0.016))]
    pts, idx = catmull(path, per=12)
    cum = arclen(pts)
    total = cum[-1]
    width = [(0.0, 0.009), (0.14, 0.021), (0.4, 0.024), (0.56, 0.014), (0.68, 0.0065), (0.92, 0.0045), (1.0, 0.0015)]
    stations, phase = [], []
    steps = 3
    plate = 0.62
    span = total * plate / steps
    for i in range(steps):
        stations += [i * span, (i + 1) * span - 0.0022]
        phase += [0.0, 1.0]
    q = plate + 0.06
    while q < 0.99:
        stations.append(total * q)
        phase.append(1.0)
        q += 0.08
    stations.append(total)
    phase.append(1.0)
    centers, frames, secs = [], [], []
    for s, u in zip(stations, phase):
        q = s / total
        C = sample(pts, cum, s)
        t = (sample(pts, cum, s + 0.004) - sample(pts, cum, s - 0.004)).normalized()
        V = (Vector((0, 0, -1)) - t * Vector((0, 0, -1)).dot(t)).normalized()
        L = t.cross(V).normalized()
        if L.x * x < 0:
            L = -L
        # Tilt the slab so its outer edge drops over the eye.
        tilt = Matrix.Rotation(-x * 0.7 * (1 if t.y < 0 else -1), 3, t)
        L, V = tilt @ L, tilt @ V
        w = interp(width, q)
        k = min(1.0, max(w / 0.012, 0.62))
        grow = 1.0 + 0.2 * (1.0 - u) * (1.0 if q < plate else 0.0)
        centers.append(C)
        frames.append((L, V))
        secs.append({"w": w * grow, "hu": 0.0044 * grow * k, "hd": 0.0042 * k, "el": 0.35, "evu": 0.6, "evd": 0.35, "cup": 0.0025 * k})

    def keel(j, c, s, lat, ven):
        # The inner half of the underside runs down into the skull, so the
        # plate grows out of the head instead of lying on it like a strip.
        k = min(1.0, secs[j]["w"] / 0.016)
        if c < 0.2 and s > 0:
            ven += 0.014 * k * min(1.0, (0.2 - c) / 0.8) * s
        return lat, ven

    return loft("brow", centers, frames, secs, 12, vert_fn=keel)


def slit(e, yaw, x, turn=0.14):
    """A slit pupil running the full height of the eye: a thin pointed inlay
    laid flush on the eyeball along the meridian `turn` radians (in the eye's
    own stretched frame) forward of its outer pole, so the glare reads from
    the side, the board camera and the front alike."""
    rx, ry, rz = (EYE_R * k for k in EYE_SCALE)
    R = Matrix.Rotation(yaw, 3, "Z")
    ts = [-1.25 + 2.5 * i / 10 for i in range(11)]
    centers, frames, secs = [], [], []
    for t in ts:
        P = Vector((x * rx * math.cos(turn) * math.cos(t), -ry * math.sin(turn) * math.cos(t), rz * math.sin(t)))
        n = Vector((P.x / rx ** 2, P.y / ry ** 2, P.z / rz ** 2)).normalized()
        d = Vector((-x * rx * math.cos(turn) * math.sin(t), ry * math.sin(turn) * math.sin(t), rz * math.cos(t))).normalized()
        across = n.cross(d).normalized()
        centers.append(e + R @ (P + n * 0.0006))
        frames.append((R @ across, -(R @ n)))
        w = 0.0062 * max(math.cos(t * math.pi / 2.6), 0.0) ** 1.2 + 0.0004
        secs.append({"w": w, "hu": 0.0011, "hd": 0.0009, "el": 0.8})
    return loft("pupil", centers, frames, secs, 6, role_fn=lambda j, c, s: "body")


def skull_level(p):
    """Below 1 inside the skull's loft (head-local), above 1 outside."""
    u = min(max(-p.y / HL, 0.0), 1.0)
    w, hu, hd = interp(H_W, u), interp(H_HU, u), interp(H_HD, u)
    ven = interp(H_Z, u) - p.z
    h, ev = (hd, 0.9) if ven > 0 else (hu, H_EVU)
    return (abs(p.x) / w) ** (2 / H_EL) + (abs(ven) / h) ** (2 / ev)


def socket(e, yaw, x, n=20):
    """The rim of the eye socket: a thin bevelled lip of hide round the line
    where the eyeball meets the skull, heavier along the lower lid and at the
    corners, so the eye sits sunk in the head and narrows to an almond."""
    rx, ry, rz = (EYE_R * k for k in EYE_SCALE)
    R = Matrix.Rotation(yaw, 3, "Z")
    axis = R @ Vector((x, 0, 0))
    pts = []
    for i in range(n):
        a = math.tau * i / n
        lo, hi = 0.0, 1.0
        for _ in range(24):
            m = (lo + hi) / 2
            r = math.sqrt(max(1 - m * m, 0.0))
            q = e + R @ Vector((x * rx * m, ry * r * math.cos(a), rz * r * math.sin(a)))
            if skull_level(q) < 1.0:
                lo = m
            else:
                hi = m
        r = math.sqrt(max(1 - hi * hi, 0.0))
        pts.append((e + R @ Vector((x * rx * hi, ry * r * math.cos(a), rz * r * math.sin(a))), a))
    centers, frames, secs = [], [], []
    for k in range(n + 1):
        p, a = pts[k % n]
        t = (pts[(k + 1) % n][0] - pts[(k - 1) % n][0]).normalized()
        out = (p - e) - axis * (p - e).dot(axis)
        L = (out - t * out.dot(t)).normalized()
        V = t.cross(L).normalized()
        if V.dot(axis) > 0:
            V = -V
        # Lower lid (sin a < 0) and the two corners (cos a = +-1) heavier.
        lid = 0.6 * max(0.0, -math.sin(a)) + 0.25 * max(0.0, -math.cos(a)) ** 3
        centers.append(p + axis * 0.0012)
        frames.append((L, V))
        secs.append({"w": 0.0034 + 0.0026 * lid, "hu": 0.0026 + 0.002 * lid, "hd": 0.0026, "el": 0.7, "evu": 0.7, "evd": 0.7})
    return loft("socket", centers, frames, secs, 6, role_fn=lambda j, c, s: "body", caps=(False, False))


def labials(x, count=4):
    """The row of pale lip scales along the upper jaw on side x, from under
    the front of the eye to the snout, each standing proud at its rear edge."""
    U = [0.7 + (0.96 - 0.7) * i / 8 for i in range(9)]
    path = []
    for u in U:
        path.append(Vector((x * (interp(H_W, u) * 0.955 + 0.0015), -HL * u, interp(H_Z, u) - interp(H_HD, u) * 0.36)))
    pts, idx = catmull(path, per=8)
    cum = arclen(pts)
    total = cum[-1]
    span = total * 0.94 / count
    stations, phase = [], []
    for i in range(count):
        stations += [i * span, (i + 1) * span - 0.0018]
        phase += [0.0, 1.0]
    stations.append(total)
    phase.append(1.0)
    centers, frames, secs = [], [], []
    for st, u in zip(stations, phase):
        C = sample(pts, cum, st)
        t = (sample(pts, cum, st + 0.004) - sample(pts, cum, st - 0.004)).normalized()
        L = t.cross(UP).normalized()
        if L.x * x < 0:
            L = -L
        V = t.cross(L).normalized()
        if V.z > 0:
            V = -V
        q = st / total
        taper = min(1.0, (1.0 - q) / 0.12 + 0.15, q / 0.08 + 0.3)
        grow = 1.0 + 0.9 * (1.0 - u)
        centers.append(C)
        frames.append((L, V))
        secs.append({"w": 0.0028 * grow * taper, "hu": 0.006 * taper, "hd": 0.0055 * taper, "el": 0.6, "evu": 0.8, "evd": 0.8})
    return loft("labial", centers, frames, secs, 6, role_fn=lambda j, c, s: "skin")


def skull_point(u, phi, lift=0.0):
    """A point on the skull's upper surface at length u, phi radians from the
    top toward +X, pushed `lift` outward."""
    c, s = math.sin(phi), -math.cos(phi)
    lat = interp(H_W, u) * spow(c, H_EL)
    ven = interp(H_HU, u) * spow(s, H_EVU)
    p = Vector((lat, -HL * u, interp(H_Z, u) - ven))
    centre = Vector((0, -HL * u, interp(H_Z, u)))
    return p + (p - centre).normalized() * lift


def crown_band(ring, n=32, thick=0.0034, tall=0.0078):
    """The coronet: a flat band with bevelled edges standing on the skull,
    following ring(a) -> (u, phi) round it."""
    pts, nrm = [], []
    for k in range(n):
        u, phi = ring(math.tau * k / n)
        p = skull_point(u, phi, lift=0.0)
        c = Vector((0, p.y, interp(H_Z, u)))
        pts.append(p)
        nrm.append((p - c).normalized())
    centers, frames, secs = [], [], []
    for k in range(n + 1):
        i = k % n
        t = (pts[(i + 1) % n] - pts[(i - 1) % n]).normalized()
        L = (nrm[i] - t * nrm[i].dot(t)).normalized()
        V = t.cross(L).normalized()
        centers.append(pts[i] + L * (thick * 0.9))
        frames.append((L, V))
        secs.append({"w": thick, "hu": tall, "hd": tall, "el": 0.3, "evu": 0.3, "evd": 0.3})
    return loft("crown_band", centers, frames, secs, 8, role_fn=lambda j, c, s: "trim", caps=(False, False))


def head_parts():
    parts = []
    # Skull and upper jaw, lofted along local -Y from the back of the skull:
    # broad venom-gland cheeks, a high crown, a blunt wedge of a snout.
    U = [0.0, 0.03, 0.08, 0.15, 0.22, 0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.85, 0.91, 0.955, 0.985, 1.0]
    centers = [Vector((0, -HL * u, interp(H_Z, u))) for u in U]
    secs = [{"w": interp(H_W, u), "hu": interp(H_HU, u), "hd": interp(H_HD, u), "el": H_EL, "evu": H_EVU, "evd": 0.9} for u in U]

    def role(j, c, s):
        u = U[min(int(j), len(U) - 1)]
        return "skin" if s > 0.35 and u > 0.3 else "body"

    parts.append(loft("skull", centers, straight_frames(len(U)), secs, 24, role_fn=role))

    # Lower jaw, hinged open.
    UJ = [0.0, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.87, 0.95, 1.0]
    LJ = 0.148
    jw = [(0.0, 0.01), (0.05, 0.04), (0.15, 0.046), (0.4, 0.038), (0.7, 0.028), (0.9, 0.021), (1.0, 0.004)]
    jd = [(0.0, 0.008), (0.05, 0.024), (0.15, 0.027), (0.5, 0.021), (0.85, 0.017), (0.95, 0.014), (1.0, 0.005)]
    ju = [(0.0, 0.006), (0.05, 0.01), (0.5, 0.009), (0.95, 0.007), (1.0, 0.004)]
    jc = [Vector((0, -LJ * u, 0)) for u in UJ]
    js = [{"w": interp(jw, u), "hu": interp(ju, u), "hd": interp(jd, u), "el": 0.85, "evd": 0.9} for u in UJ]
    jaw = loft("jaw", jc, straight_frames(len(UJ)), js, 20, role_fn=lambda j, c, s: "skin" if s < -0.3 else "body")
    lib.place(jaw, (0, -0.03, -0.026), (0.27, 0, 0))
    parts.append(jaw)
    parts.append(lib.sphere(0.03, loc=(0, -0.07, -0.02), scale=(1.2, 1.6, 0.6), role="skin", segments=14, rings=8, name="throat"))

    # A forked tongue flicking out between the fangs.
    tongue = [(0, -0.085, -0.02), (0, -0.15, -0.03), (0, -0.19, -0.045)]
    parts.append(lib.tube(tongue, [0.0055, 0.0045, 0.0035], role="trim", sides=6, resolution=3, name="tongue"))
    for x in (-1, 1):
        parts.append(lib.tube([(0, -0.188, -0.044), (x * 0.006, -0.208, -0.054), (x * 0.014, -0.222, -0.058)], [0.0034, 0.0026, 0.0006], role="trim", sides=6, resolution=3, name="tongue"))

    # Fangs and teeth.
    for x in (-1, 1):
        parts.append(lib.tube([(x * 0.018, -0.152, -0.008), (x * 0.019, -0.15, -0.027), (x * 0.016, -0.142, -0.044)], [0.006, 0.0044, 0.0006], role="metal", sides=8, resolution=3, name="fang"))
        for u in (0.48, 0.6, 0.72):
            y = -HL * u
            wx = interp(H_W, u) * 0.8
            parts.append(lib.cylinder(0.0034, 0.0, 0.012, loc=(x * wx, y, -0.018), rot=(math.pi, 0, 0), role="metal", segments=6, name="tooth"))

    # Big glowing eyes, slit-pupilled, glaring out from under heavy brows.
    for x in (-1, 1):
        e = Vector((x * EYE_AT.x, EYE_AT.y, EYE_AT.z))
        a = -x * EYE_YAW
        parts.append(lib.sphere(EYE_R, loc=e, rot=(0, 0, a), scale=EYE_SCALE, role="eye", segments=14, rings=9, name="eye"))
        parts.append(socket(e, a, x))
        parts.append(slit(e, a, x))
        parts.append(brow(x))
        parts.append(labials(x))
        # Nostrils at the sides of the snout tip.
        parts.append(lib.sphere(0.0045, loc=(x * 0.019, -0.168, 0.008), rot=(0, 0, -x * 0.6), scale=(0.7, 1.4, 0.8), role="body", segments=8, rings=5, name="nostril"))
        # A short swept-back cheek scale over the hinge of the jaw.
        parts.append(lib.tube([(x * 0.056, -0.04, -0.004), (x * 0.069, -0.03, -0.008), (x * 0.08, -0.02, -0.012)], [0.007, 0.0048, 0.0008], role="trim", sides=6, resolution=3, name="spur"))

    # The crown: a coronet band round the top of the skull, ringed with spikes
    # that grow taller toward the back, and a glowing gem at the front.
    def ring(a):
        return 0.27 + 0.115 * math.cos(a), 1.08 * math.sin(a)

    parts.append(crown_band(ring))
    for a, ln in ((0.0, 0.055), (0.75, 0.06), (-0.75, 0.06), (1.5, 0.075), (-1.5, 0.075), (2.3, 0.09), (-2.3, 0.09), (math.pi, 0.1)):
        u, phi = ring(a)
        base = skull_point(u, phi, lift=0.002)
        n = (base - Vector((0, base.y, interp(H_Z, u)))).normalized()
        d = (n * 0.45 + UP + Vector((0, 0.22, 0))).normalized()
        mid = base + d * ln * 0.55 + Vector((0, 0.006, 0)) + n * 0.004
        tip = base + d * ln + Vector((0, 0.016, 0)) + n * 0.01
        parts.append(lib.tube([base - d * 0.004, mid, tip], [0.0105, 0.0062, 0.0008], role="trim", sides=6, resolution=3, name="spike"))
        # A bevelled socket ring seats each spike on the band.
        collar = lib.lathe([(0.0108, -0.004), (0.0142, -0.0005), (0.0142, 0.0025), (0.0098, 0.0055)], role="trim", segments=6, name="collar")
        collar.data.transform(d.to_track_quat("Z", "Y").to_matrix().to_4x4())
        parts.append(lib.place(collar, base))
    for a in (0.375, -0.375, 1.125, -1.125, 1.9, -1.9):
        parts.append(lib.sphere(0.0052, loc=skull_point(*ring(a), lift=0.0085), role="eye", segments=6, rings=4, name="crown_gem"))
    gem_at = skull_point(*ring(0.0), lift=0.004)
    parts.append(lib.sphere(0.0095, loc=gem_at, scale=(1.0, 0.7, 1.0), role="eye", segments=10, rings=6, name="gem"))

    for p in parts:
        lib.place(p, (0, 0, 0), (0, 0, 0), (HEAD_SCALE,) * 3)
        lib.place(p, HEAD_AT, (HEAD_PITCH, 0, 0))
    return parts


# ------------------------------------------------------------ build


def build():
    parts = []
    add = parts.append

    body_obj, pts, idx, spine = body()
    add(body_obj)
    for p in hood_marks(body_obj):
        add(p)
    for p in dorsal_spines(spine):
        add(p)

    for i, reach, splay in ((CHEST, 0.032, 1.0), (MIDDLE, 0.0, 1.05), (HIPS, -0.018, 0.9)):
        hind = i == HIPS
        C = Vector(SPINE[i][0])
        j = idx[i]
        f = (pts[min(j + 3, len(pts) - 1)] - pts[max(j - 3, 0)]).normalized()
        w = SPINE[i][1]
        for side in (-1, 1):
            for p in leg(C, f, side, w, reach, splay, hind):
                add(p)

    for p in head_parts():
        add(p)
    return parts
