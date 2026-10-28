"""Troll: a huge hunched Dark cave troll. Massive shoulders and a hump, a small
head thrust forward and up past the shoulder line, a big warty potato nose
with a little nostril ring, a jutting underbite with four lower teeth and
banded tusks that flare out and forward to the nose's mid-height, a pot belly
sagging over the rope belt, short thick legs on big flat feet. The right arm
(-X) hangs to a fist knuckled on the ground; the left (+X) hoists a cracked,
pitted boulder onto its shoulder. A topknot tied with a trim band, a shaggy
mane over the hump, a hide loincloth, a fang necklace, a thigh bone and a
little skull at the hip. All flesh (warts and belt too) is the body role;
skin is kept for bone and stone. The figure faces -Y."""
import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

import lib

# Lumpy flesh needs a much softer edge split than the default 35 degrees, so
# the few steep creases on the limbs and legs still shade smooth.
FINISH = {"sharp_angle": 80}

POSE = {}  # points worked out while building, such as the left wrist on the rock
FLESH_TRIS = 10900
BASE_DETAIL = 0.3
PROTECT = 2.5
SHIFT_Y = 0.05
THRESHOLD = 0.6  # lib.metaball's default


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def reach(stiff):
    """Metaball radius that shows a lone element at unit size. Blender's field
    is stiffness * (1 - (d/R)^2)^3; stiffer elements blend over a shorter
    distance, so their creases stay crisper."""
    return 1 / math.sqrt(1 - (THRESHOLD / stiff) ** (1 / 3))


def ball(co, r, stiff=3.0, neg=False):
    return {"co": tuple(co), "radius": r * reach(stiff), "stiffness": stiff, "negative": neg}


def ell(co, radii, rot=(0, 0, 0), stiff=3.0, neg=False):
    base = min(radii)
    return {"co": tuple(co), "type": "ELLIPSOID", "radius": base * reach(stiff), "stiffness": stiff,
            "size": tuple(r / base for r in radii), "rot": tuple(rot), "negative": neg}


def seg(a, b, r, t=0.5, half=None, off=(0, 0, 0), stiff=3.0, neg=False):
    """An ellipsoid stretched along the bone a->b, centred at fraction t."""
    a, b = V(a), V(b)
    d = b - a
    c = a + d * t + V(off)
    r1, r2 = (r, r) if isinstance(r, (int, float)) else r
    rot = d.normalized().to_track_quat("X", "Z").to_euler()
    return ell(c, (half or d.length * 0.5, r1, r2), rot, stiff, neg)


def organic(elements, res, role, target, lumps=0.0, lump_size=0.05, relax=2, detail=None, protect=3.0, name="flesh"):
    """A fused metaball mass, roughened, relaxed and decimated to `target`
    tris. `detail(co) -> 0..1` keeps more triangles where it is high (faces,
    hands), so small features stay smooth while broad masses thin out."""
    o = lib.metaball(elements, role=role, resolution=res, name=name)
    if lumps:
        o = lib.displace(o, lumps, size=lump_size, seed=len(name))
    if relax:
        o = lib.smooth(o, relax, 0.5)
    n = lib.tris(o)
    if n > target:
        mod = o.modifiers.new("dec", "DECIMATE")
        mod.ratio = target / n
        mod.use_collapse_triangulate = False
        if detail:
            vg = o.vertex_groups.new(name="detail")
            for v in o.data.vertices:
                vg.add([v.index], max(0.0, min(1.0, detail(v.co))), "REPLACE")
            mod.vertex_group = "detail"
            mod.invert_vertex_group = True
            mod.vertex_group_factor = protect
        o = lib.apply_modifiers(o)
        o.vertex_groups.clear()
    return o


def islands(obj):
    """The mesh's connected pieces as vertex counts, largest first. A finger
    or a thumb that failed to fuse shows up as a second piece."""
    me = obj.data
    parent = list(range(len(me.vertices)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in me.edges:
        a, b = root(e.vertices[0]), root(e.vertices[1])
        if a != b:
            parent[a] = b
    sizes = {}
    for i in range(len(parent)):
        r = root(i)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values(), reverse=True)


def decimate_to(obj, target, keep=None, protect=3.0):
    """Collapses `obj` to about `target` tris. `keep(vertex) -> 0..1` charges
    a length cost where it is high, so grooves and creases there keep their
    edges while flat planes thin out. A keep of 1 never collapses at all, so
    stay below it."""
    # A single very deep collapse can stop short of its ratio, so it runs
    # again (weights re-read from the surviving vertices) until it lands.
    for _ in range(4):
        n = lib.tris(obj)
        if n <= target * 1.02:
            break
        mod = obj.modifiers.new("dec", "DECIMATE")
        mod.ratio = target / n
        mod.use_collapse_triangulate = False
        if keep:
            vg = obj.vertex_groups.new(name="keep")
            for v in obj.data.vertices:
                vg.add([v.index], max(0.0, min(1.0, keep(v))), "REPLACE")
            mod.vertex_group = "keep"
            mod.invert_vertex_group = True
            mod.vertex_group_factor = protect
        obj = lib.apply_modifiers(obj)
        obj.vertex_groups.clear()
    return obj


def flatten_below(obj, z=0.0):
    """Flattens everything below z onto it: flat soles and knuckles."""
    return lib.deform(obj, lambda v: Vector((v.x, v.y, max(v.z, z))))


def bvh_of(obj):
    me = obj.data
    return BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [tuple(p.vertices) for p in me.polygons])


def radial(angle):
    """Horizontal direction at `angle` (0 = front, -Y; positive toward +X)."""
    return Vector((math.sin(angle), -math.cos(angle), 0))


def outer_radius(bvh, axis, angle, z, start=0.3):
    """Radius of the body's outermost surface at angle and height, measured
    inward from `start` so hanging arms further out are ignored."""
    d = radial(angle)
    o = Vector((axis[0], axis[1], z))
    hit = bvh.ray_cast(o + d * start, -d, start)
    return start - hit[3] if hit[0] is not None else 0.0


def surface_point(bvh, origin, direction, offset=0.0):
    """Where a ray from inside the body leaves it, pushed out by offset."""
    d = Vector(direction).normalized()
    hit = bvh.ray_cast(Vector(origin), d, 1.0)
    if hit[0] is None:
        return Vector(origin) + d * 0.1, d
    return hit[0] + hit[1] * offset, hit[1]


def shell(rings, role, thickness, name):
    """A quad shell through rows of points, solidified to `thickness`."""
    bm = bmesh.new()
    vrows = [[bm.verts.new(p) for p in row] for row in rings]
    for a, b in zip(vrows, vrows[1:]):
        for i in range(len(a) - 1):
            bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = lib.set_role(lib._link(bpy.data.objects.new(name, me)), role)
    return lib.solidify(obj, thickness, offset=0.0)


def ring(centre, axis, radius, thick, role="trim", n=12, sides=6, name="ring"):
    """A closed torus of cord round `axis`, `n` segments round and `sides`
    round the cord: bands, rings, bindings."""
    axis = V(axis).normalized()
    t1 = axis.cross(V(0, 0, 1) if abs(axis.z) < 0.9 else V(1, 0, 0)).normalized()
    t2 = axis.cross(t1)
    bm = bmesh.new()
    rows = []
    for i in range(n):
        a = math.tau * i / n
        out = t1 * math.cos(a) + t2 * math.sin(a)
        c = V(centre) + out * radius
        rows.append([bm.verts.new(c + (out * math.cos(b) + axis * math.sin(b)) * thick)
                     for b in (math.tau * (k + 0.5) / sides for k in range(sides))])
    for i in range(n):
        r0, r1 = rows[i], rows[(i + 1) % n]
        for k in range(sides):
            bm.faces.new((r0[k], r1[k], r1[(k + 1) % sides], r0[(k + 1) % sides]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return lib.set_role(lib._link(bpy.data.objects.new(name, me)), role)


# ---------------------------------------------------------------- pose

HEAD = V(0.0, -0.34, 0.82)
HS = 1.2  # head scale: offsets given in hp() and hd() grow with it
PITCH = -0.12  # chin a little up, so the face presents to the board camera


def hv(x, y, z):
    """A direction in the head's pitched frame."""
    c, s = math.cos(PITCH), math.sin(PITCH)
    return V(x, y * c - z * s, y * s + z * c)


def hd(x, y, z):
    """An offset in the head's pitched frame, scaled with the head."""
    return hv(x, y, z) * HS


def hp(x, y, z):
    """A point in the head's pitched frame."""
    return HEAD + hd(x, y, z)


# Right arm (-X): shoulder, elbow, wrist, fist knuckled on the ground.
RS, RE, RW, RF = V(-0.27, -0.14, 0.715), V(-0.352, -0.15, 0.43), V(-0.36, -0.215, 0.145), V(-0.36, -0.235, 0.012)
# Left arm (+X): the boulder rides on the shoulder and hump; the elbow juts
# forward and the forearm climbs back to a hand spread over the rock's
# outer front, leaving its inner face clear beside the head.
BOULDER, BR = V(0.215, 0.15, 0.93), 0.19
ROCK_ROT = (0.25, 0.15, 0.4)
ROCK_POINTS = 30  # hull points scattered over the boulder's ellipsoid
LS = V(0.27, -0.12, 0.76)
GRIP = V(0.5, -0.74, 0.45).normalized()  # from the boulder's centre to the palm
GRIP_UP = V(0.3, 0.38, 0.88).normalized()  # the way the fingers curl over the rock
UPPER_ARM, FOREARM = 0.25, 0.27


def elbow(shoulder, wrist, bend, a=UPPER_ARM, b=FOREARM):
    """Two-bone reach: where the elbow sits for this shoulder and wrist, bent
    toward `bend`."""
    d = wrist - shoulder
    n = min(d.length, a + b - 1e-4)
    u = d.normalized()
    x = (a * a - b * b + n * n) / (2 * n)
    h = math.sqrt(max(a * a - x * x, 0.0))
    v = V(bend) - u * V(bend).dot(u)
    return shoulder + u * x + v.normalized() * h


# Legs: hip, knee, ankle per side.
LEGS = {s: (V(s * 0.12, 0.04, 0.30), V(s * 0.172, -0.045, 0.172), V(s * 0.178, 0.005, 0.07)) for s in (-1, 1)}


# ---------------------------------------------------------------- body


def body_elements(LW, LE):
    E = []
    # Torso pitched forward: pelvis, a pot belly sagging over the belt, a
    # barrel chest, and the hump and traps riding up behind the low head.
    E += [
        ell((0, 0.045, 0.32), (0.15, 0.115, 0.09)),
        ball((-0.075, 0.11, 0.30), 0.082), ball((0.075, 0.11, 0.30), 0.082),
        ell((0, -0.10, 0.39), (0.17, 0.19, 0.15)),  # pot belly, sagging over the belt
        ell((0, -0.2, 0.298), (0.13, 0.026, 0.013), stiff=3, neg=True),  # fold under the gut
        ell((0, -0.11, 0.58), (0.19, 0.145, 0.13)),  # chest, hunched forward
        ell((0, -0.03, 0.80), (0.2, 0.14, 0.14), rot=(-0.45, 0, 0)),  # hump
        seg((0, -0.12, 0.735), hp(0, 0.05, -0.01), (0.08, 0.07)),  # neck, thrust forward
        ell((0, 0.135, 0.62), (0.022, 0.03, 0.13), rot=(-0.4, 0, 0), neg=True),  # spine groove
        ball((0, -0.29, 0.385), 0.013, neg=True),  # navel
    ]
    for s in (-1, 1):
        sh = RS if s < 0 else LS
        E += [
            ell((s * 0.1, -0.168, 0.635), (0.1, 0.048, 0.064), rot=(0.3, 0, s * 0.15), stiff=5),  # pec
            ell((s * 0.1, -0.216, 0.572), (0.088, 0.02, 0.013), rot=(0.25, 0, s * 0.22), stiff=3, neg=True),  # pec underline
            ell((s * 0.12, -0.045, 0.808 + (0.0 if s < 0 else 0.015)), (0.12, 0.09, 0.068), rot=(0, s * 0.35, 0)),  # trap
            ell((s * 0.17, 0.02, 0.58), (0.08, 0.1, 0.13)),  # lat
            ell((s * 0.222, -0.075, 0.52), (0.015, 0.045, 0.08), rot=(0.2, s * 0.25, 0), stiff=3, neg=True),  # lat edge
            ell((s * 0.1, 0.085, 0.72), (0.085, 0.05, 0.08), rot=(-0.4, 0, s * 0.2), stiff=5),  # shoulder blade
            ell((s * 0.13, -0.02, 0.37), (0.06, 0.09, 0.06)),  # love handle
            ell(sh + V(s * 0.012, -0.005, 0.016), (0.112, 0.11, 0.098), stiff=5),  # deltoid
        ]
    # Arms: distinct masses with narrow joints between them, never a sausage.
    for S, El, W in ((RS, RE, RW), (LS, LE, LW)):
        out = V(1 if S.x > 0 else -1, 0, 0)
        bend = (S - El).normalized() + (W - El).normalized()
        bend = bend.normalized() if bend.length > 0.2 else V(0, -1, 0)
        E += [
            seg(S, El, 0.05, t=0.5),
            seg(S, El, 0.068, t=0.6, off=(0, -0.022, 0.0), stiff=5),  # biceps
            seg(S, El, 0.058, t=0.42, off=(0, 0.03, 0.0), stiff=5),  # triceps
            ball(El, 0.043),
            seg(El, W, (0.09, 0.076), t=0.3, half=0.115, off=(-0.006 if S.x < 0 else -0.012, 0, 0), stiff=5),  # forearm
            seg(El, W, 0.046, t=0.74),
            ball(W, 0.04),
            ball(S.lerp(El, 0.4) + out * 0.07 + V(0, -0.028, 0), 0.022, stiff=3, neg=True),  # deltoid insertion
            ball(El + bend * 0.052, 0.02, stiff=3, neg=True),  # elbow crease
        ]
    # Legs: thick and short, a big calf, a knee and an ankle.
    for s, (H, Kn, A) in LEGS.items():
        E += [
            seg(H, Kn, (0.102, 0.097), t=0.45),
            seg(H, Kn, 0.072, t=0.62, off=(0, -0.03, 0.01), stiff=5),
            ball(Kn, 0.058),
            seg(Kn, A, (0.08, 0.074), t=0.35, off=(0, 0.024, 0), stiff=5),
            seg(Kn, A, 0.052, t=0.72),
            ball(A, 0.05),
        ]
    return E


def foot_at(s, dx, dy, dz):
    _, _, A = LEGS[s]
    yaw = s * 0.18
    return V(A.x + dx * math.cos(yaw) - dy * math.sin(yaw), A.y + dx * math.sin(yaw) + dy * math.cos(yaw), dz)


TOES = (-0.046, 0.0, 0.046)


def foot_elements(s):
    yaw = s * 0.18
    E = [
        ell(foot_at(s, 0, -0.05, 0.036), (0.078, 0.095, 0.042), rot=(0, 0, yaw)),
        ball(foot_at(s, 0, 0.03, 0.042), 0.05),
    ]
    for dx in TOES:
        E.append(ell(foot_at(s, dx * 1.05, -0.152, 0.03), (0.024, 0.036, 0.028), rot=(0, 0, yaw), stiff=6))
    return E


FINGERS = (-0.057, -0.019, 0.019, 0.057)


def fist_elements():
    # Knuckle-walking: the back of the hand faces forward and the curled
    # fingers rest their middle joints flat on the ground.
    E = [ell(RF + V(0, 0.005, 0.062), (0.078, 0.058, 0.068), stiff=4), ball(RW, 0.046)]
    for i, dx in enumerate(FINGERS):
        k = RF + V(dx, -0.036, 0.022 + 0.005 * (i in (1, 2)))
        E.append(ball(k, 0.029, stiff=5))  # knuckle
        E.append(seg(k, k + V(0, 0.055, -0.018), 0.026, stiff=5))  # curled finger
    E.append(seg(RF + V(0.075, 0.01, 0.095), RF + V(0.086, -0.042, 0.048), 0.027, half=0.042, stiff=5))  # thumb
    return E


def grip_elements(rock_bvh):
    # The palm presses on the boulder's outer front; thick fingers, rooted in
    # the palm, curl up over its shoulder; the thumb braces its inner face.
    up = GRIP_UP
    side = GRIP.cross(up).normalized()

    def on_rock(d, lift):
        d = d.normalized()
        hit = rock_bvh.ray_cast(BOULDER, d, 1.0)
        return (hit[0] if hit[0] is not None else BOULDER + d * BR) + d * lift

    palm = on_rock(GRIP, 0.03)
    E = [ell(palm, (0.068, 0.062, 0.036), rot=GRIP.to_track_quat("Z", "Y").to_euler(), stiff=4)]
    for spread in (-0.32, -0.107, 0.107, 0.32):
        pts = [on_rock(GRIP * math.cos(t) + up * math.sin(t) + side * spread, 0.026 - 0.003 * j)
               for j, t in enumerate((0.2, 0.62, 0.98))]
        for j, (a, b) in enumerate(zip(pts, pts[1:])):
            E.append(seg(a, b, 0.027 - 0.003 * j, half=(a - b).length * 0.62, stiff=4))
    thumb = [palm + side * 0.04, on_rock(GRIP + side * 0.5 + up * 0.05, 0.03), on_rock(GRIP * 0.8 + side * 0.72 + up * 0.32, 0.026)]
    E.append(seg(thumb[0], thumb[1], 0.03, half=(thumb[1] - thumb[0]).length * 0.62, stiff=4))
    E.append(seg(thumb[1], thumb[2], 0.026, half=(thumb[2] - thumb[1]).length * 0.62, stiff=4))
    wrist = palm - up * 0.072 + GRIP * 0.012
    E.append(ball(wrist, 0.043))
    POSE["LW"] = wrist
    POSE["LE"] = elbow(LS, wrist, (0.18, -0.95, -0.3))
    POSE["palm"] = palm
    return E, wrist


NOSE = (0, -0.138, -0.024)  # the potato nose's bulb, in the head frame


def head_elements():
    E = [
        ell(hp(0, 0.02, 0.022), (0.068, 0.07, 0.056)),  # low skull
        ell(hp(0, 0.1, 0.0), (0.06, 0.09, 0.062), rot=(0.45, 0, 0)),  # nape into the hump
        ell(hp(0, -0.07, 0.036), (0.06, 0.028, 0.022), stiff=5),  # heavy brow ridge
        ball(hp(-0.05, -0.04, -0.012), 0.034), ball(hp(0.05, -0.04, -0.012), 0.034),  # cheeks
        ell(hp(0, -0.078, -0.03), (0.054, 0.044, 0.03)),  # muzzle
        ell(hp(0, -0.098, -0.084), (0.085, 0.065, 0.04), stiff=5),  # jutting lower jaw
        ball(hp(0, -0.142, -0.09), 0.034, stiff=5),  # chin
        ball(hp(-0.064, -0.028, -0.062), 0.034), ball(hp(0.064, -0.028, -0.062), 0.034),  # jaw hinge
        ell(hp(0, -0.1, 0.008), (0.022, 0.035, 0.022), rot=(0.5, 0, 0), stiff=5),  # nose bridge
        ell(hp(*NOSE), (0.046, 0.044, 0.046), stiff=5),  # potato nose
        ball(hp(-0.036, -0.122, -0.042), 0.021, stiff=5), ball(hp(0.036, -0.122, -0.042), 0.021, stiff=5),  # nostril wings
        ball(hp(-0.019, -0.15, -0.074), 0.012, neg=True), ball(hp(0.019, -0.15, -0.074), 0.012, neg=True),  # nostrils
        ell(hp(0, -0.13, -0.068), (0.072, 0.03, 0.007), neg=True),  # mouth line
        ball(hp(-0.034, -0.08, 0.01), 0.021, neg=True), ball(hp(0.034, -0.08, 0.01), 0.021, neg=True),  # sockets
    ]
    for e in E:
        e["radius"] *= HS
    return E


def flesh(rock_bvh):
    grip, LW = grip_elements(rock_bvh)
    E = body_elements(LW, POSE["LE"]) + head_elements() + fist_elements() + grip
    for s in (-1, 1):
        E += foot_elements(s)
    palm = POSE["palm"]

    def bone(co, a, b):
        """Distance from co to the segment a-b."""
        d = b - a
        t = max(0.0, min(1.0, (co - a).dot(d) / d.length_squared))
        return (co - (a + d * t)).length

    limbs = [(RS, RE), (RE, RW), (LS, POSE["LE"]), (POSE["LE"], POSE["LW"])]
    limbs += [seg_ for H, Kn, A in LEGS.values() for seg_ in ((H, Kn), (Kn, A))]

    def detail(co):
        # Decimation pays a length cost wherever this is above zero, so every
        # part of the flesh gets a floor and the edge length evens out; the
        # face, hands and limbs (the silhouette) get more.
        head = 0.55 - ((co - hp(0, -0.08, -0.03)).length - 0.07 * HS) / (0.06 * HS)
        fist = 0.35 - ((co - RF).length - 0.08) / 0.1
        grip = 0.5 - ((co - palm).length - 0.07) / 0.08
        limb = 0.45 if min(bone(co, a, b) for a, b in limbs) < 0.12 else 0.0
        return max(head, fist, grip, limb, BASE_DETAIL, 0.0)

    o = organic(E, 0.0072, "body", FLESH_TRIS, lumps=0.018, lump_size=0.07, relax=2, detail=detail, protect=PROTECT, name="troll")
    o = flatten_below(o, 0.0)
    pieces = islands(o)
    if len(pieces) > 1:
        print(f"Error: troll flesh has {len(pieces)} islands {pieces[:6]}; something failed to fuse")

    return o


# ---------------------------------------------------------------- props


def rock():
    """A heavy weathered boulder: broad chipped planes with worn, rounded
    edges, a gritty pitted surface and a few cracks."""
    rnd = random.Random(7)
    bm = bmesh.new()
    for _ in range(ROCK_POINTS):
        z = rnd.uniform(-1, 1)
        a = rnd.uniform(0, math.tau)
        rr = math.sqrt(1 - z * z)
        k = BR * rnd.uniform(0.84, 1.0)
        bm.verts.new((math.cos(a) * rr * k * 1.08, math.sin(a) * rr * k, z * k * 0.88))
    bmesh.ops.convex_hull(bm, input=bm.verts)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    # Merge near-coplanar facets into a few broad breaks.
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(10), verts=bm.verts[:], edges=bm.edges[:])
    me = bpy.data.meshes.new("boulder")
    bm.to_mesh(me)
    bm.free()
    r = lib.set_role(lib._link(bpy.data.objects.new("boulder", me)), "skin")
    # Chipped edges, worn just enough to catch the light, then a fine mesh
    # for the weathering to cut into.
    r = lib.bevel_edges(r, 0.013, 2, angle=math.radians(8))
    r.modifiers.new("tri", "TRIANGULATE")
    r = lib.apply_modifiers(r)
    mod = r.modifiers.new("sub", "SUBSURF")
    mod.subdivision_type = "SIMPLE"
    mod.levels = mod.render_levels = 3
    r = lib.apply_modifiers(r)
    r = lib.displace(r, 0.011, size=0.09, seed=4)
    r = lib.displace(r, 0.005, size=0.025, seed=9)
    # Cracks and pits, centred on the faces the cameras see (given in the
    # figure's frame and turned into the rock's own).
    inv = Euler(ROCK_ROT).to_matrix().transposed()
    cracks = []
    for w, span in (((0.85, 0.15, 0.5), 0.9), ((-0.3, -0.8, 0.5), 0.75), ((0.2, 0.7, 0.65), 0.9), ((0.05, -0.15, 1.0), 0.6)):
        a = (inv @ V(w)).normalized()
        n = a.cross(V(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))).normalized()
        cracks.append((n, a, span, rnd.uniform(0, 6)))
    pits = [((inv @ V(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.1, 1))).normalized(),
             rnd.uniform(0.07, 0.12), rnd.uniform(0.008, 0.014)) for _ in range(12)]

    def groove(p):
        """How deep the weathering cuts at p (0 where untouched)."""
        u = p.normalized()
        cut = 0.0
        for n, a, span, ph in cracks:
            ang = math.acos(max(-1.0, min(1.0, u.dot(a))))
            if ang > span:
                continue
            fade = 1 - (ang / span) ** 2
            dist = u.dot(n) + 0.06 * math.sin(ang * 7 + ph)
            w = 0.03
            if abs(dist) < w:
                cut = max(cut, 0.035 * fade * (1 - abs(dist) / w) ** 2)
        for d, rad, depth in pits:
            ang = math.acos(max(-1.0, min(1.0, u.dot(d))))
            if ang < rad:
                cut = max(cut, depth * (1 - (ang / rad) ** 2) ** 2)
        return cut

    for v in r.data.vertices:
        v.co = v.co - v.co.normalized() * groove(v.co)
    r.data.update()
    # The planes thin out; the cracks and pits keep their edges.
    r = decimate_to(r, 1000, keep=lambda v: 0.15 + 0.7 * min(1.0, groove(v.co) / 0.006), protect=3.0)
    lib.place(r, (0, 0, 0), ROCK_ROT)
    lib.place(r, BOULDER)
    return r


def loincloth(bvh):
    parts = []
    axis = (0.0, 0.03)
    n = 40
    # The belt dips under the gut in front and rides up over the rump.
    def belt_z(a):
        return 0.295 + 0.045 * (1 - math.cos(a)) / 2

    ring_ = []
    rows = []
    for i in range(n + 1):
        a = -math.pi + math.tau * i / n
        z0 = belt_z(a)
        r0 = outer_radius(bvh, axis, a, z0) + 0.004
        # Ragged hem: longer flaps front and back, short at the hips.
        flap = max(math.cos(a), 0) ** 3 * 0.13 + max(-math.cos(a), 0) ** 3 * 0.1
        tear = 0.022 * abs(math.sin(a * 5.5)) + 0.012 * abs(math.sin(a * 11 + 1))
        z1 = z0 - 0.075 - flap + tear
        col = []
        r = r0
        for k in range(6):
            t = k / 5
            z = z0 + (z1 - z0) * t
            rs = outer_radius(bvh, axis, a, z) + 0.006
            r = max(rs, r + 0.012 * t)
            fold = 1 + 0.035 * math.sin(a * 9) * t
            p = V(axis[0], axis[1], 0) + radial(a) * r * fold
            col.append(V(p.x, p.y, z))
        rows.append(col)
        ring_.append(V(axis[0], axis[1], 0) + radial(a) * (r0 + 0.004) + V(0, 0, z0))
    rings = [[rows[i][k] for i in range(n + 1)] for k in range(6)]
    parts.append(shell(rings, "cloth", 0.008, "loincloth"))
    # A rope belt, beaded so it reads as twisted cord; the knot and its ends
    # carry the accent.
    pts = ring_[::2] + [ring_[2]]
    radii = [0.012 if i % 2 else 0.0095 for i in range(len(pts))]
    parts.append(lib.tube(pts, radii, role="body", sides=6, resolution=1, name="belt"))
    knot = ring_[n // 2 + 5]
    parts.append(lib.sphere(0.024, loc=knot + V(0, -0.008, 0), scale=(1, 0.8, 1.1), role="trim", segments=10, rings=7, name="knot"))
    for dx, ln in ((-0.012, 0.1), (0.014, 0.085)):
        top = knot + V(dx, -0.014, -0.01)
        parts.append(lib.tube([top, top + V(dx, -0.014, -ln * 0.5), top + V(dx * 1.8, -0.008, -ln)],
                              [0.01, 0.009, 0.007], role="trim", sides=6, resolution=3, name="rope_end"))
    return parts, ring_


def head_details(bvh):
    parts = []

    def on_skin(p, lift=0.0):
        q, n = bvh.find_nearest(p)[:2]
        return q + n * lift

    for s in (-1, 1):
        # Small glowing eyes deep under the brow.
        parts.append(lib.sphere(0.0135 * HS, loc=hp(s * 0.034, -0.067, 0.008), role="eye", segments=10, rings=6, name="eye"))
        # Tusks from the jutting lower jaw, flaring out and forward and
        # stopping at the nose's mid-height, each bound with a band where it
        # leaves the lip.
        base = hp(s * 0.056, -0.128, -0.08)
        mid = base + hd(s * 0.01, -0.022, 0.042)
        tip = base + hd(s * 0.032, -0.034, 0.062)
        parts.append(lib.tube([base, mid, tip], [0.022 * HS, 0.016 * HS, 0.008 * HS], role="skin", sides=8, resolution=4, name="tusk"))
        parts.append(ring(base + (mid - base) * 0.55, mid - base, 0.0182 * HS, 0.0048 * HS, n=10, sides=5, name="tusk_band"))
    # Warts on the potato nose and one on the brow.
    nose = hp(*NOSE)
    for d, r in ((hv(0.6, -0.55, 0.45), 0.012), (hv(-0.5, -0.8, -0.2), 0.01), (hv(0.1, -0.5, 0.85), 0.009),
                 (hv(-0.75, -0.2, 0.5), 0.009)):
        p = on_skin(nose + d.normalized() * 0.06 * HS)
        parts.append(lib.sphere(r * HS, loc=p, scale=(1, 1, 0.85), role="body", segments=7, rings=4, name="wart"))
    parts.append(lib.sphere(0.008 * HS, loc=on_skin(hp(-0.05, -0.07, 0.04)), role="body", segments=7, rings=4, name="wart"))
    # A small ring through the rim of one nostril, clear of the jaw, teeth
    # and tusk.
    parts.append(ring(hp(0.027, -0.16, -0.07), hv(0.3, -0.9, 0.3), 0.011 * HS, 0.0035 * HS, n=10, sides=5, name="nose_ring"))
    # Lower teeth along the jutting jaw, standing above the lip.
    for x in (-0.026, -0.009, 0.009, 0.026):
        parts.append(lib.cylinder(0.0075 * HS, 0.0025 * HS, 0.028 * HS, loc=hp(x, -0.152, -0.06), rot=(PITCH, 0, 0), role="skin", segments=6, name="tooth"))
    return parts


def toenails():
    parts = []
    for s in (-1, 1):
        for dx in TOES:
            p = foot_at(s, dx * 1.05, -0.176, 0.034)
            parts.append(lib.sphere(0.017, loc=p, rot=(0.5, 0, s * 0.18), scale=(1.0, 0.75, 0.4), role="skin", segments=7, rings=4, name="nail"))
    return parts


def strand(path, widths, thicks, normals, role="cloth", sides=6, name="lock"):
    """A tapering lock through `path`: a lens section `widths` across and
    `thicks` deep, its flat face toward `normals`, closing in a soft point."""
    bm = bmesh.new()
    rings = []
    n = len(path)
    for i, p in enumerate(path):
        t = (path[min(i + 1, n - 1)] - path[max(i - 1, 0)]).normalized()
        nn = (normals[i] - t * normals[i].dot(t)).normalized()
        b = t.cross(nn)
        rings.append([bm.verts.new(p + b * math.cos(a) * widths[i] * 0.5 + nn * math.sin(a) * thicks[i] * 0.5)
                      for a in (math.tau * k / sides for k in range(sides))])
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(sides):
            bm.faces.new((r0[k], r0[(k + 1) % sides], r1[(k + 1) % sides], r1[k]))
    tip = bm.verts.new(path[-1] + (path[-1] - path[-2]).normalized() * widths[-1] * 0.7)
    for k in range(sides):
        bm.faces.new((rings[-1][k], rings[-1][(k + 1) % sides], tip))
    bm.faces.new(rings[0][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return lib.set_role(lib._link(bpy.data.objects.new(name, me)), role)


def lock(bvh, root, flow, length, width, thick, lift=0.02, droop=0.03, curl=0.0, free=0.4, sides=6, rings=5, taper=0.72, name="lock"):
    """A lock of hair rooted on the body: it lies along the skin, arches a
    little off it, droops past the body's edge and closes in a soft point."""
    loc, nrm = bvh.find_nearest(V(root))[:2]
    flow = V(flow)
    t1 = (flow - nrm * flow.dot(nrm)).normalized()
    t2 = nrm.cross(t1)
    path, ws, ts, ns = [], [], [], []
    for i in range(rings):
        f = i / (rings - 1)
        target = loc + t1 * length * f + t2 * curl * f * f
        q, n = bvh.find_nearest(target)[:2]
        # The root sinks into the skin so no cut end shows.
        off = (thick * 0.4 if i else -thick * 0.4) + lift * math.sin(math.pi * f * 0.85)
        hug = q + n * off
        loose = target + nrm * off
        path.append(hug.lerp(loose, free * f * f) + V(0, 0, -droop * f * f))
        ws.append(width * (1 + 0.3 * math.sin(math.pi * f * 0.75)) * (1 - taper * f ** 2.2))
        ts.append(thick * (1 - 0.55 * f))
        ns.append(nrm.lerp(n, 0.3 * (1 - f)).normalized())  # steady, so the lock does not twist
    return strand(path, ws, ts, ns, sides=sides, name=name)


def topknot(bvh):
    """The scalp's hair combed up from the brow, temples and nape into a knot
    above the crown, tied with a band, with a shaggy tuft spraying up and back
    out of it: an accent that lifts the small head above the shoulders."""
    parts = []
    knot = hp(0, 0.03, 0.118)
    up, back = hv(0, 0, 1), hv(0, 1, 0)
    centre = hp(0, 0.02, 0.022)
    roots = [(-0.034, -0.056, 0.062), (0.0, -0.066, 0.058), (0.034, -0.056, 0.062),
             (-0.058, -0.012, 0.05), (0.058, -0.012, 0.05), (-0.03, 0.06, 0.055), (0.03, 0.06, 0.055)]
    for x, y, z in roots:
        root = hp(x, y, z)
        path, ws, ts, ns = [], [], [], []
        for i in range(5):
            t = i / 4
            q = root.lerp(knot, t)
            # Bowed out over the dome of the skull, gathering into the knot.
            out = (q - centre).normalized()
            q = q + out * 0.012 * HS * math.sin(math.pi * t) * (1 - t)
            if t < 0.5:
                near, nrm = bvh.find_nearest(q)[:2]
                q = near + nrm * 0.006 * HS * (1 if i else -1)
            path.append(q)
            ws.append(0.03 * HS * (1 - 0.55 * t))
            ts.append(0.014 * HS * (1 - 0.3 * t))
            ns.append(out)
        parts.append(strand(path, ws, ts, ns, name="topknot"))
    # A tapering core under the strands, so the gathered hair reads as one
    # mass rather than a cage.
    crown = hp(0, 0.01, 0.055)
    d = knot - crown
    L = d.length
    core = lib.lathe([(0.05 * HS, 0), (0.044 * HS, 0.3 * L), (0.028 * HS, 0.68 * L), (0.014 * HS, L), (0, L + 0.004)],
                     role="cloth", segments=12, name="topknot_core")
    lib.place(core, (0, 0, 0), d.normalized().to_track_quat("Z", "Y").to_euler())
    parts.append(lib.place(core, crown))
    axis = (up + back * 0.45).normalized()
    side = axis.cross(back).normalized()
    parts.append(ring(knot + axis * 0.004, axis, 0.018 * HS, 0.0065 * HS, n=12, sides=5, name="topknot_band"))
    for k, spread in enumerate((-0.75, -0.3, 0.15, 0.6, 0.0)):
        d = (axis + side * spread * 0.55 + back * (0.25 if k == 4 else 0)).normalized()
        ln = (0.095 + 0.02 * math.cos(k * 1.9)) * HS
        path, ws, ts, ns = [], [], [], []
        for i in range(5):
            t = i / 4
            # Up out of the band, then flopping back and down under its weight.
            q = knot - axis * 0.006 + d * ln * t + back * ln * 0.55 * t * t - up * ln * 0.35 * t ** 3
            path.append(q)
            ws.append(0.04 * HS * (1 + 0.25 * math.sin(math.pi * t)) * (1 - 0.6 * t ** 2))
            ts.append(0.02 * HS * (1 - 0.45 * t))
            ns.append((back - d * back.dot(d)).normalized() * -1 + up * 0.001)
        parts.append(strand(path, ws, ts, ns, name="topknot"))
    return parts


def hair(bvh):
    parts = []
    parts += topknot(bvh)
    # Locks over the temples, running back into the mane.
    for x in (-0.06, 0.06):
        u = x / 0.03
        parts.append(lock(bvh, hp(x, -0.004, 0.044), hv(u * 0.28, 1.0, -0.15), 0.15 * HS, 0.028 * HS, 0.015 * HS,
                          lift=0.035, droop=0.03, curl=u * 0.01, free=0.35, sides=6, rings=6, name="crest"))
    # A mane over the hump and shoulders in shingled rows, each row lying
    # over the roots of the next and drooping past the silhouette's edge.
    core = V(0, 0.0, 0.76)
    rows = ((1.3, (-0.6, 0.0, 0.6)), (1.0, (-1.35, -0.85, -0.3, 0.3, 0.85, 1.35)),
            (0.7, (-1.65, -1.1, -0.55, 0.0, 0.55, 1.1, 1.65)), (0.4, (-1.3, -0.75, -0.25, 0.3, 0.8, 1.3)))
    for r, (e, azimuths) in enumerate(rows):
        for k, b in enumerate(azimuths):
            d = V(math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e))
            root, _ = surface_point(bvh, core, d)
            away = root - BOULDER
            if away.length < BR - 0.01:
                continue  # rooted inside the rock
            flow = V(math.sin(b) * 0.6, math.cos(b) * 0.4, -1)
            if away.length < BR + 0.08:
                # Locks beside the rock comb out and away from it.
                flow = flow.normalized() + away.normalized() * 0.8
            ln = 0.145 + 0.03 * math.sin(k * 2.3 + r * 1.1)
            wd = 0.056 + 0.008 * math.sin(k * 1.3 + r * 2.1)
            parts.append(lock(bvh, root, flow, ln, wd, 0.026, lift=0.02, droop=0.05,
                              curl=0.03 * math.sin(k * 1.7 + r), free=0.45, taper=0.55, name="mane"))
    return parts


def trinkets(bvh):
    parts = []
    # A necklace of fangs on a cord, sagging across the upper chest just
    # under the jutting jaw.
    cord = []
    for i in range(9):
        a = -1.15 + 2.3 * i / 8
        p, _ = surface_point(bvh, V(0, -0.04, 0.675), (math.sin(a), -math.cos(a), 0.32 - 0.38 * math.cos(a * 1.3)), 0.007)
        cord.append(p)
    parts.append(lib.tube(cord, 0.0065, role="cloth", sides=6, resolution=2, name="cord"))
    for i in range(1, 8):
        p = cord[i]
        ln = 0.038 if i % 2 else 0.056
        parts.append(lib.cylinder(0.011, 0.002, ln, loc=p + V(0, -0.008, -ln * 0.5), rot=(-0.2, 0, 0), role="skin", segments=7, name="fang"))
    return parts


def hip_skull(c, turn=0.52, tilt=0.25):
    """A little skull of bone: a round cranium, a narrow muzzle, cheekbones
    and two deep sockets, its face turned `turn` toward +X and tipped `tilt`
    up toward the board camera."""
    E = [
        ball((0, 0.006, 0.008), 0.034),
        ell((0, -0.02, -0.02), (0.022, 0.018, 0.017)),
        ball((-0.019, -0.016, -0.008), 0.012), ball((0.019, -0.016, -0.008), 0.012),
        ball((-0.014, -0.036, 0.0), 0.015, stiff=2.5, neg=True), ball((0.014, -0.036, 0.0), 0.015, stiff=2.5, neg=True),
        ball((0, -0.04, -0.018), 0.006, stiff=4, neg=True),
    ]
    o = lib.metaball(E, role="skin", resolution=0.0035, name="hip_skull")
    o = lib.decimate(o, min(1.0, 450 / max(lib.tris(o), 1)))
    lib.place(o, (0, 0, 0), (tilt, 0, turn))
    return lib.place(o, c)


def belt_bones(ring_):
    parts = []
    # A thigh bone and a little skull hung at the left hip.
    hip = ring_[int(len(ring_) * 0.68)]
    out = V(hip.x, hip.y, 0).normalized()
    a = hip + out * 0.024 + V(0, 0, -0.012)
    b = a + V(0.02, -0.02, -0.12)
    parts.append(lib.tube([a, b], [0.01, 0.009], role="skin", sides=8, resolution=1, name="bone"))
    for p in (a, b):
        for k in (-1, 1):
            parts.append(lib.sphere(0.012, loc=p + V(k * 0.009, 0, 0), role="skin", segments=7, rings=4, name="knuckle"))
    sk = hip + V(-0.06, -0.09, -0.07)
    parts.append(hip_skull(sk))
    parts.append(lib.tube([hip, sk + V(0, 0.004, 0.036)], 0.0045, role="cloth", sides=6, resolution=1, name="string"))
    return parts


def wrist_band(bvh, centre, axis, name="band"):
    """A rope wrapped round a wrist: a closed ring of beaded cord that hugs
    the flesh, found by casting rays out from the limb's core."""
    axis = V(axis).normalized()
    t1 = axis.cross(V(0, 0, 1) if abs(axis.z) < 0.9 else V(1, 0, 0)).normalized()
    t2 = axis.cross(t1)
    n = 14
    pts = []
    for i in range(n + 2):
        d = t1 * math.cos(math.tau * i / n) + t2 * math.sin(math.tau * i / n)
        p, _ = surface_point(bvh, centre, d, 0.0)
        pts.append(V(centre) + d * (min((p - V(centre)).length, 0.07) + 0.003))
    radii = [0.0072 if i % 2 else 0.006 for i in range(n + 2)]
    return lib.tube(pts, radii, role="trim", sides=6, resolution=1, name=name)


def build():
    parts = []
    boulder = rock()
    body = flesh(bvh_of(boulder))
    parts += [body, boulder]
    bvh = bvh_of(body)
    parts += head_details(bvh)
    parts += toenails()
    cloth, ring_ = loincloth(bvh)
    parts += cloth
    parts += belt_bones(ring_)
    parts += hair(bvh)
    parts += trinkets(bvh)
    LW = POSE["LW"]
    LE = POSE["LE"]
    parts.append(wrist_band(bvh, LW + (LE - LW).normalized() * 0.03, LW - LE, name="band"))
    # The head and tusks thrust far forward; set the whole figure back on the
    # plinth so it stays inside the footprint.
    for p in parts:
        lib.place(p, (0, SHIFT_Y, 0))
    return parts
