"""Earth: an earth elemental serving either side, a hulking heap of raw,
weathered boulders and shale slabs held together by glowing fissures. A
lopsided crown of shoulders, the left one far the larger with a crystal
cluster bursting out of its seam, a jagged crystal ridge cresting behind the
head and running down the back, gorilla arms swelling to rock-pile fists
knuckle-planted on the ground well ahead of the toes, thick pillar legs of the
darker earth, broken rubble at its feet, and a craggy head thrust forward
under one heavy brow ledge, its eyes fed by a fissure that runs up from the
chest. The figure faces -Y; its right hand is on -X and its left on +X."""
import math
import random

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector, noise
from mathutils.bvhtree import BVHTree

import lib

FINISH = {"edges": 0.3, "sharp_angle": 24, "ao_distance": 0.1}

EYE = 1  # material slot of the glow on every rock object


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


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


def to_obj(bm, role, name, low=None):
    """A mesh object whose slot 0 is `role`, slot 1 the glow and slot 2 the
    `low` role, if any."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = lib._link(bpy.data.objects.new(name, me))
    me.materials.append(lib.mat(role))
    me.materials.append(lib.mat("eye"))
    if low:
        me.materials.append(lib.mat(low))
    return obj


def rand_dir(rnd):
    while True:
        v = Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1)))
        if v.length > 1e-3:
            return v.normalized()


def clip(bm, co, no, mat=0):
    """Cuts away everything on the +no side of the plane and caps the hole.
    Returns the cap faces."""
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=Vector(co), plane_no=Vector(no), clear_outer=True)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    loose = [e for e in bm.edges if not e.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="EDGES")
    bnd = [e for e in bm.edges if e.is_boundary]
    if not bnd:
        return []
    faces = bmesh.ops.holes_fill(bm, edges=bnd, sides=0)["faces"]
    for f in faces:
        f.material_index = mat
    return faces


def bevel_sharp(bm, width, angle=25, segments=1, mat=-1, edges=None):
    if edges is None:
        bm.normal_update()
        edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0) > math.radians(angle)]
    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=width, segments=segments, profile=0.5, affect="EDGES",
                        clamp_overlap=True, material=mat)


CHIP = (0.86, 0.95)  # depth of the small chipped corners knocked off a stone


def hull_bm(radii, rnd, n=16, jitter=0.18, cuts=0, depth=(0.45, 0.8), taper=0.0, chips=0):
    """A jagged stone: the convex hull of points scattered over an ellipsoid,
    sliced by planes into broad fracture faces, then `chips` small corners
    knocked off. `taper` narrows it toward local +z (negative widens it)."""
    rx, ry, rz = radii
    bm = bmesh.new()
    for _ in range(n):
        d = rand_dir(rnd)
        k = 1 - jitter * rnd.random()
        w = k * (1 - taper * d.z)
        bm.verts.new((d.x * rx * w, d.y * ry * w, d.z * rz * k))
    bmesh.ops.convex_hull(bm, input=bm.verts[:])
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    for i in range(cuts + chips):
        d = rand_dir(rnd)
        h = math.sqrt((rx * d.x) ** 2 + (ry * d.y) ** 2 + (rz * d.z) ** 2)
        clip(bm, d * h * rnd.uniform(*(depth if i < cuts else CHIP)), d)
    return bm


def column_bm(half, r0, r1, rnd, n=14, jitter=0.25, flat=1.0, cap=0.75, cuts=0, depth=(0.6, 0.9), chips=0):
    """A tapered column of rock along local z, from -half (radius r0) to +half
    (radius r1), with blunt broken ends: a limb segment that runs all the way
    from joint to joint, so its neighbours overlap it instead of hanging off
    its tips like beads."""
    bm = bmesh.new()
    for i in range(n):
        t = -1 + 2 * (i + rnd.random()) / n
        r = (r0 + (r1 - r0) * (t + 1) / 2) * (1 + 0.15 * math.sin(math.pi * (t + 1) / 2))
        d = rand_dir(rnd)
        k = 1 - jitter * rnd.random()
        bm.verts.new((d.x * r * k, d.y * r * flat * k, t * half + d.z * r * cap * k))
    bmesh.ops.convex_hull(bm, input=bm.verts[:])
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    rx = max(r0, r1)
    for i in range(cuts + chips):
        d = rand_dir(rnd)
        h = abs(d.z) * (half + rx * cap) + math.hypot(d.x, d.y * flat) * rx
        clip(bm, d * h * rnd.uniform(*(depth if i < cuts else CHIP)), d)
    return bm


def split_edge(e, cuts):
    """Splits an edge into cuts + 1 equal parts; its faces keep their identity."""
    v1 = e.verts[1]
    cur, start = e, e.verts[0]
    for i in range(cuts):
        ne, nv = bmesh.utils.edge_split(cur, start, 1.0 / (cuts + 1 - i))
        cur = ne if v1 in ne.verts else cur
        start = nv


def fracture(bm, planes, gap, groove, amp=0.012, freq=26.0, step=0.023, band=0.035, seed=0, pick=None, glow=True):
    """Splits a stone along world planes (point, normal) into pieces parted by
    `gap`; the broken faces and a `groove` wide chamfer round them glow (or,
    with `glow` off, stay stone: a dead seam between layers). The seam wanders
    off its plane by up to `amp`, so it reads as a crack."""
    pick = pick or [None] * len(planes)
    seam = EYE if glow else 0
    pieces = [bm]
    for k, (co, no) in enumerate(planes):
        co, no = Vector(co), nrm(no)
        t1 = nrm(no.cross(V(0.31, 0.53, 0.79)))
        t2 = no.cross(t1)
        z = 3.7 * k + 1.3 * seed

        def warp(p, co=co, no=no, t1=t1, t2=t2, z=z):
            # Pushed along the normal by noise over the in-plane position,
            # fading out away from the cut; the slope stays under 1, so the
            # surface never folds, and both halves move alike at the seam.
            q = p - co
            u, v = q.dot(t1), q.dot(t2)
            w = max(0.0, 1 - abs(q.dot(no)) / band)
            return w * (amp * noise.noise(Vector((u * freq, v * freq, z)))
                        + 0.45 * amp * noise.noise(Vector((u * freq * 2.7, v * freq * 2.7, z + 9.1))))

        out = []
        only = None
        if pick[k] is not None:
            # A branch crack splits only the piece nearest the given point.
            def centre(p):
                return sum((v.co for v in p.verts), Vector()) / max(len(p.verts), 1)
            only = min(pieces, key=lambda p: (centre(p) - pick[k]).length)
        for p in pieces:
            ds = [(v.co - co).dot(no) for v in p.verts]
            if (only is not None and p is not only) or min(ds) > -1e-4 or max(ds) < 1e-4:
                out.append(p)
                continue
            q = p.copy()
            a = clip(p, co, no, seam)
            b = clip(q, co, -no, seam)
            for piece, cap, s in ((p, a, -1), (q, b, 1)):
                # More points along the seam so it can wander; both halves
                # share the seam, so they split it identically.
                ring = sorted({e for f in cap for e in f.edges}, key=lambda e: e.index)
                for e in ring:
                    c = int(e.calc_length() / step)
                    if c:
                        split_edge(e, min(c, 10))
                if groove:
                    ring = [e for f in cap if f.is_valid for e in f.edges]
                    bevel_sharp(piece, groove, mat=seam, edges=list(set(ring)))
                for v in piece.verts:
                    v.co += no * warp(v.co)
                big = [f for f in piece.faces if len(f.verts) > 4]
                if big:
                    bmesh.ops.triangulate(piece, faces=big, quad_method="BEAUTY", ngon_method="BEAUTY")
                bmesh.ops.translate(piece, vec=no * (s * gap / 2), verts=piece.verts)
                out.append(piece)
        pieces = out
    return pieces


def roughen(bm, amount, freq, seed):
    """Pushes verts along their normals by noise: weathered, pitted stone."""
    if not amount:
        return
    bm.normal_update()
    off = Vector((seed * 7.31, seed * 3.17, seed * 5.53))
    for v in bm.verts:
        q = v.co * freq + off
        # Broad lumps, then pits and chips about the size of the facets.
        v.co += v.normal * amount * (noise.noise(q) + 0.5 * noise.noise(q * 2.3) + 0.4 * noise.noise(q * 5.1))


def etch(bm, co, no, width=0.004, depth=0.005):
    """Scores a hairline seam into a stone along a plane: three parallel
    cuts, the middle one sunk, so the surface folds into a narrow dead groove
    (the parting between layers of shale), without splitting the stone."""
    co, no = Vector(co), nrm(no)
    rings = []
    for off in (-width, 0.0, width):
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=co + no * off, plane_no=no)
        rings.append([g for g in res["geom_cut"] if isinstance(g, bmesh.types.BMVert)])
    bm.normal_update()
    for v in rings[1]:
        v.co -= v.normal * depth
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method="BEAUTY", ngon_method="BEAUTY")


_SEED = [0]


def rock(c, radii, m=None, rot=(0, 0, 0), seed=None, n=16, jitter=0.18, cuts=5, depth=(0.45, 0.8), taper=0.0,
         bevel=0.004, split=(), gap=0.006, groove=0.008, crack=0.012, floor=None, subdiv=0, rough=0.0,
         rough_freq=32, seg=None, chips=0, chop=(), strata=(), low=None, glow=True, role="body", name="rock"):
    """A boulder centred at `c` with half sizes `radii` in the frame `m` (a
    3x3 or 4x4 rotation) or Euler `rot`. `split` lists fracture planes as
    (offset from c, normal) in world space; their seams glow unless `glow` is
    off. `strata` lists
    planes the same way that are only scored, as dead seams between layers.
    `chips` knocks small corners off; `chop` lists (world normal, fraction)
    broken faces sheared off it late, so they keep their aim. `floor` cuts it
    flat at that height, and `low` = (height, role) gives the faces below
    that height another role. With `seg` = (half length, r0, r1, flatness) it
    is a column along local z instead, and `radii` is ignored."""
    if seed is None:
        _SEED[0] += 1
        seed = _SEED[0]
    rnd = random.Random(seed * 7919 + 13)
    if seg:
        bm = column_bm(seg[0], seg[1], seg[2], rnd, n=n, jitter=jitter, flat=seg[3], cuts=cuts, depth=depth,
                       chips=chips)
    else:
        bm = hull_bm(radii, rnd, n=n, jitter=jitter, cuts=cuts, depth=depth, taper=taper, chips=chips)
    if subdiv:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=subdiv, use_grid_fill=True)
    rot3 = (m.to_3x3() if m is not None else Euler(rot).to_matrix())
    M = Matrix.Translation(Vector(c)) @ rot3.to_4x4()
    bmesh.ops.transform(bm, matrix=M, verts=bm.verts)
    roughen(bm, rough, rough_freq, seed)
    for no, frac in chop:
        no = nrm(no)
        h = max((v.co - Vector(c)).dot(no) for v in bm.verts)
        clip(bm, Vector(c) + no * h * frac, no)
    if bevel and not subdiv:
        bevel_sharp(bm, bevel, angle=25)
    for off, no in strata:
        etch(bm, Vector(c) + Vector(off), no)
    if floor is not None:
        clip(bm, (0, 0, floor), (0, 0, -1))
    planes = [(Vector(c) + Vector(sp[0]), nrm(sp[1])) for sp in split]
    pick = [Vector(c) + Vector(sp[2]) if len(sp) > 2 else None for sp in split]
    pieces = fracture(bm, planes, gap, groove, amp=crack, seed=seed, pick=pick, glow=glow) if planes else [bm]
    out = []
    for p in pieces:
        if low:
            for f in p.faces:
                if f.material_index == 0 and f.calc_center_median().z < low[0]:
                    f.material_index = 2
        out.append(to_obj(p, role, name, low[1] if low else None))
    return out


def slab(c, radii, normal, spin=0.0, front=(0, -1, 0), **kw):
    """A broken slab of rock whose thin axis (the third radius) lies along
    `normal`, turned by `spin` about it."""
    m = frame(c, normal, front) @ Matrix.Rotation(spin, 4, "Z")
    return rock(c, radii, m=m, **kw)


def along(a, b, t):
    a, b = Vector(a), Vector(b)
    return a + (b - a) * t


def bone(a, b, front=(0, -1, 0)):
    """Rotation whose local z runs from a to b."""
    return frame(a, Vector(b) - Vector(a), front).to_3x3()


def column(a, b, r0, r1, flat=0.9, over=0.0, front=(0, -1, 0), **kw):
    """A column of rock from joint `a` (radius r0) to joint `b` (radius r1),
    reaching `over` past each joint."""
    a, b = Vector(a), Vector(b)
    half = (b - a).length / 2 + over
    kw.setdefault("n", 14)
    kw.setdefault("jitter", 0.25)
    kw.setdefault("cuts", 5)
    kw.setdefault("depth", (0.55, 0.88))
    return rock((a + b) / 2, None, m=bone(a, b, front), seg=(half, r0, r1, flat), **kw)


# ---------------------------------------------------------------- pose

# A planted gorilla stance: short legs wide apart, the upper body hunched
# forward and wrenched round so the heavy left shoulder swings forward, the
# arms reaching down to fists set on the ground, the left one leading.
TWIST = Matrix.Translation((0, 0.03, 0.40)) @ Matrix.Rotation(-0.16, 4, "Z") @ Matrix.Rotation(0.12, 4, "X") \
    @ Matrix.Translation((0, -0.03, -0.40))
TW = TWIST.to_3x3()


def U(*a):
    return TWIST @ V(*a)


def tw(rot):
    """An Euler rotation carried round with the twisted upper body."""
    return TW @ Euler(rot).to_matrix()


PELVIS = V(0.0, 0.03, 0.36)
BELLY = U(0.0, 0.0, 0.49)
CHEST = U(0.0, -0.02, 0.635)
YOKE = U(0.0, 0.06, 0.76)
NECK = U(0.0, -0.08, 0.82)
HEAD = U(0.0, -0.15, 0.90)
SHO = {-1: U(-0.235, 0.03, 0.745), 1: U(0.245, 0.04, 0.80)}
ELB = {-1: U(-0.35, 0.0, 0.53), 1: U(0.355, 0.02, 0.55)}
# The hands bear weight on the ground, so they stay where they are planted:
# knuckle-walking, well out ahead of the toes.
WRI = {-1: V(-0.285, -0.2, 0.25), 1: V(0.295, -0.25, 0.25)}
FIST = {-1: V(-0.235, -0.28, 0.1), 1: V(0.24, -0.33, 0.1)}
HIP = {-1: V(-0.12, 0.04, 0.33), 1: V(0.12, 0.04, 0.33)}
KNEE = {-1: V(-0.165, 0.0, 0.2), 1: V(0.17, 0.01, 0.2)}
ANK = {-1: V(-0.15, 0.05, 0.08), 1: V(0.155, 0.06, 0.08)}


# ---------------------------------------------------------------- parts


def core():
    """The dark packed earth beneath the stones, seen in the deeper joints of
    the trunk and in the neck."""
    j = [PELVIS, BELLY, CHEST, YOKE, NECK, SHO[-1], SHO[1], HIP[-1], HIP[1], HEAD + V(0, 0.03, -0.01)]
    r = [(0.1, 0.08), 0.09, (0.15, 0.11), (0.13, 0.1), 0.07, 0.07, 0.07, 0.07, 0.07, 0.05]
    e = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (0, 7), (0, 8), (4, 9)]
    trunk = lib.skin([tuple(p) for p in j], e, r, role="cloth", subdiv=1, name="core")
    # The skin modifier can leave an end open; cap it.
    bm = bmesh.new()
    bm.from_mesh(trunk.data)
    bnd = [ed for ed in bm.edges if ed.is_boundary]
    if bnd:
        bmesh.ops.holes_fill(bm, edges=bnd, sides=0)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(trunk.data)
    bm.free()
    return [trunk]


def torso():
    out = []
    # The chest: a great slab tipped up toward the viewer, split by a fissure
    # that climbs from the belly to the throat, with a branch running down to
    # the right flank and another up toward the heavy left shoulder.
    out += rock(CHEST + V(0.0, -0.07, 0.0), (0.25, 0.12, 0.15), m=tw((-0.25, 0.0, 0.06)), n=16, cuts=6,
                depth=(0.5, 0.85), chips=2, subdiv=1, rough=0.02, rough_freq=22,
                split=[((0.012, 0, 0), (1, 0.1, -0.22)),
                       ((-0.08, 0, -0.03), (0.4, 0.1, 1), (-0.13, -0.05, -0.02)),
                       ((0.1, 0, 0.05), (-0.55, 0.0, 1), (0.13, -0.05, 0.07))],
                groove=0.02, gap=0.013, crack=0.02, name="chest")
    # A broken slab sheared off over the left breast, at odds with the chest.
    out += slab(CHEST + V(0.15, -0.14, 0.06), (0.11, 0.1, 0.04), (0.45, -1, 0.65), spin=0.5, n=14, cuts=4,
                chips=1, subdiv=1, rough=0.016, rough_freq=24, bevel=0, name="pec")
    # Slabs bridging the chest to the shoulders, the left one higher.
    out += slab(U(0.12, 0.0, 0.83), (0.13, 0.12, 0.05), (-0.35, 0.1, 1), spin=0.3, n=14, cuts=4,
                subdiv=1, rough=0.016, rough_freq=24, name="trapezius")
    out += slab(U(-0.12, -0.01, 0.785), (0.12, 0.11, 0.045), (0.45, 0.15, 1), spin=-0.4, n=14, cuts=4,
                subdiv=1, rough=0.016, rough_freq=24, name="trapezius")
    # The nape: a stone heaving up behind the head, so it juts from the
    # hunched mass of the shoulders instead of a neck.
    out += rock(U(0.0, -0.02, 0.84), (0.1, 0.09, 0.075), m=tw((0.5, 0, 0.2)), n=14, cuts=4, depth=(0.6, 0.9),
                subdiv=1, rough=0.016, rough_freq=24, name="nape")
    # The back: two great slabs stacked like split shale, the upper one
    # facing up and back, the lower one back; the crystal ridge breaks out
    # of the seam between them.
    out += slab(YOKE + V(0.0, 0.07, 0.02), (0.21, 0.16, 0.075), (0.1, 0.85, 1), spin=0.25, n=16, cuts=6,
                chips=2, subdiv=1, rough=0.02, rough_freq=22, split=[((0.05, 0, 0), (1, -0.15, 0.25))], groove=0.018,
                strata=[((0.0, 0.02, 0.022), (0.1, 0.85, 1)), ((0.0, -0.018, -0.02), (0.16, 0.8, 1))],
                gap=0.012, crack=0.018, name="hump")
    out += slab(CHEST + V(0.0, 0.12, -0.05), (0.2, 0.18, 0.08), (-0.1, 1, 0.15), spin=-0.3, n=14, cuts=6,
                chips=2, subdiv=1, rough=0.02, rough_freq=22,
                strata=[((0, 0.028, 0.004), (-0.1, 1, 0.15)), ((0, -0.024, -0.004), (-0.04, 1, 0.2))],
                split=[((-0.03, 0, 0.02), (1, 0.0, 0.35)), ((0.06, 0, -0.07), (0.2, 0.1, 1), (0.1, 0.05, -0.09))],
                groove=0.018, gap=0.012, crack=0.018, name="back")
    # The gut and hips: darker, packed earth and stone.
    out += rock(BELLY + V(0.03, -0.08, 0.0), (0.14, 0.095, 0.1), m=tw((0.1, 0.2, -0.1)), n=14, cuts=5,
                depth=(0.6, 0.9), subdiv=1, rough=0.018, rough_freq=22, name="belly")
    out += rock(BELLY + V(-0.085, -0.07, -0.035), (0.095, 0.075, 0.08), m=tw((-0.3, 0.4, 0.3)), seed=901, n=12,
                cuts=4, role="cloth", name="belly")
    for s in (-1, 1):
        out += rock(V(s * 0.17, 0.02, 0.52), (0.1, 0.11, 0.11), n=12, cuts=3, bevel=0, name="flank")
    out += rock(PELVIS + V(0.0, -0.01, 0.0), (0.19, 0.13, 0.09), n=14, cuts=4, role="cloth", name="pelvis")
    out += rock(V(0.0, 0.12, 0.45), (0.15, 0.08, 0.1), n=12, cuts=3, bevel=0, role="cloth", name="lowback")
    # A boulder wedged between the thighs, so no daylight shows under the
    # pelvis and the mass sits low.
    out += rock(V(0.0, -0.05, 0.205), (0.11, 0.09, 0.1), rot=(0.3, 0.2, 0.5), seed=1001, n=14, cuts=4,
                depth=(0.6, 0.9), role="cloth", name="crotch")
    return out


def shard(base, direction, length, width, front=(0, -1, 0), **kw):
    """A jagged spike of rock jutting from `base`, its foot buried."""
    d = nrm(direction)
    c = Vector(base) + d * (length * 0.32)
    kw.setdefault("n", 12)
    kw.setdefault("cuts", 3)
    return rock(c, (width, width * 0.75, length * 0.55), m=frame(c, d, front), taper=0.7, jitter=0.1,
                depth=(0.7, 0.95), **kw)


def shoulders():
    out = []
    # The left shoulder, far the larger and set higher: a deltoid boulder
    # capped by a tilted slab, the crystal cluster bursting from between them.
    L = SHO[1]
    out += rock(L + V(0.035, 0.0, 0.0), (0.155, 0.155, 0.13), m=tw((0.1, 0.3, -0.15)), n=16, cuts=6,
                chips=2, subdiv=1, rough=0.022, rough_freq=22,
                split=[((0.0, 0, 0.0), (-0.3, 1, 0.2)), ((0.04, 0, -0.02), (1, 0.2, -0.3), (0.06, -0.08, -0.03))],
                groove=0.02, gap=0.013, crack=0.02, name="shoulder")
    # Its cap is layered like shale, the hairline seams between the layers
    # scored round its rim.
    out += slab(L + V(0.0, 0.0, 0.085), (0.19, 0.16, 0.065), (0.4, 0.1, 1), spin=0.6, n=16, cuts=6,
                chips=2, subdiv=1, rough=0.02, rough_freq=22,
                split=[((-0.02, 0, 0), (1, 0.45, 0.1))], groove=0.02, gap=0.013, crack=0.018,
                strata=[((0.008, 0.002, 0.02), (0.4, 0.1, 1)), ((-0.006, 0.0, -0.016), (0.46, 0.05, 1))],
                name="shoulder")
    # The right shoulder, smaller and dropped.
    R = SHO[-1]
    out += rock(R + V(-0.03, 0.0, 0.0), (0.115, 0.115, 0.1), m=tw((0.1, -0.25, 0.2)), n=16, cuts=6,
                chips=2, subdiv=1, rough=0.018, rough_freq=22,
                split=[((0.0, 0, 0.01), (0.4, 1, 0.3))], groove=0.018, gap=0.012, crack=0.018, name="shoulder")
    out += slab(R + V(-0.01, 0.0, 0.08), (0.14, 0.12, 0.048), (-0.55, 0.15, 1), spin=-0.5, n=14, cuts=5,
                chips=1, subdiv=1, rough=0.018, rough_freq=22,
                strata=[((-0.008, 0.0, 0.014), (-0.55, 0.15, 1)), ((0.006, 0.0, -0.012), (-0.6, 0.1, 1))],
                name="shoulder")
    # Broken spurs of rock thrust out of the right shoulder and the back.
    out += shard(R + V(-0.08, 0.04, 0.07), (-0.9, 0.35, 0.8), 0.11, 0.034, name="spur")
    out += shard(YOKE + V(-0.10, 0.11, -0.02), (-0.5, 1, 0.4), 0.09, 0.03, name="spur")
    return out


def arm(s):
    """A gorilla arm: an upper arm half buried under the shoulder, a forearm
    swelling toward a rock-pile fist planted on the ground."""
    out = []
    S, E, W, F = SHO[s], ELB[s], WRI[s], FIST[s]
    k = 1.0 if s < 0 else 1.1  # the left arm is the heavier
    raw = dict(subdiv=1, rough=0.02, rough_freq=22, chips=2)
    out += column(along(E, S, 1.1), E, 0.1 * k, 0.085 * k, over=0.03, n=18, name="upper_arm", **raw)
    # A wedge stone jammed into the outside of the elbow.
    out += rock(E + V(s * 0.035, 0.02, 0.0), (0.075, 0.07, 0.08), rot=(0.4, s * 0.5, 0.3), n=12, cuts=4,
                depth=(0.6, 0.9), bevel=0, name="elbow")
    # The forearm: one long, lumpy column swelling into the fist, cracked
    # across, with a boulder heaved out of its outer flank like a club.
    out += column(E, along(E, W, 1.15), 0.09 * k, 0.135 * k, over=0.02, n=18, cuts=7,
                  split=[((0, 0, 0.01), (0.2, 0.3, 1))] if s > 0 else [], glow=False, groove=0.012, gap=0.01,
                  crack=0.016, name="forearm", **raw)
    out += rock(along(E, W, 0.72) + V(s * 0.055, 0.04, 0.0), (0.085 * k, 0.08 * k, 0.1 * k), m=bone(E, W),
                n=14, cuts=5, depth=(0.55, 0.85), subdiv=1, rough=0.016, rough_freq=24, name="cuff")
    out += slab(along(E, W, 0.3) + V(s * 0.065, 0.03, 0.0), (0.055, 0.1 * k, 0.035), (s, 0.4, 0.15),
                spin=0.2, front=(0, 0, 1), n=12, cuts=4, bevel=0, name="bracer")
    # The fist: a palm stone with a row of knuckle stones packed into its
    # front and a thumb tucked in on the inside, all set flat on the ground,
    # their undersides the darker earth.
    P = F + V(0.0, 0.02, 0.0)
    out += rock(P, (0.15 * k, 0.14 * k, 0.13 * k), n=16, cuts=5, depth=(0.6, 0.9), floor=0.0,
                low=(0.05, "cloth"), name="palm", **raw)
    for i, dx in enumerate((-0.064, 0.0, 0.064)):
        c = P + V(s * dx * k, -0.1 * k, -0.06 * k + 0.012 * (i == 1))
        out += rock(c, (0.072 * k, 0.062 * k, 0.07 * k), n=12, cuts=3, depth=(0.6, 0.9), floor=0.0,
                    subdiv=1, rough=0.014, rough_freq=26, low=(0.035, "cloth"), name="knuckle")
    out += rock(P + V(-s * 0.09 * k, -0.05, 0.01), (0.05, 0.058, 0.062), n=12, cuts=3, depth=(0.6, 0.9),
                bevel=0, name="thumb")
    return out


def leg(s):
    """A short, thick pillar of stone on a broad slab foot."""
    out = []
    H, K, A = HIP[s], KNEE[s], ANK[s]
    raw = dict(subdiv=1, rough=0.02, rough_freq=20, chips=2, n=18, depth=(0.7, 0.92))
    out += column(along(K, H, 1.15), K, 0.15, 0.14, over=0.02, name="thigh", **raw)
    out += rock(K + V(s * 0.015, -0.08, 0.0), (0.075, 0.055, 0.065), n=12, cuts=3, bevel=0, name="knee")
    out += column(K, A, 0.13, 0.15, over=0.05, floor=0.0, role="cloth", name="shin", **raw)
    out += rock(V(A.x + s * 0.01, A.y - 0.06, 0.035), (0.125, 0.175, 0.09), rot=(0, 0, s * 0.15), n=16, cuts=4,
                depth=(0.75, 0.95), floor=0.0, subdiv=1, rough=0.014, rough_freq=22, role="cloth", name="foot")
    return out


def head():
    """A craggy head thrust forward from between the shoulders: a heavy V brow
    over angular glowing eyes set in a blunt face block, and a jutting jaw
    split by a glowing mouth."""
    out = []
    H = HEAD
    out += rock(H + V(0.0, 0.02, 0.026), (0.095, 0.085, 0.08), n=16, cuts=3, depth=(0.7, 0.95), subdiv=1,
                rough=0.008, rough_freq=30, name="cranium")
    # The face block: a blunt wedge the eyes are set into.
    out += rock(H + V(0.0, -0.04, -0.005), (0.072, 0.045, 0.05), rot=(0.1, 0, 0), n=16, jitter=0.1, cuts=3,
                depth=(0.75, 0.95), name="face")
    # One heavy ledge of brow over both eyes, dipping to a V over the nose
    # and swept back at its ends, overhanging the eyes like a cliff, its
    # top sloping back into the skull.
    brow = rock(H + V(0.0, -0.062, 0.042), (0.092, 0.045, 0.034), rot=(0.35, 0, 0), seed=1002, n=16,
                jitter=0.1, cuts=4, depth=(0.65, 0.9), subdiv=1, rough=0.007, rough_freq=30, name="brow")
    for o in brow:
        lib.deform(o, lambda p: p + V(0, 2.6 * (p.x - H.x) ** 2, 0.3 * abs(p.x - H.x) - 0.008))
    out += brow
    for s in (-1, 1):
        _SEED[0] += 1  # the seed the two old brow stones drew, so the eyes keep their shape
        # Angular glowing slits slanting down to the nose: an angry glare.
        out += rock(H + V(s * 0.035, -0.075, 0.008), (0.033, 0.014, 0.02), rot=(0.2, -s * 0.3, s * 0.3), n=16,
                    jitter=0.05, cuts=2, depth=(0.75, 0.9), bevel=0, role="eye", name="eye")
    out += rock(H + V(0.0, -0.045, -0.055), (0.078, 0.055, 0.04), rot=(0.15, 0, 0), n=14, cuts=4, depth=(0.6, 0.9),
                split=[((0, 0, 0.008), (0, 0.25, 1))], groove=0.012, gap=0.01, name="jaw")
    # Chin up toward the viewer, and larger than life so the face carries.
    piv = H + V(0, 0.02, 0)
    m = Matrix.Translation(piv) @ Matrix.Scale(1.3, 4) @ Matrix.Rotation(-0.12, 4, "X") @ Matrix.Translation(-piv)
    for o in out:
        o.data.transform(m)
        o.data.update()
    return out


def crystal(base, direction, length, r, rnd, sink=0.4):
    """A six-sided crystal growing from `base` along `direction`, its foot sunk
    into the stone, its point chiselled off-centre."""
    tip = (rnd.uniform(-0.3, 0.3) * r, rnd.uniform(-0.3, 0.3) * r)
    prof = [(0.0, -sink * length), (r * 0.92, -sink * length + r * 0.3), (r, length * 0.1),
            (r * 0.96, length * 0.72), (0.0, length)]
    o = lib.lathe(prof, role="trim", segments=6, start=rnd.uniform(0, 1.05), name="crystal")

    def chisel(v):
        t = max(0.0, (v.z - length * 0.72) / (length * 0.28))
        return Vector((v.x + tip[0] * t, v.y + tip[1] * t, v.z))
    lib.deform(o, chisel)
    m = frame(base, direction, rand_dir(rnd))
    o.data.transform(m)
    o.data.update()
    return o


def cluster(base, up, size, seed, count=5, spread=0.55):
    rnd = random.Random(seed)
    up = nrm(up)
    out = [crystal(base, up, size, size * 0.24, rnd)]
    side = nrm(up.cross(V(0.31, 0.57, 0.76)))
    for k in range(count - 1):
        a = k * math.tau / (count - 1) + rnd.uniform(-0.4, 0.4)
        lat = (side * math.cos(a) + up.cross(side) * math.sin(a))
        d = nrm(up + lat * math.tan(spread * rnd.uniform(0.6, 1.2)))
        ln = size * rnd.uniform(0.38, 0.7)
        out.append(crystal(base + lat * size * 0.18, d, ln, ln * 0.27, rnd))
    return out


def ridge(points, sizes, ups, seed):
    """A jagged crest of crystals breaking out along a seam: one big crystal at
    each point with a smaller one splaying off beside it."""
    rnd = random.Random(seed)
    out = []
    for i, (p, size, up) in enumerate(zip(points, sizes, ups)):
        up = nrm(up)
        side = nrm(up.cross(V(0.31, 0.57, 0.76)))
        d = nrm(up + side * rnd.uniform(-0.2, 0.2))
        out.append(crystal(V(p), d, size, size * 0.19, rnd, sink=0.45))
        lat = side * (1 if i % 2 else -1)
        ln = size * rnd.uniform(0.45, 0.65)
        out.append(crystal(V(p) + lat * size * 0.18, nrm(up + lat * 0.6), ln, ln * 0.21, rnd, sink=0.5))
    return out


def crystals():
    out = []
    # The crown of the figure: the greatest cluster bursting out of the left
    # shoulder's seam and thrusting out past the shoulder line, a smaller one
    # over the right.
    out += cluster(SHO[1] + V(-0.02, 0.06, 0.075), (0.95, 0.4, 1.0), 0.24, 1, count=5, spread=0.6)
    out += cluster(SHO[-1] + V(-0.03, 0.05, 0.07), (-1.1, 0.45, 1.0), 0.17, 2, count=5)
    # The ridge down the back, cresting up behind the head, where the board
    # camera sees it, and running down to the middle of the back.
    pts = [YOKE + V(0.06, 0.05, 0.12), YOKE + V(0.03, 0.08, 0.09), YOKE + V(0.04, 0.14, 0.0),
           CHEST + V(0.015, 0.18, 0.08), CHEST + V(0.035, 0.195, 0.02)]
    out += ridge(pts, [0.15, 0.14, 0.13, 0.115, 0.1],
                 [(0.2, 0.5, 1), (0.1, 0.7, 1), (0.05, 0.9, 1), (0.1, 1, 0.75), (-0.05, 1, 0.5)], 3)
    # Shards along the outside of the heavy left forearm.
    out += cluster(along(ELB[1], WRI[1], 0.3) + V(0.045, 0.06, 0.02), (0.45, 0.8, 0.6), 0.1, 6, count=3)
    return out


def rubble():
    """Broken stone heaped into a low mound under the feet, kept on the plinth."""
    out = []
    rnd = random.Random(99)
    # Big broken blocks out in front, between the feet and the fists, where
    # the board camera sees them, each with a fresh broken face turned up to
    # the light; a smaller one wedged behind them.
    spots = [(0.0, -0.13, 0.074), (-0.09, -0.105, 0.066), (0.095, -0.1, 0.068), (0.02, -0.04, 0.05),
             (-0.035, -0.135, 0.036)]
    for i, (x, y, r) in enumerate(spots):
        role = "body" if i % 2 == 0 else "cloth"
        face = (rnd.uniform(-0.5, 0.1), rnd.uniform(-0.8, -0.4), 0.8)
        out += rock(V(x, y, r * 0.3), (r * 1.25, r, r * 0.8), rot=(rnd.uniform(-0.3, 0.3), 0, rnd.uniform(0, 3)),
                    seed=1100 + i, n=12, cuts=3, chop=[(face, 0.62)], floor=0.0, subdiv=1, rough=0.012,
                    rough_freq=26, role=role, name="rubble")
    return out


GROUNDED = ("palm", "knuckle", "foot")


def check(parts):
    """Prints an EARTHCHECK line for every part that touches neither another
    part nor the ground (a floating stone), and for every part that should
    bear weight but stops short of the ground. The REPORT catches neither."""
    trees = []
    for o in parts:
        me = o.data
        vs = [o.matrix_world @ v.co for v in me.vertices]
        tree = BVHTree.FromPolygons(vs, [tuple(p.vertices) for p in me.polygons]) if me.polygons else None
        lo = Vector([min(v[i] for v in vs) for i in range(3)]) if vs else Vector()
        hi = Vector([max(v[i] for v in vs) for i in range(3)]) if vs else Vector()
        trees.append((o.name, tree, lo, hi))
    for i, (name, tree, lo, hi) in enumerate(trees):
        if tree is None:
            continue
        if name.split(".")[0] in GROUNDED and lo.z > 0.003:
            print(f"EARTHCHECK SHORT {name} stops {lo.z:.4f} above the ground")
        if lo.z < 0.002:
            continue
        touch = False
        for j, (_, other, lo2, hi2) in enumerate(trees):
            if j == i or other is None or any(lo[a] > hi2[a] or lo2[a] > hi[a] for a in range(3)):
                continue
            if tree.overlap(other):
                touch = True
                break
        if not touch:
            print(f"EARTHCHECK FLOAT {name} touches nothing")


def build():
    parts = []
    parts += core()
    parts += torso()
    parts += shoulders()
    for s in (-1, 1):
        parts += arm(s)
        parts += leg(s)
    parts += head()
    parts += crystals()
    parts += rubble()
    check(parts)
    return parts
