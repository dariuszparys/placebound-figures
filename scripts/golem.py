"""Golem: a Light heavy, a massive construct of carved, dressed stone.

Courses of crisp chamfered ashlar make a barrel chest that widens to great
shoulder blocks capped with riveted iron plates. A proud wedge keystone in the
middle of the chest holds the glowing rune core in a square socket. Iron bands
and rivets bind every segment; teal bracers and girdle lead the eye from the
core to the huge squared fists, each with a riveted iron cuff and a glowing
rune, which hang heavy and forward at knee height. Every strap turns its
corners over the stone it binds.
A small mask-like head sits low between the shoulders, and the legs are two
square stone pillars on plinth feet with squared toe-stones. Symmetric and
deliberate: carved masonry, never raw rock or flesh.

The figure faces -Y; its right hand is on -X, its left on +X.
"""
import math

import bmesh
from mathutils import Matrix, Vector

import lib

# Stone wants crisp chamfer highlights and dark masonry joints.
FINISH = {"edges": 0.4, "ao_floor": 0.25, "ao_distance": 0.12, "sharp_angle": 33}

# The upper body leans forward a little about the waist.
TILT = 0.12
PIVOT = Vector((0.0, 0.0, 0.56))
M_TILT = Matrix.Translation(PIVOT) @ Matrix.Rotation(TILT, 4, "X") @ Matrix.Translation(-PIVOT)

GAP = 0.013          # masonry joint between the stones of a course
BACK_BULGE = 0.09    # how far the flat dressed backs of the courses turn in at the sides
LEG_X = 0.14         # centre of each leg pillar
ELBOW = Vector((0.332, -0.07, 0.625))
WRIST = Vector((0.327, -0.21, 0.405))


# ---------------------------------------------------------------- helpers


def xf(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def frame(a, b, roll=0.0):
    """Matrix taking local +z (from the origin) along the segment a -> b."""
    a, b = Vector(a), Vector(b)
    q = Vector((0, 0, 1)).rotation_difference((b - a).normalized())
    return Matrix.Translation(a) @ q.to_matrix().to_4x4() @ Matrix.Rotation(roll, 4, "Z")


def mirror_x(obj):
    """A copy of `obj` reflected across x = 0, faces kept outward."""
    new = lib.duplicate(obj, obj.name + "_r")
    new.data.transform(Matrix.Scale(-1, 4, (1, 0, 0)))
    new.data.flip_normals()
    new.data.update()
    return new


def _hash(x, y, z):
    h = math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return h - math.floor(h) - 0.5


def _chip(bm, P, corner, d):
    """Knocks a flat chisel facet off the stone at `corner` (i, j, k), cutting
    d[0], d[1] and d[2] back along the three edges that meet there."""
    i, j, k = corner
    C = P(i, j, k)
    pts = [C + (P(*n) - C).normalized() * dd for n, dd in
           (((1 - i, j, k), d[0]), ((i, 1 - j, k), d[1]), ((i, j, 1 - k), d[2]))]
    n = (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalized()
    if n.dot(C - pts[0]) < 0:
        n = -n
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=pts[0], plane_no=n, clear_outer=True)
    edges = [e for e in bm.edges if len(e.link_faces) == 1]
    if edges:
        bmesh.ops.holes_fill(bm, edges=edges, sides=0)


def hexa(P, chamfer=0.012, cuts=1, pillow=0.0, jitter=0.001, chips=(), role="body", name="stone"):
    """A dressed stone filling the hexahedron with corners P(i, j, k) (0 or 1
    along x, y and z): a box with flat chamfers on every edge, mapped onto the
    corners, its faces pillowed outward by `pillow` (a number, or a dict keyed
    '-x', '+x', '-y', '+y', '-z', '+z'). `jitter` nudges every vertex a
    little, mirror-symmetrically; `chips` is a list of (corner, (dx, dy, dz))
    chisel facets knocked off corners."""
    def span(axis):
        tot = 0.0
        for a in (0, 1):
            for b in (0, 1):
                if axis == 0:
                    tot += (P(1, a, b) - P(0, a, b)).length
                elif axis == 1:
                    tot += (P(a, 1, b) - P(a, 0, b)).length
                else:
                    tot += (P(a, b, 1) - P(a, b, 0)).length
        return tot / 4

    sx, sy, sz = span(0), span(1), span(2)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((sx, sy, sz)), verts=bm.verts)
    if cuts:
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    bm.normal_update()
    sharp = [e for e in bm.edges if len(e.link_faces) == 2
             and e.link_faces[0].normal.dot(e.link_faces[1].normal) < 0.5]
    c = min(chamfer, 0.3 * min(sx, sy, sz))
    if c > 0:
        bmesh.ops.bevel(bm, geom=sharp, offset=c, segments=1, profile=0.5,
                        affect="EDGES", clamp_overlap=True)
    pil = pillow if isinstance(pillow, dict) else {k: pillow for k in ("-x", "+x", "-y", "+y", "-z", "+z")}
    half = (hx, hy, hz)
    corners = {(i, j, k): P(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for v in bm.verts:
        u = [v.co[a] / half[a] for a in range(3)]
        d = [0.0, 0.0, 0.0]
        for ax, key in ((0, "x"), (1, "y"), (2, "z")):
            amt = pil.get(("+" if u[ax] > 0 else "-") + key, 0.0)
            if not amt:
                continue
            o = [u[a] for a in range(3) if a != ax]
            w = max(0.0, 1 - o[0] ** 2) * max(0.0, 1 - o[1] ** 2) * min(1.0, abs(u[ax])) ** 8
            d[ax] += math.copysign(amt * w, u[ax])
        a, b, cz = [(u[i] + d[i] / half[i] + 1) / 2 for i in range(3)]
        p = Vector((0, 0, 0))
        for (i, j, k), q in corners.items():
            p += q * ((a if i else 1 - a) * (b if j else 1 - b) * (cz if k else 1 - cz))
        if jitter:
            ax_, ay_, az_ = abs(p.x) * 91.0, p.y * 91.0, p.z * 91.0
            p += Vector((math.copysign(_hash(ax_, ay_, az_), p.x), _hash(ay_, az_, ax_), _hash(az_, ax_, ay_))) * (2 * jitter)
        v.co = p
    for corner, dd in chips:
        _chip(bm, P, corner, dd)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._from_bmesh(name, bm, role)


def box_corners(size, top=(1.0, 1.0), shift=(0.0, 0.0)):
    """Corner function of a box centred on the origin, its top face scaled by
    `top` and shifted by `shift` (x, y)."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

    def P(i, j, k):
        fx, fy = (top if k else (1.0, 1.0))
        sx, sy = (shift if k else (0.0, 0.0))
        return Vector(((2 * i - 1) * hx * fx + sx, (2 * j - 1) * hy * fy + sy, (2 * k - 1) * hz))
    return P


def stone(size, loc=(0, 0, 0), rot=(0, 0, 0), top=(1.0, 1.0), shift=(0.0, 0.0), chamfer=0.012, cuts=1,
          pillow=0.0, jitter=0.001, chips=(), role="body", name="stone"):
    o = hexa(box_corners(size, top, shift), chamfer=chamfer, cuts=cuts, pillow=pillow, jitter=jitter, chips=chips,
             role=role, name=name)
    return lib.place(o, loc, rot)


def face_frame(P, u, v, back=False, lift=0.0):
    """A frame on the front (or back) face of hexahedron P at (u, v) in [0, 1]:
    local x along the face, local z up it, local -y out of it, origin `lift`
    out from the flat face."""
    j = 1 if back else 0
    p00, p10, p01, p11 = P(0, j, 0), P(1, j, 0), P(0, j, 1), P(1, j, 1)
    p = p00 * ((1 - u) * (1 - v)) + p10 * (u * (1 - v)) + p01 * ((1 - u) * v) + p11 * (u * v)
    du = (p10 - p00) * (1 - v) + (p11 - p01) * v
    dv = (p01 - p00) * (1 - u) + (p11 - p10) * u
    n = du.cross(dv) if not back else dv.cross(du)
    n.normalize()
    ey = -n
    ez = du.normalized().cross(ey).normalized()
    if ez.z < 0:
        ez = -ez
    ex = ey.cross(ez).normalized()
    m = Matrix((ex, ey, ez)).transposed().to_4x4()
    return Matrix.Translation(p + n * lift) @ m


def course(z0, z1, w, front, back, splits, gap=GAP, bulge=0.3, back_bulge=BACK_BULGE, chamfer=0.013, cuts=1,
           pillow=None, slope=0.0, bed=0.0, fronts=None, chips=None, role="body", name="course"):
    """One masonry course of the torso: blocks side by side between z0 and z1,
    cut at the fractions `splits` (from -1 to 1) of the half width; a split may
    be a (bottom, top) pair. w, front and back are (bottom, top) pairs: half
    width, front y (negative), back y. The front and back faces curve back
    toward the sides by `bulge`, and the top front edge sits `slope` lower than
    the back one. `bed` tilts the bottom face up toward the front by that
    angle, so it lies level once the course is leaned forward by it. `fronts`
    overrides the front y of some blocks (index -> (bottom, top)), `chips`
    knocks corners off some (index -> chips).
    Returns the stones and their corner functions."""
    out, Ps = [], []
    n = len(splits) - 1
    za, zb = z0 + gap / 2, z1 - gap / 2
    fronts = fronts or {}
    chips = chips or {}
    if pillow is None:
        pillow = {"-y": 0.003, "-x": 0.002, "+x": 0.002}

    def fr(f, k):
        return f[k] if isinstance(f, tuple) else f

    for s in range(n):
        f0, f1 = splits[s], splits[s + 1]
        g0 = gap / 2 if s > 0 else 0.0
        g1 = gap / 2 if s < n - 1 else 0.0

        def P(i, j, k, f0=f0, f1=f1, g0=g0, g1=g1, s=s):
            ww = w[k]
            x = (fr(f0, k) * ww + g0) if i == 0 else (fr(f1, k) * ww - g1)
            t = min(1.0, abs(x) / ww)
            if j == 0:
                y = fronts[s][k] if s in fronts else front[k] * (1 - bulge * t * t)
            else:
                y = back[k] * (1 - back_bulge * t * t)
            z = (zb - (slope if j == 0 and s not in fronts else 0.0)) if k else za - y * math.tan(bed)
            return Vector((x, y, z))
        out.append(hexa(P, chamfer=chamfer, cuts=cuts, pillow=pillow, chips=chips.get(s, ()), role=role, name=name))
        Ps.append(P)
    return out, Ps


def ring(hx, hy, z0, z1, t=0.012, c=None, inset=0.006, role="metal", bevel=0.0025, name="band"):
    """A strap of iron around a chamfered rectangle of half sizes hx, hy,
    between z0 and z1, standing t proud of it. Its corner cut `c` is only a
    little more than the stones' chamfers, so the strap turns every corner
    over the stone instead of letting the corner poke through."""
    c = 0.018 if c is None else c

    def outline(ax, ay, cc):
        return [(ax, -ay + cc), (ax, ay - cc), (ax - cc, ay), (-ax + cc, ay),
                (-ax, ay - cc), (-ax, -ay + cc), (-ax + cc, -ay), (ax - cc, -ay)]
    outer = outline(hx + t, hy + t, c + t * 0.414)
    inner = outline(hx - inset, hy - inset, max(c - inset * 0.414, 0.002))
    bm = bmesh.new()
    loops = [[bm.verts.new((x, y, z0)) for x, y in inner], [bm.verts.new((x, y, z0)) for x, y in outer],
             [bm.verts.new((x, y, z1)) for x, y in outer], [bm.verts.new((x, y, z1)) for x, y in inner]]
    for la, lb in zip(loops, loops[1:] + loops[:1]):
        for k in range(8):
            m = (k + 1) % 8
            bm.faces.new((la[k], la[m], lb[m], lb[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    o = lib._from_bmesh(name, bm, role)
    if bevel:
        o = lib.bevel_edges(o, bevel, 1)
    return o


def ring_rivets(hx, hy, zc, t=0.012, r=0.0085, sides=("front", "side", "back"), role="metal"):
    """Rivets on the faces of a ring() strap."""
    out = []
    pts = []
    if "front" in sides:
        pts += [((0, -(hy + t), zc), (0, -1, 0))]
    if "front2" in sides:
        pts += [((s * hx * 0.5, -(hy + t), zc), (0, -1, 0)) for s in (-1, 1)]
    if "side" in sides:
        pts += [((s * (hx + t), 0, zc), (s, 0, 0)) for s in (-1, 1)]
    if "back" in sides:
        pts += [((0, hy + t, zc), (0, 1, 0))]
    for p, n in pts:
        out.append(rivet(p, n, r, role))
    return out


def rivet(p, n, r=0.0085, role="metal"):
    prof = [(r, 0.0), (r * 0.92, r * 0.35), (r * 0.6, r * 0.7), (0, r * 0.8)]
    o = lib.lathe(prof, role=role, segments=8, name="rivet")
    n = Vector(n).normalized()
    q = Vector((0, 0, 1)).rotation_difference(n)
    return xf(o, Matrix.Translation(Vector(p) - n * r * 0.15) @ q.to_matrix().to_4x4())


RUNES = {
    "othala": [((0, 0.5), (0.32, 0.12)), ((0, 0.5), (-0.32, 0.12)), ((0.32, 0.12), (-0.3, -0.5)), ((-0.32, 0.12), (0.3, -0.5))],
    "algiz": [((0, -0.5), (0, 0.5)), ((0, 0.02), (0.36, 0.42)), ((0, 0.02), (-0.36, 0.42))],
    "dagaz": [((-0.4, -0.45), (-0.4, 0.45)), ((0.4, -0.45), (0.4, 0.45)), ((-0.4, 0.45), (0.4, -0.45)), ((-0.4, -0.45), (0.4, 0.45))],
    "ingwaz": [((0, 0.5), (0.34, 0)), ((0.34, 0), (0, -0.5)), ((0, -0.5), (-0.34, 0)), ((-0.34, 0), (0, 0.5))],
    "tiwaz": [((0, -0.5), (0, 0.5)), ((0, 0.5), (0.34, 0.16)), ((0, 0.5), (-0.34, 0.16))],
}


def glyph(rune, size, m, width=0.011, depth=0.005, proud=0.002, role="eye"):
    """A rune of glowing strokes inlaid in a face: each stroke stands only
    `proud` out of it. In the local frame the face is the XZ plane looking
    toward -Y; `m` places it."""
    out = []
    for (u0, v0), (u1, v1) in RUNES[rune]:
        du, dv = (u1 - u0) * size, (v1 - v0) * size
        L = math.hypot(du, dv) + width
        ang = math.atan2(dv, du)
        o = lib.box((L, depth, width), loc=((u0 + u1) / 2 * size, depth / 2 - proud, (v0 + v1) / 2 * size),
                    rot=(0, -ang, 0), role=role, name="rune")
        out.append(xf(o, m))
    return out


def crack(points, width, back_y, thick=0.03, name="crack"):
    """A cutter for a carved crack on a face looking toward -Y: a zigzag
    groove through `points` (x, z), `width` wide at its start and tapering
    to a hairline, reaching back to `back_y` inside the stone."""
    left, right = [], []
    n = len(points)
    for i, (x, z) in enumerate(points):
        a, b = points[max(i - 1, 0)], points[min(i + 1, n - 1)]
        dx, dz = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dz)
        nx, nz = -dz / L, dx / L
        w = width / 2 * (1 - i / (n - 1)) + 0.0004
        left.append((x + nx * w, z + nz * w))
        right.append((x - nx * w, z - nz * w))
    o = lib.extrude(left + right[::-1], thick, bevel=0.0, name=name)
    return lib.place(o, (0, back_y - thick / 2, 0))


def cramp(loc, facing=-1, length=0.056, role="metal"):
    """An iron cramp across a joint: a flat bar with a square head at each
    end, on a face looking toward -Y (facing -1) or +Y (facing 1)."""
    x, y, z = loc
    out = [lib.box((length, 0.008, 0.014), loc=(x, y, z), role=role, bevel=0.0015, name="cramp")]
    for s in (-1, 1):
        out.append(lib.box((0.018, 0.011, 0.024), loc=(x + s * length / 2, y + facing * 0.0015, z), role=role,
                           bevel=0.002, name="cramp_head"))
    return out


# ---------------------------------------------------------------- parts


def core():
    """The inner body, mortar-coloured, that shows in the joints between stones."""
    x = LEG_X
    j = [
        (0.0, 0.0, 0.47),   # 0 pelvis
        (0.0, 0.0, 0.56),   # 1 waist
        (0.0, 0.0, 0.73),   # 2 chest
        (0.0, -0.03, 0.89),  # 3 neck
        (0.3, 0.0, 0.84),   # 4 left shoulder
        (-0.3, 0.0, 0.84),  # 5 right shoulder
        (x, 0.0, 0.43),     # 6 left hip
        (x, 0.0, 0.275),    # 7 left knee
        (x, 0.0, 0.1),      # 8 left ankle
        (-x, 0.0, 0.43),    # 9
        (-x, 0.0, 0.275),   # 10
        (-x, 0.0, 0.1),     # 11
    ]
    j = [tuple(M_TILT @ Vector(p)) if i in (2, 3, 4, 5) else p for i, p in enumerate(j)]
    e = [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (0, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11)]
    r = [(0.17, 0.1), (0.15, 0.1), (0.16, 0.11), 0.05, 0.07, 0.07, 0.08, 0.082, 0.07, 0.08, 0.082, 0.07]
    return lib.skin(j, e, r, role="cloth", subdiv=1, name="core")


# The keystone: a wedge through the middle of the chest, wider at the top,
# standing proud of the courses either side of it.
KEY_Z0, KEY_Z1 = 0.675 + GAP / 2, 0.85
KEY_F0, KEY_F1 = -0.174, -0.212
KEY_ZC = 0.765
TRAP_TOP = 0.97      # inner top edge of the trapezius capstones


def key_half(z):
    return 0.076 + (z - 0.68) * 0.24


def key_P(i, j, k):
    z = KEY_Z1 if k else KEY_Z0
    y = (KEY_F1 if k else KEY_F0) if j == 0 else -0.02
    return Vector(((2 * i - 1) * key_half(z), y, z))


def key_frame():
    """Frame on the keystone face at the core: local -y out of the face."""
    v = (KEY_ZC - KEY_Z0) / (KEY_Z1 - KEY_Z0)
    return face_frame(key_P, 0.5, v)


def keystone():
    out = []
    ks = hexa(key_P, chamfer=0.012, cuts=1, jitter=0.0, name="keystone")
    F = key_frame()
    # A square socket sunk into the face holds the core.
    cutter = lib.box((0.124, 0.04, 0.124), loc=(0, -0.02 + 0.012, 0), bevel=0.003, name="socket")
    xf(cutter, F)
    ks = lib.boolean(ks, cutter)
    out.append(ks)
    # The teal bezel sits down in the socket, the faceted gem domes out of it.
    out.append(xf(lib.lathe([(0.04, 0.0), (0.057, 0.0), (0.059, 0.005), (0.055, 0.010), (0.046, 0.011),
                             (0.04, 0.006), (0.04, 0.0)],
                            loc=(0, 0.012, 0), rot=(math.pi / 2, 0, 0), role="trim", segments=24, name="bezel"), F))
    out.append(xf(lib.lathe([(0.045, -0.002), (0.045, 0.008), (0.039, 0.018), (0.024, 0.026), (0, 0.029)],
                            loc=(0, 0.012, 0), rot=(math.pi / 2, 0, 0), role="eye", segments=8, start=math.pi / 8,
                            name="core_gem"), F))
    # Rays of the rune: flush glowing grooves running from the socket out to
    # the sides of the keystone.
    side_len = key_half(KEY_ZC) - 0.077
    for ang, ln in ((math.pi / 2, side_len), (-math.pi / 2, side_len)):
        ray = lib.extrude([(-0.008, 0.0), (0.008, 0.0), (0.0, ln)], 0.005, role="eye", bevel=0.001, name="ray")
        lib.place(ray, (0, 0, 0.064), (0, 0, 0))
        lib.place(ray, (0, 0.0, 0), (0, ang, 0))
        xf(ray, F @ Matrix.Translation((0, 0.0005, 0)))
        out.append(ray)
    return out


def edge_at(P, i, j, z):
    """The point at height z on the upright edge (i, j) of hexahedron P."""
    a, b = P(i, j, 0), P(i, j, 1)
    return a.lerp(b, (z - a.z) / (b.z - a.z))


def joint_cramp(Ps, s, j, z):
    """An iron cramp across the joint between blocks s and s + 1 of a course,
    on its front (j = 0) or back (j = 1) face, at height z."""
    a, b = edge_at(Ps[s], 1, j, z), edge_at(Ps[s + 1], 0, j, z)
    y = max(a.y, b.y) + 0.001 if j else min(a.y, b.y) - 0.001
    return cramp(((a.x + b.x) / 2, y, z), 1 if j else -1)


def torso():
    """Courses above the waist, the keystone with the rune core and the yoke,
    in the upright frame (tilted by the caller)."""
    out = []
    # Abdomen: three blocks in running bond, a middle stone under the keystone.
    ab = dict(z0=0.575, z1=0.675, w=(0.172, 0.215), front=(-0.133, -0.16), back=(0.12, 0.145))
    fa = 0.42
    blocks, Pa = course(ab["z0"], ab["z1"], ab["w"], ab["front"], ab["back"], [-1, -fa, fa, 1], bulge=0.35, bed=TILT,
                       chips={0: [((0, 0, 0), (0.035, 0.03, 0.03))], 2: [((1, 0, 0), (0.035, 0.03, 0.03))]},
                       name="abdomen")
    out += blocks
    # Lower chest: two rib stones either side of the keystone, and a backing
    # stone behind it that shows on the back.
    rb = dict(z0=0.675, z1=0.79, w=(0.215, 0.28), front=(-0.16, -0.2), back=(0.145, 0.175))
    za, zb = 0.675 + GAP / 2, 0.79 - GAP / 2
    kr = ((key_half(za) + GAP / 2) / 0.215, (key_half(zb) + GAP / 2) / 0.28)
    blocks, Pr = course(rb["z0"], rb["z1"], rb["w"], rb["front"], rb["back"],
                        [-1, (-kr[0], -kr[1]), kr, 1], bulge=0.4, fronts={1: (-0.03, -0.03)},
                        pillow={"-x": 0.002, "+x": 0.002}, name="ribs")
    out += blocks
    # Upper chest: two great pectoral stones, their tops sloping down to the
    # front, a chisel facet off each outer corner; the keystone rises between.
    pc = dict(z0=0.79, z1=0.9, w=(0.28, 0.255), front=(-0.208, -0.18), back=(0.175, 0.177))
    za, zb = 0.79 + GAP / 2, 0.9 - GAP / 2
    kp = ((key_half(za) + GAP / 2) / 0.28, (key_half(zb) + GAP / 2) / 0.255)
    blocks, Pp = course(pc["z0"], pc["z1"], pc["w"], pc["front"], pc["back"], [-1, (-kp[0], -kp[1]), kp, 1],
                        bulge=0.4, slope=0.035, fronts={1: (-0.06, -0.06)},
                        chips={0: [((0, 0, 1), (0.05, 0.04, 0.03))], 2: [((1, 0, 1), (0.05, 0.04, 0.03))]},
                        pillow={"-y": 0.002, "-x": 0.002, "+x": 0.002}, name="pecs")
    out += blocks
    # Runes inlaid in the rib stones, flanking the core.
    for P, u in ((Pr[0], 0.5), (Pr[2], 0.5)):
        out += glyph("algiz", 0.062, face_frame(P, u, 0.5))
    out += keystone()
    # Trapezius capstones behind the head, sloping down toward the shoulders,
    # so the head sits low in a notch between them.
    for side in (-1, 1):
        def P(i, j, k, side=side):
            x = (0.004 if i == 0 else (0.2 if k == 0 else 0.194)) * side
            y = (-0.06 if k == 0 else -0.046) if j == 0 else (0.165 if k == 0 else 0.145)
            z = 0.902 if k == 0 else (TRAP_TOP if i == 0 else 0.95)
            return Vector((x, y, z))
        if side < 0:
            P0 = P
            P = (lambda i, j, k, P0=P0: P0(1 - i, j, k))
        outer = 1 if side > 0 else 0
        out.append(hexa(P, chamfer=0.007, cuts=1, pillow=0.0,
                        chips=[((outer, 1, 1), (0.034, 0.03, 0.02))], name="trapezius"))
    # Iron cramps tie the stones across their joints, front, back and top.
    for s in (0, 1):
        out += joint_cramp(Pa, s, 0, 0.625)
        out += joint_cramp(Pa, s, 1, 0.625)
        out += joint_cramp(Pr, s, 1, 0.735)
        out += joint_cramp(Pp, s, 1, 0.845)
    # One across the trapezius joint, lying on top behind the head.
    for o in cramp((0, 0, 0), -1, length=0.07):
        out.append(xf(o, Matrix.Translation((0, 0.075, TRAP_TOP - 0.001)) @ Matrix.Rotation(-math.pi / 2, 4, "X")))
    # A rune on the back of the keystone's backing stone.
    out += glyph("tiwaz", 0.062, face_frame(Pr[1], 0.5, 0.5, back=True, lift=0.002))
    return out


def head():
    """The small mask-like head: a heavy V brow over glowing eyes, a square
    jaw and the golem's word of life, a glowing rune, on the forehead."""
    out = []

    def P(i, j, k):
        x = (0.064 if k == 0 else 0.07) * (2 * i - 1)
        y = (-0.062 if j == 0 else 0.058) + (-0.004 if k else 0)
        return Vector((x, y, 0.126 * k))
    out.append(hexa(P, chamfer=0.008, cuts=1, pillow={"-x": 0.002, "+x": 0.002, "+y": 0.002}, name="cranium"))
    for s in (-1, 1):
        out.append(stone((0.08, 0.042, 0.03), loc=(s * 0.038, -0.066, 0.078), rot=(0, -s * 0.24, 0), chamfer=0.009,
                         cuts=0, name="brow"))
        out.append(stone((0.044, 0.016, 0.022), loc=(s * 0.034, -0.069, 0.049), rot=(0, -s * 0.2, 0), chamfer=0.005,
                         cuts=0, role="eye", name="eye"))
    out.append(stone((0.122, 0.024, 0.046), loc=(0, -0.058, 0.049), chamfer=0.004, cuts=0, role="cloth", name="sockets"))
    out.append(stone((0.024, 0.036, 0.052), loc=(0, -0.075, 0.042), top=(0.8, 1.0), chamfer=0.007, cuts=0, name="nose"))
    out.append(stone((0.136, 0.104, 0.046), loc=(0, -0.03, 0.004), shift=(0, -0.008), chamfer=0.012, cuts=0,
                     name="jaw"))
    # The word on the brow.
    out += glyph("ingwaz", 0.026, Matrix.Translation((0, -0.0655, 0.1)) @ Matrix.Rotation(-0.033, 4, "X"),
                 width=0.0075, depth=0.006, proud=0.003)
    m = Matrix.Translation((0, -0.101, 0.862)) @ Matrix.Rotation(-0.24, 4, "X") @ Matrix.Scale(1.14, 4)
    for o in out:
        xf(o, m)
    return out


def shoulder():
    """The left shoulder block with its iron capping plate, band and runes,
    in the upright frame."""
    out = []
    m = Matrix.Translation((0.33, 0.0, 0.88)) @ Matrix.Rotation(-0.2, 4, "Y")
    blk = stone((0.172, 0.23, 0.165), top=(1.06, 1.04), cuts=1, chamfer=0.016,
                pillow={"-y": 0.002, "+y": 0.002},
                chips=[((1, 0, 1), (0.045, 0.055, 0.04)), ((1, 1, 1), (0.03, 0.035, 0.03))], name="shoulder")
    # A crack runs down the front face from its top edge.
    blk = lib.boolean(blk, crack([(-0.022, 0.095), (-0.006, 0.06), (-0.026, 0.034), (-0.01, 0.004), (-0.03, -0.024)],
                                 0.007, -0.11))
    out.append(xf(blk, m))
    # The iron capping plate, riveted at its corners, a rune glowing on top.
    cap = stone((0.148, 0.2, 0.026), loc=(0, -0.004, 0.0925), top=(0.92, 0.92), chamfer=0.007, cuts=0, jitter=0.0,
                role="metal", name="cap")
    out.append(xf(cap, m))
    ct = 0.0925 + 0.013
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(xf(rivet((sx * 0.052, -0.004 + sy * 0.075, ct), (0, 0, 1), 0.009), m))
    out += glyph("othala", 0.092, m @ Matrix.Translation((0, -0.004, ct)) @ Matrix.Rotation(-math.pi / 2, 4, "X"),
                 proud=0.0025)
    out.append(xf(ring(0.09, 0.118, -0.06, -0.03, t=0.012, name="shoulder_band"), m))
    for o in ring_rivets(0.09, 0.118, -0.045, t=0.012, sides=("front2", "side")):
        out.append(xf(o, m))
    # A rune on the outer face of the block.
    fx = 0.086 * (1 + 0.06 * 0.59)
    gm = m @ Matrix.Translation((fx, 0, 0.015)) @ Matrix.Rotation(math.pi / 2, 4, "Z")
    out += glyph("othala", 0.085, gm)
    return out


def fist():
    """The left fist in the forearm frame (origin at the wrist, +z up the arm,
    -y forward, +x outward). The back of the hand faces forward, a squared
    course of four knuckle stones makes its lower front edge, the fingers are
    one grooved slab tucked under and behind them, and the thumb closes the
    inner side."""
    out = []
    xo = -0.014
    out.append(stone((0.245, 0.178, 0.135), loc=(xo, 0.0, -0.077), top=(0.93, 0.93), chamfer=0.012, cuts=1,
                     pillow={"-y": 0.002}, name="hand"))
    for i in range(4):
        dx = xo + (i - 1.5) * 0.0615
        # Knuckles: a squared course along the lower front edge, facing
        # forward and down.
        out.append(stone((0.058, 0.066, 0.058), loc=(dx, -0.074, -0.15), rot=(0.32, 0, 0), top=(0.95, 0.95),
                         chamfer=0.007, cuts=0, name="knuckle"))
        # Fingers: one flat slab, grooved into four, tucked under the hand
        # behind the knuckles.
        out.append(stone((0.0585, 0.116, 0.047), loc=(dx, 0.01, -0.163), chamfer=0.006, cuts=0, name="finger"))
    thumb = stone((0.058, 0.09, 0.115), top=(0.88, 0.88), chamfer=0.009, cuts=0, name="thumb")
    out.append(lib.place(thumb, (xo - 0.133, -0.012, -0.125), (0.15, 0.0, 0.12)))
    # A riveted iron cuff round the wrist end of the hand, on its flat faces
    # below the top chamfer.
    tw = Matrix.Translation((xo, 0, 0))
    out.append(xf(ring(0.118, 0.088, -0.058, -0.024, t=0.01, c=0.016, name="hand_band"), tw))
    for o in ring_rivets(0.118, 0.088, -0.041, t=0.01, r=0.008, sides=("front2", "side")):
        out.append(xf(o, tw))
    # A rune of power glowing on the back of the fist, between cuff and
    # knuckles.
    hand_P = box_corners((0.245, 0.178, 0.135), top=(0.93, 0.93))
    out += glyph("ingwaz", 0.04, tw @ Matrix.Translation((0, 0, -0.077)) @ face_frame(hand_P, 0.5, 0.4, lift=0.002))
    return out


def arm(shoulder_joint):
    """The left arm, hanging from the shoulder joint to the fist, in world space."""
    out = []
    sj = Vector(shoulder_joint)
    elbow, wrist = ELBOW, WRIST
    um = frame(elbow, sj)
    out.append(xf(drum(0.064, 0.058, (sj - elbow).length, chamfer=0.009, name="upper_arm"), um))
    # Upper arm: a squared stone round the drum, under the shoulder block.
    ul = (sj - elbow).length
    out.append(xf(hexa(lambda i, j, k: Vector(((2 * i - 1) * (0.066 if k else 0.072), (2 * j - 1) * (0.068 if k else 0.07),
                                                0.045 + (ul - 0.105) * k)),
                       chamfer=0.013, cuts=1, pillow={"-y": 0.002, "+x": 0.002, "-x": 0.002, "+y": 0.002}, name="bicep"),
                  um))
    out.append(xf(ring(0.071, 0.07, 0.066, 0.094, t=0.011, inset=0.008, name="arm_band"), um))
    for o in ring_rivets(0.071, 0.07, 0.08, t=0.011, r=0.0075, sides=("front", "side")):
        out.append(xf(o, um))
    # Elbow: a squared joint stone set at the mean angle of the two bones,
    # the dark core showing round it.
    mean = ((sj - elbow).normalized() - (wrist - elbow).normalized()).normalized()
    em = frame(elbow, elbow + mean)
    out.append(xf(stone((0.132, 0.128, 0.1), top=(0.94, 0.94), chamfer=0.014, cuts=0,
                        chips=[((1, 0, 1), (0.03, 0.03, 0.025))], name="elbow"), em))
    out.append(xf(lib.sphere(0.06, role="cloth", segments=12, rings=8, name="elbow_core"), em))
    # Forearm: a stone widening toward the wrist, bound by a teal bracer.
    fm = frame(wrist, elbow)
    L = (elbow - wrist).length
    fw0, fw1, fd0, fd1 = 0.084, 0.064, 0.077, 0.061
    out.append(xf(hexa(lambda i, j, k: Vector(((2 * i - 1) * (fw1 if k else fw0), (2 * j - 1) * (fd1 if k else fd0),
                                                L * k)),
                       chamfer=0.014, cuts=1, pillow={"-y": 0.002, "-x": 0.002, "+y": 0.002}, name="forearm"),
                  fm))
    out.append(xf(ring(0.083, 0.076, 0.014, 0.062, t=0.01, role="trim", name="bracer"), fm))
    out.append(xf(ring(0.07, 0.065, L - 0.07, L - 0.042, t=0.011, inset=0.009, name="forearm_band"), fm))
    for o in ring_rivets(0.083, 0.076, 0.038, t=0.01, sides=("front2",)):
        out.append(xf(o, fm))
    # A rune on the outer face of the forearm, tilted to its taper.
    gz = 0.13
    gx = fw0 + (fw1 - fw0) * gz / L
    tilt = math.atan2(fw0 - fw1, L)
    gm = Matrix.Translation((gx, 0, gz)) @ Matrix.Rotation(-tilt, 4, "Y") @ Matrix.Rotation(math.pi / 2, 4, "Z")
    out += [xf(o, fm) for o in glyph("algiz", 0.075, gm)]
    out.append(xf(lib.sphere(0.055, role="cloth", segments=12, rings=8, name="wrist"), fm))
    curl = Matrix.Rotation(-0.1, 4, "X")
    out += [xf(o, fm @ curl) for o in fist()]
    return out


def drum(r0, r1, h, sides=8, chamfer=0.008, role="body", name="drum"):
    """An octagonal (by default) column drum along local z from 0 to h."""
    c = chamfer
    prof = [(0, 0), (r0 - c, 0), (r0, c), (r1, h - c), (r1 - c, h), (0, h)]
    return lib.lathe(prof, role=role, segments=sides, start=math.pi / sides, name=name)


def leg():
    """The left leg: a square stone pillar of a tall shin drum and a thigh
    over a recessed knee, on a plinth foot fronted by squared toe-stones."""
    out = []
    x = LEG_X
    out.append(stone((0.24, 0.265, 0.06), loc=(x, 0.006, 0.03), top=(0.95, 0.94), chamfer=0.01, name="foot"))
    for dx in (-0.075, 0.0, 0.075):
        out.append(stone((0.069, 0.08, 0.05), loc=(x + dx, -0.142, 0.025), top=(0.96, 0.92), chamfer=0.006, cuts=0,
                         name="toe"))
    # The foot turns out a little for a planted stance.
    splay = Matrix.Translation((x, 0, 0)) @ Matrix.Rotation(0.08, 4, "Z") @ Matrix.Translation((-x, 0, 0))
    for o in out:
        xf(o, splay)
    pil = {"-y": 0.003, "+x": 0.002, "-x": 0.002, "+y": 0.002}
    # Shin: one tall drum, a chisel facet off its outer front top corner.
    out.append(stone((0.23, 0.23, 0.193), loc=(x, 0.0, 0.1665), top=(0.95, 0.95), chamfer=0.014, cuts=1, pillow=pil,
                     chips=[((1, 0, 1), (0.03, 0.03, 0.035))], name="shin"))
    out.append(stone((0.1, 0.05, 0.075), loc=(x, -0.098, 0.277), top=(0.86, 0.8), chamfer=0.01, cuts=0,
                     name="kneecap"))
    # Thigh over the recessed knee.
    out.append(stone((0.222, 0.222, 0.12), loc=(x, 0.0, 0.355), top=(1.04, 1.03), chamfer=0.014, cuts=1, pillow=pil,
                     chips=[((1, 0, 0), (0.03, 0.03, 0.03))], name="thigh"))
    t = Matrix.Translation((x, 0, 0))
    hs = 0.115 * (1 - 0.05 * 0.8)
    out.append(xf(ring(hs, hs, 0.201, 0.241, t=0.012, inset=0.008, name="shin_band"), t))
    for o in ring_rivets(hs, hs, 0.221, t=0.012, sides=("front2", "side")):
        out.append(xf(o, t))
    ht = 0.111 * (1 + 0.035 * 0.5)
    out.append(xf(ring(ht, ht, 0.335, 0.375, t=0.012, inset=0.008, name="thigh_band"), t))
    for o in ring_rivets(ht, ht, 0.355, t=0.012, sides=("front", "side")):
        out.append(xf(o, t))
    return out


def pelvis():
    out = []
    blocks, _ = course(0.42, 0.535, (0.23, 0.22), (-0.13, -0.13), (0.12, 0.12), [-1, -0.35, 0.35, 1], bulge=0.3,
                       chips={0: [((0, 0, 0), (0.04, 0.03, 0.03))], 2: [((1, 0, 0), (0.04, 0.03, 0.03))]},
                       name="pelvis")
    out += blocks
    out.append(stone((0.31, 0.22, 0.08), loc=(0, 0.0, 0.56), top=(1.03, 1.03), chamfer=0.012, cuts=1, role="cloth",
                     name="waist"))
    out.append(ring(0.157, 0.112, 0.534, 0.582, t=0.012, role="trim", name="girdle"))
    for p, n in (((0.0, -0.124, 0.558), (0, -1, 0)), ((0.085, -0.124, 0.558), (0, -1, 0)),
                 ((-0.085, -0.124, 0.558), (0, -1, 0)), ((0.169, 0.0, 0.558), (1, 0, 0)),
                 ((-0.169, 0.0, 0.558), (-1, 0, 0)), ((0.0, 0.124, 0.558), (0, 1, 0))):
        out.append(rivet(p, n, 0.008))
    out.append(stone((0.115, 0.04, 0.135), loc=(0, -0.135, 0.44), rot=(0.1, 0, 0), top=(1.15, 1.0), chamfer=0.011,
                     cuts=1, name="apron"))
    out += glyph("dagaz", 0.065, Matrix.Translation((0, -0.155, 0.435)) @ Matrix.Rotation(0.1, 4, "X"))
    return out


def build():
    upper = torso() + head()
    left_upper = shoulder()
    upper += left_upper + [mirror_x(o) for o in left_upper]
    for o in upper:
        xf(o, M_TILT)
    sj = M_TILT @ Vector((0.338, 0.0, 0.83))
    left = arm(sj) + leg()
    return upper + left + [mirror_x(o) for o in left] + pelvis() + [core()]
