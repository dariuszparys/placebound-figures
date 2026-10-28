"""Banshee: a Dark flyer, the radial scream. A wailing spirit hovering above
the board: a gaunt skull-like face thrown back in an open scream, hollow
glowing eyes, wild hair streaming up and back, skeletal arms flung out with
clawed fingers, and a tattered shroud whose ragged hems trail down into wisps
instead of feet. Broken shackles hang from her wrists and torn ribbons stream
from her girdle. The figure faces -Y; her right hand is on -X."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import lib

TAU = math.tau
DEBUG = True
SCALE = 1.10
SKULL_TRIS = 3000
HAND_TRIS = 780
CURL = 0.80
# Smooth shading across the turns of thin tubes (fingers, claws, hair locks)
# so they read round, not as prisms; cloth edges stay crisp.
FINISH = {"sharp_angle": 62}


def note(name, obj):
    if DEBUG:
        print(f"BANSHEE {name}: {lib.tris(obj)} tris")
    return obj


# ---------------------------------------------------------------- helpers


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def nrm(v):
    v = Vector(v)
    return v.normalized() if v.length > 1e-9 else Vector((0, 0, 1))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def hash01(*k):
    """A repeatable pseudo-random number in [0, 1) from a few numbers."""
    s = math.sin(sum(v * c for v, c in zip(k, (12.9898, 78.233, 37.719, 4.581))) + 0.5) * 43758.5453
    return s - math.floor(s)


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


def cut(obj, where):
    """Deletes the faces whose centre satisfies `where`."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if where(f.calc_center_median())], context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


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
    m = r - r.dot(tang[0]) * tang[0]
    if m.length < 1e-6:
        m = Vector((1, 0, 0)) - tang[0].x * tang[0]
    norms = [m.normalized()]
    for i in range(1, n):
        m = norms[-1] - norms[-1].dot(tang[i]) * tang[i]
        norms.append(m.normalized())
    binorms = [t.cross(m) for t, m in zip(tang, norms)]
    return tang, norms, binorms


def grid(fn, nu, nv, role, name, roles=None, wrap=False):
    """An open sheet: fn(i, j) gives the vertex for column i (0..nu) and row j
    (0..nv); roles(i, j) may pick a role per quad. `wrap` joins the last
    column back to the first."""
    cols = nu if wrap else nu + 1
    verts = [fn(i, j) for j in range(nv + 1) for i in range(cols)]
    faces, fr = [], []
    for j in range(nv):
        for i in range(nu):
            a, b = j * cols + i, j * cols + (i + 1) % cols
            faces.append((a, b, b + cols, a + cols))
            fr.append(roles(i, j) if roles else role)
    return mesh_object(name, verts, faces, role, fr if roles else None)


def torus(R, r, m, role="metal", seg=12, sides=6, name="torus", sx=1.0):
    """A ring of radius R and tube r round local z, stretched sx along local x,
    put in frame m."""
    verts, faces = [], []
    for i in range(seg):
        a = TAU * i / seg
        c, s = math.cos(a), math.sin(a)
        for j in range(sides):
            b = TAU * j / sides
            rr = R + r * math.cos(b)
            verts.append(m @ V(rr * c * sx, rr * s, r * math.sin(b)))
    for i in range(seg):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c2, d2 = ((i + 1) % seg) * sides + (j + 1) % sides, ((i + 1) % seg) * sides + j
            faces.append((a, b, c2, d2))
    return mesh_object(name, verts, faces, role)


# Metaball elements with visible (not influence) sizes.


def kfac(stiff):
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def along(d, up="Z"):
    return Vector(d).to_track_quat("X", up).to_euler()


def ell(co, semi, rot=(0, 0, 0), stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": 1.0 / kfac(stiff), "type": "ELLIPSOID", "size": tuple(semi),
            "rot": tuple(rot), "stiffness": stiff, "negative": neg}


def caps(a, b, r, stiff=2.0, neg=False):
    a, b = Vector(a), Vector(b)
    d = b - a
    return {"co": tuple((a + b) / 2), "radius": r / kfac(stiff), "type": "CAPSULE", "size": (d.length / 2, 1, 1),
            "rot": tuple(along(d)), "stiffness": stiff, "negative": neg}


def ball(co, r, stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": r / kfac(stiff), "stiffness": stiff, "negative": neg}


def fine_metaball(elements, resolution, role, name, K=4.0):
    """lib.metaball, but finer than Blender allows: a metaball's resolution
    is clamped to 0.005, far too coarse for a head 0.1 across (the surface
    comes out terraced and faceted). Build the field K times larger, then
    shrink the mesh back."""
    big = []
    for e in elements:
        e = dict(e, co=tuple(c * K for c in e["co"]))
        kind = e.get("type", "BALL")
        if kind == "ELLIPSOID":  # extent is radius * size
            e["size"] = tuple(s * K for s in e["size"])
        elif kind == "CAPSULE":  # radius and half length are both absolute
            e["radius"] *= K
            e["size"] = (e["size"][0] * K,) + tuple(e["size"][1:])
        else:
            e["radius"] *= K
        big.append(e)
    obj = lib.metaball(big, role=role, resolution=resolution * K, name=name)
    return put(obj, Matrix.Scale(1 / K, 4))


# ---------------------------------------------------------------- pose

PELVIS = V(0.0, 0.016, 0.600)
WAIST = V(0.0, 0.014, 0.650)
CHEST = V(0.0, 0.016, 0.728)
GIRDLE = V(0.0, 0.020, 0.780)
NECK0 = V(0.0, 0.022, 0.810)
NECK1 = V(0.0, 0.010, 0.858)
# The head is thrown back in the scream: it turns about the top of the neck.
HEAD_TILT = -0.27
HEAD_SCALE = 1.12  # heroic: the screaming face is what names her on the board
HEAD_M = (Matrix.Translation(NECK1) @ Matrix.Rotation(HEAD_TILT, 4, "X") @ Matrix.Scale(HEAD_SCALE, 4)
          @ Matrix.Translation(-NECK1))
HEAD_C = V(0.0, -0.006, 0.914)

# Arms flung wide and back in anguish: elbows dropped below the line from
# shoulder to wrist, hands clawing at shoulder height; the left a touch
# higher and further back. Hair above, arms out and the wisp below make an X.
R_SHO, R_ELB, R_WRI = V(-0.098, 0.026, 0.796), V(-0.207, 0.026, 0.760), V(-0.314, 0.062, 0.856)
L_SHO, L_ELB, L_WRI = V(0.098, 0.026, 0.796), V(0.210, 0.036, 0.766), V(0.317, 0.078, 0.868)


def arm_at(sho, elb, wri, u):
    """The point a share u of the way along the arm, shoulder to wrist."""
    l1, l2 = (elb - sho).length, (wri - elb).length
    d = u * (l1 + l2)
    return sho.lerp(elb, d / l1) if d < l1 else elb.lerp(wri, (d - l1) / l2)


def H(x, y, z):
    """A point given in head space (from the skull centre), put in the world."""
    return HEAD_M @ (HEAD_C + V(x, y, z))


# ---------------------------------------------------------------- body


def body():
    """Neck, shoulders and bare skeletal arms; the ribs live under the bodice.
    The trapezius slopes from the neck down to each shoulder."""
    joints = [PELVIS, WAIST, CHEST, GIRDLE, NECK0, NECK1]  # 0-5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    radii = [(0.052, 0.042), (0.046, 0.036), (0.058, 0.044), (0.044, 0.034), 0.023, 0.019]
    for s, (sho, elb, wri) in ((-1, (R_SHO, R_ELB, R_WRI)), (1, (L_SHO, L_ELB, L_WRI))):
        b = len(joints)
        trap = V(s * 0.050, 0.026, 0.800)
        joints += [trap, sho, sho.lerp(elb, 0.28), sho.lerp(elb, 0.62), elb, elb.lerp(wri, 0.22), elb.lerp(wri, 0.62), wri]
        edges += [(3, b)] + [(b + i, b + i + 1) for i in range(7)]
        # Trapezius, deltoid, thin upper arm, knobbed elbow, forearm swelling
        # below it, bony wrist.
        radii += [(0.022, 0.026), 0.023, 0.018, 0.0128, 0.0162, 0.0150, 0.0090, 0.0110]
    sk = lib.skin([tuple(j) for j in joints], edges, radii, role="skin", name="body")
    # Bones pressing through: the point of the elbow.
    for sho, elb, wri in ((R_SHO, R_ELB, R_WRI), (L_SHO, L_ELB, L_WRI)):
        point = nrm(nrm(elb - sho) - nrm(wri - elb))

        def knob(v, c=elb + point * 0.012, s=0.010, h=0.009):
            d = (v - c).length
            if d < 2.5 * s:
                v += nrm(v - elb) * h * math.exp(-(d / s) ** 2)
            return v
        lib.deform(sk, knob)
    # Hidden under the bodice: not worth triangles.
    cut(sk, lambda c: c.z < 0.765 and abs(c.x) < 0.075)
    sk = lib.decimate(sk, 1800 / lib.tris(sk))
    return [sk]


# ---------------------------------------------------------------- head


# Head-space hollows, shared by the cutters and the dark paint inside them.
MOUTH_C, MOUTH_S = V(0, -0.034, -0.051), (0.014, 0.037, 0.027)  # the Munch oval
EYE_C, EYE_S = V(0.0155, -0.042, -0.001), (0.0135, 0.018, 0.0120)  # right socket mirrored
NOSE_C, NOSE_S = V(0.004, -0.033, -0.017), (0.0042, 0.009, 0.0065)  # one lobe of the nasal cavity


def skull():
    """The gaunt head: high cranium, hollow sockets, sharp cheekbones over
    sunken cheeks, and the jaw dropped wide in the scream."""
    els = [
        ell((0, 0.010, 0.016), (0.041, 0.045, 0.042)),  # cranium
        ell((0, -0.022, 0.012), (0.034, 0.022, 0.024)),  # forehead
        ell((0, -0.034, 0.004), (0.031, 0.010, 0.007), stiff=3),  # brow ridge
        ell((0, -0.034, -0.026), (0.018, 0.014, 0.012)),  # upper jaw
        ell((0, -0.041, -0.003), (0.004, 0.003, 0.006), stiff=3),  # thin nasal bone, flush between the sockets
        ball((0, -0.034, -0.090), 0.012),  # chin, dropped in the scream
        ball((0, -0.050, -0.054), 0.016, neg=True),  # mouth
        # The scream: a long Munch oval cut deep into the head, widest at
        # mid-height and narrowing to the chin. Stiff, so it cuts cleanly
        # through the summed field of the jaw and throat, not only dents it.
        ell(MOUTH_C, MOUTH_S, stiff=5, neg=True),
        # The back of the throat and the cheek walls close the scream into a
        # pocket open only at the front, so it fills with shadow.
        ell((0, -0.008, -0.050), (0.020, 0.023, 0.030)),
        ell((0, -0.018, -0.081), (0.017, 0.019, 0.009)),  # floor of the mouth, chin to throat
    ]
    for s in (-1, 1):
        els += [
            # The nasal cavity: two hollows leaning together at the top, an
            # upturned heart of shadow where the nose was.
            ell((s * NOSE_C.x, NOSE_C.y, NOSE_C.z), NOSE_S, rot=(0, -s * 0.35, 0), stiff=4, neg=True),
            ell((s * 0.026, -0.030, -0.010), (0.013, 0.011, 0.009), stiff=3),  # cheekbone
            caps((s * 0.028, -0.026, -0.010), (s * 0.038, 0.002, -0.008), 0.0065),  # cheek arch
            caps((s * 0.033, -0.002, -0.022), (s * 0.023, -0.011, -0.064), 0.0080),  # jaw ramus
            caps((s * 0.023, -0.011, -0.064), (s * 0.006, -0.034, -0.092), 0.0080),  # jaw line
            # Eye socket: a tunnel bored deep under the brow.
            ell((s * EYE_C.x, EYE_C.y, EYE_C.z), EYE_S, neg=True),
            # Cheek walls: skin stretched from the upper jaw down to the jaw
            # line, closing the scream at the sides so it is a pocket of shadow.
            ell((s * 0.019, -0.031, -0.054), (0.0065, 0.017, 0.029)),
            ell((s * 0.034, -0.034, -0.032), (0.008, 0.012, 0.012), neg=True),  # hollow under the cheekbone
            ell((s * 0.044, -0.014, 0.010), (0.008, 0.012, 0.012), neg=True),  # temple
        ]
    els = [dict(e, co=tuple(HEAD_C + V(e["co"]))) for e in els]
    h = fine_metaball(els, 0.0026, "skin", "skull")
    h = lib.smooth(h, iterations=3, factor=0.5)
    h = lib.decimate(h, SKULL_TRIS / lib.tris(h))

    # Paint the hollows dark (cloth is darker than skin in both palettes),
    # throat and nose holes, leaving a rim of skin round each, as a
    # miniature painter would.
    def q(p, c, semi):
        return sum(((a - b) / r) ** 2 for a, b, r in zip(p, c, semi))

    def hollow(c):
        p = c - HEAD_C
        m = abs(p.x), p.y, p.z
        return ((q(p, MOUTH_C, MOUTH_S) < 1.6 and p.y > -0.037)
                or (q(m, NOSE_C, (0.0060, NOSE_S[1], NOSE_S[2])) < 1.4 and p.y > -0.031))
    lib.paint(h, "cloth", hollow)
    # The finished bone, still in head space (offset by HEAD_C), to seat the
    # glow and the teeth on its real surface rather than on guesses.
    bone = BVHTree.FromPolygons([v.co.copy() for v in h.data.vertices], [p.vertices[:] for p in h.data.polygons])

    def hit(origin, direction, far=0.05):
        loc, _, _, _ = bone.ray_cast(HEAD_C + origin, nrm(direction), far)
        return None if loc is None else loc - HEAD_C
    put(h, HEAD_M)
    parts = [h]
    # Pinpoints of glow deep in the sockets (their floor is at y -0.035).
    for s in (-1, 1):
        e = lib.sphere(0.0055, loc=HEAD_C + V(s * 0.0155, -0.0335, -0.001), scale=(1.05, 0.6, 0.9), role="eye", segments=10, rings=6, name="eye")
        parts.append(put(e, HEAD_M))
    # The glow of the throat, half sunk into its back wall.
    wall = hit(V(0, -0.090, -0.050), V(0, 1, 0), 0.12)
    if wall is None:
        wall = V(0, -0.012, -0.050)
    if DEBUG:
        print(f"BANSHEE throat wall at {tuple(round(c, 4) for c in wall)}")
    m = lib.sphere(0.0085, loc=HEAD_C + wall + V(0, 0.0004, 0), scale=(0.95, 0.4, 1.1), role="eye", segments=12, rings=8, name="maw")
    parts.append(put(m, HEAD_M))
    # Bone teeth along both rims of the scream: each is cast back along its
    # own axis from inside the open mouth onto the rim, then driven into it.
    for k in range(6):
        u = (k - 2.5) / 2.5
        for top in (True, False):
            if top:
                base = V(u * 0.011, -0.046 + 0.004 * u * u, -0.031)
                d = nrm(V(0, -0.001, -0.007))
            else:
                base = V(u * 0.010, -0.042 + 0.004 * u * u, -0.079)
                d = nrm(V(0, -0.002, 0.006))
            # Where the lip is too thin at this depth, look a little further in.
            for back in (0.0, 0.003, 0.006, 0.009, 0.012):
                rim = hit(base + V(0, back, 0) + d * 0.022, -d, 0.04)
                if rim is not None:
                    break
            if rim is None:
                print(f"BANSHEE tooth {k} {'top' if top else 'bottom'} found no rim")
                continue
            base, tip = rim - d * 0.003, rim + d * (0.0075 if top else 0.0065)
            t = lib.cylinder(0.0028, 0.0006, (tip - base).length, role="skin", segments=5, name="tooth")
            put(t, frame((base + tip) / 2, tip - base))
            put(t, Matrix.Translation(HEAD_C))
            parts.append(put(t, HEAD_M))
    return parts


def lock(ctrl, width, thick, role="cloth", n=13, around=4, twist=0.0, name="lock", tip=None, tip_rows=2):
    """A clump of hair along the curve through ctrl: lens-shaped in section,
    full near the root, tapering to a fine point, its flat turning `twist`
    radians along the way. `tip` paints the last `tip_rows` rows."""
    pts = curve(ctrl, n)
    tang, norms, binorms = frames(pts, ref=pts[0] - H(0, 0.010, 0.016))
    verts, faces = [], []
    for i, p in enumerate(pts):
        t = i / (n - 1)
        a0 = twist * t
        out = norms[i] * math.cos(a0) + binorms[i] * math.sin(a0)
        side = tang[i].cross(out)
        w = width * (0.75 + 0.50 * math.sin(min(t * 3.0, 1) * math.pi / 2)) * (1 - t) ** 0.75 + 0.0007
        th = thick * (1 - 0.6 * t) * (0.35 + 0.65 * smoothstep(0.0, 0.2, t)) + 0.0004
        for j in range(around):
            a = TAU * j / around
            verts.append(p + side * math.cos(a) * w + out * math.sin(a) * th * (1 + 0.25 * abs(math.cos(a))))
    fr = []
    for i in range(n - 1):
        for j in range(around):
            a, b = i * around + j, i * around + (j + 1) % around
            faces.append((a, b, b + around, a + around))
            fr.append(tip if tip and i >= n - 1 - tip_rows else role)
    faces.append(tuple(range(around - 1, -1, -1)))
    faces.append(tuple((n - 1) * around + j for j in range(around)))
    fr += [role, tip or role]
    return mesh_object(name, verts, faces, role, fr if tip else None)


# The highest the hair may reach before build() grows the figure: 1.205 on
# the finished model, under the flyer cap of 1.22.
HAIR_TOP = 0.18 + (1.205 - 0.18) / SCALE
SCALP_C = V(0, 0.010, 0.016)
SCALP_S = V(0.0445, 0.049, 0.0465)


def on_scalp(d, lift=1.0):
    """The point of the scalp (head space) in direction d from its centre."""
    d = nrm(d)
    return SCALP_C + V(SCALP_S.x * d.x, SCALP_S.y * d.y, SCALP_S.z * d.z) * lift


def hair():
    """A scalp cap and wild clumps of hair blasting up and out from the
    scream. Every clump first lies back along the skull, then rises and
    splays outward, curling like a tongue of flame; the back clumps stream
    out behind her."""
    parts = []
    capm = lib.sphere(1.0, role="cloth", segments=24, rings=14, name="scalp")

    def bare(c):
        # Leave the face and the underside open.
        return (c.y < -0.12 and c.z < 0.72) or c.z < -0.55 or (c.z < -0.1 and c.y < 0.2)
    cut(capm, bare)
    lib.place(capm, loc=SCALP_C, scale=tuple(SCALP_S * 0.985))
    capm = lib.solidify(capm, 0.003)
    parts.append(put(lib.place(capm, HEAD_C), HEAD_M))
    # The burst: (root round the head seen from the front, 0 = crown; how far
    # forward (-) or back (+) the root sits; length; half width; the angle it
    # leaves at in the front view; how far it curls; how far back it streams;
    # whether its tip carries the accent)
    spec = []
    for k, phi in enumerate((-1.15, -0.70, -0.25, 0.22, 0.68, 1.12)):  # hairline
        spec.append((phi, -1.2, 0.21, 0.013, phi * 0.55, 0.30 * math.copysign(1, phi), 0.80, False))
    for k, phi in enumerate((-1.55, -1.00, -0.45, 0.05, 0.52, 1.02, 1.58)):  # crown
        spec.append((phi, 0.05, 0.25 - 0.03 * abs(phi), 0.013, phi * 0.85, (0.30 if abs(phi) > 0.2 else 0.18) * math.copysign(1, phi + 0.01), 0.85,
                     abs(phi) < 1.3))
    for phi in (-0.55, 0.0, 0.62):  # a few long thin strands, wilder than the rest
        spec.append((phi, -0.30, 0.27, 0.008, phi * 1.4, 0.45 * math.copysign(1, phi + 0.01), 0.70, False))
    for k, (phi, lean, L, w, psi, curl, beta, lit) in enumerate(spec):
        r1, r2, r3 = hash01(k, 1), hash01(k, 2), hash01(k, 3)
        sp, cp = math.sin(phi), math.cos(phi)
        root = on_scalp(V(sp, lean, cp), 0.95)
        over = on_scalp(V(sp * 1.06, lean + 0.55, cp), 1.12)
        L *= 0.90 + 0.22 * r2
        psi += 0.15 * (r1 - 0.5)
        curl *= 0.8 + 0.4 * r3
        wave = (0.26 + 0.08 * r2) * (1 if k % 2 else -1)

        def path(L, steps=6):
            ctrl, p = [root, over], over.copy()
            for i in range(steps):
                s = (i + 0.5) / steps
                a = psi + curl * s * s + wave * math.sin(s * TAU * 1.0 + r3)
                bb = beta + 0.35 * math.sin(s * math.pi + r1 * 3) * (r2 - 0.5)
                d = V(math.sin(a) * math.cos(bb), math.sin(bb), math.cos(a) * math.cos(bb))
                p = p + d * L / steps
                ctrl.append(p.copy())
            return [HEAD_M @ (HEAD_C + c) for c in ctrl]
        ctrl = path(L)
        # A lock that would rise past the flyer's height cap is cut shorter.
        while max(c.z for c in ctrl) > HAIR_TOP and L > 0.08:
            L *= 0.96
            ctrl = path(L)
        # Full clumps get a rounder six-sided lens; the wild strands stay thin.
        parts.append(lock(ctrl, w * (0.9 + 0.2 * r3), 0.0085, n=9, around=6 if w > 0.01 else 4, twist=1.0 * (r1 - 0.5),
                          tip="trim" if lit else None, tip_rows=3))
    # The long hair: from the back of the head and behind the ears it streams
    # back, then falls in S-waves down past the waist.
    for k, (phi, lean, L, w, out) in enumerate((
            (-0.85, 0.85, 0.41, 0.013, -0.30), (-0.25, 0.85, 0.44, 0.013, -0.10), (0.30, 0.85, 0.43, 0.013, 0.12),
            (0.90, 0.85, 0.40, 0.013, 0.32), (-1.95, 0.30, 0.38, 0.012, -0.55), (1.95, 0.30, 0.39, 0.012, 0.55))):
        r1, r2, r3 = hash01(k, 11), hash01(k, 12), hash01(k, 13)
        sp, cp = math.sin(phi), math.cos(phi)
        root = on_scalp(V(sp, lean, cp), 0.95)
        over = on_scalp(V(sp * 1.06, lean + 0.55, cp), 1.12)
        L *= 0.94 + 0.12 * r2
        ctrl = [root, over]
        p = over.copy()
        steps = 8
        wave = (0.30 + 0.12 * r1) * (1 if k % 2 else -1)
        for i in range(steps):
            s = (i + 0.5) / steps
            # Back and a little up off the skull, turning down as it goes.
            f = smoothstep(0.0, 0.75, s)
            d = nrm(V(out * (0.5 + 0.5 * f), 1.0 - 0.45 * f, 0.45 - 1.55 * f))
            side = nrm(V(1, 0, 0) - d.x * d)
            d = nrm(d + side * wave * math.sin(s * TAU * 1.1 + r3 * 2))
            p = p + d * L / steps
            ctrl.append(p.copy())
        ctrl = [HEAD_M @ (HEAD_C + c) for c in ctrl]
        # The two locks streaming out behind the ears carry the accent too.
        parts.append(lock(ctrl, w * (0.9 + 0.2 * r3), 0.0085, n=11, twist=0.8 * (r1 - 0.5), tip="trim" if k >= 4 else None))
    return parts


# ---------------------------------------------------------------- shroud


def bodice():
    """Rags clinging to the ribs, the ribs and breastbone pressing through,
    with a torn neckline below the collarbones."""
    nu, nv = 30, 7

    def prof(z):
        # Gaunt but female: a pinched waist under a lifted ribcage and bust.
        table = [(0.585, 0.054), (0.620, 0.046), (0.650, 0.038), (0.690, 0.048), (0.735, 0.064), (0.768, 0.065), (0.800, 0.058)]
        for (z0, r0), (z1, r1) in zip(table, table[1:]):
            if z0 <= z <= z1:
                return r0 + (r1 - r0) * smoothstep(z0, z1, z)
        return table[-1][1]

    def fn(i, j):
        a = TAU * i / nu
        t = j / nv
        front = max(0.0, math.cos(a))
        # A torn neckline: a sawtooth of rags, low over the breastbone, higher
        # over the left shoulder than the right.
        rag = 0.007 * ((i % 3) - 1) + 0.010 * (hash01(i, 7) - 0.5)
        ztop = 0.784 + 0.008 * (1 - front) - 0.018 * front ** 4 + 0.010 * math.sin(a) + rag
        z = 0.585 + (ztop - 0.585) * t
        r = prof(z)
        # Folds gathered up from the girdle, fading over the chest, and the
        # hollow under the breastbone.
        gather = 0.0035 * math.sin(a * 9 + 0.7 * math.sin(a * 4)) * (1 - smoothstep(0.66, 0.76, z)) * smoothstep(0.60, 0.65, z)
        sternum = 0.004 * math.exp(-(math.sin(a) / 0.10) ** 2) * front * smoothstep(0.68, 0.74, z)
        bust = 0.010 * front ** 1.5 * math.exp(-((z - 0.742) / 0.022) ** 2) * (0.4 + 0.6 * min(1.0, abs(math.sin(a)) / 0.45))
        r += gather - sternum + bust
        return (math.sin(a) * r * 1.16, -math.cos(a) * r * 0.80 + 0.016, z)

    obj = grid(fn, nu, nv, "body", "bodice", wrap=True)
    return [note("bodice", lib.solidify(obj, 0.004))]


def skirt(n_tat, k, nv, z0, length, r0, rmax, rtip, notch, spin, back, role, name, seed=0, hem=None,
          fold=0.012, fold2=0.003, sy=0.84, wide_at=0.45, converge=0.88, theta0=0.0, curl=0.3, hem_rows=2):
    """A robe that hangs whole from the waist and tears, lower down, into
    `n_tat` ragged strips that trail back and narrow into points. Each strip
    is `k` columns wide; its edges end higher than its middle, so the hem is
    a run of deep V notches. length(theta) gives each strip's length."""
    N = n_tat * k
    dA = TAU / n_tat
    tat = []
    for g in range(n_tat):
        th = theta0 + dA * (g + 0.5)
        tat.append({
            "theta": th,
            "L": length(th) * (0.92 + 0.16 * hash01(seed, g, 1)),
            "notch": notch * (0.75 + 0.5 * hash01(seed, g, 2)),
            "curl": curl * (hash01(seed, g, 3) - 0.5) * 2,
            "ph": hash01(seed, g, 4) * TAU,
        })
    # Rows: even down the strip, then a thin last row for the hem band.
    ts = [j / (nv - 1) * 0.95 for j in range(nv)] + [1.0]

    def fn(i, j):
        g, c = i // k, (i % k) / k - 0.5
        T = tat[g]
        e = 2 * abs(c)
        tend = 1 - T["notch"] * e ** 1.5 - 0.06 * hash01(seed, i, 7) * e
        t = ts[j] * tend
        r = r0 + (rmax - r0) * smoothstep(0.0, wide_at, t) - (rmax - rtip) * smoothstep(wide_at * 0.85, 1.0, t) ** 1.3
        conv = smoothstep(1 - T["notch"] - 0.1, 1.0, t)
        a = T["theta"] + c * dA * (1 - converge * conv) + spin * t * t + T["curl"] * smoothstep(0.55, 1.0, t) ** 2
        f = fold * math.cos(c * TAU) * smoothstep(0.0, 0.3, t) + fold2 * math.sin(c * 3 * TAU + T["ph"]) * t
        rr = r + f
        z = z0 - T["L"] * t
        return V(math.sin(a) * rr, -math.cos(a) * rr * sy + back * t ** 1.7 + 0.012, z)

    roles = (lambda i, j: hem if j >= nv - hem_rows else role) if hem else None
    return lib.solidify(grid(fn, N, nv, role, name, roles=roles, wrap=True), 0.0035)


def shroud():
    """Two layers of tattered robe: long cloth wisps beneath, a shorter ragged
    body-coloured robe over them, the whole a teardrop that trails back and
    down as she flies. The torn ends glow, so the accent runs down the wisps."""
    outer = skirt(7, 5, 12, 0.660, lambda th: 0.27 + 0.10 * (1 - math.cos(th)) / 2, 0.054, 0.106, 0.072, 0.42,
                  0.30, 0.15, "body", "shroud", seed=1, hem="trim", fold=0.012, wide_at=0.55, theta0=0.2)
    under = skirt(6, 4, 14, 0.648, lambda th: 0.42 + 0.05 * (1 - math.cos(th)) / 2, 0.047, 0.096, 0.018, 0.62,
                  0.80, 0.22, "cloth", "underrobe", seed=2, hem="trim", hem_rows=1, fold=0.010, wide_at=0.42, converge=0.96, curl=0.5)
    return [outer, under]


def girdle():
    """A knotted cord round the waist; its torn ends stream back as ribbons."""
    parts = []
    nu = 24

    def fn(i, j):
        a = TAU * i / nu
        z = 0.648 + (j - 0.5) * 0.012 + 0.004 * math.sin(a * 3)
        r = 0.057 + 0.003 * j
        return (math.sin(a) * r * 1.12, -math.cos(a) * r * 0.84 + 0.014, z)
    parts.append(lib.solidify(grid(fn, nu, 1, "cloth", "girdle", wrap=True), 0.004))
    knot = V(0.036, -0.032, 0.648)
    parts.append(lib.sphere(0.009, loc=knot, scale=(1.2, 0.8, 1.0), role="trim", segments=10, rings=6, name="knot"))
    for k, (off, ctrl) in enumerate((
        # Both fly clear of the robe: out to the side first, then back.
        (0.0, [(0.042, -0.034, 0.646), (0.078, -0.046, 0.618), (0.122, -0.012, 0.592), (0.160, 0.050, 0.585), (0.186, 0.122, 0.600)]),
        (0.5, [(0.036, -0.036, 0.642), (0.068, -0.056, 0.600), (0.118, -0.036, 0.552), (0.155, 0.030, 0.515), (0.168, 0.096, 0.508)]),
    )):
        parts.append(ribbon([V(c) for c in ctrl], 0.014 - 0.002 * k, 0.003, "cloth", seed=k + 3, tip="trim"))
    return parts


def ribbon(ctrl, width, thick, role, seed=0, n=16, twist=1.2, tip=None):
    """A torn strip along the curve through ctrl, turning as it flutters and
    ending in a ragged fork; `tip` paints the last quarter and the fork."""
    pts = curve(ctrl, n)
    tang, norms, binorms = frames(pts, ref=(0, 0, 1))
    verts, faces = [], []
    ph = hash01(seed, 1) * TAU
    for i, p in enumerate(pts):
        t = i / (n - 1)
        a = twist * math.sin(t * 3.0 + ph)
        wv = norms[i] * math.cos(a) + binorms[i] * math.sin(a)
        nv = tang[i].cross(wv)
        w = width / 2 * (1 - 0.35 * t)
        for u, v in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            verts.append(p + wv * u * w + nv * v * thick / 2)
    for i in range(n - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            faces.append((a, b, b + 4, a + 4))
    faces.append((3, 2, 1, 0))
    # Fork: the two edges run on past the centre.
    last = (n - 1) * 4
    d = (pts[-1] - pts[-2])
    tipa, tipb = len(verts), len(verts) + 1
    verts.append(verts[last] + d * 1.8)
    verts.append(verts[last + 2] + d * 2.4)
    faces.append((last, last + 1, tipa))
    faces.append((last + 3, last, tipa))
    faces.append((last + 1, last + 2, tipb))
    faces.append((last + 2, last + 3, tipb))
    faces.append((last + 1, tipb, last + 3))
    faces.append((last + 1, last + 3, tipa))
    if not tip:
        return mesh_object("ribbon", verts, faces, role)
    lit = (n - 1) * 3 // 4
    fr = [tip if i >= lit else role for i in range(n - 1) for _ in range(4)] + [role] + [tip] * 6
    return mesh_object("ribbon", verts, faces, role, fr)


def sleeve(side, sho, elb, wri):
    """A torn sleeve over the upper arm, and below it a ragged drape of cloth
    that hangs from the arm and streams back into three points."""
    parts = []
    axis = elb - sho
    L = axis.length
    m = frame(sho, axis, (0, 0.25, -1))  # local -y hangs down and a little back
    nu, nv = 16, 4
    jag = [0.30 * hash01(side, i) + (0.25 if i % 3 == 1 else 0.0) for i in range(nu)]

    def fn(i, j):
        a = TAU * i / nu
        t = j / nv
        down = max(0.0, math.cos(a))  # 1 on the hanging side
        r = 0.0215 + 0.003 * t + 0.0015 * math.sin(a * 3 + side)
        z = -0.010 + L * (0.10 + 0.40 * t * (1 - 0.5 * jag[i] * t)) + 0.016 * down * t * t
        sag = 0.008 * down ** 1.5 * t ** 1.6
        x, y = math.sin(a) * r, -math.cos(a) * r - sag
        return m @ V(x, y, z)

    parts.append(lib.solidify(grid(fn, nu, nv, "cloth", "sleeve", wrap=True), 0.003))
    # The rag-wing: a torn drape hung along the whole arm, shoulder to near
    # the wrist, its rows falling down and streaming back into ragged points.
    nu, nv = 12, 7
    hang = nrm(V(0, 0.90, -1))
    out = nrm(V(side, 0, 0))
    pts_jag = [0.0 if i % 2 == 0 else 0.30 + 0.35 * hash01(side, i, 5) for i in range(nu + 1)]
    pts_jag[0] = pts_jag[-1] = 0.35

    rows = (0.0, 0.20, 0.39, 0.57, 0.73, 0.86, 0.94, 1.0)  # finer toward the torn points

    def dfn(i, j):
        s = i / nu
        t = rows[j]
        a = arm_at(sho, elb, wri, 0.06 + 0.80 * s) + V(0, 0.008, -0.012)
        length = (0.10 + 0.12 * math.sin(math.pi * (0.15 + 0.85 * s))) * (1 - 0.6 * pts_jag[i])
        p = a + hang * length * t + V(0, 0.050, 0) * t * t
        p += out * 0.014 * math.sin(math.pi * t) + V(0, 0.006, 0) * math.sin(s * 11 + side) * t
        return p

    # Only the torn points glow: the accent belongs to the scream.
    roles = lambda i, j: "trim" if j >= nv - 1 else "cloth"
    parts.append(lib.solidify(grid(dfn, nu, nv, "cloth", "drape", roles=roles), 0.003))
    return parts


# ---------------------------------------------------------------- hands


def hand(side, elb, wri):
    """A long bony hand, fingers splayed and hooked into claws."""
    parts = []
    fore = nrm(wri - elb)
    # The palm faces down and forward, so the talons hook in the silhouette
    # rather than straight at the viewer.
    palm_n = nrm(V(0, -0.7, -1) - V(0, -0.7, -1).dot(fore) * fore)
    across = nrm(fore.cross(palm_n)) * side  # toward the thumb is -across... keep fingers ordered
    root = wri + fore * 0.006
    # The hand's skin starts well up the forearm at the forearm's own width
    # and swallows its rounded end, so wrist and palm are one bony joint
    # widening into a flat hand, not beads on a string.
    joints = [wri - fore * 0.030, wri - fore * 0.012, root, root + fore * 0.022]
    edges, radii = [(0, 1), (1, 2), (2, 3)], [0.0080, 0.0100, 0.0124, (0.0150, 0.0080)]
    tips = []
    for f in range(5):
        thumb = f == 0
        if thumb:
            d = nrm(fore * 0.45 - across * 0.85 + palm_n * 0.25)
            base = root + fore * 0.006 - across * 0.010
            segs, rr, curl = (0.020, 0.016, 0.012), (0.0060, 0.0050, 0.0040), 0.45
        else:
            fan = (f - 2.5) * 0.28
            d = nrm(fore * math.cos(fan) + across * math.sin(fan))
            base = root + fore * 0.024 + across * (f - 2.5) * 0.007
            k = (1.0, 1.08, 1.02, 0.86)[f - 1]
            segs, rr, curl = tuple(s * k for s in (0.026, 0.019, 0.015)), (0.0058, 0.0048, 0.0040), CURL
        b = len(joints)
        joints.append(base)
        radii.append(rr[0] * 1.15)
        edges.append((2 if thumb else 3, b))
        p = base
        for s, (L, r) in enumerate(zip(segs, rr)):
            # Each joint hooks further toward the palm; bone pinches between
            # the swollen knuckles.
            d = nrm(d + palm_n * curl * (0.3, 0.8, 1.0)[s])
            joints.append(p + d * L * 0.5)
            radii.append(r * 0.85)
            edges.append((len(joints) - 2, len(joints) - 1))
            p = p + d * L
            joints.append(p)
            radii.append(r * (1.15 if s < 2 else 0.95))
            edges.append((len(joints) - 2, len(joints) - 1))
        tips.append((p, d))
    hd = lib.skin([tuple(j) for j in joints], edges, radii, role="skin", subdiv=1, name="hand")
    hd = lib.decimate(hd, HAND_TRIS / lib.tris(hd))
    parts.append(hd)
    # Long hooked claws on every finger.
    for p, d in tips:
        c = nrm(d + palm_n * 0.9)
        pts = [p - d * 0.003, p + d * 0.010, p + d * 0.016 + c * 0.008]
        parts.append(lib.tube([tuple(q) for q in pts], [0.0042, 0.0030, 0.0003], role="trim", sides=6, resolution=2, name="claw"))
    return parts


def band(R, depth, width, m, a0, a1, seg, role="metal", name="cuff", chamfer=0.0008):
    """A squared iron band: an arc of radius R (to the middle of its section)
    from angle a0 to a1 round local z, `depth` thick radially and `width` wide
    along z, its corners chamfered and both broken ends capped; in frame m."""
    hd, hw, c = depth / 2, width / 2, chamfer
    sec = [(hd, -hw + c), (hd, hw - c), (hd - c, hw), (-hd + c, hw), (-hd, hw - c), (-hd, -hw + c), (-hd + c, -hw), (hd - c, -hw)]
    k = len(sec)
    verts, faces = [], []
    for i in range(seg + 1):
        a = a0 + (a1 - a0) * i / seg
        ca, sa = math.cos(a), math.sin(a)
        for dr, dz in sec:
            verts.append(m @ V((R + dr) * ca, (R + dr) * sa, dz))
    for i in range(seg):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            faces.append((a, b, b + k, a + k))
    faces.append(tuple(range(k - 1, -1, -1)))
    faces.append(tuple(seg * k + j for j in range(k)))
    return mesh_object(name, verts, faces, role)


def shackle(side, elb, wri):
    """A broken iron cuff hanging loose on the wrist bone, its top resting on
    the bone and a gap torn out of its underside, with a few links dangling
    from one of the broken ends."""
    parts = []
    fore = nrm(wri - elb)
    down = V(0, 0, -1)
    down = nrm(down - down.dot(fore) * fore)
    R = 0.0140
    # Hung 3 mm low, so its top sits on the wrist bone as a loose ring would.
    c = elb.lerp(wri, 0.80) + down * 0.003
    m = frame(c, fore, down)  # local -y points down
    # 15 of 18 segments: the gap is torn out of the underside, toward the front.
    gap = TAU * 3 / 18
    mid = -math.pi / 2 - side * 0.7  # toward the front, where the board camera sees it
    a0, a1 = mid + gap / 2, mid + TAU - gap / 2
    parts.append(band(R, 0.0042, 0.0075, m, a0, a1, 15, name="cuff"))
    # The chain hangs from the broken end nearer the hand's outer side.
    a = a0 if side > 0 else a1
    p = m @ V(R * math.cos(a), R * math.sin(a), 0)
    d = nrm(V(side * 0.15, 0.35, -1))
    for k in range(3):
        q = p + d * (0.004 + k * 0.0125)
        # Oval links, each turned a quarter from the last.
        lm = frame(q, V(1, 0, 0) if k % 2 else V(0, 1, 0), d)
        parts.append(torus(0.0050, 0.0018, lm @ Matrix.Rotation(math.pi / 2, 4, "Z"), role="metal", seg=8, sides=4,
                           name="link", sx=1.45))
    return parts


# ---------------------------------------------------------------- figure


def sash():
    """A torn strip of shroud slung from the left shoulder across the chest to
    the right hip, its loose end flying out behind."""
    pts = [V(0.070, 0.040, 0.808), V(0.060, -0.010, 0.806), V(0.030, -0.052, 0.770), V(-0.010, -0.058, 0.725),
           V(-0.048, -0.050, 0.680), V(-0.068, -0.020, 0.650), V(-0.072, 0.030, 0.640), V(-0.060, 0.080, 0.620),
           V(-0.070, 0.140, 0.585), V(-0.100, 0.200, 0.570)]
    c = curve(pts, 26)
    n = len(c)
    tang = frames(c)[0]
    axis = V(0, 0.016, 0)

    def fn(i, j):
        t = j / (n - 1)
        p = c[j]
        out = p - V(0, 0.016, p.z)
        out = nrm(out - out.dot(tang[j]) * tang[j])
        across = nrm(tang[j].cross(out))
        free = smoothstep(0.62, 0.75, t)
        if free > 0:
            across = nrm(across.lerp(V(0, 0, 1), 0.6 * free))
        w = 0.016 * (1 - 0.6 * smoothstep(0.7, 1.0, t)) * (1 + 0.15 * math.sin(t * 23))
        u = 2 * i / 2 - 1
        return p + across * u * w + out * 0.003 * math.cos(u * 2)

    obj = grid(fn, 2, n - 1, "cloth", "sash", roles=lambda i, j: "trim" if j >= n - 3 else "cloth")
    parts = [lib.solidify(obj, 0.003)]
    # A tarnished ring brooch pinning it at the shoulder, a pin through it.
    at, facing = V(0.062, -0.014, 0.806), nrm(V(0.25, -1, 0.55))
    parts.append(torus(0.0085, 0.0026, frame(at, facing), role="metal", seg=14, sides=6, name="brooch"))
    pin = lib.cylinder(0.0016, 0.0008, 0.012, role="metal", segments=6, name="pin")
    parts.append(put(pin, frame(at + facing * 0.002, nrm(V(1, 0.1, 0.45)))))
    return parts


def build():
    parts = []
    for name, fn in (("body", body), ("head", skull), ("hair", hair),
                     ("bodice", bodice), ("sash", sash), ("shroud", shroud), ("girdle", girdle)):
        objs = fn()
        print(f"BANSHEE {name}: {sum(lib.tris(o) for o in objs)} tris")
        parts += objs
    for side, (sho, elb, wri) in ((-1, (R_SHO, R_ELB, R_WRI)), (1, (L_SHO, L_ELB, L_WRI))):
        for name, objs in (("sleeve", sleeve(side, sho, elb, wri)), ("hand", hand(side, elb, wri)),
                           ("shackle", shackle(side, elb, wri))):
            print(f"BANSHEE {name}: {sum(lib.tris(o) for o in objs)} tris")
            parts += objs
    print(f"BANSHEE total: {sum(lib.tris(o) for o in parts)} tris in {len(parts)} parts")
    # Modelled at a slight human scale; grown about her lowest wisp so she
    # holds her own beside the other flyers on the board.
    grow = Matrix.Translation((0, 0, 0.18)) @ Matrix.Scale(SCALE, 4) @ Matrix.Translation((0, 0, -0.18))
    for o in parts:
        put(o, grow)
    return parts
