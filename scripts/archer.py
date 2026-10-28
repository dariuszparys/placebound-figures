"""Archer: a lithe elven ranger at full draw. A deep rounded cowl with a soft
point folded back behind the crown frames a sharp three-quarter face with
glowing eyes, and pointed elf ears push out through slits in its sides. A big
recurved bow, broadside to the viewer and square to an arrow aimed a little
down the board, is drawn to the chin. A slim quiver rides over the cloak on
the back, arrows sprayed over the right shoulder, held by a baldric across the
chest. A leather jerkin with a laced V neck whose trimmed front edges run down
to the belt, two overlapping leaf flaps in front and leaf-cut side panels,
tapered bracers, a trimmed cloak streaming away from the aim and soft boots;
no plate anywhere.

Modelled in an "archer frame" (the arrow flies along +X, the chest faces -Y),
then the whole figure is turned by YAW so the aim points front-left. The
figure's right hand (drawing) is on -X, its left (bow) hand on +X."""
import math

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

import lib

# Whole-figure turn: the arrow points a little toward the front, so the bow is
# broadside to the Light board camera and the face turns three-quarter to it.
YAW = -0.28

# Head pose in the archer frame: turned toward the aim and dipped only a little,
# so the eyes still show under the brow from the board camera.
# The face sits a little forward of the hood so the nose and chin break the
# opening.
HEAD_C = Vector((0.006, -0.012, 0.554))
HEAD_YAW = 0.5
HEAD_TILT = 0.03
FACE_FWD = 0.008

# The arrow is aimed a little down the board; the bow stays square to it, so
# its top limb leans toward the aim.
AIM_PITCH = math.radians(12)
BOW_CANT = math.radians(14)
DRAW = 0.25


# ---------------------------------------------------------------- helpers


def head_matrix():
    return Matrix.LocRotScale(HEAD_C, Euler((HEAD_TILT, 0, HEAD_YAW)), Vector((1, 1, 1)))


def axes_matrix(origin, x, y, z):
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = x[i], y[i], z[i], origin[i]
    return m


def frame(origin, zdir, xhint=(1, 0, 0)):
    """A 4x4 frame at `origin` whose local z runs along `zdir`."""
    z = Vector(zdir).normalized()
    x = Vector(xhint)
    if abs(x.normalized().dot(z)) > 0.95:
        x = Vector((0, 1, 0))
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = x[i], y[i], z[i], origin[i]
    return m


def xform(obj, m):
    obj.data.transform(m)
    obj.data.update()
    return obj


def band(center, axis, r, h, t, role="trim", segments=20, scale=(1, 1), name="band", round_=False):
    """A short collar of radius `r`, height `h` and thickness `t` round `axis`
    through `center`: belts, cuffs, rings, ferrules. `round_` adds rounded
    shoulders for the bigger ones."""
    if round_:
        prof = [(r, -h / 2), (r + t * 0.8, -h / 2), (r + t, -h / 4), (r + t, h / 4), (r + t * 0.8, h / 2), (r, h / 2)]
    else:
        prof = [(r, -h / 2), (r + t, -h / 3), (r + t, h / 3), (r, h / 2)]
    b = lib.lathe(prof, role=role, segments=segments, scale=(scale[0], scale[1], 1), name=name)
    return xform(b, frame(center, axis))


def open_lathe(rings, segments=32, role="cloth", name="shell"):
    """A lathe whose rings are either closed or open at the front. `rings` is a
    list of (r, z, gap): gap None closes the ring, otherwise the ring spans
    angles gap..tau-gap measured from -Y toward +X, leaving the front open."""
    bm = bmesh.new()
    loops = []
    if rings[0][0] <= 1e-6:
        # A closed point: fan the first ring in from a single vertex.
        tip = bm.verts.new((0, 0, rings[0][1]))
        rings = rings[1:]
    else:
        tip = None
    for r, z, gap in rings:
        if gap is None:
            angles = [math.tau * i / segments for i in range(segments)]
        else:
            angles = [gap + (math.tau - 2 * gap) * i / segments for i in range(segments + 1)]
        loops.append(([bm.verts.new((math.sin(a) * r, -math.cos(a) * r, z)) for a in angles], gap is None))
    n = segments
    if tip is not None:
        first = loops[0][0]
        for i in range(n):
            bm.faces.new((tip, first[(i + 1) % n], first[i]))
    for (a, ca), (b, cb) in zip(loops, loops[1:]):
        for i in range(n):
            if ca and cb:
                bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
            elif ca and not cb:
                bm.faces.new((a[i], a[(i + 1) % n], b[i + 1], b[i]))
            elif cb and not ca:
                bm.faces.new((a[i], a[i + 1], b[(i + 1) % n], b[i]))
            else:
                bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, role)


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def feather(length, width, depth=0.0025, role="trim", name="vane", resolution=3):
    """A fletching vane along local +z, standing out along local +x."""
    outline = [(0.0, 0.0), (width * 0.7, length * 0.12), (width, length * 0.55), (width * 0.75, length * 0.92),
               (0.0, length)]
    return lib.extrude(outline, depth, role=role, bevel=0.0, smooth=True, resolution=resolution, name=name)


def arc(center, r, angles, z, dz=0.0):
    """Points on a horizontal arc round `center` (angles in radians from +X
    toward +Y), sinking by `dz` from first to last."""
    n = len(angles) - 1
    return [center + Vector((math.cos(a) * r, math.sin(a) * r, z - dz * i / n)) for i, a in enumerate(angles)]


# ---------------------------------------------------------------- key points

M_HEAD = head_matrix()
M_FACE = M_HEAD @ Matrix.Translation((0, -FACE_FWD, 0))
CHIN = M_FACE @ Vector((0.0, -0.033, -0.046))
NOCK = Vector((CHIN.x + 0.004, -0.062, CHIN.z - 0.012))   # arrow nock on the string, under the chin
AIM = Vector((math.cos(AIM_PITCH), 0.008, -math.sin(AIM_PITCH))).normalized()
REST = NOCK + AIM * DRAW                                    # arrow rest above the bow fist
BOW_U = Vector((math.sin(BOW_CANT), 0, math.cos(BOW_CANT)))  # up the bow
BOW_X = Vector((math.cos(BOW_CANT), 0, -math.sin(BOW_CANT)))  # away from the archer, square to the limbs
GRIP = REST - BOW_U * 0.013 + Vector((0, 0.004, 0))         # centre of the bow grip
M_BOW = axes_matrix(GRIP, BOW_X, Vector((0, 1, 0)), BOW_U)
DRAW_HAND = NOCK - AIM * 0.024 + Vector((0, -0.003, -0.004))
QUIVER_TOP = Vector((-0.05, 0.078, 0.512))
QUIVER_BOTTOM = QUIVER_TOP + Vector((0.098, 0.02, -0.174)).normalized() * 0.2
L_SHOULDER = Vector((0.074, 0.002, 0.464))
L_WRIST = GRIP + (L_SHOULDER - GRIP).normalized() * 0.026 + Vector((0, 0.003, 0))
L_ELBOW = L_SHOULDER.lerp(L_WRIST, 0.47) + Vector((0.0, -0.012, 0.012))
R_SHOULDER = Vector((-0.074, 0.004, 0.464))
R_ELBOW = Vector((-0.136, -0.028, 0.482))
R_WRIST = DRAW_HAND + Vector((-0.022, -0.002, 0.004))


def bow_pt(u, v, w=0.0):
    """A point in the bow's plane: u away from the archer, v up the bow."""
    return GRIP + BOW_X * u + BOW_U * v + Vector((0, w, 0))


# ---------------------------------------------------------------- hood


# The cowl as rings of (radius, height, front opening) round the head centre:
# a rounded crown close over the skull whose soft point folds back, a pointed
# opening framing the face, and a cape over the shoulders.
HOOD_RINGS = [
    (0.0, 0.068, None),
    (0.015, 0.0665, None),
    (0.0265, 0.0605, None),
    (0.0345, 0.051, None),
    (0.0405, 0.039, None),
    (0.0445, 0.025, None),
    (0.046, 0.012, 0.46),
    (0.048, 0.001, 0.70),
    (0.049, -0.016, 0.84),
    (0.047, -0.033, 0.76),
    (0.043, -0.048, 0.68),
    (0.042, -0.060, 0.62),
    (0.050, -0.074, 0.58),
    (0.062, -0.086, 0.58),
]
def hood_shape(v):
    """Pulls the brow forward, drapes the cowl between the shoulders and folds
    the soft point of the crown back into the shell."""
    x, y, z = v
    a = math.atan2(x, -y)
    front = max(0.0, math.cos(a))
    back = max(0.0, -math.cos(a))
    # Brow overhang.
    y -= 0.01 * front ** 2 * smoothstep(-0.012, 0.012, z) * (1 - smoothstep(0.03, 0.05, z))
    # The crown leans back a little and a soft point stands out behind it,
    # folded down toward the nape.
    c = smoothstep(0.012, 0.068, z)
    p = back * back * max(0.0, 1 - ((z - 0.05) / 0.02) ** 2)
    y += 0.008 * c * c + 0.013 * p
    z -= 0.004 * p
    # Cowl drape at the back.
    z -= 0.025 * back * smoothstep(-0.05, -0.09, z)
    # Soft folds in the cowl.
    k = smoothstep(-0.05, -0.086, z)
    f = 1 + 0.06 * math.sin(a * 6) * k
    return Vector((x * f, y * f, z))


def hood(add):
    shell = open_lathe(HOOD_RINGS, segments=20, role="cloth", name="hood")
    lib.deform(shell, hood_shape)
    shell = lib.solidify(shell, 0.005)
    add(xform(shell, M_HEAD))
    # Trim along the opening: down one edge, up the other.
    rim = []
    opened = [(r, z, g) for r, z, g in HOOD_RINGS if g is not None]
    for r, z, g in reversed(opened):
        rim.append((math.sin(g) * r, -math.cos(g) * r, z))
    r, z, _ = [h for h in HOOD_RINGS if h[2] is None][-1]
    rim.append((0.0, -r, z))
    for r, z, g in opened:
        rim.append((-math.sin(g) * r, -math.cos(g) * r, z))
    rim = [tuple(M_HEAD @ hood_shape(Vector(p))) for p in rim]
    add(lib.tube(rim, 0.0042, role="trim", sides=6, resolution=2, name="hood_rim"))


# ---------------------------------------------------------------- head


def face(v):
    """A narrow elven face: tapered jaw to a pointed chin, high cheekbones."""
    x, y, z = v
    t = smoothstep(0.0, -0.036, z)
    x *= 1 - 0.42 * t
    y = y * (1 - 0.2 * t) - 0.012 * t
    z *= 1 + 0.3 * t
    for sx in (-1, 1):
        # Cheekbones high and to the side, eye sockets just above them.
        d = (Vector((x, y, z)) - Vector((sx * 0.02, -0.026, -0.011))).length
        y -= 0.005 * max(0.0, 1 - d / 0.012)
        d = (Vector((x, y, z)) - Vector((sx * 0.016, -0.03, 0.001))).length
        y += 0.0025 * max(0.0, 1 - d / 0.009)
    return Vector((x, y * 0.95, z))


def head(add):
    skull = lib.sphere(0.035, role="skin", segments=20, rings=14, name="head")
    lib.deform(skull, face)
    add(xform(skull, M_FACE))
    nose = lib.tube([(0, -0.031, 0.005), (0, -0.0395, -0.01), (0, -0.0355, -0.016)], [0.0035, 0.0046, 0.003],
                    role="skin", sides=8, resolution=3, name="nose")
    add(xform(nose, M_FACE))
    brow = lib.tube([(-0.026, -0.024, 0.009), (-0.012, -0.0325, 0.0105), (0.0, -0.0335, 0.007),
                     (0.012, -0.0325, 0.0105), (0.026, -0.024, 0.009)], 0.0035, role="skin", sides=6, resolution=2,
                    name="brow")
    add(xform(brow, M_FACE))
    # Upswept almond eyes that glow out of the shadow of the hood, a clear
    # bridge of skin between them.
    for side in (-1, 1):
        eye = lib.extrude([(-0.009, -0.001), (0.0, 0.004), (0.011, 0.0055), (0.001, -0.0025)], 0.006,
                          loc=(side * 0.017, -0.0315, -0.001), rot=(0, side * 0.32, side * 0.55),
                          scale=(side * 1.0, 1, 1.15), role="eye", bevel=0.0, smooth=True, resolution=3, name="eye")
        add(xform(eye, M_FACE))
    # A small set mouth above the pointed chin.
    add(xform(lib.tube([(-0.0065, -0.0285, -0.0285), (0.0, -0.0312, -0.0302), (0.0065, -0.0285, -0.0285)],
                       [0.0009, 0.0014, 0.0009], role="skin", sides=6, resolution=2, name="mouth"), M_FACE))
    # Long pointed elf ears pushed out through slits in the sides of the hood
    # at eye level, sweeping out and back with a little lift.
    ear = [(-0.008, 0.0), (0.007, 0.003), (0.011, 0.0173), (0.006, 0.0302), (0.0, 0.0446), (-0.004, 0.0302),
           (-0.009, 0.0144)]
    for side in (-1, 1):
        d = Vector((side * 0.85, 0.22, 0.30)).normalized()
        n = Vector((side * 0.3, -0.35, 0.89)).normalized()
        x = n.cross(d).normalized()
        base = Vector((side * 0.041, 0.010, -0.002))
        e = lib.extrude([(u * side, v) for u, v in ear], 0.004, role="skin", bevel=0.0, smooth=True, resolution=3,
                        name="ear")
        add(xform(e, M_FACE @ frame(base, d, x)))
        # The slit: a cloth lip round the ear root where it leaves the hood,
        # and a sliver of lining showing inside it.
        slit = base + d * 0.0098
        lip = lib.lathe([(0.0068, -0.003), (0.0088, -0.0015), (0.0088, 0.0015), (0.0068, 0.003)], role="cloth",
                        segments=14, scale=(1.65, 0.62, 1), name="ear_lip")
        add(xform(lip, M_FACE @ frame(slit, d, x)))
        liner = lib.sphere(0.0068, role="body", segments=10, rings=4, scale=(1.6, 0.6, 0.22), name="ear_lining")
        add(xform(liner, M_FACE @ frame(slit - d * 0.0008, d, x)))


# ---------------------------------------------------------------- body


def body(add):
    J = {
        "pelvis": ((0.0, 0.006, 0.30), (0.058, 0.043)),
        "waist": ((0.0, 0.008, 0.36), (0.047, 0.034)),
        "chest": ((0.0, 0.006, 0.418), (0.064, 0.045)),
        "upper": ((0.0, 0.006, 0.456), (0.062, 0.04)),
        "neck": ((0.004, -0.004, 0.515), 0.019),
        "r_hip": ((-0.044, 0.006, 0.29), 0.042),
        "r_thigh": ((-0.06, 0.0, 0.238), 0.043),
        "r_vast": ((-0.069, -0.006, 0.198), 0.034),
        "r_knee": ((-0.075, -0.008, 0.168), 0.027),
        "r_shin": ((-0.087, 0.002, 0.1), 0.022),
        "l_hip": ((0.044, 0.004, 0.29), 0.042),
        "l_thigh": ((0.062, -0.004, 0.238), 0.043),
        "l_vast": ((0.071, -0.011, 0.198), 0.034),
        "l_knee": ((0.078, -0.016, 0.168), 0.027),
        "l_shin": ((0.09, -0.006, 0.1), 0.022),
    }
    chain = [
        ("pelvis", "waist", "chest", "upper", "neck"),
        ("pelvis", "r_hip", "r_thigh", "r_vast", "r_knee", "r_shin"),
        ("pelvis", "l_hip", "l_thigh", "l_vast", "l_knee", "l_shin"),
    ]
    names = list(J)
    edges = []
    for c in chain:
        for a, b in zip(c, c[1:]):
            edges.append((names.index(a), names.index(b)))
    sk = lib.skin([J[n][0] for n in names], edges, [J[n][1] for n in names], role="body", name="body")
    lib.paint(sk, "cloth", lambda c: c.z < 0.29)
    lib.paint(sk, "skin", lambda c: c.z > 0.49 and abs(c.x) < 0.03)
    # The jerkin opens in a laced V over a cloth undershirt, and its two
    # front edges run on down to the belt over a strip of the shirt.
    lib.paint(sk, "cloth", lambda c: c.y < -0.01 and 0.444 < c.z <= 0.49 and abs(c.x) < (c.z - 0.444) * 0.52)
    lib.paint(sk, "cloth", lambda c: c.y < -0.01 and 0.355 < c.z <= 0.444 and abs(c.x) < 0.003 + (0.444 - c.z) * 0.16)
    add(sk)
    # Laid on the chest: the surface runs from y -0.0315 at the V's foot to
    # -0.02 under the chin.
    vee = [Vector((0.026, -0.0135, 0.495)), Vector((0.013, -0.0285, 0.468)), Vector((0.0, -0.0335, 0.443)),
           Vector((-0.013, -0.0285, 0.468)), Vector((-0.026, -0.0135, 0.495))]
    add(lib.tube(vee, 0.0032, role="trim", sides=6, resolution=3, name="collar"))
    lace = []
    for i, z in enumerate((0.452, 0.46, 0.468, 0.476, 0.484)):
        w = (z - 0.444) * 0.5 * (1 if i % 2 else -1)
        lace.append(Vector((w, -0.0302 + (z - 0.452) * 0.17 + abs(w) * 0.1, z)))
    add(lib.tube(lace, 0.0014, role="metal", sides=4, smooth=False, name="lacing"))
    for side in (-1, 1):
        edge = [Vector((side * 0.003, -0.0325, 0.443)), Vector((side * 0.0085, -0.0328, 0.425)),
                Vector((side * 0.0115, -0.0307, 0.405)), Vector((side * 0.0138, -0.0272, 0.385)),
                Vector((side * 0.015, -0.025, 0.368)), Vector((side * 0.0162, -0.0243, 0.356))]
        add(lib.tube(edge, 0.0022, role="trim", sides=6, resolution=2, name="jerkin_edge"))
    # Sleeved arms: a swelling deltoid and bicep, a slim elbow, a forearm
    # that tapers to the wrist.
    for sh, el, wr, inner in ((L_SHOULDER, L_ELBOW, L_WRIST, Vector((0.04, 0.006, 0.458))),
                              (R_SHOULDER, R_ELBOW, R_WRIST, Vector((-0.04, 0.006, 0.458)))):
        add(lib.tube([inner, sh, sh.lerp(el, 0.45), el, el.lerp(wr, 0.3), wr],
                     [0.026, 0.026, 0.0225, 0.0165, 0.0168, 0.0128], role="cloth", sides=14, resolution=3,
                     name="arm"))
        add(lib.sphere(0.0175, loc=el, role="cloth", segments=12, rings=8, name="elbow"))
    J = {n: Vector(J[n][0]) for n in names}
    J["r_ankle"] = Vector((-0.094, 0.006, 0.05))
    J["l_ankle"] = Vector((0.097, 0.0, 0.05))
    return J


def hands(add):
    # Bow hand, modelled in the bow's frame (x toward the target, y toward the
    # archer's back, z up the bow): the back of the hand faces the viewer, a
    # knuckle row runs down the front of the grip and four tapering fingers
    # curl round its far side, their tips tucked short of the back. The thumb
    # lies over the top with the arrow resting on it.
    add(xform(lib.sphere(0.0165, loc=(-0.0145, -0.004, -0.001), scale=(0.82, 1.0, 1.28), role="skin", segments=14,
                         rings=10, name="palm"), M_BOW))
    knuckles, middle = [], []
    for i, (dz, r, tip) in enumerate(((0.0095, 0.0052, 78), (0.0005, 0.0054, 84), (-0.0085, 0.005, 76),
                                      (-0.0165, 0.0044, 66))):
        start = -128 + i * 4
        angles = [start, start + 52, -18 + i * 2, 34, tip]
        pts = arc(Vector((0, 0, 0)), 0.0192 - abs(i - 1.2) * 0.0004, [math.radians(a) for a in angles], dz + 0.0012,
                  0.0034)
        # Bulging middle knuckles, tapering to the fingertip.
        add(xform(lib.tube(pts, [r, r * 0.9, r * 0.96, r * 0.84, 0.0038 * r / 0.0052], role="skin", sides=8,
                           resolution=2, name="finger"), M_BOW))
        knuckles.append(pts[0] * 1.06)
        middle.append(pts[2] * 1.05)
    # The knuckle row along the front of the fist.
    add(xform(lib.tube([knuckles[0] + Vector((0, 0, 0.004))] + knuckles + [knuckles[-1] - Vector((0, 0, 0.003))],
                       [0.0042, 0.0056, 0.0058, 0.0055, 0.0048, 0.0038], role="skin", sides=8, resolution=2,
                       name="knuckle_row"), M_BOW))
    # The middle knuckles line up down the far side of the grip.
    add(xform(lib.tube(middle, [0.0047, 0.005, 0.0047, 0.004], role="skin", sides=6, resolution=2,
                       name="knuckle_row"), M_BOW))
    add(xform(lib.tube([(-0.019, -0.012, 0.006), (-0.008, -0.02, 0.0155), (0.006, -0.0175, 0.0185),
                        (0.0145, -0.008, 0.0165)], [0.0062, 0.0056, 0.0047, 0.0036], role="skin", sides=8,
                       resolution=3, name="thumb"), M_BOW))
    # Draw hand: the flat back of the hand to the viewer with a knuckle ridge,
    # one finger above the arrow and two below, hooked round the string.
    add(lib.extrude([(-0.012, -0.0075), (0.009, -0.0115), (0.0135, 0.0), (0.009, 0.0112), (-0.012, 0.0072)], 0.012,
                    loc=DRAW_HAND, rot=(0, AIM_PITCH, 0), role="skin", bevel=0.0035, smooth=False, resolution=3,
                    name="draw_hand"))
    add(lib.tube([DRAW_HAND + Vector((0.013, -0.004, 0.011)), DRAW_HAND + Vector((0.016, -0.006, 0.0)),
                  DRAW_HAND + Vector((0.014, -0.005, -0.012))], [0.0042, 0.0048, 0.0042], role="skin", sides=8,
                 resolution=2, name="knuckles"))
    for dz in (0.008, -0.005, -0.015):
        k = DRAW_HAND + Vector((0.015, -0.005, dz * 0.8))
        add(lib.tube([k, NOCK + Vector((0.005, -0.006, dz)), NOCK + Vector((0.008, 0.002, dz)),
                      NOCK + Vector((0.001, 0.007, dz))], [0.0052, 0.0048, 0.0045, 0.0038],
                     role="skin", sides=6, resolution=2, name="draw_finger"))
    add(lib.tube([DRAW_HAND + Vector((0.004, 0.006, -0.01)), DRAW_HAND + Vector((0.014, 0.008, -0.018)),
                  DRAW_HAND + Vector((0.016, 0.0, -0.022))], [0.0055, 0.005, 0.004], role="skin", sides=8,
                 resolution=2, name="draw_thumb"))
    # Tapered bracers over the forearm sleeves, clear of the elbow bend,
    # capped with trim at both ends and laced once.
    for a, b in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        s, e = a.lerp(b, 0.25), a.lerp(b, 0.94)
        add(lib.tube([s, s.lerp(e, 0.45), e], [0.019, 0.0182, 0.016], role="body", sides=12, resolution=3,
                     name="bracer"))
        add(band(s, e - s, 0.0186, 0.005, 0.003, role="trim", segments=12, name="bracer_cuff"))
        add(band(e, e - s, 0.0158, 0.006, 0.0032, role="trim", segments=12, name="bracer_cuff"))
        add(band(s.lerp(e, 0.55), e - s, 0.0178, 0.003, 0.0018, role="metal", segments=10, name="lace"))


def boots(add, J):
    for side, turn in ((-1, -0.25), (1, 0.55)):
        pre = "r_" if side < 0 else "l_"
        ankle, knee = J[pre + "ankle"], J[pre + "knee"]
        top = ankle.lerp(knee, 0.8)
        axis = top - ankle
        boot = lib.tube([ankle - Vector((0, 0, 0.024)), ankle + Vector((0, 0, 0.008)), ankle.lerp(top, 0.45),
                         ankle.lerp(top, 0.78), top], [0.021, 0.0175, 0.0235, 0.0272, 0.0255], role="body", sides=14,
                        resolution=3, name="boot")

        def slouch(v, ankle=ankle):
            # Soft wrinkles gathered over the instep.
            h = v.z - ankle.z
            k = 0.0025 * math.sin(h * 260) * (1 - smoothstep(0.0, 0.05, abs(h - 0.02)))
            d = Vector((v.x - ankle.x, v.y - ankle.y, 0))
            return v + d.normalized() * k if d.length > 1e-6 else v
        add(lib.deform(boot, slouch))
        # Turned-down cuff with a pointed tab at the front.
        cuff = lib.lathe([(0.024, -0.004), (0.03, 0.0), (0.035, 0.014), (0.034, 0.026)], role="body", segments=12,
                         name="cuff")
        lib.deform(cuff, lambda v: Vector((v.x, v.y, v.z - 0.014 * max(0.0, -v.y / 0.034) ** 6)))
        add(xform(lib.solidify(cuff, 0.003), frame(top - axis.normalized() * 0.018, axis, (1, 0, 0))))
        # A strap with a buckle across the ankle.
        add(band(ankle + Vector((0, 0, 0.012)), (0, 0, 1), 0.0195, 0.006, 0.002, role="cloth", segments=12,
                 name="strap"))
        add(lib.box((0.008, 0.004, 0.008), loc=(ankle.x + side * 0.01, ankle.y - 0.018, ankle.z + 0.012),
                    role="metal", name="boot_buckle"))
        foot = lib.tube([(0, 0.024, 0.02), (0, -0.008, 0.023), (0, -0.04, 0.016), (0, -0.064, 0.009), (0, -0.078, 0.017)],
                        [0.019, 0.021, 0.016, 0.008, 0.003], role="body", sides=14, resolution=3, name="foot")
        lib.deform(foot, lambda v: Vector((v.x, v.y, max(v.z, 0.0))))
        lib.place(foot, (ankle.x, ankle.y, 0), (0, 0, turn))
        add(foot)


def panel(center, half, rmul, hem):
    """A leaf-cut skirt panel hanging from under the belt, with soft folds and
    a pointed hem, and the trim tube along that hem."""
    p = lib.lathe([(0.052 * rmul, 0.345), (0.06 * rmul, 0.322), (0.07 * rmul, 0.295), (0.08 * rmul, 0.255),
                   (0.088 * rmul, 0.215)], role="body", segments=8, start=center - half, sweep=2 * half,
                  scale=(1.08, 0.86, 1), name="panel")

    def shape(v):
        a = math.atan2(v.x / 1.08, -v.y / 0.86)
        d = min(1.0, abs((a - center + math.pi) % math.tau - math.pi) / half)
        low = smoothstep(0.30, 0.215, v.z)
        target = hem[0] + (hem[1] - hem[0]) * d ** 1.5
        fold = 1 + 0.035 * math.cos(d * math.pi * 2.0) * smoothstep(0.33, 0.215, v.z)
        return Vector((v.x * fold, v.y * fold, v.z + (target - 0.215) * low))
    lib.deform(p, shape)
    hem_pts = []
    for i in range(9):
        a = center - half + 2 * half * i / 8
        hem_pts.append(shape(Vector((math.sin(a) * 0.088 * rmul * 1.08, -math.cos(a) * 0.088 * rmul * 0.86, 0.215))))
    edge = lib.tube(hem_pts, 0.0026, role="trim", sides=6, resolution=1, name="panel_hem")
    return lib.solidify(p, 0.005), edge


def jerkin(add):
    # Layered leaf panels: long front and back flaps, shorter ones over the hips.
    for center, half, rmul, hem in ((0.36, 0.44, 1.0, (0.207, 0.252)), (-0.34, 0.44, 1.035, (0.196, 0.25)),
                                    (math.pi, 0.7, 1.0, (0.205, 0.25)),
                                    (1.95, 0.5, 1.07, (0.235, 0.27)), (-1.95, 0.5, 1.07, (0.235, 0.27))):
        for part in panel(center, half, rmul, hem):
            add(part)
    add(band((0, 0.008, 0.352), (0, 0, 1), 0.049, 0.016, 0.005, role="trim", segments=28, scale=(1.0, 0.78), name="belt",
               round_=True))
    add(lib.box((0.02, 0.008, 0.018), loc=(0.004, -0.038, 0.352), role="metal", bevel=0.003, name="buckle"))
    # Baldric from the quiver's upper wrap over the right shoulder, across the
    # chest to the left hip and back round into the lower wrap.
    qa = QUIVER_TOP - QUIVER_BOTTOM
    hi = QUIVER_BOTTOM + qa * 0.75 + Vector((-0.004, -0.012, 0.004))
    lo = QUIVER_BOTTOM + qa * 0.25 + Vector((0.008, -0.014, 0.0))
    # It lies on the chest: each point sits a strap's thickness off the body.
    add(lib.tube([hi, Vector((-0.052, 0.046, 0.492)), Vector((-0.062, -0.004, 0.494)), Vector((-0.046, -0.022, 0.466)),
                  Vector((-0.024, -0.0335, 0.436)), Vector((0.0, -0.0355, 0.41)), Vector((0.019, -0.0302, 0.389)),
                  Vector((0.035, -0.0188, 0.37)), Vector((0.05, -0.0125, 0.352)), Vector((0.064, 0.004, 0.332)),
                  Vector((0.07, 0.03, 0.318)), lo], 0.0058, role="cloth", sides=8, resolution=3, name="baldric"))
    # Belt pouch and a hunting knife on the right hip.
    add(lib.sphere(0.02, loc=(-0.058, -0.026, 0.33), scale=(0.75, 0.6, 1.0), role="body", segments=12, rings=8,
                   name="pouch"))
    add(lib.box((0.026, 0.01, 0.012), loc=(-0.058, -0.034, 0.345), role="body", bevel=0.002, segments=1,
                name="pouch_flap"))
    add(lib.tube([(-0.044, -0.04, 0.34), (-0.05, -0.045, 0.285)], [0.0065, 0.004], role="metal", sides=8,
                 smooth=False, name="sheath"))
    add(lib.tube([(-0.043, -0.04, 0.345), (-0.041, -0.037, 0.37)], 0.0042, role="cloth", sides=8, smooth=False,
                 name="knife_grip"))


def lining(obj, axis_point, role):
    """Paints the faces of a solidified shell that face the vertical axis
    through `axis_point` (its inside) with `role`."""
    m = lib.mat(role)
    names = [x.name for x in obj.data.materials]
    if role not in names:
        obj.data.materials.append(m)
        names.append(role)
    idx = names.index(role)
    for p in obj.data.polygons:
        out = Vector((p.center.x - axis_point.x, p.center.y - axis_point.y, 0))
        if p.normal.dot(out) < 0:
            p.material_index = idx


CLOAK_PROFILE = [(0.062, 0.482), (0.074, 0.455), (0.087, 0.41), (0.1, 0.36), (0.113, 0.305), (0.126, 0.25),
                 (0.138, 0.195), (0.15, 0.14), (0.16, 0.09)]
CLOAK_START, CLOAK_SWEEP, CLOAK_SEGMENTS = math.pi * 0.6, math.pi * 0.8, 28


def cloak_flow(v):
    """Streams the cloak away from the aim in waving folds, with a leaf-dagged hem."""
    a = math.atan2(v.x, -v.y)
    t = smoothstep(0.47, 0.09, v.z)
    wave = math.sin(a * 7 + 0.6) + 0.45 * math.sin(a * 13 + 1.3)
    f = 1 + 0.085 * wave * t
    x, y = v.x * f, v.y * f
    x -= 0.11 * t ** 1.4
    y += 0.05 * t * t
    z = v.z + 0.012 * wave * t + 0.045 * t * t * max(0.0, -math.sin(a))
    z += 0.012 * abs(math.sin(a * 9)) * smoothstep(0.13, 0.09, v.z)
    return Vector((x, y, z))


def cloak(add):
    c = lib.lathe(CLOAK_PROFILE, role="cloth", segments=CLOAK_SEGMENTS, start=CLOAK_START, sweep=CLOAK_SWEEP,
                  scale=(1, 0.8, 1), name="cloak")
    c = lib.solidify(c, 0.006)
    # The lining is the inner shell, found before the cloak is blown out.
    lining(c, Vector((0.0, 0.0, 0.0)), "body")
    lib.deform(c, cloak_flow)
    add(c)

    def at(a, r, z):
        return cloak_flow(Vector((math.sin(a) * r, -math.cos(a) * r * 0.8, z)))
    end = CLOAK_START + CLOAK_SWEEP
    left = [at(CLOAK_START, r, z) for r, z in CLOAK_PROFILE]
    right = [at(end, r, z) for r, z in reversed(CLOAK_PROFILE)]
    r, z = CLOAK_PROFILE[-1]
    hem = [at(CLOAK_START + CLOAK_SWEEP * i / CLOAK_SEGMENTS, r, z) for i in range(CLOAK_SEGMENTS + 1)]
    add(lib.tube(left[1:], 0.0034, role="trim", sides=6, resolution=2, name="cloak_edge"))
    add(lib.tube(right[:-1], 0.0034, role="trim", sides=6, resolution=2, name="cloak_edge"))
    add(lib.tube(hem, 0.0034, role="trim", sides=6, smooth=False, name="cloak_hem"))
    # Leaf brooch where the cloak is pinned below the cowl.
    leaf = [(0.0, 0.016), (0.008, 0.004), (0.0, -0.012), (-0.008, 0.004)]
    add(lib.extrude(leaf, 0.005, loc=(-0.05, -0.04, 0.47), rot=(0.2, 0, 0.5), role="metal", bevel=0.0,
                    smooth=True, resolution=4, name="brooch"))


def quiver(add):
    axis = QUIVER_TOP - QUIVER_BOTTOM
    m = frame(QUIVER_BOTTOM, axis, (1, 0, 0))
    length = axis.length
    q = lib.lathe([(0.0, 0.0), (0.013, 0.0), (0.017, 0.007), (0.019, length * 0.5), (0.0205, length - 0.008),
                   (0.022, length)], role="body", segments=16, name="quiver")
    add(xform(lib.solidify(q, 0.003), m))
    add(xform(lib.lathe([(0.0, -0.005), (0.012, -0.005), (0.0175, 0.0), (0.018, 0.01)], role="metal", segments=16,
                        name="quiver_cap"), m))
    for f in (0.012, length - 0.01):
        add(xform(band(Vector((0, 0, f)), (0, 0, 1), 0.0182 + f * 0.017, 0.01, 0.0035, role="trim", segments=14,
                       name="quiver_band"), m))
    # Leather wraps where the baldric holds it.
    for f in (0.25, 0.75):
        add(xform(band(Vector((0, 0, length * f)), (0, 0, 1), 0.0185 + f * 0.0016, 0.008, 0.0022, role="cloth",
                       segments=14, name="quiver_wrap"), m))
    # A pointed leaf tooled into the leather, facing outward.
    leaf = lib.extrude([(0.0, 0.036), (0.004, 0.024), (0.008, 0.006), (0.0, -0.016), (-0.008, 0.006),
                        (-0.004, 0.024)], 0.003, role="trim", bevel=0.0, smooth=True, resolution=3, name="quiver_leaf")
    add(xform(leaf, m @ Matrix.Translation((0, 0.0195, length * 0.5)) @ Matrix.Rotation(-0.06, 4, "X")))
    # Arrows sprayed out of the mouth, each with two trim vanes showing.
    for i, (dx, dy, lift) in enumerate(((0.0, 0.0, 0.12), (0.018, 0.009, 0.1), (-0.018, 0.008, 0.108),
                                        (0.012, -0.016, 0.094), (-0.013, -0.016, 0.115), (0.0, 0.019, 0.09),
                                        (0.022, -0.004, 0.104))):
        base = Vector((dx * 0.5, dy * 0.5, length - 0.06))
        end = Vector((dx * 1.7, dy * 1.7, length + lift))
        d = (end - base).normalized()
        add(xform(lib.tube([base, end], 0.003, role="body", sides=6, smooth=False, name="q_shaft"), m))
        for k in range(2):
            v = feather(0.04, 0.0095, role="trim", name="q_vane", resolution=1)
            lib.place(v, (0, 0, 0), (0, 0, k * math.pi + i))
            add(xform(v, m @ frame(end - d * 0.045, d, (1, 0, 0))))


# The bow limbs as (u, v) in the bow's frame: u away from the archer, v up.
# Each limb bends back toward the archer at full draw and flicks forward at
# the tip into a recurve ear.
BOW_TOP = [(-0.002, 0.055), (-0.005, 0.072), (-0.017, 0.12), (-0.038, 0.176), (-0.062, 0.222), (-0.077, 0.254),
           (-0.073, 0.275), (-0.057, 0.287), (-0.036, 0.292)]
BOW_LOW = [(-0.002, -0.055), (-0.005, -0.072), (-0.017, -0.118), (-0.038, -0.172), (-0.061, -0.216),
           (-0.075, -0.247), (-0.071, -0.268), (-0.055, -0.28), (-0.034, -0.285)]
BOW_RADII = [0.0135, 0.015, 0.0145, 0.0132, 0.0115, 0.0097, 0.008, 0.0066, 0.0052]


def bow(add):
    for limb in (BOW_TOP, BOW_LOW):
        add(lib.tube([bow_pt(u, v) for u, v in limb], BOW_RADII, role="trim", sides=8, resolution=3, name="limb"))
    add(lib.tube([bow_pt(0, -0.064), GRIP, bow_pt(0, 0.064)], [0.0145, 0.0165, 0.0145], role="body", sides=12,
                 resolution=4, name="riser"))
    for s in (-1, 1):
        add(band(bow_pt(-0.002, s * 0.06), BOW_U, 0.0135, 0.009, 0.0032, role="trim", segments=12,
                 name="riser_ring"))
    # The string, drawn back to the nock from where the ears turn forward.
    add(lib.tube([bow_pt(-0.074, 0.272), NOCK, bow_pt(-0.072, -0.265)], 0.003, role="body", sides=6, smooth=False,
                 name="string"))
    # The arrow, its glowing head and trim fletching.
    tail = NOCK - AIM * 0.008
    tip = NOCK + AIM * 0.31
    add(lib.tube([tail, tip], 0.0046, role="body", sides=8, smooth=False, name="shaft"))
    head = lib.extrude([(-0.004, 0.004), (0.009, 0.0105), (0.044, 0.0), (0.009, -0.0105), (-0.004, -0.004)], 0.005,
                       role="eye", bevel=0.0015, smooth=False, name="arrowhead")
    add(xform(head, frame(tip, (0, 0, 1), AIM)))
    for k in range(3):
        v = feather(0.056, 0.0105, role="trim", name="vane")
        lib.place(v, (0, 0, 0), (0, 0, k * math.tau / 3))
        add(xform(v, frame(NOCK + AIM * 0.03, AIM, (0, 0, 1))))


def build():
    parts = []
    add = parts.append
    J = body(add)
    head(add)
    hood(add)
    hands(add)
    boots(add, J)
    jerkin(add)
    cloak(add)
    quiver(add)
    bow(add)
    for p in parts:
        lib.place(p, (0, 0, 0), (0, 0, YAW))
    return parts
