"""Valkyrie: a Light flyer, a winged warrior maiden hovering in flight. Her right
arm is cocked back above her shoulder with a long spear held level, ready to
hurl, and a round shield rides her left arm. Big feathered wings with gilded
leading edges rise up and back from her shoulder blades, a winged helm crowns
her, locks of hair frame her face and a long braid streams out behind. Her
armoured skirt and trailing greaved legs, ending in laminated sabatons, hang
below.

The figure faces -Y; her right hand is on -X. Z is up; the origin is the plinth
top, and she hovers above it with nothing solid below z = 0.3.
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

V = Vector


# ---------------------------------------------------------------- helpers


def log(obj):
    # FIGURE_PARTS=1 prints each part's triangle count, for budgeting.
    if os.environ.get("FIGURE_PARTS"):
        print(f"PART {obj.name} {lib.tris(obj)}")
    return obj


def basis(z, up=(0, 0, 1)):
    """Rotation whose local +Z is `z`, with local +Y leaning toward `up`."""
    z = V(z).normalized()
    x = V(up).cross(z)
    if x.length < 1e-6:
        x = V((1, 0, 0)).cross(z)
    x.normalize()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed()


def orient(obj, loc, z, up=(0, 0, 1)):
    """Bakes a frame into the mesh: local +Z onto `z`, the origin onto `loc`."""
    m = basis(z, up).to_4x4()
    m.translation = V(loc)
    obj.data.transform(m)
    obj.data.update()
    return obj


def _cr(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3)


def spline(points, per):
    """Catmull-Rom through `points`, `per` samples per span."""
    pts = [V(p) for p in points]
    ext = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    out = []
    for i in range(len(pts) - 1):
        for k in range(per):
            out.append(_cr(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], k / per))
    out.append(pts[-1].copy())
    return out


def resample(points, n):
    """`n` + 1 points evenly spaced by arc length along a spline through `points`."""
    dense = spline(points, 24)
    acc = [0.0]
    for a, b in zip(dense, dense[1:]):
        acc.append(acc[-1] + (b - a).length)
    total = acc[-1]
    out, j = [], 0
    for i in range(n + 1):
        d = total * i / n
        while j < len(acc) - 2 and acc[j + 1] < d:
            j += 1
        seg = max(acc[j + 1] - acc[j], 1e-9)
        out.append(dense[j].lerp(dense[j + 1], min(max((d - acc[j]) / seg, 0), 1)))
    return out


def cut(obj, where):
    """Deletes the faces whose local centre satisfies `where`."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = [f for f in bm.faces if where(f.calc_center_median())]
    bmesh.ops.delete(bm, geom=faces, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


class Plumage:
    """Collects feathers into one bmesh per role, so a wing is a few objects."""

    def __init__(self):
        self.bms = {}

    def bm(self, role):
        if role not in self.bms:
            self.bms[role] = bmesh.new()
        return self.bms[role]

    def objects(self, name):
        out = []
        for role, bm in self.bms.items():
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            me = bpy.data.meshes.new(f"{name}_{role}")
            bm.to_mesh(me)
            bm.free()
            obj = bpy.data.objects.new(f"{name}_{role}", me)
            bpy.context.scene.collection.objects.link(obj)
            out.append(lib.set_role(obj, role))
        self.bms = {}
        return out


def feather(pl, base, axis, normal, length, width, role="body", nl=6, nw=4, camber=0.14, thick=0.0042,
            bend=0.0, sweep=0.0, asym=0.0, narrow=None, tip=0.34, quill=0.3, point=0.25):
    """One feather: a lens-section shell with a raised rachis, cupped toward
    `normal` (its convex side), running `length` from `base` along `axis`.
    `width` is the half width; `asym` narrows the vane on the `narrow` side;
    `bend` curls the tip toward `normal`, `sweep` sideways."""
    axis = V(axis).normalized()
    across = V(normal).cross(axis).normalized()
    normal = axis.cross(across).normalized()
    sgn = 1.0 if narrow is None or across.dot(V(narrow)) >= 0 else -1.0
    bm = pl.bm(role)
    base = V(base)

    def spine(s):
        return base + axis * (length * s) + normal * (bend * length * s * s) + across * (sweep * length * s * s)

    def wprof(s):
        if s < 0.28:
            return quill + (1 - quill) * math.sin(s / 0.28 * math.pi / 2)
        if s < 1 - tip:
            return 1.0
        k = min((s - (1 - tip)) / tip, 1.0)
        return (1 - point) * math.sqrt(max(0.0, 1 - k * k)) + point * (1 - k)

    T, B = [], []
    for i in range(nl):
        s = 1 - (1 - i / nl) ** 1.4
        c = spine(s)
        w = width * wprof(s)
        th = thick * (1 - 0.55 * s)
        trow, brow = [], []
        for j in range(nw + 1):
            t = -1 + 2 * j / nw
            narrowing = (1 - asym) if (t * sgn) > 0 else (1 + asym * 0.4)
            x = t * w * narrowing
            mid = camber * w * (1 - t * t)
            ht = th * (0.35 + 0.65 * math.exp(-(t / 0.22) ** 2)) * math.sqrt(max(0.0, 1 - t * t))
            p = c + across * x + normal * mid
            tv = bm.verts.new(p + normal * ht)
            trow.append(tv)
            brow.append(tv if j in (0, nw) else bm.verts.new(p - normal * ht))
        T.append(trow)
        B.append(brow)
    tipv = bm.verts.new(spine(1.0))

    def face(*vs):
        uniq = []
        for v in vs:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) >= 3:
            try:
                bm.faces.new(uniq)
            except ValueError:
                pass

    for i in range(nl - 1):
        for j in range(nw):
            face(T[i][j], T[i][j + 1], T[i + 1][j + 1], T[i + 1][j])
            face(B[i][j + 1], B[i][j], B[i + 1][j], B[i + 1][j + 1])
    last = nl - 1
    for j in range(nw):
        face(T[last][j], T[last][j + 1], tipv)
        face(B[last][j + 1], B[last][j], tipv)
        face(T[0][j + 1], T[0][j], B[0][j], B[0][j + 1])


# ---------------------------------------------------------------- pose

TILT = 0.35  # torso pitched forward into the throw
UPT = V((0, -math.sin(TILT), math.cos(TILT)))
P0 = V((0, 0.04, 0.56))  # pelvis
P1 = P0 + UPT * 0.068  # waist
P2 = P1 + UPT * 0.068  # chest
P3 = P2 + UPT * 0.05  # neck base
NT = V((0, -0.036, 0.766))  # neck top
HC = V((0, -0.046, 0.797))  # head centre

# The throwing arm is cocked back and up, the fist high and behind her
# shoulder and out to her right, so the spear lies near level and points
# ahead, a little down and wide of her body.
R_SH, R_EL, R_WR, R_FIST = V((-0.088, -0.014, 0.722)), V((-0.182, 0.04, 0.832)), V((-0.217, 0.07, 0.93)), V((-0.215, 0.08, 0.955))
L_SH, L_EL, L_WR, L_FIST = V((0.088, -0.014, 0.722)), V((0.17, -0.045, 0.655)), V((0.213, -0.125, 0.645)), V((0.222, -0.142, 0.643))
# Her legs trail in a flying split: the right one long and trailing, the left
# knee brought forward under the skirt hem with its shin swept back. The feet
# break off the shin line, toes pointed but not in line with the greave.
R_HIP, R_KNEE, R_ANK = V((-0.052, 0.045, 0.548)), V((-0.048, 0.095, 0.43)), V((-0.034, 0.195, 0.375))
L_HIP, L_KNEE, L_ANK = V((0.052, 0.045, 0.548)), V((0.062, 0.03, 0.44)), V((0.06, 0.15, 0.39))
R_FOOT, L_FOOT = V((0, 0.63, -0.78)), V((0, 0.71, -0.71))

SPEAR = V((-0.40, -0.91, -0.07)).normalized()

FINISH = {"sharp_angle": 48}


CUIRASS = [(0.049, -0.016), (0.053, 0.006), (0.061, 0.032), (0.068, 0.056), (0.067, 0.076), (0.058, 0.096), (0.043, 0.11), (0.032, 0.116)]


def torso_local(c):
    """A world point in the torso frame (origin at the waist, z up the spine)."""
    q = c - P1
    s, k = math.sin(TILT), math.cos(TILT)
    return V((q.x, q.y * k + q.z * s, -q.y * s + q.z * k))


def cuirass_radius(z):
    for (r0, z0), (r1, z1) in zip(CUIRASS, CUIRASS[1:]):
        if z0 <= z <= z1:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return 0.0


def body():
    joints = [
        P0, P1, P2, P3, NT,  # 0-4 spine
        R_HIP, R_HIP.lerp(R_KNEE, 0.5), R_KNEE, R_KNEE.lerp(R_ANK, 0.4), R_ANK,  # 5-9
        L_HIP, L_HIP.lerp(L_KNEE, 0.5), L_KNEE, L_KNEE.lerp(L_ANK, 0.4), L_ANK,  # 10-14
        R_SH, R_SH.lerp(R_EL, 0.5), R_EL, R_EL.lerp(R_WR, 0.4), R_WR,  # 15-19
        L_SH, L_SH.lerp(L_EL, 0.5), L_EL, L_EL.lerp(L_WR, 0.4), L_WR,  # 20-24
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4),
             (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
             (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
             (2, 15), (15, 16), (16, 17), (17, 18), (18, 19),
             (2, 20), (20, 21), (21, 22), (22, 23), (23, 24)]
    radii = [(0.066, 0.05), (0.05, 0.04), (0.062, 0.046), 0.024, 0.021,
             0.046, 0.04, 0.029, 0.03, 0.019,
             0.046, 0.04, 0.029, 0.03, 0.019,
             0.031, 0.028, 0.021, 0.023, 0.016,
             0.031, 0.028, 0.021, 0.023, 0.016]
    sk = lib.skin([tuple(j) for j in joints], edges, radii, role="skin", name="body")

    def hidden(c):
        # Inside the cuirass: never seen, so not worth triangles.
        q = torso_local(c)
        r = cuirass_radius(q.z) - 0.002
        return -0.014 < q.z < 0.112 and r > 0 and (q.x / r) ** 2 + (q.y / (0.8 * r)) ** 2 < 1

    def in_capsule(c, a, b, r, t0, t1):
        ab = b - a
        t = (c - a).dot(ab) / ab.length_squared
        return t0 < t < t1 and (a + ab * t - c).length < r

    def covered(c):
        return (hidden(c)
                or in_capsule(c, R_KNEE, R_ANK, 0.031, 0.16, 1.3) or in_capsule(c, L_KNEE, L_ANK, 0.031, 0.16, 1.3)
                or in_capsule(c, R_EL, R_WR, 0.021, 0.3, 0.95) or in_capsule(c, L_EL, L_WR, 0.021, 0.3, 0.95))

    cut(sk, covered)
    sk = lib.decimate(sk, 0.45)
    # Leggings below the skirt.
    lib.paint(sk, "cloth", lambda c: c.z < 0.53 and abs(c.x) > 0.01)
    return sk


HEAD_SCALE = (1.1, 1.1, 1.1)
HEAD_ROT = (-0.2, 0, 0)  # chin up, so the face shows under the brow from the board camera


def head():
    parts = []
    h = lib.sphere(0.041, loc=(0, 0, 0), scale=(0.86, 0.95, 1.06), role="skin", segments=16, rings=12, name="head")

    def jaw(p):
        # Narrow the jaw, push the chin forward and sink the eye sockets.
        if p.z < 0:
            k = min(-p.z / 0.043, 1.0)
            p.x *= 1 - 0.28 * k
            if p.y < 0:
                p.y *= 1 + 0.2 * k * (1 - 0.6 * min(abs(p.x) / 0.02, 1))
        if p.y < 0:
            for s in (-1, 1):
                p.y += 0.003 * math.exp(-((p.x - s * 0.0155) / 0.008) ** 2 - ((p.z + 0.001) / 0.007) ** 2)
        return p

    lib.deform(h, jaw)
    parts.append(h)
    parts.append(lib.sphere(0.0085, loc=(0, -0.0425, -0.008), scale=(0.68, 1.05, 1.25), role="skin", segments=8, rings=6, name="nose"))
    # Brow ridge shading the eyes, and parted lips under the nose.
    parts.append(lib.sphere(0.01, loc=(0, -0.0352, 0.0088), scale=(2.4, 0.62, 0.5), role="skin", segments=10, rings=5, name="brow_ridge"))
    parts.append(lib.sphere(0.0058, loc=(0, -0.0392, -0.0215), scale=(1.65, 0.6, 0.5), role="skin", segments=8, rings=4, name="lip"))
    parts.append(lib.sphere(0.0052, loc=(0, -0.0372, -0.0278), scale=(1.5, 0.62, 0.55), role="skin", segments=8, rings=4, name="lip"))
    for s in (-1, 1):
        parts.append(lib.sphere(0.0066, loc=(s * 0.0155, -0.0335, -0.001), scale=(1.25, 0.55, 0.62), role="eye", segments=8, rings=4, name="eye"))
        # Locks of hair framing the face under the helm brow.
        parts.append(lib.tube([V((s * 0.033, -0.031, 0.008)), V((s * 0.037, -0.031, -0.012)), V((s * 0.036, -0.027, -0.032)),
                               V((s * 0.03, -0.028, -0.048))], [0.0072, 0.0068, 0.005, 0.0016], role="cloth", sides=8,
                              resolution=2, name="lock"))
    for p in parts:
        lib.place(p, HC, HEAD_ROT, HEAD_SCALE)
    return parts


def helm():
    parts = []
    prof = [(0.0, 0.064), (0.02, 0.062), (0.036, 0.055), (0.047, 0.043), (0.052, 0.029), (0.054, 0.014),
            (0.054, -0.004), (0.051, -0.024), (0.046, -0.042)]
    shell = lib.lathe(prof, role="metal", segments=20, name="helm")
    # Open the face below the brow.
    cut(shell, lambda c: c.z < 0.012 and c.y < 0 and abs(math.atan2(c.x, -c.y)) < 0.95)
    shell = lib.solidify(shell, 0.0045, offset=1.0)
    lib.place(shell, (0, 0, 0), (0, 0, 0), (0.93, 1.06, 1.0))
    parts.append(shell)
    brow = lib.lathe([(0.055, 0.007), (0.059, 0.011), (0.059, 0.019), (0.055, 0.024)], role="trim", segments=20, scale=(0.93, 1.06, 1.0), name="brow")
    parts.append(brow)
    # A Norse brow: the band dips to a point between the eyes and runs on
    # down as a nasal bar, wrapped round the curve of the helm.
    nasal = [(-0.024, 0.02), (0.024, 0.02), (0.02, 0.009), (0.008, 0.004), (0.0042, -0.003), (0.0034, -0.019), (0.0, -0.023),
             (-0.0034, -0.019), (-0.0042, -0.003), (-0.008, 0.004), (-0.02, 0.009)]
    peak = lib.extrude(nasal, 0.006, role="trim", bevel=0.0012, smooth=False, name="nasal")
    lib.deform(peak, lambda p: V((p.x, p.y - 0.0615 + 0.0062 * (p.x / 0.024) ** 2 + 0.003 * max(0.0, -p.z) / 0.023, p.z)))
    parts.append(peak)
    ridge = lib.tube([(0, -0.058, 0.03), (0, -0.034, 0.058), (0, 0.0, 0.07), (0, 0.035, 0.058), (0, 0.055, 0.03)],
                     [0.004, 0.006, 0.007, 0.006, 0.004], role="metal", sides=8, resolution=2, name="ridge")
    parts.append(ridge)
    # Feathered wings sweeping up and back from the temples.
    pl = Plumage()
    for s in (-1, 1):
        root = V((s * 0.05, 0.004, 0.022))
        dirs = [(0.34, 0.08, 0.94), (0.46, 0.34, 0.82), (0.5, 0.6, 0.62), (0.44, 0.84, 0.32)]
        lens = [0.112, 0.104, 0.088, 0.068]
        for k, (d, ln) in enumerate(zip(dirs, lens)):
            d = V((s * d[0], d[1], d[2]))
            feather(pl, root + V((s * 0.002 * k, 0.004 * k, -0.003 * k)), d, V((s * 0.8, -0.6, 0.0)), ln, 0.0125 - 0.001 * k,
                    role="trim", nl=6, nw=2, camber=0.25, thick=0.0038, bend=-0.12, asym=0.3, narrow=V((0, -1, 0.3)), point=0.45)
        parts.append(lib.sphere(0.012, loc=root, scale=(0.7, 1.2, 1.0), role="metal", segments=8, rings=5, name="wing_boss"))
    parts += pl.objects("helm_wings")
    for p in parts:
        lib.place(p, HC, HEAD_ROT, HEAD_SCALE)
    return parts


def braid_mesh(path, r0, taper=0.45, period=0.115, per=7, sides=6, pinches=(), name="braid"):
    """A real three-strand plait through `path`: each strand swings from side
    to side once per `period` and dives under the others twice, so the edges
    show alternating stitches and the face a herringbone. The plait is
    squeezed tight at each arc length in `pinches`, where a tie binds it.
    Returns the object, the centre line samples and their arc lengths."""
    dense = resample(path, 200)
    acc = [0.0]
    for p, q in zip(dense, dense[1:]):
        acc.append(acc[-1] + (q - p).length)
    total = acc[-1]
    m = max(8, round(total / period * per))
    pts = resample(path, m)
    arcs = [total * i / m for i in range(m + 1)]
    frames = []
    for i, c in enumerate(pts):
        T = (pts[min(i + 1, m)] - pts[max(i - 1, 0)]).normalized()
        side = T.cross(V((0, 0, 1))).normalized()
        out = side.cross(T).normalized()
        frames.append((T, side, out))
    bm = bmesh.new()
    for k in range(3):
        phase = math.tau * k / 3
        centres, radii = [], []
        for i, (c, (T, side, out)) in enumerate(zip(pts, frames)):
            s = arcs[i]
            r = r0 * (1 - taper * s / total)
            squeeze = 1 - 0.35 * sum(math.exp(-((s - p) / 0.011) ** 2) for p in pinches)
            th = math.tau * s / period - phase
            centres.append(c + side * (0.56 * r * squeeze * math.sin(th)) + out * (0.4 * r * squeeze * math.sin(2 * th)))
            radii.append(0.47 * r)
        rings = []
        for i, (c, rho) in enumerate(zip(centres, radii)):
            t = (centres[min(i + 1, m)] - centres[max(i - 1, 0)]).normalized()
            u = t.cross(frames[i][2]).normalized()
            w = u.cross(t).normalized()
            rings.append([bm.verts.new(c + u * (rho * math.cos(a)) + w * (rho * 0.86 * math.sin(a)))
                          for a in (math.tau * j / sides for j in range(sides))])
        for ra, rb in zip(rings, rings[1:]):
            for j in range(sides):
                q = (j + 1) % sides
                bm.faces.new((ra[j], ra[q], rb[q], rb[j]))
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, "cloth"), pts, arcs


def braid():
    """Three-strand braid from under the helm rim at the nape, streaming back
    from head height in a loose S over her right wing and settling a little
    in the slipstream, bound at the nape, the middle and the end, and
    finishing in a short fluted brush of loose hair."""
    parts = []
    path = [HC + V((0.0, 0.03, -0.042)), HC + V((0.0, 0.075, -0.046)), HC + V((-0.012, 0.125, -0.04)),
            HC + V((-0.04, 0.175, -0.04)), HC + V((-0.075, 0.222, -0.05)), HC + V((-0.095, 0.27, -0.068)),
            HC + V((-0.125, 0.315, -0.08)), HC + V((-0.17, 0.35, -0.09))]
    r0, taper = 0.023, 0.45
    line = resample(path, 200)
    length = sum((q - p).length for p, q in zip(line, line[1:]))
    ties = (0.05, length * 0.55, length - 0.004)
    obj, pts, arcs = braid_mesh(path, r0, taper=taper, pinches=ties)
    parts.append(obj)
    m = len(pts) - 1
    for at in ties:
        i = min(range(m + 1), key=lambda j: abs(arcs[j] - at))
        r = r0 * (1 - taper * arcs[i] / arcs[-1])
        T = (pts[min(i + 1, m)] - pts[max(i - 1, 0)]).normalized()
        tie = lib.cylinder(r * 0.92, r * 0.92, 0.0095, role="trim", segments=12, name="braid_tie")
        orient(tie, pts[i], T)
        parts.append(tie)
    # The loose end: a brush little wider than the tie, fluted into locks and
    # closing to a soft point that trails down in the wind.
    end, T = pts[-1], (pts[-1] - pts[-2]).normalized()
    side = T.cross(V((0, 0, 1))).normalized()
    up = side.cross(T).normalized()
    brush = lib.lathe([(0.0112, -0.006), (0.0138, 0.008), (0.0142, 0.018), (0.012, 0.028), (0.0082, 0.037), (0.0038, 0.044),
                       (0.0, 0.048)], role="cloth", segments=10, scale=(1.0, 0.8, 1.0), name="braid_brush")

    def flute(p):
        a = math.atan2(p.y, p.x)
        k = 1 + 0.14 * math.cos(5 * a) * min(max(p.z, 0.0) / 0.01, 1.0)
        p.x *= k
        p.y *= k
        p.y -= 0.22 * (max(p.z, 0.0) / 0.048) ** 2 * 0.048  # trail down
        return p

    lib.deform(brush, flute)
    orient(brush, end, T, up)
    parts.append(brush)
    return parts


def cuirass():
    parts = []
    c = lib.lathe(CUIRASS, role="metal", segments=32, scale=(1.0, 0.8, 1.0), name="cuirass")

    def bust(p):
        if p.y < 0:
            for s in (-1, 1):
                d = ((p.x - s * 0.031) / 0.024) ** 2 + ((p.z - 0.064) / 0.024) ** 2
                p.y -= 0.011 * math.exp(-d)
            # A keel down the centre of the waist.
            p.y -= 0.004 * math.exp(-(p.x / 0.01) ** 2) * (1 if p.z < 0.05 else 0.3)
        return p

    lib.deform(c, bust)
    parts.append(c)
    neck = lib.lathe([(0.03, 0.112), (0.036, 0.117), (0.03, 0.122)], role="trim", segments=24, scale=(1.0, 0.85, 1.0), name="gorget")
    parts.append(neck)
    belt = lib.lathe([(0.05, -0.03), (0.056, -0.026), (0.057, -0.012), (0.052, -0.006)], role="trim", segments=32, scale=(1.0, 0.82, 1.0), name="belt")
    parts.append(belt)
    parts.append(lib.box((0.026, 0.01, 0.022), loc=(0, -0.05, -0.018), role="metal", bevel=0.004, segments=1, name="buckle"))
    # A winged brooch at the top of the sternum.
    brooch = [(0.0, -0.007), (0.006, -0.002), (0.017, 0.002), (0.026, 0.009), (0.014, 0.008), (0.005, 0.006), (0.0, 0.009),
              (-0.005, 0.006), (-0.014, 0.008), (-0.026, 0.009), (-0.017, 0.002), (-0.006, -0.002)]
    parts.append(lib.extrude(brooch, 0.004, loc=(0, -0.0545, 0.09), rot=(-0.42, 0, 0), role="trim", bevel=0.0, name="brooch"))
    parts.append(lib.sphere(0.0042, loc=(0, -0.057, 0.0895), scale=(1, 0.7, 1), role="eye", segments=8, rings=5, name="brooch_gem"))
    for p in parts:
        lib.place(p, P1, (TILT, 0, 0))
    return parts


def skirt():
    """Tasset plates over a cloth underskirt, swept back by the wind."""
    parts = []
    origin = P1 + UPT * -0.024
    tilt = (0.43, 0, 0)

    def wind(k):
        def f(p):
            d = max(0.0, -p.z) / 0.16
            p.y += k * d * d
            return p
        return f

    under = lib.lathe([(0.052, 0.0), (0.066, -0.04), (0.08, -0.09), (0.09, -0.13), (0.094, -0.158)], role="cloth", segments=24,
                      scale=(1.0, 0.85, 1.0), name="underskirt")

    def folds(p):
        a = math.atan2(p.x, -p.y)
        d = max(0.0, -p.z) / 0.158
        k = 1 + 0.07 * math.sin(a * 8) * d
        p.x *= k
        p.y *= k
        return p

    lib.deform(under, folds)
    lib.deform(under, wind(0.03))
    under = lib.solidify(under, 0.004)
    parts.append(lib.place(under, origin, tilt))
    n = 9
    for i in range(n):
        a0 = math.tau * i / n
        plate = lib.lathe([(0.056, 0.002), (0.07, -0.035), (0.086, -0.08), (0.0915, -0.1), (0.094, -0.112)], role="metal", segments=3,
                          start=a0 - 0.29, sweep=0.58, scale=(1.0, 0.86, 1.0), name="tasset")
        lib.paint(plate, "trim", lambda c: c.z < -0.098)
        lib.deform(plate, wind(0.026))
        plate = lib.solidify(plate, 0.005, offset=1.0)
        parts.append(lib.place(plate, origin, tilt))
    return parts


# Lathe profile (radius, distance toward the toe) of a sabaton: a collar inside
# the greave, two overlapping lames each stepping in under the one above with
# a gilded lower edge, and a rounded toe cap.
SABATON = [(0.0, -0.014), (0.017, -0.012),
           (0.021, 0.003), (0.025, 0.015), (0.0256, 0.021), (0.0195, 0.0225),
           (0.0205, 0.031), (0.0214, 0.039), (0.0165, 0.0405),
           (0.017, 0.051), (0.0152, 0.064), (0.0118, 0.074), (0.0065, 0.081), (0.0, 0.084)]
SABATON_TRIM = ((0.0145, 0.0228), (0.0345, 0.0408))


def sabaton(ank, shin, foot, front):
    parts = []
    s = lib.lathe(SABATON, role="metal", segments=10, scale=(1.0, 0.76, 1.0), name="sabaton")
    lib.paint(s, "trim", lambda c: any(z0 < c.z < z1 for z0, z1 in SABATON_TRIM))
    # The instep (local +Y) faces the same way as the knee.
    orient(s, ank - foot * 0.004, foot, front)
    parts.append(s)
    # A small spur off the heel, opposite the instep.
    sole = (-(front - foot * front.dot(foot))).normalized()
    spur_dir = (sole * 0.75 - foot * 0.45 + shin * 0.2).normalized()
    spur = lib.cylinder(0.0062, 0.0008, 0.024, loc=(0, 0, 0.012), role="trim", segments=8, name="heel_spur")
    orient(spur, ank + sole * 0.016 - foot * 0.002, spur_dir)
    parts.append(spur)
    return parts


def spear_hand():
    """The throwing hand closed round the shaft: the palm and heel of the hand
    toward the wrist, fingers curled over the far side with their knuckles
    outward, and the thumb locked over them toward the spearhead."""
    parts = []
    a = SPEAR
    f = R_FIST - R_WR
    f = (f - a * f.dot(a)).normalized()  # from the wrist across the shaft
    q = a.cross(f).normalized()  # the back of the hand
    if q.x > 0:
        q = -q
    palm = lib.sphere(1.0, scale=(0.0165, 0.0175, 0.0205), role="skin", segments=12, rings=7, name="fist")
    # Local z along the shaft, local y toward the back of the hand.
    orient(palm, R_FIST - f * 0.006 + q * 0.004, a, q)
    parts.append(palm)
    for k, off in enumerate((-0.0085, 0.0075)):
        r = 0.0168
        arc = [R_FIST + a * off + (q * math.cos(t) + f * math.sin(t)) * r for t in (-0.15, 0.75, 1.65, 2.45)]
        parts.append(lib.tube(arc, [0.0068, 0.0064, 0.006, 0.0052], role="skin", sides=8, resolution=2, name="fingers"))
    thumb = lib.tube([R_FIST - q * 0.012 - f * 0.008 + a * 0.004, R_FIST - q * 0.013 + f * 0.004 + a * 0.016,
                      R_FIST - q * 0.004 + f * 0.012 + a * 0.022], [0.0068, 0.0058, 0.0048], role="skin", sides=8,
                     resolution=2, name="thumb")
    parts.append(thumb)
    return parts


def limbs():
    parts = []
    # Pauldrons: two layered lames over each shoulder, following the upper arm.
    for sh, el, s in ((R_SH, R_EL, -1), (L_SH, L_EL, 1)):
        out = ((sh - P2).normalized() + (el - sh).normalized() * 0.6).normalized()
        for k, (r, off) in enumerate(((0.044, 0.0), (0.038, 0.024))):
            lame = lib.lathe([(0.0, 0.017), (0.02, 0.015), (0.035, 0.006), (r, -0.006), (r + 0.002, -0.012)], role="metal", segments=10,
                             scale=(1.0, 1.12, 1.0), name="pauldron")
            lib.paint(lame, "trim", lambda c: c.z < -0.008)
            lame = lib.solidify(lame, 0.004, offset=1.0)
            orient(lame, sh + out * 0.012 + (el - sh).normalized() * off, out, V((0, -1, 0)))
            parts.append(lame)
            if k == 0:
                for ang in (-0.5, 0.5):
                    rv = lib.sphere(0.0036, loc=(math.sin(ang) * 0.036, -math.cos(ang) * 0.036 * 1.12, 0.004), scale=(1, 1, 0.7),
                                    role="metal", segments=5, rings=3, name="pauldron_rivet")
                    orient(rv, sh + out * 0.012, out, V((0, -1, 0)))
                    parts.append(rv)
        # Bracer on the forearm.
        wr = R_WR if s < 0 else L_WR
        a, b = el.lerp(wr, 0.25), wr.lerp(el, 0.05)
        parts.append(lib.tube([a, a.lerp(b, 0.5), b], [0.024, 0.023, 0.02], role="metal", sides=12, resolution=3, name="bracer"))
        for p, r in ((a, 0.025), (b, 0.0215)):
            ring = lib.cylinder(r, r, 0.006, role="trim", segments=14, name="bracer_rim")
            orient(ring, p, b - a)
            parts.append(ring)
    parts += spear_hand()
    lf = lib.sphere(0.02, scale=(0.95, 0.8, 1.15), role="skin", segments=12, rings=7, name="fist")
    orient(lf, L_FIST, L_FIST - L_WR)
    parts.append(lf)
    # Greaves swelling over the calf and narrowing to a real ankle, knee cops,
    # a buckled strap round the calf and laminated sabatons.
    for hip, knee, ank, foot in ((R_HIP, R_KNEE, R_ANK, R_FOOT), (L_HIP, L_KNEE, L_ANK, L_FOOT)):
        shin = (ank - knee).normalized()
        parts.append(lib.tube([knee.lerp(ank, 0.1), knee.lerp(ank, 0.38), knee.lerp(ank, 0.72), ank + shin * 0.006],
                              [0.03, 0.036, 0.026, 0.02], role="metal", sides=12, resolution=3, name="greave"))
        front = ((knee - hip).normalized() - shin).normalized()
        sp = knee.lerp(ank, 0.5)
        strap = lib.cylinder(0.0345, 0.0335, 0.007, role="cloth", segments=12, name="greave_strap")
        orient(strap, sp, shin)
        parts.append(strap)
        back = (-(front - shin * front.dot(shin))).normalized()
        buckle = lib.box((0.013, 0.006, 0.011), role="trim", bevel=0.0015, segments=1, name="greave_buckle")
        orient(buckle, sp + back * 0.0345, shin, back)
        parts.append(buckle)
        cop = lib.lathe([(0.0, 0.02), (0.008, 0.016), (0.02, 0.008), (0.029, -0.002), (0.031, -0.008)], role="metal", segments=12,
                        scale=(1.0, 0.85, 1.0), name="poleyn")
        lib.paint(cop, "trim", lambda c: c.z < -0.004)
        cop = lib.solidify(cop, 0.004, offset=1.0)
        orient(cop, knee + front * 0.016, front, shin)
        parts.append(cop)
        ridge_a, ridge_b = knee.lerp(ank, 0.2), knee.lerp(ank, 0.85)
        parts.append(lib.tube([ridge_a + front * 0.031, ridge_a.lerp(ridge_b, 0.5) + front * 0.031, ridge_b + front * 0.023],
                              [0.004, 0.0045, 0.003], role="metal", sides=6, resolution=2, name="greave_ridge"))
        parts += sabaton(ank, shin, foot.normalized(), front)
        rim = lib.cylinder(0.033, 0.033, 0.008, role="trim", segments=14, name="greave_rim")
        orient(rim, knee.lerp(ank, 0.12), shin)
        parts.append(rim)
    return parts


def spear():
    """A winged spear: trim shaft, a broad leaf head with a glowing rune,
    socket lugs like small wings, and a butt spike."""
    parts = []
    parts.append(lib.cylinder(0.0092, 0.0092, 0.58, loc=(0, 0, 0.02), role="trim", segments=12, name="shaft"))
    parts.append(lib.cylinder(0.0112, 0.0112, 0.05, loc=(0, 0, 0.012), role="cloth", segments=12, name="grip_wrap"))
    parts.append(lib.cylinder(0.011, 0.0078, 0.036, loc=(0, 0, 0.318), role="metal", segments=12, name="socket"))
    for z in (0.302, 0.322):
        parts.append(lib.cylinder(0.0122, 0.0122, 0.005, loc=(0, 0, z), role="trim", segments=12, name="binding"))
    leaf = [(0.0, 0.0), (0.029, 0.025), (0.04, 0.066), (0.031, 0.119), (0.0, 0.185), (-0.031, 0.119), (-0.04, 0.066), (-0.029, 0.025)]
    head = lib.extrude(leaf, 0.0115, loc=(0, 0, 0.33), role="metal", bevel=0.002, smooth=True, resolution=3, name="spearhead")

    def diamond(p):
        # Thin the blade toward its edges.
        k = 1 - min(abs(p.x) / 0.042, 1) * 0.65
        p.y *= k
        return p

    lib.deform(head, diamond)
    parts.append(head)
    rune = [(0.0, 0.0), (0.006, 0.045), (0.0, 0.13), (-0.006, 0.045)]
    parts.append(lib.extrude(rune, 0.0102, loc=(0, 0, 0.35), role="eye", bevel=0.001, smooth=False, name="rune"))
    for s in (-1, 1):
        lug = [(0.0, 0.0), (s * 0.032, -0.014), (s * 0.04, -0.005), (s * 0.012, 0.014)]
        lug = lug if s > 0 else lug[::-1]
        parts.append(lib.extrude(lug, 0.005, loc=(0, 0, 0.312), role="metal", bevel=0.0, name="lug"))
    parts.append(lib.cylinder(0.0092, 0.0025, 0.04, loc=(0, 0, -0.29), role="metal", segments=10, name="butt_spike"))
    parts.append(lib.cylinder(0.0112, 0.0112, 0.008, loc=(0, 0, -0.268), role="trim", segments=12, name="butt_ring"))
    # Grip a little ahead of the balance point, then turn the flat of the
    # blade toward the board camera.
    for p in parts:
        lib.place(p, (0, 0, -0.012))
        orient(p, R_FIST, SPEAR, V((0, -0.67, 0.64)))
    return parts


def shield():
    parts = []
    normal = V((0.45, -0.87, 0.14)).normalized()
    centre = L_FIST + normal * 0.026
    parts.append(lib.lathe([(0.0, 0.016), (0.03, 0.0145), (0.06, 0.01), (0.088, 0.003), (0.093, 0.0), (0.093, -0.006), (0.0, -0.004)],
                           role="body", segments=28, name="shield"))
    parts.append(lib.lathe([(0.088, 0.002), (0.097, 0.006), (0.103, 0.0), (0.099, -0.007), (0.09, -0.006)], role="trim", segments=28,
                           name="shield_rim"))
    parts.append(lib.lathe([(0.0, 0.034), (0.012, 0.031), (0.02, 0.024), (0.024, 0.016), (0.026, 0.012), (0.0, 0.01)], role="metal",
                           segments=16, name="boss"))
    parts.append(lib.sphere(0.008, loc=(0, 0, 0.034), scale=(1, 1, 0.6), role="eye", segments=8, rings=6, name="boss_gem"))
    # Emblem: a pair of spread wings in trim around the boss.
    half = [(0.026, 0.008), (0.042, 0.024), (0.06, 0.036), (0.078, 0.046), (0.072, 0.03), (0.076, 0.022), (0.066, 0.012),
            (0.068, 0.002), (0.056, -0.004), (0.056, -0.014), (0.042, -0.012), (0.03, -0.006)]
    for s in (-1, 1):
        outline = [(s * u * 0.95, v * 0.95) for u, v in half]
        if s < 0:
            outline = outline[::-1]
        em = lib.extrude(outline, 0.004, role="trim", bevel=0.0, smooth=False, name="emblem")
        lib.place(em, (0, 0, 0.0), (-math.pi / 2, 0, 0))
        lib.deform(em, lambda p: V((p.x, p.y, p.z + 0.0172 - 1.78 * (p.x * p.x + p.y * p.y))))
        parts.append(em)
    for k in range(8):
        a = math.tau * (k + 0.5) / 8
        parts.append(lib.sphere(0.0042, loc=(math.cos(a) * 0.0955, math.sin(a) * 0.0955, 0.0065), scale=(1, 1, 0.7), role="metal",
                                segments=4, rings=3, name="rivet"))
    for p in parts:
        orient(p, centre, normal, V((0, 0, 1)))
    return parts


def lerp_dir(keys, u):
    for (u0, d0), (u1, d1) in zip(keys, keys[1:]):
        if u <= u1:
            t = max(0.0, min(1.0, (u - u0) / (u1 - u0)))
            return d0.lerp(d1, t).normalized()
    return keys[-1][1]


def wing(side):
    """A raised feathered wing, displayed toward the front so it reads from the
    board camera. side +1 is her left (+X), -1 her right."""
    pl = Plumage()

    def X(x, y, z):
        return V((x * side, y, z))

    R, E, W, H = X(0.05, 0.085, 0.70), X(0.152, 0.125, 0.79), X(0.24, 0.17, 0.945), X(0.265, 0.21, 1.025)
    a = (H - R).normalized()
    b0 = X(0.62, -0.15, -0.77).normalized()
    b = (b0 - a * a.dot(b0)).normalized()
    nv = a.cross(b).normalized()
    if nv.y > 0:
        nv = -nv
    nd = -nv  # dorsal, toward her back
    tert = (b * 0.4 + X(0.06, 0.62, -0.78).normalized()).normalized()
    outer = (a * 0.9 + b * 0.1).normalized()
    keys = [(0.0, tert), (0.35, b), (0.78, (b + a * 0.16).normalized()), (1.0, outer)]

    def flow(u):
        return lerp_dir(keys, u)

    N = 40
    path = resample([R, E, W, H], N)

    def at(u):
        return path[max(0, min(N, round(u * N)))]

    # The whole arm is gilded, so one unbroken leading edge runs from the
    # shoulder to the wrist on both faces.
    arm = lib.tube([R, R.lerp(E, 0.5) + nd * 0.004, E, W, H], [0.026, 0.022, 0.019, 0.014, 0.007], role="trim", sides=8,
                   resolution=3, name="wing_arm")
    parts = [arm]

    # Primaries fanned from the hand, outermost first.
    lens = [0.182, 0.203, 0.212, 0.208, 0.196, 0.183, 0.17, 0.158, 0.15]
    for k, ln in enumerate(lens):
        u = 1.0 - 0.21 * k / (len(lens) - 1)
        d = flow(u)
        feather(pl, at(u) + d * 0.004 + nd * (-0.004 + 0.0008 * k), d, nd, ln, 0.025, nl=5, nw=3, camber=0.1, bend=0.05,
                asym=0.35, narrow=a - b, tip=0.42, point=0.18)
    # Secondaries along the forearm.
    K = 8
    for j in range(K):
        u = 0.76 - 0.39 * j / (K - 1)
        d = flow(u)
        feather(pl, at(u) + d * 0.004 + nd * (0.003 + 0.0009 * j), d, nd, 0.15, 0.028, nl=5, nw=3, camber=0.1, bend=0.04,
                asym=0.15, narrow=a, tip=0.48, point=0.04)
    # Tertials hanging down her back.
    for j, (u, ln) in enumerate(((0.29, 0.148), (0.19, 0.152), (0.09, 0.14))):
        d = flow(u)
        feather(pl, at(u) + nd * (0.011 + 0.0015 * j), d, nd, ln, 0.03, nl=5, nw=3, camber=0.12, bend=0.05, tip=0.5, point=0.0)
    # Coverts: shingled rows on each face over the bases of the flight feathers.
    # The marginal rows straddle the arm on both faces as narrow gilded scale
    # feathers, each leaning toward the wrist and overlapping the next by half
    # its length, so the leading edge reads as one scalloped band.
    rows = (
        (nv, 0.011, 0.33, 0.97, 8, 0.075, 0.022, 0.006, "body", 4, 0.0),  # under greater coverts
        (nv, 0.015, 0.08, 0.92, 7, 0.051, 0.019, -0.004, "body", 4, 0.0),  # under median coverts
        (nv, 0.019, 0.03, 0.9, 14, 0.05, 0.012, -0.026, "trim", 3, 0.6),  # marginal coverts along the leading edge
        (nd, 0.017, 0.33, 0.97, 8, 0.079, 0.022, 0.006, "body", 4, 0.0),  # greater coverts
        (nd, 0.021, 0.04, 0.9, 7, 0.053, 0.019, -0.004, "body", 4, 0.0),  # lesser coverts
        (nd, 0.024, 0.03, 0.92, 14, 0.05, 0.012, -0.032, "trim", 3, 0.6),  # dorsal marginal coverts over the arm
    )
    for n, off, u0, u1, count, ln, w, shift, role, nl, lean in rows:
        scale = lean > 0
        for i in range(count):
            u = u0 + (u1 - u0) * i / (count - 1)
            d = flow(u)
            boost = 1.12 if u > 0.8 else 1.0
            # Scale feathers lean toward the wrist, so each one lies over the
            # base of the next like a roof tile, and close with a round tip.
            axis = (d + a * lean).normalized()
            sw = 0.1 if V(n).cross(axis).dot(a) > 0 else -0.1
            feather(pl, at(u) + d * shift + n * (off + 0.0006 * i), axis, n, ln * boost, w, role=role, nl=nl, nw=2, camber=0.04,
                    bend=-0.07, sweep=sw if scale else 0.0, tip=0.55 if scale else 0.5, point=0.12 if scale else 0.45,
                    thick=0.002)
    # Alula: a little gilded thumb tuft at the wrist.
    for i in range(2):
        d = (a * (0.75 - 0.15 * i) + b * (0.25 + 0.15 * i)).normalized()
        feather(pl, at(0.8) + nv * (0.007 + 0.003 * i), d, nv, 0.055 - 0.01 * i, 0.011, role="trim", nl=4, nw=2, camber=0.15,
                tip=0.5, point=0.4)
    parts += pl.objects("wing")
    return parts


def build():
    parts = []

    def add(objs):
        for o in objs if isinstance(objs, list) else [objs]:
            parts.append(log(o))

    add(body())
    add(head())
    add(helm())
    add(braid())
    add(cuirass())
    add(skirt())
    add(limbs())
    add(spear())
    add(shield())
    add(wing(1))
    add(wing(-1))
    return parts
