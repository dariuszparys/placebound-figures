"""Djinni: a Light flyer, the travelling whirlwind. A powerful bare-chested genie
with a jewelled onion turban and pointed beard; the right hand is raised,
conjuring a small tornado, the left fist rests on the sash. Below the waist the
body becomes a smoke tail of billowing spiral bands that runs out to his left,
crosses under him in a shallow bowl and sweeps far out to his right, ending in
a glowing curl. The figure's right hand is on -X (it faces -Y).

The board settles flyers toward their plinth (as the game does), so
figure z of about 0.31 lands on the plinth top. The tail wins its length
sideways rather than downward and its bowl stays near 0.42, so a clear gap
shows above the plinth in both palettes and the genie never reads as a bust
on a pedestal."""
import math
import os

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

import lib

DEBUG = bool(os.environ.get("DJ_DEBUG"))
# Decimated organic surfaces keep smooth normals; only real corners split.
FINISH = {"sharp_angle": 60}


def note(name, obj):
    if DEBUG:
        print(f"DJ {name}: {lib.tris(obj)} tris")
    return obj


# ---------------------------------------------------------------- helpers


def kfac(stiff):
    """Visible radius of a lone metaball element as a share of its radius."""
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def along(d, up="Z"):
    """Euler that turns local +X onto direction d."""
    return Vector(d).to_track_quat("X", up).to_euler()


def ell(co, semi, rot=(0, 0, 0), stiff=2.0, neg=False):
    """Metaball ellipsoid with the given visible half sizes."""
    return {"co": tuple(co), "radius": 1.0 / kfac(stiff), "type": "ELLIPSOID", "size": tuple(semi),
            "rot": tuple(rot), "stiffness": stiff, "negative": neg}


def cap(a, b, r, stiff=2.0, neg=False):
    """Metaball capsule from a to b with visible radius r."""
    a, b = Vector(a), Vector(b)
    d = b - a
    return {"co": tuple((a + b) / 2), "radius": r / kfac(stiff), "type": "CAPSULE", "size": (d.length / 2, 1, 1),
            "rot": tuple(along(d)), "stiffness": stiff, "negative": neg}


def ball(co, r, stiff=2.0, neg=False):
    return {"co": tuple(co), "radius": r / kfac(stiff), "stiffness": stiff, "negative": neg}


def lerp(a, b, t):
    return Vector(a).lerp(Vector(b), t)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
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


def coil(f, sharp=0.55):
    """A rounded band across one period with creases at its ends."""
    return math.sin(math.pi * min(max(f, 0.0), 1.0)) ** sharp


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
    nrm = r - r.dot(tang[0]) * tang[0]
    if nrm.length < 1e-6:
        nrm = Vector((1, 0, 0)) - tang[0].x * tang[0]
    norms = [nrm.normalized()]
    for i in range(1, n):
        m = norms[-1] - norms[-1].dot(tang[i]) * tang[i]
        norms.append(m.normalized())
    binorms = [t.cross(m) for t, m in zip(tang, norms)]
    return tang, norms, binorms


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


def sweep(pts, radius, around, role="body", name="sweep", ref=(0, -1, 0), cap_start=True, tip=True, roles=None, twist=None):
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
            th = math.tau * c + tw
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
        froles += [role] * around
    if tip:
        step = (pts[-1] - pts[-2]).length
        verts.append(pts[-1] + tang[-1] * step * 0.8)
        c = len(verts) - 1
        base = (n - 1) * around
        faces += [(base + j, base + (j + 1) % around, c) for j in range(around)]
        froles += [roles(1.0, 0.5) if roles else role] * around
    return mesh_object(name, verts, faces, role, froles if roles else None)


def helix_shell(prof, height, turns, per_turn, rows, bulge, dip, role="body", name="shell"):
    """A dome wound from one continuous helical band, so the creases between
    turns lie exactly on mesh edges. prof is [(t, radius)] up the height;
    bulge(s, t, phi) shapes the band across its width (s 0 at its lower crease,
    1 at its upper; it must match at 0 and 1); dip(phi) raises or lowers the wraps
    round the dome (phi from the front toward +X)."""
    p = height / turns
    k0, k1 = -2 * per_turn, (int(math.ceil(turns)) + 2) * per_turn
    verts, faces = [], []
    for k in range(k0, k1 + 1):
        phi = math.tau * k / per_turn
        for m in range(rows + 1):
            sv = m / rows
            z = min(max(p * (phi / math.tau + sv) + dip(phi), 0.0), height)
            t = z / height
            r = interp(prof, t) * (1 + bulge(sv, t, phi))
            verts.append((r * math.sin(phi), -r * math.cos(phi), z))
    for a in range(k1 - k0):
        for m in range(rows):
            i = a * (rows + 1) + m
            faces.append((i, i + rows + 1, i + rows + 2, i + 1))
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([vs[i] for i in f])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, role)


def ribbon(pts, width, thick, normal, role="trim", name="ribbon", lens=True):
    """A thin strip along pts with a lens (or diamond) section. width(t) is its
    full width, normal(i) the direction its flat face looks."""
    tang = frames(pts)[0]
    verts, faces = [], []
    n = len(pts)
    sec = [(-1, 0), (-0.55, 1), (0.55, 1), (1, 0), (0.55, -1), (-0.55, -1)] if lens else [(-1, 0), (0, 1), (1, 0), (0, -1)]
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
    faces.append(tuple(range(k - 1, -1, -1)))
    faces.append(tuple((n - 1) * k + j for j in range(k)))
    return mesh_object(name, verts, faces, role)


def torus(R, r, loc=(0, 0, 0), rot=(0, 0, 0), role="metal", seg=24, sides=8, name="torus"):
    verts, faces = [], []
    for i in range(seg):
        a = math.tau * i / seg
        c, s = math.cos(a), math.sin(a)
        for j in range(sides):
            b = math.tau * j / sides
            rr = R + r * math.cos(b)
            verts.append((rr * c, rr * s, r * math.sin(b)))
    for i in range(seg):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c2, d2 = ((i + 1) % seg) * sides + (j + 1) % sides, ((i + 1) % seg) * sides + j
            faces.append((a, b, c2, d2))
    return lib.place(mesh_object(name, verts, faces, role), loc, rot)


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
    s = 0.0 if L2 == 0 else min(max(((p[0] - ax) * dx + (p[1] - az) * dz) / L2, 0.0), 1.0)
    qx, qz = ax + s * dx, az + s * dz
    return math.hypot(p[0] - qx, p[1] - qz), s


def groove(line, width, depth, facing=(0, -1, 0), fade=True):
    """A crease drawn as a polyline in the x-z plane, pressed into the surface
    where it faces `facing`."""
    fv = Vector(facing)

    def fn(p, n):
        w = smoothstep(0.15, 0.6, n.dot(fv))
        if w <= 0:
            return 0.0
        best, where = 1e9, 0.0
        total = len(line) - 1
        for k, (a, b) in enumerate(zip(line, line[1:])):
            d, s = seg_dist((p.x, p.z), a, b)
            if d < best:
                best, where = d, (k + s) / total
        end = smoothstep(0.0, 0.12, where) * smoothstep(1.0, 0.88, where) if fade else 1.0
        return -depth * w * end * math.exp(-(best / width) ** 2)

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


def to_budget(obj, target, mirror=False):
    """Decimates to about `target` triangles; `mirror` keeps the result
    symmetric across x = 0 so a face stays even."""
    n = lib.tris(obj)
    if n <= target:
        return obj
    if not mirror:
        return lib.decimate(obj, target / n)
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = target / n
    mod.use_symmetry = True
    mod.symmetry_axis = "X"
    mod.use_collapse_triangulate = False
    return lib.apply_modifiers(obj)


def fit_radius(obj, a, b, frac_, slab=0.006):
    """Largest distance from the axis a-b of obj's vertices near `frac_` along it."""
    a, b = Vector(a), Vector(b)
    d = (b - a).normalized()
    c = a.lerp(b, frac_)
    best = 0.0
    for v in obj.data.vertices:
        w = v.co - c
        h = w.dot(d)
        if abs(h) < slab:
            r = (w - d * h).length
            if r < 0.046:
                best = max(best, r)
    return best


def drop_hidden(obj, hidden):
    """Deletes the faces whose every corner satisfies hidden(co): skin that
    lies wholly under a turban, sash or bracer and would only cost triangles."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    gone = [f for f in bm.faces if all(hidden(v.co) for v in f.verts)]
    bmesh.ops.delete(bm, geom=gone, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def along_segment(p, a, b):
    """Share along a-b of p's projection, and p's distance from the line."""
    d = b - a
    h = (p - a).dot(d) / d.length_squared
    return h, (p - a - d * h).length


# ---------------------------------------------------------------- pose

# Right arm (-X) raised, conjuring; left fist (+X) on the sash.
SH_R = Vector((-0.142, 0.008, 0.787))
EL_R = Vector((-0.268, 0.022, 0.83))
WR_R = Vector((-0.248, -0.03, 0.911))
SH_L = Vector((0.142, 0.008, 0.787))
EL_L = Vector((0.232, 0.054, 0.672))
WR_L = Vector((0.146, 0.006, 0.6))

HEAD = Vector((0.0, -0.006, 0.902))
HS = 1.08  # head scale
SCALE = 1.05  # whole figure, about z 0.2
# Then lifted, so the tail has room to taper and curl above the board's plinth top.
LIFT = 0.04
# The head turns a little toward the raised hand and looks up at the whirlwind.
NECK = Vector((0.0, 0.012, 0.85))
HEAD_TURN = Euler((-0.08, 0.0, -0.15))


def H(x, y, z):
    """A point on the head, in head units from its centre."""
    return HEAD + Vector((x, y, z)) * HS


# ---------------------------------------------------------------- body


def torso():
    els = [
        ell((0, 0.012, 0.722), (0.094, 0.066, 0.098)),  # ribcage
        ell((0, 0.008, 0.772), (0.112, 0.06, 0.042)),  # shoulder girdle
        ell((0, -0.004, 0.64), (0.07, 0.054, 0.07)),  # abdomen
        ell((0, -0.034, 0.636), (0.042, 0.026, 0.07), stiff=3),  # rectus
        ell((0, 0.004, 0.57), (0.074, 0.057, 0.05)),  # waist
        ell((0, 0.048, 0.752), (0.08, 0.034, 0.06)),  # upper back
        ell((0, 0.016, 0.84), (0.036, 0.036, 0.052)),  # neck
    ]
    for s in (-1, 1):
        els += [
            ell((s * 0.05, -0.04, 0.738), (0.054, 0.03, 0.04), rot=(0, s * 0.25, 0), stiff=3.5),  # pec
            ell((s * 0.076, 0.034, 0.705), (0.04, 0.045, 0.075), rot=(0, -s * 0.3, 0)),  # lat
            ell((s * 0.062, -0.002, 0.6), (0.03, 0.044, 0.05)),  # oblique
            ell((s * 0.046, 0.03, 0.812), (0.06, 0.034, 0.03), rot=(0, -s * 0.45, 0)),  # trap
            ell((s * 0.038, 0.026, 0.82), (0.016, 0.012, 0.035), rot=(0, -s * 0.3, 0), stiff=3),  # neck muscle
        ]
    for sh, el, wr in ((SH_R, EL_R, WR_R), (SH_L, EL_L, WR_L)):
        up, fore = el - sh, wr - el
        dn = up.normalized()
        s = 1 if sh.x > 0 else -1
        els += [
            # Deltoid: a cap set a little down the arm so it melts into pec and
            # trap, then a long head running down to its insertion.
            ell(sh + dn * 0.012 + Vector((s * 0.005, 0.0, 0.006)), (0.027, 0.032, 0.029), stiff=2.5),
            ell(lerp(sh, el, 0.27) + Vector((s * 0.004, 0.0, 0.005)), (up.length * 0.32, 0.037, 0.035), rot=along(up), stiff=2.5),
            cap(sh, el, 0.026),
            # Biceps peaked in front, triceps long behind, a valley between them.
            ell(lerp(sh, el, 0.57) + Vector((0, -0.017, 0.0)), (up.length * 0.27, 0.029, 0.029), rot=along(up), stiff=4),
            ell(lerp(sh, el, 0.4) + Vector((0, 0.019, 0.0)), (up.length * 0.34, 0.027, 0.027), rot=along(up), stiff=4),
            ball(el, 0.025),
            cap(el, wr, 0.019),
            ell(lerp(el, wr, 0.27), (fore.length * 0.3, 0.032, 0.029), rot=along(fore), stiff=3.5),  # forearm
            ball(wr, 0.019),
        ]
    body = lib.metaball(els, role="skin", resolution=0.0042, name="torso")
    note("torso raw", body)
    # Relax the metaball's ripples before the anatomy is carved in.
    body = lib.smooth(body, iterations=4, factor=0.5)

    front, back = (0, -1, 0), (0, 1, 0)
    fns = [
        groove([(0, 0.585), (0, 0.708)], 0.0055, 0.0045, front),  # linea alba
        groove([(0, 0.705), (0, 0.79)], 0.007, 0.004, front),  # sternum
        groove([(-0.037, 0.674), (0, 0.669), (0.037, 0.674)], 0.0045, 0.0035, front, fade=False),
        groove([(-0.037, 0.641), (0, 0.637), (0.037, 0.641)], 0.0045, 0.0032, front, fade=False),
        groove([(-0.035, 0.608), (0, 0.605), (0.035, 0.608)], 0.0045, 0.003, front, fade=False),
        groove([(0, 0.79), (0, 0.6)], 0.008, 0.005, back),  # spine
        bump((0, -0.06, 0.593), (0.006, 0.03, 0.007), -0.0045, front),  # navel
    ]
    for s in (-1, 1):
        fns += [
            # Lower edge of the pecs, sweeping up to the armpit.
            groove([(s * 0.004, 0.703), (s * 0.05, 0.698), (s * 0.088, 0.71), (s * 0.108, 0.745)], 0.0065, 0.0065, front),
            # Outer edge of the six-pack.
            groove([(s * 0.04, 0.692), (s * 0.044, 0.645), (s * 0.04, 0.6), (s * 0.03, 0.578)], 0.005, 0.0035, front),
            # Collar bone and the line between pec and deltoid.
            groove([(s * 0.1, 0.792), (s * 0.108, 0.76), (s * 0.112, 0.735)], 0.006, 0.005, front),
            bump((s * 0.022, -0.06, 0.8), (0.04, 0.03, 0.006), 0.003, front),
        ]
        for k in range(3):
            # Serratus fingers along the ribs.
            fns.append(bump((s * 0.088, -0.02, 0.672 - k * 0.02), (0.012, 0.02, 0.006), 0.0035, (s, -0.4, 0)))
        fns.append(groove([(s * 0.03, 0.79), (s * 0.06, 0.72)], 0.008, 0.004, back))  # shoulder blade
    sculpt(body, fns)
    return note("torso sculpt", body)


def hand_open():
    """Right hand, palm up, fingers cupped round the whirlwind (local frame:
    palm at origin, fingers along +X, palm normal +Z, thumb toward -Y)."""
    els = [ell((0, 0, 0), (0.028, 0.025, 0.011), stiff=3), ell((-0.024, 0, -0.004), (0.014, 0.019, 0.012), stiff=3)]
    for k, y in enumerate((-0.017, -0.0058, 0.0058, 0.017)):
        L = (0.95, 1.05, 1.0, 0.82)[k]
        # Long fingers fanned wide and curled up hard, cradling the funnel.
        p0 = Vector((0.022, y, 0.002))
        p1 = Vector((0.022 + 0.03 * L, y * 1.3, 0.009))
        p2 = Vector((0.022 + 0.05 * L, y * 1.6, 0.034 * L))
        els += [cap(p0, p1, 0.0066, stiff=3), cap(p1, p2, 0.0058, stiff=3), ball(p0, 0.0075, stiff=3)]
    # The thumb swings well out from the palm and curls up beside the fingers.
    els += [cap((-0.01, -0.022, 0.0), (0.006, -0.046, 0.008), 0.0076, stiff=3),
            cap((0.006, -0.046, 0.008), (0.024, -0.052, 0.028), 0.0064, stiff=3)]
    obj = lib.metaball(els, role="skin", resolution=0.0026, name="hand_r")
    return note("hand", to_budget(obj, 720))


def fist():
    """Left fist: knuckles toward -X (pressed into the sash), thumb in front."""
    els = [ell((0, 0, 0), (0.025, 0.027, 0.029), stiff=3)]
    for z in (-0.02, -0.0065, 0.0065, 0.02):
        els.append(ell((-0.021, -0.006, z), (0.012, 0.018, 0.0068), stiff=3))
        els.append(ball((-0.012, -0.024, z * 0.95), 0.0075, stiff=3))
    els.append(cap((0.008, -0.026, 0.022), (-0.012, -0.03, 0.006), 0.0078, stiff=3))
    obj = lib.metaball(els, role="skin", resolution=0.003, name="fist_l")
    return note("fist", to_budget(obj, 470))


# ---------------------------------------------------------------- head


def head():
    els = [
        ell(H(0, 0.008, 0.008), (0.047 * HS, 0.055 * HS, 0.057 * HS)),  # cranium
        ell(H(0, -0.018, -0.036), (0.041 * HS, 0.038 * HS, 0.032 * HS)),  # jaw
        ell(H(0, -0.042, -0.052), (0.021 * HS, 0.016 * HS, 0.016 * HS), stiff=3),  # chin
        ell(H(0, -0.048, 0.008), (0.042 * HS, 0.012 * HS, 0.011 * HS), stiff=3),  # brow
        ell(H(0, -0.06, -0.012), (0.008 * HS, 0.012 * HS, 0.018 * HS), rot=(-0.3, 0, 0), stiff=3.5),  # nose
        ell(H(0, -0.064, -0.026), (0.011 * HS, 0.009 * HS, 0.007 * HS), stiff=3.5),  # nose tip
    ]
    for s in (-1, 1):
        els += [
            ell(H(s * 0.031, -0.04, -0.012), (0.016 * HS, 0.012 * HS, 0.01 * HS), stiff=3),  # cheekbone
            ball(H(s * 0.019, -0.058, -0.002), 0.008 * HS, stiff=3, neg=True),  # eye socket
        ]
    obj = lib.metaball(els, role="skin", resolution=0.0036, name="head")
    # The crown sits wholly inside the turban.
    up = Vector((0, 0, 1))
    up.rotate(TURBAN_TILT)
    drop_hidden(obj, lambda co: (co - TURBAN_BASE).dot(up) > 0.012)
    return note("head", to_budget(obj, 950, mirror=True))


def face():
    parts = []
    for s in (-1, 1):
        # Glowing almond eyes under a heavy, stern brow.
        parts.append(lib.sphere(0.0105 * HS, loc=H(s * 0.019, -0.052, -0.003), rot=(0, s * -0.25, s * 0.12),
                                scale=(1.0, 0.5, 0.5), role="eye", segments=8, rings=6, name="eye"))
        parts.append(lib.tube([H(s * 0.006, -0.062, 0.006), H(s * 0.022, -0.061, 0.01), H(s * 0.04, -0.05, 0.019)],
                              [0.0045, 0.0055, 0.0025], role="cloth", sides=6, resolution=3, name="brow"))
        # Mustache sweeping out and curling up at the tips.
        parts.append(lib.tube([H(s * 0.004, -0.066, -0.03), H(s * 0.022, -0.062, -0.035), H(s * 0.04, -0.05, -0.044),
                               H(s * 0.054, -0.042, -0.04), H(s * 0.058, -0.04, -0.028)],
                              [0.0055, 0.0065, 0.005, 0.0032, 0.0012], role="cloth", sides=6, resolution=3, name="mustache"))
        # Pointed ear swept back under the turban.
        ear = lib.extrude([(0.0, -0.012), (0.011, -0.005), (0.026, 0.012), (0.04, 0.028), (0.014, 0.011), (0.0, 0.011)],
                          0.006, role="skin", bevel=0.0012, smooth=True, resolution=1, name="ear")
        lib.place(ear, (0, 0, 0), (0, 0, math.pi / 2))
        lib.place(ear, H(s * 0.046, 0.006, -0.008), (0, 0, -s * 0.35))
        parts.append(ear)
    # Gold hoop in the right ear.
    parts.append(torus(0.012, 0.0024, loc=H(-0.05, 0.008, -0.034), rot=(0, math.pi / 2, 0.2), role="metal", seg=14, sides=5, name="earring"))

    # Pointed beard from the chin, bound near its end by a gold-thread cuff,
    # its tip curling forward.
    pts = curve([H(0, -0.034, -0.045), H(0, -0.052, -0.074), H(0, -0.061, -0.108), H(0, -0.06, -0.142), H(0, -0.072, -0.172)], 26)
    prof = [(0.0, 1.0), (0.25, 0.96), (0.55, 0.64), (0.8, 0.36), (1.0, 0.07)]
    CUFF = (0.66, 0.74)

    def br(t, th, c):
        k = interp(prof, t)
        rx, ry = 0.039 * HS * k, 0.021 * HS * k
        r = rx * ry / math.sqrt((rx * math.cos(th)) ** 2 + (ry * math.sin(th)) ** 2)
        cuff = smoothstep(CUFF[0] - 0.02, CUFF[0] + 0.01, t) * smoothstep(CUFF[1] + 0.02, CUFF[1] - 0.01, t)
        return r * (1 - 0.05 * abs(math.sin(3.5 * th + t * 3.0)) * (1 - cuff)) * (1 + 0.28 * cuff)

    parts.append(sweep(pts, br, 15, role="cloth", name="beard",
                       roles=lambda t, c: "trim" if CUFF[0] - 0.005 < t < CUFF[1] + 0.005 else "cloth"))
    # A short jaw beard from ear to chin.
    for s in (-1, 1):
        parts.append(lib.tube([H(s * 0.045, 0.004, -0.012), H(s * 0.042, -0.016, -0.04), H(s * 0.026, -0.036, -0.054), H(s * 0.004, -0.044, -0.058)],
                              [0.006, 0.0085, 0.01, 0.011], role="cloth", sides=8, resolution=3, name="jawbeard"))
    return parts


TURBAN_BASE = H(0, 0.006, 0.016)
TURBAN_TILT = Euler((-0.12, 0, 0))
TURBAN_H = 0.172 * HS
# An onion bulb: tight over the brow, swelling well past the head into a round
# dome, then drawn up into a short point.
TURBAN_PROF = [(0.0, 0.058), (0.1, 0.07), (0.2, 0.083), (0.3, 0.093), (0.4, 0.098), (0.5, 0.097), (0.6, 0.093), (0.7, 0.086),
               (0.8, 0.073), (0.88, 0.057), (0.94, 0.039), (0.98, 0.02), (1.0, 0.004)]
JEWEL_T = 0.3


def turban_point(t, out=0.0):
    """The front of the turban at height share t, pushed out by `out`."""
    p = Vector((0, -(interp(TURBAN_PROF, t) * HS + out), TURBAN_H * t))
    p.rotate(TURBAN_TILT)
    return TURBAN_BASE + p


def turban():
    """Wrapped turban whose folds spiral up to a point, a jewel in a gold
    setting at the front and a plume rising from it."""
    def wrap(sv, t, phi):
        # Each turn of cloth rises in a lip over the one below, then tucks under
        # the next. The lips stand full over the forehead and flatten toward the
        # back, where the cloth is pulled tight, so it reads as wound cloth from
        # behind rather than stacked discs.
        lip = smoothstep(0.0, 0.25, sv) * (1 - smoothstep(0.25, 1.0, sv)) - 0.5
        facing = 0.6 + 0.4 * max(0.0, math.cos(phi))
        wraps = smoothstep(0.2, 0.32, t) * (1 - 0.45 * smoothstep(0.6, 0.85, t)) * smoothstep(0.98, 0.86, t)
        # A short tuck crease pinched into each wrap at the back, at a different
        # place on each turn, where the cloth was pulled through.
        n = math.floor(phi / math.tau)
        d = (phi - math.tau * n) - math.pi - (-0.45, 0.1, 0.55)[n % 3]
        tuck = -0.16 * math.exp(-(d / 0.2) ** 2) * math.sin(math.pi * sv) ** 0.7
        return (0.22 * facing * lip + tuck) * wraps

    prof = [(t_, r * HS) for t_, r in TURBAN_PROF]
    turns = 3.2

    def dip(phi):
        # The wraps dip to a V over the forehead and climb diagonally round the
        # head, rising on its left and falling on its right, like wound cloth.
        return -0.03 * math.cos(phi) ** 2 * (1 if math.cos(phi) > 0 else 0) + 0.04 * math.sin(phi)

    t = helix_shell(prof, TURBAN_H, turns, 36, 8, wrap, dip, role="body", name="turban")

    def on_lip(c):
        # Where across its wrap a face lies: the crown of each lip gets a stripe
        # of the second cloth, so the spiral of the wrapping reads in both palettes.
        phi = math.atan2(c.x, -c.y)
        sv = frac((c.z - dip(phi)) / (TURBAN_H / turns) - phi / math.tau)
        return 0.25 < sv < 0.5 and 0.22 < c.z / TURBAN_H < 0.9

    lib.paint(t, "cloth", on_lip)
    t.data.transform(TURBAN_TILT.to_matrix().to_4x4())
    lib.place(t, TURBAN_BASE)
    band = lib.lathe([(0.058 * HS, -0.008), (0.07 * HS, -0.005), (0.0745 * HS, 0.004), (0.0752 * HS, 0.016), (0.0738 * HS, 0.022),
                      (0.068 * HS, 0.027), (0.06 * HS, 0.028)],
                     role="trim", segments=32, name="turban_band")
    band.data.transform(TURBAN_TILT.to_matrix().to_4x4())
    lib.place(band, TURBAN_BASE)
    parts = [note("turban", t), band]
    # A small gold finial closes the crown.
    top = Vector((0, 0, TURBAN_H))
    top.rotate(TURBAN_TILT)
    parts.append(lib.sphere(0.0085, loc=TURBAN_BASE + top + Vector((0, 0, 0.002)), scale=(1, 1, 1.25), role="metal", segments=10, rings=6, name="finial"))
    # Jewel: a big faceted gem in a ring setting at the front of the swell.
    jl = turban_point(JEWEL_T, 0.004)
    parts.append(lib.ico(0.022 * HS, loc=jl + Vector((0, -0.004, 0)), rot=(-0.2, 0, 0), scale=(0.9, 0.6, 1.25), role="eye", subdiv=1, name="jewel"))
    parts.append(torus(0.023 * HS, 0.0042, loc=jl, rot=(math.pi / 2 - 0.2, 0, 0), role="metal", seg=16, sides=6, name="setting"))
    # Tall plume rising from behind the jewel and curling back over the crown.
    pb = jl + Vector((0, -0.006, 0.02))
    plume = lib.extrude([(0.0, 0.0), (0.0135, 0.02), (0.02, 0.054), (0.016, 0.088), (0.005, 0.12), (-0.006, 0.096),
                         (-0.014, 0.058), (-0.0115, 0.022)], 0.006, role="trim", bevel=0.002, smooth=True, resolution=3, name="plume")
    lib.deform(plume, lambda v: Vector((v.x + 0.05 * v.z ** 2, v.y + 2.0 * v.z ** 2, v.z)))
    lib.place(plume, pb, (-0.1, 0, 0))
    parts.append(plume)
    rib = lib.tube([Vector((0.05 * z * z, 2.0 * z * z - 0.0035, z)) for z in (0.004, 0.04, 0.076, 0.11)],
                   [0.0024, 0.002, 0.0013, 0.0005], role="body", sides=6, resolution=3, name="rib")
    parts.append(lib.place(rib, pb, (-0.1, 0, 0)))
    # Gold quill binding the plume to the setting.
    parts.append(lib.cylinder(0.005, 0.0065, 0.016, loc=pb + Vector((0, 0.0, -0.002)), rot=(-0.1, 0, 0), role="metal", segments=10, name="quill"))
    # The loose end of the cloth hangs from the back of the wrap onto the shoulders.
    end = lib.extrude([(-0.03, 0.0), (0.03, 0.0), (0.038, -0.065), (0.035, -0.13), (0.019, -0.185), (0.002, -0.166),
                       (-0.014, -0.19), (-0.03, -0.13), (-0.036, -0.062)], 0.01, role="cloth", bevel=0.003,
                      smooth=True, resolution=3, name="turban_end")

    def drape(v):
        d = -v.z  # how far down the end hangs
        # The cloth turns a little as it falls, so it reads as a soft twisted
        # length of cloth rather than a cut card.
        a = 0.4 * d / 0.19
        x, y = v.x * math.cos(a) - v.y * math.sin(a), v.x * math.sin(a) + v.y * math.cos(a)
        return Vector((x * (1 - 0.9 * d) + 1.6 * d * d, y + 1.6 * d * d - 0.27 * d + 0.006 * math.sin(x * 140), v.z))

    lib.deform(end, drape)
    back = Vector((0, interp(TURBAN_PROF, 0.22) * HS - 0.006, TURBAN_H * 0.22))
    back.rotate(TURBAN_TILT)
    lib.place(end, TURBAN_BASE + back, (0.12, 0, 0.0))
    parts.append(end)
    return parts


# ---------------------------------------------------------------- smoke

# The tail runs out to the genie's left, crosses under him in a shallow bowl and
# sweeps far out to his right, where it turns up and curls over in a glowing
# wisp: a wide inverted teardrop under the torso, staying near y = 0 so both
# board turns see the whole flourish. After SCALE and LIFT its lowest point
# sits near z 0.42, well clear of the plinth top the board puts at about 0.31.
TAIL_CTRL = [(0.0, 0.004, 0.6), (0.03, 0.006, 0.535), (0.065, 0.0, 0.48), (0.06, -0.008, 0.44), (0.0, -0.014, 0.415),
             (-0.075, -0.02, 0.41), (-0.145, -0.024, 0.42), (-0.19, -0.026, 0.445), (-0.205, -0.026, 0.475),
             (-0.188, -0.024, 0.5)]
# A steady taper, never wider than the sash, full through the bowl.
TAIL_PROF = [(0.0, 0.076), (0.1, 0.07), (0.2, 0.062), (0.3, 0.054), (0.4, 0.047), (0.5, 0.04), (0.6, 0.033), (0.7, 0.025),
             (0.8, 0.017), (0.9, 0.011), (1.0, 0.007)]
TURNS, LOBES = 5.0, 2
LIP = 0.3  # share of each band taken by the roll that rises out of the crease above it
WISP = 0.9  # from here on the tail is the glowing wisp
PUFFS = 9  # billows along each band


def tail_twist(t):
    return -math.tau * TURNS * t / LOBES


def band(f):
    """One spiral band across its width, top (0) to bottom (1): a soft roll
    rising out of the crease above, then a full slope tucking in to the crease
    below, so the bands billow over one another like rolling smoke."""
    if f < LIP:
        return math.sin(math.pi / 2 * f / LIP)
    return math.cos(math.pi / 2 * (f - LIP) / (1 - LIP))


def stripe(t):
    """Width (share of a band) of the glowing seam, gathering toward the point."""
    return 0.06 + 0.08 * smoothstep(0.2, 0.8, t)


def tail():
    pts = curve(TAIL_CTRL, 60)

    def radius(t, th, c):
        r = interp(TAIL_PROF, t)
        fade = smoothstep(0.0, 0.06, t) * smoothstep(WISP + 0.02, WISP - 0.08, t)
        u = LOBES * c
        f = frac(u)
        # Each band swells in a chain of soft puffs pinched between them, so
        # the outline rolls like smoke rather than a turned horn; the puffs
        # settle toward the thin end.
        puff = math.cos(math.tau * PUFFS * t + math.pi * int(u)) * math.sin(math.pi * f)
        billow = 1 + 0.11 * puff * smoothstep(0.8, 0.5, t) * smoothstep(0.02, 0.12, t)
        return (r * 0.8 + (min(0.013, 0.32 * r) + 0.002) * band(f) * fade) * billow

    def roles(t, c):
        if t > WISP:
            return "trim"
        u = LOBES * c
        if t > 0.17 and frac(u) < stripe(t):
            return "trim"
        return "body" if int(u) % 2 == 0 else "cloth"

    parts = [note("tail", sweep(pts, radius, 28, role="body", name="tail", roles=roles, twist=tail_twist))]
    # The wisp: a round curl that turns three quarters of the way over toward
    # the body, thinning to a point.
    e = pts[-1]
    tv = (pts[-1] - pts[-2]).normalized()
    nv = Vector((tv.z, 0.0, -tv.x)).normalized()
    R0 = 0.024
    curl = [pts[-3], pts[-2]]
    K = 8
    for k in range(K):
        a = 1.5 * math.pi * k / (K - 1)
        R = R0 * (1 - 0.55 * k / (K - 1))
        centre = e + nv * R0
        curl.append(centre + R * (-nv * math.cos(a) + tv * math.sin(a)))
    radii = [0.0066, 0.0064] + [0.0062 - 0.0042 * (k / (K - 1)) ** 1.2 for k in range(K)]
    parts.append(note("wisp", lib.tube(curl, radii, role="trim", sides=8, resolution=2, name="wisp")))
    return parts


WHIRL_H = 0.155
# The tornado leans back and a little out, so the board camera sees its side, not its mouth.
WHIRL_TILT = Euler((-0.25, -0.08, 0.0))


def coil_radius(s):
    return 0.007 + 0.035 * s ** 0.9


def whirlwind(base):
    """The conjured tornado: a slim funnel of glowing wind spinning up from the
    palm round a smoky core, with open gaps between its turns."""
    parts = []
    core = curve([Vector((0, 0, WHIRL_H * 0.9)), Vector((-0.005, 0, WHIRL_H * 0.55)), Vector((0.004, 0, WHIRL_H * 0.22)),
                  Vector((0, 0, 0.002))], 22)
    prof = [(0.0, 0.02), (0.25, 0.016), (0.6, 0.009), (1.0, 0.003)]

    def cr(t, th, c):
        # A smooth smoky column swelling in soft puffs, not a ridged horn.
        return interp(prof, t) * (1 + 0.13 * math.sin(math.tau * 3.5 * t) * smoothstep(1.0, 0.75, t))

    parts.append(note("core", sweep(core, cr, 10, role="cloth", name="whirl_core")))

    N, turns = 60, 3.8
    path, nrm = [], []
    for i in range(N):
        s = i / (N - 1)
        a = turns * math.tau * s
        radial = Vector((math.cos(a), math.sin(a), 0))
        z = 0.006 + WHIRL_H * s ** 1.1
        wob = Vector((0.005 * math.sin(s * 4.0), 0.003 * math.cos(s * 3.0), 0))
        path.append(wob + radial * coil_radius(s) + Vector((0, 0, z)))
        nrm.append(radial + Vector((0, 0, -0.3)))
    width = lambda s: (0.006 + 0.014 * s) * smoothstep(0.0, 0.06, s) * (1 - 0.8 * smoothstep(0.93, 1.0, s))
    parts.append(note("coil", ribbon(path, width, 0.005, lambda i: nrm[i], role="eye", name="whirl_coil", lens=False)))
    for o in parts:
        lib.place(o, base, WHIRL_TILT)
    return parts


# ---------------------------------------------------------------- dress


def sash():
    prof = [(0.074, 0.518), (0.085, 0.52), (0.0915, 0.524), (0.0905, 0.53), (0.091, 0.539), (0.0935, 0.556), (0.0935, 0.568),
            (0.092, 0.584), (0.0915, 0.592), (0.0935, 0.597), (0.089, 0.6015), (0.072, 0.604)]
    s = lib.lathe(prof, role="cloth", segments=30, name="sash")

    def folds(v):
        a = math.atan2(v.x, -v.y)
        k = 1 + 0.03 * math.sin(a * 7 + v.z * 40) * smoothstep(0.53, 0.542, v.z) * smoothstep(0.594, 0.582, v.z)
        return Vector((v.x * k, v.y * k, v.z))

    def tuck(v):
        # The top rolls in tight against the small of the back, which is
        # shallower than the sash is round.
        return Vector((v.x, v.y * (1 - 0.14 * smoothstep(0.594, 0.604, v.z) * smoothstep(0.0, 0.06, v.y)), v.z))

    lib.deform(s, folds)
    lib.paint(s, "trim", lambda c: (c.z < 0.5295 or c.z > 0.5915) and math.hypot(c.x, c.y) > 0.087)
    lib.deform(s, tuck)
    parts = [s]
    # Knot at the front left with two tails fluttering down into the smoke.
    knot = (0.05, -0.084, 0.56)
    parts.append(lib.sphere(0.015, loc=(knot[0], knot[1] - 0.001, knot[2] + 0.001), rot=(0, 0.2, 0), scale=(1.55, 0.68, 0.85),
                            role="cloth", segments=10, rings=6, name="knot"))
    for k, (rot, length, w) in enumerate((((0.12, -0.42, -0.35), 0.075, 0.036), ((0.06, -0.12, 0.25), 0.06, 0.03))):
        tail_ = lib.extrude([(-w / 2, 0.0), (w / 2, 0.0), (w * 0.55, -length * 0.55), (w * 0.3, -length), (0.0, -length * 0.82),
                             (-w * 0.35, -length * 0.95), (-w * 0.5, -length * 0.5)], 0.009, role="cloth", bevel=0.0025,
                            smooth=True, resolution=2, name="sash_tail")

        def flutter(v, k=k):
            return Vector((v.x + 0.012 * math.sin(v.z * 40 + k) + 0.02 * (v.z / 0.12) ** 2, v.y - 0.012 * (v.z / 0.12) ** 2 - 0.004 * math.sin(v.x * 80), v.z))

        lib.deform(tail_, flutter)
        lib.place(tail_, (knot[0] + (0.006 if k else -0.004), knot[1] + 0.006, knot[2] - 0.006), rot)
        parts.append(tail_)
    # Medallion at the front centre.
    parts.append(lib.cylinder(0.017, 0.017, 0.006, loc=(-0.012, -0.096, 0.56), rot=(math.pi / 2 - 0.15, 0, 0), role="metal",
                              segments=14, bevel=0.0015, name="medallion"))
    parts.append(lib.ico(0.008, loc=(-0.012, -0.1, 0.56), scale=(1, 0.6, 1), role="eye", subdiv=1, name="medal_gem"))
    return parts


def ring_on(a, b, frac_, profile, role, rims=None, segments=16, name="band"):
    """axis_ring that paints faces whose height along the axis satisfies rims(h) with trim."""
    a, b = Vector(a), Vector(b)
    obj = lib.lathe(profile, role=role, segments=segments, name=name)
    if rims:
        lib.paint(obj, "trim", lambda c: rims(c.z))
    rot = (b - a).normalized().to_track_quat("Z", "Y").to_euler()
    return lib.place(obj, a.lerp(b, frac_), rot)


def bracers(body):
    parts = []
    for sh, el, wr in ((SH_R, EL_R, WR_R), (SH_L, EL_L, WR_L)):
        L = (wr - el).length
        r0 = fit_radius(body, wr, el, 0.06) + 0.003
        r1 = fit_radius(body, wr, el, 0.55) + 0.004
        if DEBUG:
            print(f"DJ bracer radii {r0:.4f} {r1:.4f}")
        prof = [(r0 - 0.005, -0.004), (r0 + 0.002, -0.0035), (r0 + 0.0045, 0.0015), (r0 + 0.002, 0.007), (r0 + 0.003, 0.012),
                (r1, L * 0.5), (r1 + 0.003, L * 0.555), (r1 + 0.0065, L * 0.6), (r1 + 0.003, L * 0.645), (r1 - 0.006, L * 0.65)]
        # Only the wrist rim glows, leading the eye to the hands.
        parts.append(ring_on(wr, el, 0.0, prof, "metal", rims=lambda h: h < 0.0075, name="bracer"))
        # An oval gem set in the outer face of the bracer.
        d = (el - wr).normalized()
        c = wr.lerp(el, 0.3)
        o = Vector((c.x, c.y + 0.03, 0.0))
        o = (o - o.dot(d) * d).normalized()
        rg = r0 + (r1 - r0) * 0.5 + 0.004
        gem = lib.ico(0.0075, scale=(1.0, 0.55, 1.35), role="eye", subdiv=1, name="bracer_gem")
        gem.data.transform(o.to_track_quat("-Y", "Z").to_matrix().to_4x4())
        lib.place(gem, c + o * rg)
        parts.append(gem)
        parts.append(lib.place(torus(0.0085, 0.0018, role="metal", seg=10, sides=4, name="gem_setting"), (0, 0, 0), (math.pi / 2, 0, 0)))
        parts[-1].data.transform(o.to_track_quat("-Y", "Z").to_matrix().to_4x4())
        lib.place(parts[-1], c + o * (rg - 0.002))
        parts.append(armlet(body, sh, el, 0.36))
    return parts


def section(obj, a, b, frac_, rot, bins=24, slab=0.004, heights=(-0.012, -0.006, 0.0, 0.006, 0.012)):
    """Radius of obj's surface round the axis a-b at `frac_` along it, in `bins`
    sectors measured as lib.lathe measures angles (from local -Y toward +X) in
    the frame `rot`. In each sector the nearest surface is the arm's own skin,
    so the torso never inflates it; the widest of several cuts across the band
    is kept, so the skin cannot poke through at the band's edges."""
    centre = Vector(a).lerp(Vector(b), frac_)
    inv = rot.to_matrix().transposed()
    cuts = {h: [None] * bins for h in heights}
    for v in obj.data.vertices:
        w = inv @ (v.co - centre)
        for h in heights:
            if abs(w.z - h) < slab:
                r = math.hypot(w.x, w.y)
                if r < 0.07:
                    k = int(frac(math.atan2(w.x, -w.y) / math.tau) * bins) % bins
                    row = cuts[h]
                    row[k] = r if row[k] is None else min(row[k], r)
    best = [max((cuts[h][k] for h in heights if cuts[h][k] is not None), default=None) for k in range(bins)]
    known = [r for r in best if r is not None]
    fill = sum(known) / len(known)
    for k in range(bins):
        if best[k] is None:
            best[k] = fill
    return [(best[k - 1] + 2 * best[k] + best[(k + 1) % bins]) / 4 for k in range(bins)]


def armlet(body, sh, el, frac_, segments=18):
    """A heavy gold armlet sunk into the upper arm: it follows the arm's real
    (oval) section instead of standing off it as a round bangle, with a raised
    bead round its middle."""
    sh, el = Vector(sh), Vector(el)
    rot = (el - sh).normalized().to_track_quat("Z", "Y").to_euler()
    sec = section(body, sh, el, frac_, rot)
    n = len(sec)

    def skin_r(ang):
        x = frac(ang / math.tau) * n - 0.5
        i = math.floor(x)
        f = x - i
        return sec[i % n] * (1 - f) + sec[(i + 1) % n] * f

    # (height across the band, offset from the skin); the edges tuck 0.003 under it.
    prof = [(-0.003, -0.014), (0.002, -0.0136), (0.0045, -0.0108), (0.0038, -0.0058), (0.0068, -0.0024), (0.0085, 0.0),
            (0.0068, 0.0024), (0.0038, 0.0058), (0.0045, 0.0108), (0.002, 0.0136), (-0.003, 0.014)]
    ring = lib.lathe([(1.0 + d - 0.002, z) for d, z in prof], role="metal", segments=segments, name="armband")

    def hug(v):
        d = math.hypot(v.x, v.y)
        ang = math.atan2(v.x, -v.y)
        r = skin_r(ang) + (d - 1.0)
        return Vector((math.sin(ang) * r, -math.cos(ang) * r, v.z))

    lib.deform(ring, hug)
    if DEBUG:
        print(f"DJ armlet section {min(sec):.4f}..{max(sec):.4f}")
    return lib.place(ring, sh.lerp(el, frac_), rot)


def build():
    parts = []
    add = parts.append
    body = torso()
    add(body)
    sleeves = bracers(body)

    def hidden(co):
        if co.z < 0.58 and math.hypot(co.x, co.y) < 0.1:  # inside the sash and the top of the tail
            return True
        if co.z > 0.872 and math.hypot(co.x, co.y - 0.016) < 0.045:  # the neck inside the head
            return True
        for el, wr in ((EL_R, WR_R), (EL_L, WR_L)):
            h, dist = along_segment(co, wr, el)
            if 0.13 < h < 0.57 and dist < 0.04:  # inside the bracer
                return True
        return False

    drop_hidden(body, hidden)
    body = to_budget(body, 2700)
    parts[0] = body
    heads = [head()] + face() + turban()
    pivot = Matrix.Translation(NECK) @ HEAD_TURN.to_matrix().to_4x4() @ Matrix.Translation(-NECK)
    for o in heads:
        o.data.transform(pivot)
        o.data.update()
    parts += heads
    parts += tail()
    parts += sash()
    parts += sleeves

    # Right hand raised, palm up, the whirlwind spinning from it.
    h = hand_open()
    palm = WR_R + Vector((0.004, -0.008, 0.03))
    lib.place(h, palm, (0.0, -0.12, -2.45))
    add(h)
    parts += whirlwind(palm + Vector((0, 0, 0.01)))
    f = fist()
    lib.place(f, WR_L + Vector((-0.014, -0.008, -0.02)), (0, 0, 0.0))
    add(f)

    # Contrapposto: the chest turns toward the conjuring hand above the waist.
    def twist(v):
        a = -0.12 * smoothstep(0.56, 0.8, v.z)
        c, sn = math.cos(a), math.sin(a)
        x, y = v.x, v.y - 0.01
        return Vector((x * c - y * sn, x * sn + y * c + 0.01, v.z))

    lower = {"tail", "wisp", "sash", "knot", "sash_tail", "medallion", "medal_gem"}
    for o in parts:
        if o.name.split(".")[0] not in lower:
            lib.deform(o, twist)
    # A touch larger overall, grown from the tail tip, so the genie holds its own
    # beside the other flyers on the board.
    for o in parts:
        lib.deform(o, lambda v: Vector((v.x * SCALE, v.y * SCALE, 0.2 + (v.z - 0.2) * SCALE + LIFT)))
    if DEBUG:
        tally, area = {}, {}
        for p in parts:
            key = p.name.split(".")[0]
            tally[key] = tally.get(key, 0) + lib.tris(p)
            names = [m.name.split(".")[0] for m in p.data.materials]
            for poly in p.data.polygons:
                r = names[poly.material_index] if poly.material_index < len(names) else names[0]
                area[(key, r)] = area.get((key, r), 0.0) + poly.area
        print("DJ tally " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])))
        print("DJ area " + ", ".join(f"{k}/{r}={a * 1e4:.0f}" for (k, r), a in sorted(area.items(), key=lambda kv: -kv[1])))
    return parts
