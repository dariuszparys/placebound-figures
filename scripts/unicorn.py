"""Unicorn: a Light beast. A noble white horse in a proud, collected prance,
right foreleg raised, neck arched and head held high, with a long spiral horn
on its brow, a flowing mane and tail, feathered fetlocks and cloven hooves.
Faces -Y; its right is -X."""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

import lib

DEBUG = os.environ.get("UNICORN_DEBUG")
# Smooth the organic coat over its decimated facets; hooves, horn ridges and
# the ear rims stay crisp. A stronger baked edge highlight lets the topline,
# the neck arch and the leg edges catch light in the charcoal palette.
FINISH = {"sharp_angle": 50, "edges": 0.22}
X = Vector((1, 0, 0))
Y = Vector((0, 1, 0))
Z = Vector((0, 0, 1))


# ---------------------------------------------------------------- helpers


def visible(stiff):
    """Visible radius of a lone metaball element over its `radius` (threshold 0.6)."""
    return math.sqrt(1 - (0.6 / stiff) ** (1 / 3))


def frame(axis, up=(0, 0, 1)):
    """3x3 matrix whose columns are local x, y (along `axis`) and z (toward `up`)."""
    y = Vector(axis).normalized()
    u = Vector(up)
    z = u - u.dot(y) * y
    if z.length < 1e-6:
        z = Vector((1, 0, 0)) - y.x * y
    z.normalize()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed()


def ell(c, half, axis=None, up=(0, 0, 1), stiff=2.0, neg=False):
    """Metaball ellipsoid centred on `c` whose visible half extents are `half`
    (x, along `axis`, toward `up`)."""
    m = max(half)
    e = {"co": tuple(c), "radius": m / visible(stiff), "type": "ELLIPSOID",
         "size": tuple(h / m for h in half), "stiffness": stiff, "negative": neg}
    if axis is not None:
        e["rot"] = tuple(frame(axis, up).to_euler())
    return e


def seg(a, b, r_side, r_up, up=(0, 0, 1), stiff=2.0):
    """Ellipsoid spanning a to b."""
    a, b = Vector(a), Vector(b)
    return ell((a + b) / 2, (r_side, (b - a).length / 2, r_up), axis=b - a, up=up, stiff=stiff)


def neg_seg(a, b, r_side, r_up, up=(0, 0, 1)):
    """A negative ellipsoid spanning a to b: carves a soft groove when it sits
    just under the surface of a well-filled mass."""
    e = seg(a, b, r_side, r_up, up=up)
    e["negative"] = True
    return e


def orient(obj, loc, axis, up=(0, 1, 0)):
    """Bakes a frame into the mesh: local z runs along `axis`, local y toward `up`."""
    z = Vector(axis).normalized()
    u = Vector(up)
    y = u - u.dot(z) * z
    y.normalize()
    x = y.cross(z)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(loc)
    obj.data.transform(m)
    obj.data.update()
    return obj


def catmull(points, n):
    """n points sampled along a Catmull-Rom spline through `points`."""
    pts = [Vector(p) for p in points]
    pts = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    out = []
    segs = len(points) - 1
    for i in range(n):
        t = i / (n - 1) * segs
        k = min(int(t), segs - 1)
        f = t - k
        p0, p1, p2, p3 = pts[k], pts[k + 1], pts[k + 2], pts[k + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3))
    return out


def lock(points, radii, normal, flat=0.5, role="cloth", sides=8, resolution=3, name="lock"):
    """A hair clump: a tapered tube squashed to `flat` of its thickness along
    normal(point), so it lies on the surface it grows from like a ribbon."""
    t = lib.tube(points, radii, role=role, sides=sides, resolution=resolution, name=name)
    centre = catmull(points, 36)
    ns = []
    for i, c in enumerate(centre):
        a = (centre[min(i + 1, 35)] - centre[max(i - 1, 0)]).normalized()
        n = Vector(normal(c))
        n = n - n.dot(a) * a
        ns.append(n.normalized() if n.length > 1e-6 else Vector((1, 0, 0)))
    for v in t.data.vertices:
        p = v.co
        i = min(range(36), key=lambda k: (centre[k] - p).length_squared)
        o = p - centre[i]
        v.co = p - ns[i] * (o.dot(ns[i]) * (1 - flat))
    t.data.update()
    return t


def smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- anatomy

# Head frame: the poll, the face direction (toward the muzzle, 39 degrees off
# vertical, so the nose reaches forward of the chest) and the brow normal.
POLL = Vector((0, -0.232, 0.724))
TILT = math.radians(39)
D = Vector((0, -math.sin(TILT), -math.cos(TILT)))
F = Vector((0, -math.cos(TILT), math.sin(TILT)))


def head(u, v, x=0.0):
    return POLL + D * u + F * v + X * x


# The neck between its crest (top line, a convex arch from the withers to the
# poll) and its throat (from the breast up to a clean throatlatch behind the
# jowl). Each section spans crest to throat.
CREST = [(0, -0.085, 0.492), (0, -0.105, 0.572), (0, -0.13, 0.636), (0, -0.163, 0.684), (0, -0.195, 0.71), (0, -0.216, 0.718)]
THROAT = [(0, -0.262, 0.40), (0, -0.25, 0.47), (0, -0.222, 0.535), (0, -0.198, 0.585), (0, -0.184, 0.618), (0, -0.18, 0.64)]
NECK_W = [0.064, 0.052, 0.042, 0.036, 0.031, 0.027]


def neck_samples(n):
    cs, ts = catmull(CREST, n), catmull(THROAT, n)
    out = []
    for i in range(n):
        c, t = cs[i], ts[i]
        back = (c - t).normalized()  # toward the crest
        a = Vector((0, -back.z, back.y))  # along the neck, toward the head
        k = i / (n - 1) * (len(NECK_W) - 1)
        j = min(int(k), len(NECK_W) - 2)
        f = k - j
        w = NECK_W[j] * (1 - f) + NECK_W[j + 1] * f
        out.append(((c + t) / 2, a, back, w, (c - t).length / 2))
    return out


NECK_FINE = neck_samples(40)


# Leg skeletons: (point, radius) from inside the body down to the coronet.
# Index 1 elbow/stifle, 4 knee/point of hock, 7 fetlock, 9 coronet.
LEGS = {
    "fore_l": [((0.056, -0.205, 0.35), 0.05), ((0.06, -0.172, 0.29), 0.043), ((0.06, -0.177, 0.24), 0.034),
               ((0.059, -0.181, 0.195), 0.018), ((0.059, -0.182, 0.16), 0.026), ((0.059, -0.183, 0.13), 0.0155),
               ((0.059, -0.184, 0.092), 0.014), ((0.059, -0.181, 0.056), 0.022), ((0.059, -0.19, 0.036), 0.0145),
               ((0.059, -0.198, 0.024), 0.0165)],
    "fore_r": [((-0.056, -0.205, 0.35), 0.05), ((-0.06, -0.19, 0.30), 0.043), ((-0.061, -0.235, 0.29), 0.034),
               ((-0.062, -0.288, 0.272), 0.018), ((-0.062, -0.318, 0.259), 0.026), ((-0.062, -0.322, 0.233), 0.0155),
               ((-0.062, -0.316, 0.197), 0.014), ((-0.062, -0.303, 0.164), 0.022), ((-0.062, -0.289, 0.149), 0.0145),
               ((-0.062, -0.275, 0.14), 0.0165)],
    "hind_r": [((-0.05, 0.14, 0.34), 0.064), ((-0.067, 0.082, 0.272), 0.05), ((-0.066, 0.135, 0.218), 0.032),
               ((-0.062, 0.186, 0.174), 0.023), ((-0.06, 0.21, 0.152), 0.023), ((-0.06, 0.202, 0.122), 0.0155),
               ((-0.06, 0.2, 0.09), 0.0145), ((-0.06, 0.198, 0.056), 0.022), ((-0.06, 0.188, 0.036), 0.0145),
               ((-0.06, 0.18, 0.024), 0.0165)],
    "hind_l": [((0.05, 0.14, 0.34), 0.064), ((0.067, 0.066, 0.268), 0.05), ((0.066, 0.106, 0.216), 0.032),
               ((0.062, 0.15, 0.175), 0.023), ((0.06, 0.171, 0.153), 0.023), ((0.06, 0.161, 0.122), 0.0155),
               ((0.06, 0.15, 0.09), 0.0145), ((0.06, 0.138, 0.056), 0.022), ((0.06, 0.126, 0.036), 0.0145),
               ((0.06, 0.117, 0.024), 0.0165)],
}


def torso_elements():
    e = []
    m = 3.0  # muscles hold their shape better with a stiffer field
    # The base masses: ribcage, deep chest, withers, back, loin, croup and
    # belly. An uphill topline, the withers standing above a short croup, and
    # a waist at the flank so the muscle groups stand proud of the barrel.
    e.append(ell((0, -0.045, 0.364), (0.073, 0.125, 0.09), axis=(0, 1, -0.06)))
    # The rear ribcage and loin are broad enough that the barrel flows into
    # the croup from above instead of pinching into a waist.
    e.append(ell((0, 0.07, 0.38), (0.06, 0.08, 0.06)))
    e.append(ell((0, -0.16, 0.366), (0.07, 0.075, 0.098)))
    e.append(ell((0, -0.105, 0.448), (0.045, 0.07, 0.05)))
    e.append(ell((0, 0.015, 0.41), (0.064, 0.115, 0.04), axis=(0, 1, -0.1)))
    e.append(ell((0, 0.115, 0.366), (0.032, 0.07, 0.036), axis=(0, 1, -0.4)))
    e.append(ell((0, -0.075, 0.304), (0.06, 0.095, 0.048)))
    # The breast prow: the deep chest projects past the forearms.
    e.append(ell((0, -0.236, 0.33), (0.044, 0.03, 0.048), stiff=m))
    for s in (-1, 1):
        # Forehand: pectoral, the shoulder blade sloping from the withers to a
        # crisp point of shoulder, the upper arm back to the elbow and the
        # triceps filling the angle behind it.
        e.append(ell((s * 0.034, -0.222, 0.335), (0.032, 0.03, 0.05), stiff=m))
        e.append(seg((s * 0.05, -0.1, 0.452), (s * 0.062, -0.212, 0.342), 0.03, 0.052, up=(0, -1, 1), stiff=4.0))
        e.append(ell((s * 0.05, -0.224, 0.347), (0.03, 0.026, 0.032), stiff=4.0))
        e.append(seg((s * 0.056, -0.214, 0.335), (s * 0.06, -0.176, 0.29), 0.03, 0.03, up=(0, -1, 0), stiff=m))
        e.append(ell((s * 0.058, -0.148, 0.336), (0.027, 0.038, 0.046), stiff=m))
        # Long back muscles either side of the spine.
        e.append(seg((s * 0.022, -0.06, 0.43), (s * 0.024, 0.08, 0.405), 0.024, 0.028, stiff=m))
        # Hindquarters: a double croup, the point of hip running down into the
        # stifle, the haunch and a crisp buttock behind it.
        e.append(ell((s * 0.036, 0.118, 0.382), (0.04, 0.075, 0.05), axis=(0, 1, -0.4), stiff=4.0))
        e.append(seg((s * 0.056, 0.074, 0.396), (s * 0.062, 0.082, 0.33), 0.015, 0.013, up=(0, -1, 0), stiff=m))
        e.append(ell((s * 0.05, 0.138, 0.342), (0.05, 0.076, 0.082), stiff=m))
        e.append(ell((s * 0.034, 0.2, 0.312), (0.034, 0.032, 0.058), stiff=4.0))
    # The seam between the haunch and the buttock: a thin negative element just
    # under the surface carves a soft groove down the back of each thigh.
    for s in (-1, 1):
        e.append(neg_seg((s * 0.064, 0.211, 0.365), (s * 0.061, 0.219, 0.272), 0.009, 0.009, up=(0, 1, 0)))
    # Hind thighs: a deep, flat wedge of muscle from the hip and buttock down
    # into a continuous gaskin that tapers into the hock.
    for name in ("hind_r", "hind_l"):
        hip, stifle, gaskin, hock = (Vector(LEGS[name][i][0]) for i in (0, 1, 2, 3))
        e.append(seg(hip + Vector((0, 0.02, 0.0)), gaskin, 0.043, 0.062, up=(0, 1, 0)))
        e.append(seg(gaskin + Vector((0, 0.014, 0.03)), hock + Vector((0, 0.006, 0.008)), 0.025, 0.032, up=(0, 1, 0.2), stiff=m))
    return e


def neck_elements():
    e = []
    samples = neck_samples(10)
    for p, a, back, w, d in samples:
        e.append(ell(p + back * d * 0.1, (w, 0.036, d * 0.8), axis=a, up=back))
    # The arched crest along the top line.
    for p, a, back, w, d in samples[1:9]:
        e.append(ell(p + back * d * 0.5, (w * 0.8, 0.036, d * 0.45), axis=a, up=back, stiff=3.0))
    return e


def head_elements():
    e = []
    s3, s4 = 3.0, 4.0
    # Cranium, a broad flat forehead plane, crisp round jowls, a long tapering
    # nasal bridge and a neat muzzle with flared nostrils and a soft chin.
    e.append(ell(head(0.025, -0.006), (0.037, 0.044, 0.04), axis=D, up=F))
    e.append(ell(head(0.05, 0.014), (0.036, 0.046, 0.022), axis=D, up=F, stiff=s3))
    e.append(ell(head(0.062, -0.034), (0.047, 0.05, 0.044), axis=D, up=F, stiff=s4))
    e.append(ell(head(0.112, -0.002), (0.021, 0.066, 0.024), axis=D, up=F))
    e.append(ell(head(0.163, -0.012), (0.02, 0.026, 0.027), axis=D, up=F))
    e.append(ell(head(0.182, -0.016), (0.018, 0.014, 0.02), axis=D, up=F))
    e.append(ell(head(0.163, -0.039), (0.014, 0.022, 0.013), axis=D, up=F))
    e.append(seg(head(0.078, -0.064), head(0.15, -0.041), 0.011, 0.01, up=F))
    e.append(ell(head(0.112, -0.034), (0.019, 0.048, 0.024), axis=D, up=F))
    for s in (-1, 1):
        e.append(ell(head(0.05, 0.014, s * 0.03), (0.013, 0.018, 0.012), axis=D, up=F, stiff=s3))
        # The upper lid and brow ridge hooding the eye, and the facial crest
        # running from below the eye down toward the jowl.
        e.append(ell(head(0.046, 0.022, s * 0.036), (0.012, 0.018, 0.006), axis=D, up=F, stiff=s3))
        e.append(seg(head(0.068, -0.002, s * 0.035), head(0.12, -0.013, s * 0.021), 0.0045, 0.0045, up=F, stiff=s3))
        e.append(ell(head(0.165, 0.006, s * 0.018), (0.0105, 0.014, 0.0075), axis=D, up=F, stiff=s3))
        e.append(ell(head(0.172, 0.012, s * 0.0185), (0.0062, 0.011, 0.006), axis=D, up=F, stiff=4.0, neg=True))
    return e


def leg_back(leg):
    """The heel-side direction of a leg's cannon: perpendicular to it, toward +Y."""
    knee, fet = Vector(leg[4][0]), Vector(leg[7][0])
    axis = (fet - knee).normalized()
    b = Y - Y.dot(axis) * axis
    return axis, b.normalized()


def coat():
    meta = lib.metaball(torso_elements() + neck_elements() + head_elements(), role="body", resolution=0.006, name="torso")
    parts = [meta]
    for name, leg in LEGS.items():
        skel = leg[2:] if name.startswith("hind") else leg
        pts = [p for p, _ in skel]
        radii = [r for _, r in skel]
        parts.append(lib.skin(pts, [(i, i + 1) for i in range(len(pts) - 1)], radii, name=name))
        # The flexor tendon: a clean ridge down the back of the cannon.
        axis, back = leg_back(leg)
        knee, fet = Vector(leg[4][0]), Vector(leg[7][0])
        a = knee + axis * 0.022 + back * 0.0105
        b = fet - axis * 0.014 + back * 0.0115
        parts.append(lib.skin([a, (a + b) / 2, b], [(0, 1), (1, 2)], [0.008, 0.0085, 0.009], name="tendon"))
        if name.startswith("fore"):
            # The knee: a broad, flat-fronted carpus.
            parts.append(lib.skin([knee - back * 0.011 - axis * 0.012, knee - back * 0.011 + axis * 0.007], [(0, 1)],
                                  [0.0145, 0.0135], name="knee"))
        else:
            # The point of the hock, standing up behind the joint.
            parts.append(lib.skin([knee + back * 0.004 - axis * 0.016, knee + back * 0.01 - axis * 0.002], [(0, 1)],
                                  [0.0125, 0.0105], name="hock"))
    c = lib.join(parts, "coat")
    c = lib.remesh(c, voxel=0.0045)
    c = lib.smooth(c, iterations=2, factor=0.5)
    # A soft muzzle: cut a clean edge loop across the face, then paint the lips
    # and nostrils beyond it.
    muzzle = head(0.186, -0.012)
    cut = POLL + D * 0.166
    bm = bmesh.new()
    bm.from_mesh(c.data)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5, plane_co=cut, plane_no=D)
    bm.to_mesh(c.data)
    bm.free()
    lib.paint(c, "skin", lambda p: (p - cut).dot(D) > 0 and (p - muzzle).length < 0.03)
    # Decimate, sparing a band along the edge of the paint so it stays smooth
    # (a vertex weight of 0 is left alone, 1 is decimated freely).
    me = c.data
    mats = [set() for _ in me.vertices]
    for poly in me.polygons:
        for vi in poly.vertices:
            mats[vi].add(poly.material_index)
    edge = {i for i, m in enumerate(mats) if len(m) > 1}
    band = set(edge)
    for e in me.edges:
        a, b = e.vertices
        if a in edge or b in edge:
            band.update((a, b))
    vg = c.vertex_groups.new(name="decimate")
    vg.add([i for i in range(len(me.vertices)) if i not in band], 1.0, "REPLACE")
    vg.add(list(band), 0.0, "REPLACE")
    mod = c.modifiers.new("dec", "DECIMATE")
    if DEBUG:
        print("TRIS coat before decimation", lib.tris(c))
    mod.ratio = min(1.0, 8000 / lib.tris(c))
    mod.use_collapse_triangulate = False
    mod.vertex_group = "decimate"
    return lib.apply_modifiers(c)


# ---------------------------------------------------------------- features


HORN_BASE = head(0.05, 0.03)
HORN_AXIS = Vector((0, 0.17, 1)).normalized()  # a touch back of vertical
HORN_TOP = 0.877  # the beast tier tops out at 0.88


def horn():
    """A long slender lance of a horn on the brow: three shallow ridges
    twisting tightly to a glowing point, with magic glowing along the grooves
    so the spiral reads and the horn is the brightest thing on the board."""
    L = (HORN_TOP - HORN_BASE.z) / HORN_AXIS.z
    # About 20 degrees of twist per ring against the 20-degree lathe segments,
    # so the ridges run smoothly instead of as straight chords.
    r0, turns = 0.024, 2.75
    n = 48
    # The base flange sits inside the skull, well under the collar.
    prof = [(0.0, -0.03), (r0 * 1.05, -0.03)] + [(r0 * (1 - t) ** 0.9 + 0.0008, t * L) for t in (i / n for i in range(n))] + [(0, L)]
    h = lib.lathe(prof, segments=18, role="trim", name="horn")

    def twist(p):
        r = math.hypot(p.x, p.y)
        if r < 1e-7:
            return p
        a = math.atan2(p.x, -p.y)
        t = max(0.0, p.z / L)
        k = 1 + 0.12 * math.cos(3 * (a - math.pi / 6)) * min(1.0, t * 5 + 0.25)
        a += t * turns * math.tau
        return Vector((math.sin(a) * r * k, -math.cos(a) * r * k, p.z))

    lib.deform(h, twist)

    def band(c):
        t = max(0.0, c.z / L)
        a = math.atan2(c.x, -c.y) - t * turns * math.tau
        return math.cos(3 * (a - math.pi / 6)) < -0.5 and 0.08 < t < 0.86

    lib.paint(h, "eye", band)
    lib.paint(h, "eye", lambda c: c.z > L * 0.84)
    # The root of the horn is set in metal, so nothing bright shows under the
    # collar where it tilts onto the brow.
    lib.paint(h, "metal", lambda c: c.z < 0.01)
    orient(h, HORN_BASE, HORN_AXIS)
    # A tall collar seated deep in the sloping brow, with a bead round its
    # middle.
    ring = lib.lathe([(0.0, -0.018), (0.0275, -0.018), (0.031, -0.0145), (0.031, -0.0078), (0.0326, -0.0052), (0.031, -0.0026),
                      (0.031, 0.0028), (0.0278, 0.006), (0.0, 0.006)], segments=18, role="metal", name="horn_ring")
    orient(ring, HORN_BASE, (HORN_AXIS + F).normalized())
    return [h, ring]


def ear(side):
    """A pricked ear: an open cone cupped toward the front, lined with skin."""
    prof = [(0.013, -0.004), (0.0155, 0.014), (0.0125, 0.034), (0.005, 0.05), (0.0, 0.057)]
    o = lib.lathe(prof, segments=12, start=math.radians(55), sweep=math.radians(250), scale=(1, 0.8, 1), role="body", name="ear")
    o = lib.solidify(o, 0.003, offset=1)
    inner = lib.lathe([(r * 0.82, z * 0.95 + 0.003) for r, z in prof[:-1]] + [(0.0, 0.052)], segments=10,
                      start=math.radians(115), sweep=math.radians(130), scale=(1, 0.8, 1), role="skin", name="ear_in")
    root = head(-0.006, 0.004, side * 0.025)
    rot = (math.radians(-8), side * math.radians(14), side * math.radians(-20))
    return [lib.place(o, root, rot), lib.place(inner, root, rot)]


def eyes():
    out = []
    for s in (-1, 1):
        e = lib.sphere(0.0095, segments=10, rings=6, scale=(0.55, 1.15, 0.75), role="eye", name="eye")
        m = frame(D, F).to_4x4()
        m.translation = head(0.052, 0.01, s * 0.039)
        e.data.transform(m)
        out.append(e)
    return out


def hoof(sole, up=(0, 0, 1), name="hoof"):
    """A cloven hoof with a thin accent band round the coronet, so each leg
    ends in a point of colour in both palettes."""
    prof = [(0.0, 0.0), (0.0195, 0.0), (0.0205, 0.003), (0.019, 0.014), (0.0165, 0.025), (0.0, 0.026)]
    h = lib.lathe(prof, segments=16, scale=(1, 1.18, 1), role="metal", name=name)
    band = lib.lathe([(0.0164, 0.0266), (0.019, 0.0236), (0.0177, 0.0196)], segments=16, scale=(1, 1.18, 1), role="trim", name="coronet")
    for o in (h, band):
        lib.deform(o, lambda p: Vector((p.x, p.y + 0.3 * p.z, p.z)))
    cleft = lib.box((0.0055, 0.036, 0.05), loc=(0, -0.02, 0.012), role="metal", name="cleft")
    h = lib.boolean(h, cleft)
    u = Vector(up).normalized()
    ang = math.atan2(-u.y, u.z)
    return [lib.place(o, Vector(sole), (ang, 0, 0)) for o in (h, band)]


def feather(fetlock, coronet, floor, raised=False, name="feather"):
    """Feathering: long hair round the back of the pastern. It springs
    steeply from inside the fetlock, so it grows out of the joint without a
    cuff line, and parts into six rounded locks with pointed tips that are
    longest at the heel, where they fall over the bulbs and fan out behind
    the hoof. It wraps only the sides and back: its edges sink into the
    pastern so the coat and the cloven toe show at the front. On the raised
    leg the locks hang down with gravity."""
    top, bot = Vector(fetlock), Vector(coronet)
    axis = (top - bot).normalized()
    h = (top - bot).length + 0.014
    th = 0.0035
    outer = [(0.006, 0.01), (0.011, 0.0), (0.0205, -h * 0.3), (0.0208, -h * 0.62), (0.0196, -h)]
    inner = [(0.0196 - th * 0.8, -h + th * 0.7), (0.004, 0.006)]
    start, sweep = math.radians(50), math.radians(260)
    f = lib.lathe(outer + inner, segments=24, start=start, sweep=sweep, role="cloth", name=name)
    locks = 6
    heel_len = 0.5 if raised else 1.0
    # On the lifted leg the heel faces up, so the hair drapes down instead of
    # sweeping out behind.
    flare = 0.0 if raised else 0.017
    vary = (0.86, 1.08, 0.8, 1.14, 0.92, 1.0)
    rec = []

    def comb(p):
        a = math.atan2(p.x, -p.y)
        w = min(1.0, max(0.0, ((a if a >= -1e-6 else a + math.tau) - start) / sweep))
        t = max(0.0, -p.z / h)
        back = 0.5 - 0.5 * math.cos(a)  # 0 at the toe, 1 at the heel
        q = w * locks
        k = min(int(q), locks - 1)
        frac = q - k
        edge = smoothstep(0.0, 0.14, w) * smoothstep(1.0, 0.86, w)
        bump = math.sin(math.pi * frac)  # 1 down a lock's middle, 0 in a groove
        r = (1 + 0.2 * (bump - 0.5) * t ** 0.8) * (0.72 + 0.28 * edge) * (0.92 + 0.08 * back)
        # Pointed lock tips, split deepest at the heel where they are longest.
        tip = 1 - (0.14 + 0.22 * back) * abs(2 * frac - 1) ** 1.5
        z = p.z * (0.35 + heel_len * back) * (tip * vary[k] if t > 0.5 else 1)
        rec.append((back, t))
        return Vector((p.x * r, p.y * r * 1.08 + 0.004 * back * t + flare * back * t ** 1.5, z))

    lib.deform(f, comb)
    ang = math.atan2(-axis.y, axis.z)
    lib.place(f, (0, 0, 0), (ang, 0, 0))
    lib.place(f, top, (0, 0, 0))
    if raised:
        # The hair hangs with gravity off the lifted pastern instead of
        # wrapping it like a glove.
        for v, (back, t) in zip(f.data.vertices, rec):
            v.co.z -= 0.02 * (0.5 + 0.5 * back) * t ** 2
        f.data.update()
    lo = min(v.co.z for v in f.data.vertices)
    if lo < floor:
        lib.deform(f, lambda p: Vector((p.x, p.y, p.z + (floor - lo) * max(0.0, (top.z - p.z) / (top.z - lo)) ** 2)))
    return [f]


def surface(fn, nu, nv, role="cloth", name="surface", relax=0):
    """A grid of nu x nv points fn(u, v) with u, v in [0, 1], as quads. Vertex
    i * nv + j is (u_i, v_j). `relax` passes of grid smoothing soften creases
    inside the grid while its border (the outline and the tips) stays put."""
    pts = [[Vector(fn(i / (nu - 1), j / (nv - 1))) for j in range(nv)] for i in range(nu)]
    for _ in range(relax):
        nxt = [row[:] for row in pts]
        for i in range(1, nu - 1):
            for j in range(1, nv - 1):
                avg = (pts[i - 1][j] + pts[i + 1][j] + pts[i][j - 1] + pts[i][j + 1]) / 4
                nxt[i][j] = pts[i][j].lerp(avg, 0.4)
        pts = nxt
    bm = bmesh.new()
    rows = [[bm.verts.new(pts[i][j]) for j in range(nv)] for i in range(nu)]
    for i in range(nu - 1):
        for j in range(nv - 1):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return lib.set_role(obj, role)


def solidify_taper(obj, thickness, weight, floor=0.25):
    """Solidify whose thickness follows weight(vertex index) in [0, 1], down to
    `floor` of it, so hair thins to its tips instead of ending in a slab."""
    vg = obj.vertex_groups.new(name="thick")
    for i in range(len(obj.data.vertices)):
        vg.add([i], max(0.0, min(1.0, weight(i))), "REPLACE")
    mod = obj.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.vertex_group = "thick"
    mod.thickness_vertex_group = floor
    return lib.apply_modifiers(obj)


def neck_at(u):
    """Neck sample interpolated at u in [0, 1], from the poll to the withers."""
    s = NECK_FINE[::-1]
    k = max(0.0, min(1.0, u)) * (len(s) - 1)
    i = min(int(k), len(s) - 2)
    f = k - i
    a, b = s[i], s[i + 1]
    return tuple(x * (1 - f) + y * f for x, y in zip(a, b))


NECK_LEN = 0.27  # crest length, poll to withers
NECK_R = 0.055  # arc length per radian around the neck


def on_neck(side, u, th, off):
    """A point `off` above the neck surface at u (0 poll .. 1 withers) and
    angle th round the section (0 on the crest, pi/2 on the `side` flank)."""
    p, a, back, w, d = neck_at(u)
    back = back.normalized()
    return p + back * (d + off) * math.cos(th) + X * side * (w + off) * math.sin(th)


# The mane sheet: clump boundaries along the neck (uneven widths), and per
# clump its tip length, flick back along the neck and curl off the coat.
CLUMPS = (0.0, 0.14, 0.31, 0.46, 0.63, 0.79, 1.0)
CLUMP_LEN = (0.9, 1.08, 0.94, 1.14, 0.98, 0.84)
CLUMP_FLICK = (0.02, -0.03, 0.035, -0.015, 0.03, 0.0)
CLUMP_CURL = (0.006, 0.003, 0.007, 0.004, 0.006, 0.003)
SHEET = {"th0": -0.45, "th1": 1.32, "notch": 0.5, "puff": 0.02}


def mane_sheet(side, u0, u1, sweep, wave, per=8, nv=13, name="mane"):
    """The body of the mane: one continuous layer of hair that springs from
    inside the crest, rises over it and falls down the right side of the
    neck in a soft S. Over its lower part it parts into pointed clumps of
    uneven width and length, each crowned, split into two strands by a
    shallow groove and flicked off the coat at its tip. Its ends and the
    notches between the clumps tuck into the coat, so no slab edge shows."""
    n = len(CLUMP_LEN)
    nu = n * per + 1
    th0, th1, notch, puff = SHEET["th0"], SHEET["th1"], SHEET["notch"], SHEET["puff"]

    def fn(u, v):
        j = round(u * (nu - 1))
        k = min(j // per, n - 1)
        frac = (j - k * per) / per
        a = CLUMPS[k] + (CLUMPS[k + 1] - CLUMPS[k]) * frac
        shape = math.sin(math.pi * frac) ** 0.5  # 1 down a clump's middle, 0 in the notch
        point = math.sin(math.pi * frac) ** 1.6  # a narrower profile for the tip
        end = th1 * CLUMP_LEN[k] - notch * (1 - point)
        th = th0 + (end - th0) * v
        un = (u0 + (u1 - u0) * a + sweep * v ** 1.4 + wave * math.sin(math.pi * v) * v
              + CLUMP_FLICK[k] * smoothstep(0.55, 1.0, v) * shape)
        sep = smoothstep(0.3, 0.8, v)  # the clumps part toward their tips
        strands = 0.5 - 0.5 * math.cos(frac * math.tau * 2)
        lift = 0.016 + puff * (1 - v) ** 1.5 + 0.008 * shape * (0.35 + 0.65 * sep) + 0.004 * strands * sep
        ends = smoothstep(0.0, 0.05, a) * smoothstep(1.0, 0.93, a)
        off = -0.012 + lift * smoothstep(0.0, 0.15, v) * ends
        off += CLUMP_CURL[k] * smoothstep(0.7, 1.0, v) ** 2 * point - 0.008 * smoothstep(0.75, 1.0, v) * (1 - shape)
        return on_neck(side, un, th, off)

    m = surface(fn, nu, nv, name=name, relax=1)
    return solidify_taper(m, 0.004, lambda i: 1 - smoothstep(0.55, 1.0, (i % nv) / (nv - 1)))


def crest_strip(side, u0, u1, th1, name="mane"):
    """A thin strip of mane along the crest on the far side, its edge softly
    scalloped."""

    def fn(s, t):
        u = u0 + (u1 - u0) * t
        edge = th1 * (0.85 + 0.15 * math.sin(t * math.pi * 4) ** 2) * min(1.0, 0.4 + 2.5 * min(t, 1 - t))
        th = -0.4 + (edge + 0.4) * s
        off = -0.006 + 0.012 * smoothstep(0.0, 0.25, s) * (1 - 0.5 * s)
        return on_neck(side, u, th, off)

    m = surface(fn, 4, 14, name=name)
    return lib.solidify(m, 0.005)


def mane():
    out = []
    # The mane falls down the right side of the neck from poll to withers:
    # a continuous sheet over the crest that parts into clumps.
    out.append(mane_sheet(-1, 0.0, 0.9, 0.12, 0.05, name="mane_r"))
    out.append(crest_strip(1, 0.0, 0.95, 0.5, name="mane_l"))
    # Forelock: three locks spring from the poll behind the horn collar. The
    # long one passes outside the collar and falls forward across the right
    # brow beside the horn; the two short ones sweep off to the right temple.
    pts = [head(-0.036, 0.0, 0.004), head(-0.012, 0.034, -0.012), head(0.022, 0.036, -0.038), head(0.05, 0.03, -0.043),
           head(0.076, 0.022, -0.03)]
    out.append(lock(pts, [0.01, 0.011, 0.009, 0.0065, 0.002], lambda c: Vector(c) - head(0.05, -0.01), flat=0.6, sides=6, name="forelock"))
    for x0, du, dv in ((-0.004, -0.008, 0.004), (0.014, 0.006, -0.004)):
        pts = [head(-0.036, 0.0, x0), head(-0.016 + du * 0.5, 0.034, x0 * 0.4 - 0.016), head(0.006 + du, 0.034 + dv, -0.036),
               head(0.026 + du, 0.016 + dv, -0.045)]
        out.append(lock(pts, [0.01, 0.011, 0.008, 0.0025], lambda c: Vector(c) - head(0.05, -0.01), flat=0.6, sides=6, name="forelock"))
    return out


def tail():
    """The tail springs from a dock under the croup and falls in a soft S:
    five thick locks that fan apart toward the hem, end at different heights
    and taper to fine tips that flick out."""
    core = [Vector(p) for p in [(0, 0.19, 0.375), (0, 0.228, 0.37), (0, 0.262, 0.34), (0, 0.285, 0.285), (0, 0.297, 0.21),
                                (0, 0.302, 0.14), (0, 0.314, 0.08)]]
    core_s = catmull(core, 40)

    def radial(c):
        q = Vector(c)
        p = min(core_s, key=lambda s: (s - q).length_squared)
        return q - p

    out = []
    angles = [0.3, 1.55, 2.8, 4.05, 5.3]
    for k, ang in enumerate(angles):
        pts = []
        out_dir = Vector((math.cos(ang), math.sin(ang) * 0.7, 0))
        for i, c in enumerate(core):
            t = i / (len(core) - 1)
            # The locks fan apart toward the hem and end at different heights.
            spread = 0.004 + 0.042 * t ** 0.8
            sway = -0.025 * math.sin(t * math.pi * 1.1)
            drop = -0.05 * t * ((k * 37) % 5) / 4
            pts.append(c + out_dir * spread + Vector((sway, 0, drop)))
        # A tapering tip that flicks out.
        end, prev = pts[-1], pts[-2]
        dirn = (end - prev).normalized()
        pts.append(end + dirn * 0.02 + out_dir * 0.008 + Vector((0, 0.006, 0.0)))
        pts.append(end + dirn * 0.032 + out_dir * 0.018 + Vector((0, 0.012, 0.008)))
        radii = [0.017, 0.026, 0.03, 0.03, 0.027, 0.021, 0.014, 0.007, 0.0015]
        out.append(lock(pts, radii, radial, flat=0.75, resolution=2, name="tail"))
    return out


def build():
    groups = {"coat": [coat()], "horn": horn(), "ears": ear(-1) + ear(1), "eyes": eyes(),
              "mane": mane(), "tail": tail(), "hooves": [], "feathers": []}
    for name, leg in LEGS.items():
        fet, cor = Vector(leg[7][0]), Vector(leg[9][0])
        if name == "fore_r":
            up = (fet - cor).normalized()
            groups["hooves"] += hoof(cor - up * 0.022, up=up)
            groups["feathers"] += feather(fet, cor, 0.0, raised=True)
        else:
            groups["hooves"] += hoof((cor.x, cor.y - 0.006, 0.0))
            groups["feathers"] += feather(fet, cor, 0.006)
    if DEBUG:
        for g, objs in groups.items():
            print("TRIS", g, sum(lib.tris(o) for o in objs))
    return [o for objs in groups.values() for o in objs]
