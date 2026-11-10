"""Procedural puffer coat on a 1.68 m mannequin.
blender -b -P build.py -- <long|short> <res%> <samples> <outdir> [views] [--save]"""
import bpy, bmesh, math, sys, os
import numpy as np
from mathutils import Vector, noise

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
VARIANT = argv[0] if len(argv) > 0 else "long"
PCT = int(argv[1]) if len(argv) > 1 else 50
SAMPLES = int(argv[2]) if len(argv) > 2 else 32
OUTDIR = argv[3] if len(argv) > 3 else "/tmp/out"
VIEWS = argv[4].split(",") if len(argv) > 4 and argv[4] else ["front", "side", "back"]
SAVE = "--save" in argv
HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    # long: 10 cm above the knee (knee ~0.47 m for 1.68 m), stand collar, drawstring hem
    "long": dict(hem=0.57, color="#EFE6C8", hood=False, collar_h=0.06, cords=True,
                 snaps=[0.66, 0.78, 0.90, 1.02, 1.14, 1.26, 1.38], pocket=((0.10, 0.92), (0.19, 1.05))),
    # short: to the hip bones, hood, banded hem; same block as the long one
    "short": dict(hem=0.90, color="#C8141F", hood=True, collar_h=0.05, cords=False,
                  snaps=[1.0, 1.12, 1.24, 1.36], pocket=((0.10, 0.99), (0.19, 1.11))),
}
V = VARIANTS[VARIANT]
V["collar_h"] = float(os.environ.get("COLLAR_H", V["collar_h"]))
HEM = V["hem"]
NECK_Z = 1.47

# Keep in sync with mannequin.py
P0X, P0Z, ABD, ARM_DY = 0.175, 1.335, math.radians(22), -0.03


def lin(hexs):
    h = hexs.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c) + (1.0,)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def spline(keys, x):
    xs = [k[0] for k in keys]
    n = len(keys)
    if x <= xs[0]:
        return list(keys[0][1])
    if x >= xs[-1]:
        return list(keys[-1][1])
    i = 0
    while xs[i + 1] < x:
        i += 1
    dim = len(keys[0][1])

    def tan(j):
        a, b = max(j - 1, 0), min(j + 1, n - 1)
        return [(keys[b][1][k] - keys[a][1][k]) / (xs[b] - xs[a]) for k in range(dim)]
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    m0, m1 = tan(i), tan(i + 1)
    h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
    return [h00*keys[i][1][k] + h10*h*m0[k] + h01*keys[i+1][1][k] + h11*h*m1[k] for k in range(dim)]


# ---------------------------------------------------------------- torso
# z: (half width, half depth, y centre, superellipse exponent)
TORSO_KEYS = [
    (0.50, (.285, .185, .012, 2.6)),
    (0.70, (.272, .178, .010, 2.6)),
    (0.90, (.258, .170, .006, 2.7)),
    (1.10, (.250, .166, .000, 2.8)),
    (1.25, (.255, .165, -.004, 2.8)),
    (1.33, (.258, .158, .000, 2.7)),
    (1.38, (.245, .142, .004, 2.5)),
    (1.42, (.205, .122, .008, 2.3)),
    (1.45, (.150, .105, .010, 2.1)),
    (1.47, (.100, .092, .012, 2.0)),
]


def torso_base(z, psi):
    a, b, cy, n = spline(TORSO_KEYS, z)
    s, c = math.sin(psi), math.cos(psi)
    r = (abs(s / a) ** n + abs(c / b) ** n) ** (-1 / n)
    return r * s, cy - r * c, cy


def torso_disp(z, psi, x):
    front = math.cos(psi) > 0
    top_fade = smoothstep(NECK_Z, 1.36, z)
    amp = (0.0025 + 0.010 * smoothstep(1.25, 0.62, z)) * top_fade
    near_plk = math.exp(-(x / 0.05) ** 2) if front else 0.0
    cs, sn = math.cos(psi), math.sin(psi)
    d = amp * 1.4 * noise.noise(Vector((cs * 2.6 + 3.1, sn * 2.6 + 7.7, z * 1.4))) * (1 - 0.75 * near_plk)
    d += 0.0035 * noise.noise(Vector((cs * 7 + 11, sn * 7 + 5, z * 7))) * top_fade
    if front:  # placket with edge seams
        ax = abs(x)
        d += 0.0035 * (1 - smoothstep(0.026, 0.032, ax))
        d -= 0.0022 * math.exp(-((ax - 0.031) / 0.0035) ** 2)
    dz = z - HEM
    if V["cords"]:  # drawstring channel, gathered
        zone = 1 - smoothstep(0.02, 0.045, dz)
        d += zone * (-0.010 + 0.0035 * math.sin(psi * 38))
        d -= 0.0022 * math.exp(-((dz - 0.045) / 0.003) ** 2)
    else:  # hem band
        zone = 1 - smoothstep(0.04, 0.05, dz)
        d += zone * (-0.008 + 0.0015 * math.sin(psi * 30))
        d -= 0.0022 * math.exp(-((dz - 0.047) / 0.003) ** 2)
    d -= 0.006 * (1 - min(dz / 0.008, 1)) ** 2
    return d


def torso_point(z, psi):
    x, y, cy = torso_base(z, psi)
    rad = Vector((x, y - cy, 0)).normalized()
    return Vector((x, y, z)) + rad * torso_disp(z, psi, x)


def torso_normal(z, psi):
    e = 1e-3
    n = (torso_point(z, psi + e) - torso_point(z, psi - e)).cross(torso_point(z + e, psi) - torso_point(z - e, psi))
    return n.normalized()


def psi_for_x(z, x):
    lo, hi = 0.0, math.pi / 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if torso_base(z, mid)[0] < abs(x):
            lo = mid
        else:
            hi = mid
    return math.copysign(lo, x)


# ---------------------------------------------------------------- sleeves
SLEEVE_KEYS = [(0.0, (.080,)), (0.06, (.093,)), (0.15, (.093,)), (0.30, (.088,)), (0.42, (.084,)),
               (0.49, (.083,)), (0.515, (.072,)), (0.53, (.056,)), (0.545, (.050,)), (0.565, (.049,))]


def arm_frame(side):
    p0 = Vector((P0X * side, 0, P0Z))
    d = Vector((math.sin(ABD) * side, ARM_DY, -math.cos(ABD))).normalized()
    u = Vector((0, 1, 0)).cross(d).normalized()
    return p0, d, u, d.cross(u).normalized()


def sleeve_point(side, t, phi):
    p0, d, u, v = arm_frame(side)
    r = spline(SLEEVE_KEYS, t)[0]
    dirv = u * math.cos(phi) + v * math.sin(phi)
    sx, sy, sz = (17, 3, 9) if side > 0 else (4, 21, 13)
    cs, sn = math.cos(phi), math.sin(phi)
    dd = 0.006 * noise.noise(Vector((cs * 1.8 + sx, sn * 1.8 + sy, t * 8 + sz))) * smoothstep(0, 0.08, t) * (1 - smoothstep(0.5, 0.53, t))
    dd += 0.004 * noise.noise(Vector((cs * 1.2 + sy, sn * 1.2 + sz, t * 28 + sx))) * smoothstep(0.18, 0.30, t) * (1 - smoothstep(0.42, 0.5, t))
    if t > 0.5:
        dd += 0.0025 * math.cos(phi * 22) * smoothstep(0.5, 0.525, t)
    dd -= 0.002 * math.exp(-((t - 0.528) / 0.003) ** 2)
    return p0 + d * t + dirv * (r + dd)


# ---------------------------------------------------------------- mesh helpers
def link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def grid_mesh(name, rows, closed=True):
    bm = bmesh.new()
    vs = [[bm.verts.new(p) for p in row] for row in rows]
    m = len(rows[0])
    for i in range(len(rows) - 1):
        for j in (range(m) if closed else range(m - 1)):
            j2 = (j + 1) % m
            try:
                bm.faces.new((vs[i][j], vs[i][j2], vs[i + 1][j2], vs[i + 1][j]))
            except ValueError:
                pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return link(bpy.data.objects.new(name, me))


def orient(ob, ref):
    """Flip faces so normals point away from ref(point)."""
    score = sum(1 if p.normal.dot(p.center - ref(p.center)) > 0 else -1 for p in ob.data.polygons)
    if score < 0:
        bm = bmesh.new(); bm.from_mesh(ob.data)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(ob.data); bm.free()


def padded(ob, thick, mat, subsurf=1):
    sol = ob.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness, sol.offset, sol.use_rim, sol.use_even_offset = thick, -1, True, True
    if subsurf:
        sub = ob.modifiers.new("Subsurf", "SUBSURF")
        sub.levels = sub.render_levels = subsurf
    ob.data.materials.append(mat)
    return ob


# ---------------------------------------------------------------- materials
def principled(name, hexc, rough=0.5, sheen=0.0, metal=0.0, bump=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = lin(hexc)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Sheen Roughness"].default_value = 0.35
    if bump:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 900
        tex.inputs["Detail"].default_value = 2
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.0005
        nt.links.new(tc.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    return m


# ---------------------------------------------------------------- build
def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fabric = principled("Polyester_" + VARIANT, V["color"], rough=0.42, sheen=0.35 if VARIANT == "long" else 0.12, bump=0.05)
    metal = principled("Snap_metal", "#8E8D8A", rough=0.3, metal=1.0)
    cord_m = principled("Cord", V["color"], rough=0.7, sheen=0.5)
    skin = principled("Mannequin", "#A3A3A3", rough=0.55)

    # mannequin
    dat = np.load(os.path.join(HERE, "mannequin.npz"))
    me = bpy.data.meshes.new("Mannequin")
    me.from_pydata(dat["v"].tolist(), [], dat["f"].tolist())
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    man = link(bpy.data.objects.new("Mannequin", me))
    man.data.materials.append(skin)
    sm = man.modifiers.new("Smooth", "CORRECTIVE_SMOOTH")
    sm.iterations, sm.use_only_smooth = 4, True

    # torso
    NP = 240
    zs = [HEM + i * 0.004 for i in range(int((NECK_Z - HEM) / 0.004) + 1)]
    zs[-1] = NECK_Z
    rows = [[torso_point(z, 2 * math.pi * j / NP) for j in range(NP)] for z in zs]
    torso = grid_mesh("Coat_body", rows)
    orient(torso, lambda c: Vector((0, 0.005, c.z)))
    padded(torso, 0.011, fabric)

    # sleeves
    for side in (1, -1):
        ts = [i * 0.004 for i in range(int(0.565 / 0.004) + 1)]
        rows = [[sleeve_point(side, t, 2 * math.pi * j / 120) for j in range(120)] for t in ts]
        sl = grid_mesh("Sleeve_" + ("L" if side > 0 else "R"), rows)
        p0, d, _, _ = arm_frame(side)
        orient(sl, lambda c, p0=p0, d=d: p0 + d * (c - p0).dot(d))
        padded(sl, 0.010, fabric)

    # stand collar (open at the front)
    h, g = V["collar_h"], 0.12
    rows = []
    for i in range(int(h / 0.004) + 1):
        z = 1.455 + min(i * 0.004, h)
        s = (z - 1.455) / h
        k = 1 + 0.10 * s ** 1.5
        row = []
        for j in range(151):
            th = g + (2 * math.pi - 2 * g) * j / 150
            x, y = 0.100 * k * math.sin(th), 0.012 - 0.092 * k * math.cos(th)
            rad = Vector((x, y - 0.012, 0)).normalized()
            row.append(Vector((x, y, z)) + rad * 0.003 * noise.noise(Vector((math.cos(th) * 3 + 8, math.sin(th) * 3 + 8, z * 10))))
        rows.append(row)
    col = grid_mesh("Collar", rows, closed=False)
    orient(col, lambda c: Vector((0, 0.012, c.z)))
    padded(col, 0.012, fabric)

    # hood
    if V["hood"]:
        C, R = Vector((0, 0.03, 1.55)), Vector((0.135, 0.16, 0.165))
        f = Vector((0, -1, 0.30)).normalized()
        e1 = Vector((1, 0, 0)); e2 = f.cross(e1).normalized()
        a0 = math.radians(58)
        rows = []
        for i in range(71):
            al = a0 + (math.pi - a0) * i / 70
            row = []
            for j in range(120):
                be = 2 * math.pi * j / 120
                dv = f * math.cos(al) + (e1 * math.cos(be) + e2 * math.sin(be)) * math.sin(al)
                p = Vector((C.x + R.x * dv.x, C.y + R.y * dv.y, C.z + R.z * dv.z))
                row.append(p + dv * 0.004 * noise.noise(dv * 3 + Vector((5, 1, 9))))
            rows.append(row)
        hood = grid_mesh("Hood", rows)
        orient(hood, lambda c: C)
        padded(hood, 0.02, fabric)

    # snaps on the placket
    for z in V["snaps"]:
        p, n = torso_point(z, 0.0), torso_normal(z, 0.0)
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0085, depth=0.004, location=p + n * 0.003)
        sn = bpy.context.object
        sn.rotation_mode = "QUATERNION"; sn.rotation_quaternion = n.to_track_quat("Z", "Y")
        sn.name = "Snap"
        bv = sn.modifiers.new("Bevel", "BEVEL"); bv.width, bv.segments = 0.0015, 3
        sn.data.materials.append(metal)
        bpy.ops.object.shade_smooth()

    # slanted welt pockets
    (x1, z1), (x2, z2) = V["pocket"]
    for side in (1, -1):
        rows = [[], []]
        for k in range(41):
            t = k / 40
            x, z = (x1 + (x2 - x1) * t) * side, z1 + (z2 - z1) * t
            ps = psi_for_x(z, x)
            p, n = torso_point(z, ps), torso_normal(z, ps)
            tang = Vector(((x2 - x1) * side, 0, z2 - z1)).normalized()
            w = n.cross(tang).normalized()
            rows[0].append(p + n * 0.002 - w * 0.011)
            rows[1].append(p + n * 0.002 + w * 0.011)
        wl = grid_mesh("Welt", rows, closed=False)
        orient(wl, lambda c: Vector((0, 0.005, c.z)))
        padded(wl, 0.004, fabric)

    # drawstring cords
    if V["cords"]:
        for side in (1, -1):
            zc = HEM + 0.022
            ps = psi_for_x(zc, 0.036 * side)
            p, n = torso_point(zc, ps), torso_normal(zc, ps)
            p = p + n * 0.004
            pts = [p, p + Vector((0.004 * side, -0.012, -0.07)), p + Vector((0.008 * side, -0.010, -0.17))]
            cu = bpy.data.curves.new("Cord", "CURVE"); cu.dimensions = "3D"
            sp = cu.splines.new("NURBS"); sp.points.add(len(pts) - 1)
            for q, pt in zip(sp.points, pts):
                q.co = (*pt, 1)
            sp.use_endpoint_u = True; sp.order_u = 3
            cu.bevel_depth, cu.bevel_resolution = 0.0024, 3
            cord = link(bpy.data.objects.new("Cord", cu)); cord.data.materials.append(cord_m)
            for loc, rad, dep, mat in ((pts[-1] + Vector((0, 0, 0.03)), 0.0065, 0.016, metal),
                                       (pts[-1] + Vector((0, 0, 0.004)), 0.0035, 0.012, metal)):
                bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=rad, depth=dep, location=loc)
                bpy.context.object.data.materials.append(mat); bpy.ops.object.shade_smooth()
            bpy.ops.mesh.primitive_torus_add(major_radius=0.0055, minor_radius=0.0015, location=p)
            ey = bpy.context.object
            ey.rotation_mode = "QUATERNION"; ey.rotation_quaternion = n.to_track_quat("Z", "Y")
            ey.data.materials.append(metal)


def stage():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1200, 1600, PCT
    sc.view_settings.view_transform = "AgX" if VARIANT == "long" else "Standard"
    sc.view_settings.look = "AgX - Medium High Contrast" if VARIANT == "long" else "None"
    sc.view_settings.exposure = 0.0 if VARIANT == "long" else -0.6
    w = bpy.data.worlds.new("World"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.87, 0.9, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25

    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.is_shadow_catcher = True

    rig = link(bpy.data.objects.new("Rig", None))
    cam_d = bpy.data.cameras.new("Cam"); cam_d.sensor_fit = "VERTICAL"; cam_d.sensor_height = 24; cam_d.lens = 84
    cam = link(bpy.data.objects.new("Cam", cam_d)); cam.location = (0, -7, 0.86); cam.rotation_euler = (math.pi / 2, 0, 0)
    cam.parent = rig; sc.camera = cam

    def area(name, loc, energy, size):
        ld = bpy.data.lights.new(name, "AREA"); ld.energy, ld.size = energy, size
        ob = link(bpy.data.objects.new(name, ld)); ob.location = loc
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y")
        ob.parent = rig
    area("Key", (-2.6, -3.6, 3.2), 520, 3.0)
    area("Fill", (3.2, -3.0, 1.8), 160, 4.0)
    area("Rim", (0.5, 4.0, 3.0), 380, 3.0)
    return sc, rig


build()
sc, rig = stage()
os.makedirs(OUTDIR, exist_ok=True)
if SAVE:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTDIR, f"puffer_{VARIANT}.blend"))
for view in VIEWS:
    rig.rotation_euler = (0, 0, {"front": 0, "side": math.pi / 2, "back": math.pi}[view])
    sc.render.filepath = os.path.join(OUTDIR, f"{VARIANT}_{view}.png")
    bpy.ops.render.render(write_still=True)
