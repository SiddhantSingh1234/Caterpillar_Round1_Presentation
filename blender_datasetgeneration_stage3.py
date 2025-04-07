import bpy
import os
import random
import math
import mathutils

# ---------- CONFIG ----------
output_dir = "E:/Caterpillar_Latest_Solution/depth_dataset/stage3"
img_dir = os.path.join(output_dir, "images")
depth_dir = os.path.join(output_dir, "depths")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

num_images = 50
image_res = (512, 512)

# ---------- CLEANUP ----------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------- SCENE SETTINGS ----------
scene = bpy.context.scene
scene.render.resolution_x = image_res[0]
scene.render.resolution_y = image_res[1]
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("World")

# ---------- CAMERA ----------
camera = bpy.data.cameras.new("Camera")
camera_obj = bpy.data.objects.new("Camera", camera)
bpy.context.collection.objects.link(camera_obj)
scene.camera = camera_obj
camera_obj.location = (0, 0, 5)

# ---------- LIGHTING ----------
def setup_lighting():
    key_light = bpy.data.lights.new(name="KeyLight", type='SUN')
    key_light.energy = 5.0
    key_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    bpy.context.collection.objects.link(key_obj)

    fill_light = bpy.data.lights.new(name="FillLight", type='POINT')
    fill_light.energy = 2.0
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light)
    fill_obj.location = (-3, -3, 3)
    bpy.context.collection.objects.link(fill_obj)

setup_lighting()

# ---------- MATERIAL ----------
def create_random_material():
    mat = bpy.data.materials.new(name="RandomMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    # Random color or simple texture
    if random.random() < 0.5:
        bsdf.inputs[0].default_value = (
            random.uniform(0.1, 0.5),
            random.uniform(0.1, 0.5),
            random.uniform(0.1, 0.5),
            1
        )
    else:
        tex = nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = random.uniform(5, 20)
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Metallic"].default_value = random.uniform(0.0, 0.5)
    bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.8)
    return mat

# ---------- OBJECTS ----------
def add_random_object_at(location):
    object_types = ['sphere', 'cube', 'cylinder', 'cone', 'torus']
    choice = random.choice(object_types)

    if choice == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=random.uniform(0.8, 1.2),
            location=location,
            segments=32,
            ring_count=16
        )
    elif choice == 'cube':
        bpy.ops.mesh.primitive_cube_add(
            size=random.uniform(1.0, 1.5),
            location=location
        )
    elif choice == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(
            radius=random.uniform(0.6, 1.0),
            depth=random.uniform(1.0, 2.0),
            location=location,
            vertices=32
        )
    elif choice == 'cone':
        bpy.ops.mesh.primitive_cone_add(
            radius1=random.uniform(0.8, 1.2),
            depth=random.uniform(1.5, 2.5),
            location=location,
            vertices=32
        )
    elif choice == 'torus':
        bpy.ops.mesh.primitive_torus_add(
            major_radius=random.uniform(0.8, 1.2),
            minor_radius=random.uniform(0.2, 0.4),
            location=location,
            major_segments=48,
            minor_segments=16
        )

    obj = bpy.context.active_object
    obj.name = f"Object_{choice}"
    obj.rotation_euler = (
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2)
    )
    obj.data.materials.append(create_random_material())
    return obj

def add_multiple_random_objects(min_count=3, max_count=5):
    objects = []
    count = random.randint(min_count, max_count)
    for _ in range(count):
        loc = (
            random.uniform(-2.5, 2.5),
            random.uniform(-2.5, 2.5),
            0
        )
        obj = add_random_object_at(loc)
        objects.append(obj)
    return objects

# ---------- SKY BACKGROUND ----------
def set_sky_background():
    world = bpy.context.scene.world
    world.use_nodes = True
    tree = world.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    bg = nodes.new(type='ShaderNodeBackground')
    sky = nodes.new(type='ShaderNodeTexSky')
    output = nodes.new(type='ShaderNodeOutputWorld')

    sky.sun_elevation = random.uniform(0.1, 1.0)
    sky.sun_rotation = random.uniform(0.0, 6.28)
    sky.turbidity = random.uniform(2.0, 6.0)

    links.new(sky.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])

def set_random_background():
    if random.random() < 0.5:
        set_sky_background()
    else:
        world = bpy.context.scene.world
        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()

        bg = tree.nodes.new(type='ShaderNodeBackground')
        output = tree.nodes.new(type='ShaderNodeOutputWorld')
        bg.inputs[0].default_value = (
            random.uniform(0.2, 0.6),
            random.uniform(0.2, 0.6),
            random.uniform(0.2, 0.6),
            1
        )
        tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# ---------- DEPTH SETUP ----------
def enable_depth_output():
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    scene.view_layers["ViewLayer"].use_pass_z = True

    rl = tree.nodes.new(type='CompositorNodeRLayers')
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    invert = tree.nodes.new(type="CompositorNodeInvert")
    comp = tree.nodes.new('CompositorNodeComposite')

    depth_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_output.base_path = depth_dir
    depth_output.format.file_format = 'PNG'
    depth_output.format.color_mode = 'BW'
    depth_output.format.color_depth = '16'

    rgb_output = tree.nodes.new('CompositorNodeOutputFile')
    rgb_output.base_path = img_dir
    rgb_output.format.file_format = 'PNG'

    tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
    tree.links.new(rl.outputs['Depth'], normalize.inputs[0])
    # tree.links.new(normalize.outputs[0], invert.inputs[1])
    # tree.links.new(invert.outputs[0], depth_output.inputs[0])
    tree.links.new(normalize.outputs[0], depth_output.inputs[0])
    tree.links.new(rl.outputs['Image'], rgb_output.inputs[0])

    return depth_output, rgb_output

# ---------- MAIN LOOP ----------
depth_output, rgb_output = enable_depth_output()

for i in range(num_images):
    # Clear previous mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Add random objects
    objects = add_multiple_random_objects()

    # Set random background
    set_random_background()

    # Position camera at random angle
    distance = random.uniform(3.5, 4.5)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.pi/6, math.pi/3)

    cam_x = distance * math.sin(phi) * math.cos(theta)
    cam_y = distance * math.sin(phi) * math.sin(theta)
    cam_z = distance * math.cos(phi)
    camera_obj.location = (cam_x, cam_y, cam_z)

    # Look at center (0,0,0)
    direction = mathutils.Vector((0, 0, 0)) - camera_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()

    # Set output names
    depth_output.file_slots[0].path = f"depth_{i:04d}"
    rgb_output.file_slots[0].path = f"rgb_{i:04d}"

    # Render
    bpy.ops.render.render(write_still=True)

print("[✓] Multi-object dataset with sky/textured backgrounds generated!")