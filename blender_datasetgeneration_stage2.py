# Blender Python Script to Generate Synthetic RGB and Depth Dataset with Single Objects
# Save as generate_dataset.py and run with: blender --background --python generate_dataset.py

import bpy
import os
import random
import math
import mathutils

# ---------- Config ----------
output_dir = "E:\Caterpillar_Latest_Solution\depth_dataset\stage2"
img_dir = os.path.join(output_dir, "images")
depth_dir = os.path.join(output_dir, "depths")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

num_images = 50
image_res = (512, 512)  # Higher resolution for better quality
bpy.context.scene.render.resolution_x = image_res[0]
bpy.context.scene.render.resolution_y = image_res[1]
bpy.context.scene.render.film_transparent = False  # Set to False for solid background

# ---------- Cleanup ----------
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create a new world
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world

# ---------- Camera ----------
camera = bpy.data.cameras.new("Camera")
camera_obj = bpy.data.objects.new("Camera", camera)
bpy.context.collection.objects.link(camera_obj)
bpy.context.scene.camera = camera_obj
camera_obj.location = (0, 0, 5)

# ---------- Lighting ----------
def setup_lighting():
    # Create a key light
    key_light = bpy.data.lights.new(name="KeyLight", type='SUN')
    key_light.energy = 5.0
    key_light_obj = bpy.data.objects.new(name="KeyLight", object_data=key_light)
    bpy.context.collection.objects.link(key_light_obj)
    key_light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Create a fill light
    fill_light = bpy.data.lights.new(name="FillLight", type='POINT')
    fill_light.energy = 2.0
    fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light)
    bpy.context.collection.objects.link(fill_light_obj)
    fill_light_obj.location = (-3, -3, 3)

setup_lighting()

# ---------- Background ----------
def set_background():
    bpy.context.scene.world.use_nodes = True
    tree = bpy.context.scene.world.node_tree
    bg = tree.nodes['Background']
    # Set a light gray background for better contrast with objects
    bg.inputs[0].default_value = (0.9, 0.9, 0.9, 1)

set_background()

# ---------- Add Object ----------
def random_color_material():
    mat = bpy.data.materials.new(name="ObjectMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # Generate vibrant colors for better visibility
    r = random.uniform(0.05, 0.3)
    g = random.uniform(0.05, 0.3)
    b = random.uniform(0.05, 0.3)
    bsdf.inputs[0].default_value = (r, g, b, 1)
    # Add some metallic and roughness variation
    bsdf.inputs["Metallic"].default_value = random.uniform(0.1, 0.5)
    bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.8)
    return mat

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
    obj.data.materials.append(random_color_material())
    return obj

def add_two_random_objects():
    # Place objects at slight offsets so they don't overlap
    offset = random.uniform(1.5, 2.5)
    loc1 = (offset, 0, 0)
    loc2 = (-offset, 0, 0)
    
    obj1 = add_random_object_at(loc1)
    obj2 = add_random_object_at(loc2)
    
    return [obj1, obj2]

# ---------- Depth Setup ----------
def enable_depth_output():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Enable depth pass in render layers
    scene.view_layers["ViewLayer"].use_pass_z = True
    
    rl = tree.nodes.new(type='CompositorNodeRLayers')
    
    # Create a normalize node for depth
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    
    # Create an invert node to get white=far, black=near
    invert = tree.nodes.new(type="CompositorNodeInvert")
    
    comp = tree.nodes.new('CompositorNodeComposite')
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.base_path = depth_dir
    file_output.format.file_format = 'PNG'
    file_output.format.color_mode = 'BW'
    file_output.format.color_depth = '16'  # 16-bit for better depth precision
    
    # Connect nodes
    tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
    
    # For depth output, use the Z output
    tree.links.new(rl.outputs['Depth'], normalize.inputs[0])
    # tree.links.new(normalize.outputs[0], invert.inputs[1])
    # tree.links.new(invert.outputs[0], file_output.inputs[0])
    tree.links.new(normalize.outputs[0], file_output.inputs[0])
    
    # Create a separate output for RGB images
    rgb_output = tree.nodes.new('CompositorNodeOutputFile')
    rgb_output.base_path = img_dir
    rgb_output.format.file_format = 'PNG'
    tree.links.new(rl.outputs['Image'], rgb_output.inputs[0])

    return file_output, rgb_output

# ---------- Main Loop ----------
depth_output, rgb_output = enable_depth_output()

for i in range(num_images):
    # Clear previous objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Add a single random object
    obj = add_two_random_objects()
    
    # Randomize camera position for different viewpoints
    # Keep distance relatively constant for consistent depth maps
    distance = random.uniform(3.5, 4.5)
    theta = random.uniform(0, math.pi*2)
    phi = random.uniform(math.pi/6, math.pi/3)  # Limit vertical angle
    
    cam_x = distance * math.sin(phi) * math.cos(theta)
    cam_y = distance * math.sin(phi) * math.sin(theta)
    cam_z = distance * math.cos(phi)
    
    camera_obj.location = (cam_x, cam_y, cam_z)
    
    # Point camera at the object
    # Convert tuple to Vector for proper subtraction
    direction = mathutils.Vector((0, 0, 0))  # Object is at origin
    camera_obj.rotation_euler = (0, 0, 0)
    looking_direction = direction - camera_obj.location
    rot_quat = looking_direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()
    
    # Set output paths
    depth_output.file_slots[0].path = f"depth_{i:04d}"
    rgb_output.file_slots[0].path = f"rgb_{i:04d}"

    # Render
    bpy.ops.render.render(write_still=True)

print("[✓] Dataset generation with single objects complete!")