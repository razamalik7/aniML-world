"""Headless Blender script to inspect deer_standing.blend object transforms."""
import bpy
import math

print("\n" + "=" * 70)
print("BLENDER SCENE INSPECTION")
print("=" * 70)

for obj in bpy.data.objects:
    print(f"\nObject: '{obj.name}'  Type: {obj.type}")
    print(f"  Location:    ({obj.location.x:.4f}, {obj.location.y:.4f}, {obj.location.z:.4f})")
    rot = obj.rotation_euler
    print(f"  Rotation:    ({math.degrees(rot.x):.2f}, {math.degrees(rot.y):.2f}, {math.degrees(rot.z):.2f}) deg  order={rot.order}")
    print(f"  Scale:       ({obj.scale.x:.4f}, {obj.scale.y:.4f}, {obj.scale.z:.4f})")
    if obj.parent:
        print(f"  Parent:      '{obj.parent.name}'")
    else:
        print(f"  Parent:      None")

    if obj.type == 'ARMATURE':
        arm = obj.data
        print(f"  Bones ({len(arm.bones)}):")
        for bone in arm.bones:
            h = bone.head_local
            t = bone.tail_local
            print(f"    {bone.name:20s}  head=({h.x:.4f}, {h.y:.4f}, {h.z:.4f})  tail=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})  parent={bone.parent.name if bone.parent else 'None'}")

    if obj.type == 'MESH':
        mesh = obj.data
        verts = mesh.vertices
        if len(verts) > 0:
            xs = [v.co.x for v in verts]
            ys = [v.co.y for v in verts]
            zs = [v.co.z for v in verts]
            print(f"  Vertices: {len(verts)}")
            print(f"  Bounds X: [{min(xs):.4f}, {max(xs):.4f}]")
            print(f"  Bounds Y: [{min(ys):.4f}, {max(ys):.4f}]")
            print(f"  Bounds Z: [{min(zs):.4f}, {max(zs):.4f}]")

        # Check vertex groups
        if obj.vertex_groups:
            print(f"  Vertex Groups ({len(obj.vertex_groups)}):")
            for vg in obj.vertex_groups:
                print(f"    {vg.name}")

print("\n" + "=" * 70)
