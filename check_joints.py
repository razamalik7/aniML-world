import xml.etree.ElementTree as ET

for urdf, label in [
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_standing.urdf', 'deer_standing.urdf'),
    (r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf\deer_world.urdf', 'deer_world.urdf'),
]:
    tree = ET.parse(urdf)
    print(f"\n=== {label} ===")
    for j in tree.getroot().findall('joint'):
        if j.get('type') == 'fixed':
            continue
        name = j.get('name')
        jtype = j.get('type')
        origin = j.find('origin')
        xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
        axis = j.find('axis')
        ax = axis.get('xyz', '1 0 0') if axis is not None else '?'
        limit = j.find('limit')
        lo = limit.get('lower', '?') if limit is not None else '?'
        hi = limit.get('upper', '?') if limit is not None else '?'
        print(f"  {name:<35} axis={ax}  limits=[{lo}, {hi}]")
