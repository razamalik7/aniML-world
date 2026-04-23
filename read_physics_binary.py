"""
Parse deer_standing_v2_physics.usd (binary USDC) to extract physics:mass
and physics:centerOfMass values. Binary USDC stores strings in a string table
and floats as raw bytes. We extract them via pattern matching.
"""
import struct
import re
from pathlib import Path

path = Path(r'C:\Users\nars7\Downloads\aniML_world\configuration\deer_standing_v2_physics.usd')
data = path.read_bytes()

print(f"File: {path.name}  size={len(data)} bytes")
print()

# Extract all ASCII strings (length >= 4)
strings = re.findall(rb'[\x20-\x7e]{4,}', data)
strings = [s.decode('ascii') for s in strings]

# Find all link/joint names
link_names = [s for s in strings if any(
    kw in s for kw in ['pelvis', 'chest', 'neck', 'skull', 'thigh', 'shin', 'foot']
)]
print("Link/joint names found in binary:")
for s in sorted(set(link_names)):
    print(f"  {s}")
print()

# Find physics attributes
physics_attrs = [s for s in strings if 'physics' in s.lower() or 'mass' in s.lower() or 'center' in s.lower()]
print("Physics attribute strings:")
for s in sorted(set(physics_attrs)):
    print(f"  {s}")
print()

# Extract all float triplets (potential CoM values)
# Scan for 3 consecutive floats that could be CoM coordinates (small values, -5 to 5)
print("Potential CoM / float triplets (scanning binary):")
floats_found = []
for i in range(0, len(data) - 12, 4):
    try:
        a, b, c = struct.unpack_from('<fff', data, i)
        # CoM values should be small (within a few meters)
        if all(-10 < v < 10 for v in (a,b,c)) and not all(v == 0 for v in (a,b,c)):
            # Check that at least one is non-trivial
            if any(abs(v) > 0.01 for v in (a,b,c)):
                floats_found.append((i, a, b, c))
    except:
        pass

# Too many matches — look for ones near known strings
# Find offsets of key strings
string_offsets = {}
for s in ['centerOfMass', 'physics:mass', 'mass', 'center']:
    sb = s.encode('ascii')
    idx = 0
    while True:
        pos = data.find(sb, idx)
        if pos == -1:
            break
        string_offsets.setdefault(s, []).append(pos)
        idx = pos + 1

print(f"String 'centerOfMass' found at offsets: {string_offsets.get('centerOfMass', [])}")
print(f"String 'mass' found at offsets: {string_offsets.get('mass', [])[:10]}")
print()

# Read floats near centerOfMass string occurrences
if 'centerOfMass' in string_offsets:
    print("Floats near 'centerOfMass' occurrences:")
    for pos in string_offsets['centerOfMass']:
        # Search forward and backward 200 bytes for float triplets
        for offset in range(pos - 100, pos + 200, 4):
            if offset < 0 or offset + 12 > len(data):
                continue
            try:
                a, b, c = struct.unpack_from('<fff', data, offset)
                if all(-10 < v < 10 for v in (a,b,c)) and any(abs(v) > 0.05 for v in (a,b,c)):
                    print(f"  +{offset-pos:+4d} bytes: ({a:.4f}, {b:.4f}, {c:.4f})")
            except:
                pass
        print(f"  --- (at offset {pos}) ---")

# Also look for double triplets (USDC often uses doubles)
print()
print("Double triplets near 'centerOfMass':")
if 'centerOfMass' in string_offsets:
    for pos in string_offsets['centerOfMass']:
        for offset in range(pos - 200, pos + 400, 8):
            if offset < 0 or offset + 24 > len(data):
                continue
            try:
                a, b, c = struct.unpack_from('<ddd', data, offset)
                if all(-10 < v < 10 for v in (a,b,c)) and any(abs(v) > 0.05 for v in (a,b,c)):
                    print(f"  +{offset-pos:+4d} bytes: ({a:.4f}, {b:.4f}, {c:.4f})")
            except:
                pass
        print(f"  --- (at offset {pos}) ---")
