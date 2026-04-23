import json
path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Look at what message field contains
print('Checking message field structure on first 5 events:')
for i, line in enumerate(lines[:5]):
    ev = json.loads(line)
    msg = ev.get('message', '')
    print(f'Line {i}: role={ev.get("role")} message type={type(msg).__name__} len={len(str(msg))}')
    print(f'  sample: {str(msg)[:300]}')
    print()

# Now search all events for CoM-related content
keywords = [
    'deer_standing', 'analyze_com', 'center of mass', 'CoM',
    'top heavy', 'top-heavy', 'deer_world.urdf', 'SPAWN_Z', 
    'urdf', 'mass-weighted', '0.93', '0.652',
]

hits = []
for i, line in enumerate(lines):
    ev = json.loads(line)
    role = ev.get('role', '')
    msg = str(ev.get('message', ''))
    for kw in keywords:
        if kw.lower() in msg.lower():
            hits.append((i, role, kw, msg))
            break

print(f'Found {len(hits)} hits:\n')
for i, role, kw, msg in hits[-30:]:
    print(f'event {i} role={role} kw={kw!r}')
    for line2 in msg.split('\n'):
        if kw.lower() in line2.lower():
            print('  ', line2[:200])
    print()
