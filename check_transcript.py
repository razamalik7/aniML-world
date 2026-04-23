import json

path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
events = []
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except:
            pass

keywords = [
    'deer_standing', 'analyze_com', 'center of mass', 'Centre of mass',
    'CoM', 'top.heavy', 'top heavy', 'deer_world.urdf', 'deer_world_comfix',
    'urdf com', 'joint origin', 'spawn z', 'SPAWN_Z', '0.93', '0.652',
    'URDF', 'urdf', 'mass-weighted',
]

print(f'Total events: {len(events)}\n')
hits = []
for i, ev in enumerate(events):
    role = ev.get('role','')
    content = ev.get('content','')
    if isinstance(content, list):
        content = ' '.join(str(c) for c in content)
    content = str(content)
    for kw in keywords:
        if kw.lower() in content.lower():
            hits.append((i, role, kw, content[:500]))
            break

print(f'Found {len(hits)} hits:\n')
for i, role, kw, snip in hits[-40:]:
    print(f'event {i} role={role} kw={kw!r}')
    for line in snip.split('\n'):
        if kw.lower() in line.lower():
            print('  ', line[:200])
    print()
