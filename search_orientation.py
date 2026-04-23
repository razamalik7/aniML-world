import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
events = []
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except:
            pass

keywords = [
    'orientation', 'Rx(-90', 'rotation', 'spawn_rot', 'SPAWN_ROT',
    'upside down', 'sideways', 'rotated', 'spawning like', 'heads through',
    'butts in the air', '90 degrees', 'lying', 'upright', '0.7071',
    'quaternion', 'init_state.rot', '_DEFAULT_ROT',
]

hits = []
for i, ev in enumerate(events):
    role = ev.get('role','')
    msg = ev.get('message', {})
    if isinstance(msg, dict):
        content = msg.get('content', [])
        if isinstance(content, list):
            text = ' '.join(str(c.get('text','') if isinstance(c,dict) else c) for c in content)
        else:
            text = str(content)
    else:
        text = str(msg)
    
    for kw in keywords:
        if kw.lower() in text.lower():
            hits.append((i, role, kw, text))
            break

print(f'Found {len(hits)} orientation-related events\n')

# Show the most relevant ones - user messages and key assistant actions
for i, role, kw, text in hits:
    if role == 'user':
        print(f'=== EVENT {i} [USER] kw={kw!r} ===')
        # extract just the text parts
        for line in text.split('\n'):
            if kw.lower() in line.lower():
                print(f'  {line[:300]}')
        print()
    elif role == 'assistant' and any(k in kw for k in ['0.7071', 'Rx(-90', 'SPAWN_ROT', '_DEFAULT_ROT', 'init_state.rot', 'quaternion']):
        print(f'--- event {i} [ASSISTANT] kw={kw!r} ---')
        for line in text.split('\n'):
            if kw.lower() in line.lower():
                print(f'  {line[:300]}')
        print()
