import re
with open(r'C:\Users\nars7\Downloads\aniML_world\tools\measure_com_live.py') as f:
    content = f.read()
content2 = re.sub(r'\bprint\(', 'log(', content)
with open(r'C:\Users\nars7\Downloads\aniML_world\tools\measure_com_live.py', 'w') as f:
    f.write(content2)
print('Done')
