"""Open a USD in Isaac Sim viewer WITHOUT physics. Just renders the static model.

Run: _isaac_sim\\python.bat view_usd_static.py <usd_path>
"""
import sys, time

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

usd_path = sys.argv[1]
print(f"Opening: {usd_path}", flush=True)

omni.usd.get_context().open_stage(usd_path)
for _ in range(30):
    app.update()

stage = omni.usd.get_context().get_stage()

print("\nPrim hierarchy:", flush=True)
for prim in stage.Traverse():
    indent = "  " * (len(str(prim.GetPath()).split("/")) - 1)
    print(f"{indent}{prim.GetPath()}", flush=True)

print("\n--- Static view (no physics). Close the window or Ctrl+C to exit. ---", flush=True)
while app.is_running():
    app.update()
    time.sleep(0.05)

app.close()
