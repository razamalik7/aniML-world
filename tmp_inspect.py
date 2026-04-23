from isaaclab_tasks.utils import get_checkpoint_path
import inspect
src = inspect.getsource(get_checkpoint_path)
print(src)
