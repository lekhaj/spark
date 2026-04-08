# GPU Workers package
from .base_worker   import BaseWorker, WorkerConfig
from .auto_shutdown import AutoShutdown

__all__ = ["BaseWorker", "WorkerConfig", "AutoShutdown"]

# Lazy-imported workers (avoid heavy imports at package level):
# SD15Worker      — from .sd15_image_worker import SD15Worker
# TRELLISWorker   — from .trellis_worker    import TRELLISWorker
# RigWorker       — from .rig_worker        import RigWorker
