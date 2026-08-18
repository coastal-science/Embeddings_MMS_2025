import os

def _make_id(file_path: str, label: str, time_start: float, duration: float) -> str:
    """Generate uid from fpath, label, and time"""
    return f"{os.path.basename(file_path)}_{label}_{time_start:.3f}_{duration:.3f}"