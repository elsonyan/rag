from pathlib import Path


def get_file_type(path):
    types = {
        "yaml": "yaml",
        "yml": "yaml",
    }
    # 1. 检查扩展名
    ext = path.split(".")[-1]
    return types.get(ext, ext)

