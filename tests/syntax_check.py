import compileall
from pathlib import Path


def test_python_files_compile():
    project_root = Path(__file__).resolve().parents[1]
    assert compileall.compile_dir(project_root / "src", quiet=1)
    assert compileall.compile_file(project_root / "ImageEnhancement.py", quiet=1)
