import zipfile
from pathlib import Path

wheel = next(Path("dist").glob("*.whl"))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert "chess_video/assets/face.svg" in names
    assert "chess_video/assets/pieces.svg" in names
    assert "chess_video/cli.py" in names
print(f"Distribution contains renderer assets: {wheel.name}")
