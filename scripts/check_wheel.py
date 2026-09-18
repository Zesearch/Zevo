"""Check all runtime Python modules ship in a built wheel, without importing it."""
from pathlib import Path
import sys
import zipfile

wheel_dir = Path(sys.argv[1])
wheels = list(wheel_dir.glob("zevo-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"Expected one Zevo wheel, found {len(wheels)}")
expected = {str(path.relative_to("src")) for path in Path("src/zevo").rglob("*.py")}
with zipfile.ZipFile(wheels[0]) as archive:
    missing = expected - set(archive.namelist())
if missing:
    raise SystemExit("Runtime modules missing from wheel: " + ", ".join(sorted(missing)))
print(f"Wheel contains all {len(expected)} runtime Python modules")
