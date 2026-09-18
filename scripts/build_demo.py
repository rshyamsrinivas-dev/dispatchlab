"""Embed recorded Python simulator outputs in an offline, dependency-free replay."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
template = (ROOT / "demo" / "template.html").read_text()
payload = (ROOT / "demo" / "replays.json").read_text()
assert template.count("__PAYLOAD__") == 1
(ROOT / "demo" / "index.html").write_text(template.replace("__PAYLOAD__", payload))
print("Created demo/index.html; open it directly in a browser.")
