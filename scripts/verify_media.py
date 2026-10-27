import json
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
    capture_output=True,
    text=True,
    check=True,
)
data = json.loads(probe.stdout)
video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
assert video["codec_name"] == "h264"
assert video["pix_fmt"] == "yuv420p"
assert video["width"] < video["height"]
assert float(data["format"]["duration"]) > 8
subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
print(json.dumps({"file": path.name, "width": video["width"], "height": video["height"], "duration": data["format"]["duration"]}))
