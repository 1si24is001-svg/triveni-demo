"""Assemble web/template.html + web/triveni.js + web/bundle.json into one
self-contained page: dist/triveni_demo.html (works offline too)."""
import json
from pathlib import Path

b = json.load(open("web/bundle.json"))
py = b["python"]
b["python"] = {k: py[k] for k in ("metrics", "metrics_threshold", "metrics_stage1", "by_country", "all_pairs")}
data = json.dumps(b, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
engine = Path("web/triveni.js").read_text()
assert "</script" not in engine.lower()
html = Path("web/template.html").read_text().replace("/*BUNDLE*/", data).replace("/*ENGINE*/", engine)
Path("dist").mkdir(exist_ok=True)
Path("dist/triveni_demo.html").write_text(html)
print("dist/triveni_demo.html", round(len(html.encode()) / 1024), "KB")
