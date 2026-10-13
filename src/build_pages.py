"""Assemble the HTML pages from web templates + data/results -> dist/draft-board.html, dist/draft-plans.html, dist/draft-tree-<slug>.html per league config (config.json = 10 team, config_*.json) that has results/tree_<slug>.json
Run: python src/build_pages.py"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda *p: open(os.path.join(ROOT, *p), encoding="utf-8").read()
cfg = json.load(open(os.path.join(ROOT, "config.json")))
players = json.load(open(os.path.join(ROOT, "data", "players.json")))
excl = set(cfg.get("first_four_exclude", []))
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)

board = rd("web", "board_template.html").replace("__DATA__", json.dumps(players, separators=(",", ":")))
open(os.path.join(ROOT, "dist", "draft-board.html"), "w", encoding="utf-8").write(board)

plans = json.load(open(os.path.join(ROOT, "results", "plans.json")))
four = json.load(open(os.path.join(ROOT, "results", "foursomes.json")))
slim = {s: {"total": d["total"], "top": [{k: t[k] for k in ("names", "adp", "picks", "cats", "se", "gap")} for t in d["top"][:5]]}
        for s, d in four.items()}
pg = rd("web", "plans_template.html").replace("__PLANS__", json.dumps(plans, separators=(",", ":"))).replace("__FOUR__", json.dumps(slim, separators=(",", ":")))
open(os.path.join(ROOT, "dist", "draft-plans.html"), "w", encoding="utf-8").write(pg)
import glob
built = ["draft-board.html", "draft-plans.html"]
for cp in ["config.json"] + sorted(glob.glob(os.path.join(ROOT, "config_*.json"))):
    c = json.load(open(os.path.join(ROOT, cp)))
    slug, name = c.get("slug", "10team"), c.get("name", "10 team")
    tp = os.path.join(ROOT, "results", f"tree_{slug}.json"); top_p = os.path.join(ROOT, "results", f"top_{slug}.json")
    if os.path.exists(tp):
        html = rd("web", "tree_template.html").replace("__TREE__", open(tp).read()).replace("__LEAGUE__", name)
        lp = os.path.join(ROOT, "results", f"live_{slug}.json")
        html = html.replace("__LIVE__", open(lp).read() if os.path.exists(lp) else "null")
        html = html.replace("__TOP__", open(top_p).read() if os.path.exists(top_p) else "{}")
        open(os.path.join(ROOT, "dist", f"draft-tree-{slug}.html"), "w", encoding="utf-8").write(html); built.append(f"draft-tree-{slug}.html")
print("built", ", ".join("dist/" + b for b in built))
