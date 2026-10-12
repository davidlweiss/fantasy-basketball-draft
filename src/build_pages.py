"""Assemble the two HTML pages from web templates + data/results -> dist/draft-board.html, dist/draft-plans.html
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
print("built dist/draft-board.html, dist/draft-plans.html")
