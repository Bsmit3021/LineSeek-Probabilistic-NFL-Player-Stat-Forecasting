"""Phase 0 feasibility probe for The Odds API (NFL player props). Spends <= ~15 credits."""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HOST = "https://api.the-odds-api.com"
SPORT = "americanfootball_nfl"
PROBE_MARKETS = ["player_rush_yds", "player_reception_yds", "player_receptions"]


def get(path: str, **params) -> tuple[object, dict]:
    params["apiKey"] = os.environ["ODDS_API_KEY"]
    url = f"{HOST}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            hdrs = {k: v for k, v in r.headers.items() if k.lower().startswith("x-requests")}
            return json.load(r), hdrs
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:300]}, {}


def main() -> None:
    sports, h = get("/v4/sports")
    nfl = [s for s in sports if isinstance(s, dict) and s.get("key") == SPORT]
    print("sports:", nfl, h)

    events, h = get(f"/v4/sports/{SPORT}/events")
    print("upcoming events:", len(events) if isinstance(events, list) else events, h)
    if not isinstance(events, list) or not events:
        sys.exit(1)
    ev = events[0]
    print("first event:", ev)

    odds, h = get(
        f"/v4/sports/{SPORT}/events/{ev['id']}/odds",
        regions="us", markets=",".join(PROBE_MARKETS), oddsFormat="american",
    )
    print("current event odds headers:", h)
    if "bookmakers" in odds:
        books = odds["bookmakers"]
        print("bookmakers:", [b["key"] for b in books])
        for b in books[:1]:
            for m in b["markets"]:
                print(" ", b["key"], m["key"], m.get("last_update"), m["outcomes"][:2])
    else:
        print(odds)

    hist, h = get(
        f"/v4/historical/sports/{SPORT}/events/{ev['id']}/odds",
        regions="us", markets="player_rush_yds", date=ev["commence_time"],
    )
    print("historical headers:", h)
    if isinstance(hist, dict) and "data" in hist:
        print("historical snapshot ts:", hist["timestamp"], "prev:", hist.get("previous_timestamp"))
        print("books:", [b["key"] for b in hist["data"].get("bookmakers", [])])
    else:
        print("historical:", hist)


if __name__ == "__main__":
    main()
