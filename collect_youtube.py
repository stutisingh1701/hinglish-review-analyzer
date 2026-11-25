"""Collect YouTube comments from Indian phone-review videos (YouTube Data API v3).

Get a free API key: Google Cloud Console -> enable "YouTube Data API v3" -> Credentials -> API key.

    export YT_API_KEY=...
    python scripts/collect_youtube.py --query "phone review hindi" --videos 15 --per-video 300
    python scripts/collect_youtube.py --video-ids abc123 def456
Output: data/own/raw_comments.csv (comment_id, video_id, video_title, text, likes, published_at)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

API = "https://www.googleapis.com/youtube/v3"
OUT = Path(__file__).resolve().parent.parent / "data" / "own" / "raw_comments.csv"


def get(endpoint, key, **params):
    params["key"] = key
    for attempt in range(3):
        r = requests.get(f"{API}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403 and "commentsDisabled" in r.text:
            return None
        time.sleep(2 ** attempt)
    r.raise_for_status()


def search_videos(key, query, n):
    out, token = [], None
    while len(out) < n:
        data = get("search", key, part="snippet", q=query, type="video", maxResults=min(50, n - len(out)),
                   regionCode="IN", relevanceLanguage="hi", pageToken=token or "")
        for item in data.get("items", []):
            out.append((item["id"]["videoId"], item["snippet"]["title"]))
        token = data.get("nextPageToken")
        if not token:
            break
    return out


def comments(key, video_id, limit):
    token, n = None, 0
    while n < limit:
        data = get("commentThreads", key, part="snippet", videoId=video_id, maxResults=100,
                   textFormat="plainText", order="relevance", pageToken=token or "")
        if data is None:
            return
        for item in data.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            yield item["id"], s["textDisplay"], s.get("likeCount", 0), s.get("publishedAt", "")
            n += 1
            if n >= limit:
                return
        token = data.get("nextPageToken")
        if not token:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="smartphone review hindi")
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--video-ids", nargs="*")
    ap.add_argument("--per-video", type=int, default=300)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    key = os.environ.get("YT_API_KEY")
    if not key:
        sys.exit("Set YT_API_KEY first (see the docstring).")
    videos = [(v, "") for v in args.video_ids] if args.video_ids else search_videos(key, args.query, args.videos)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    seen, total = set(), 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comment_id", "video_id", "video_title", "text", "likes", "published_at"])
        for vid, title in videos:
            got = 0
            for cid, text, likes, ts in comments(key, vid, args.per_video):
                text = " ".join(text.split())
                if text in seen:
                    continue
                seen.add(text)
                w.writerow([cid, vid, title, text, likes, ts])
                got += 1
            total += got
            print(f"{vid} {title[:60]!r}: {got} comments")
    print(f"Saved {total} comments to {args.out}")


if __name__ == "__main__":
    main()
