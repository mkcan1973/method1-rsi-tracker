"""
Discord webhook notifications for method1's daily scan: best-effort,
never allowed to break the actual scan/signal logic if it fails.

Webhook URL resolution: the same Discord channel/webhook already used by
other projects on this machine. Local runs read it from
notify_config.json (gitignored, never committed -- method1-tracker is a
public repo). The deployed GitHub Actions run instead reads it from the
DISCORD_WEBHOOK_URL environment variable, set as an encrypted repo
secret (`gh secret set DISCORD_WEBHOOK_URL --repo mkcan1973/method1-rsi-tracker`),
never stored in plaintext in the repo.
"""

import json
import os
import urllib.request
from pathlib import Path

CONFIG_FILE = Path(__file__).with_name("notify_config.json")


def get_webhook_url() -> str | None:
    env_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_url:
        return env_url
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text()).get("discord_webhook_url")
        except Exception:
            return None
    return None


def notify_discord(message: str, webhook_url: str | None = None) -> None:
    webhook_url = webhook_url or get_webhook_url()
    if not webhook_url:
        return
    try:
        data = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=data, headers={
            "Content-Type": "application/json",
            # Discord's Cloudflare front-end 403s urllib's default UA - use a browser one.
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        })
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[notification] Discord webhook failed (non-fatal): {type(e).__name__}: {e}")
