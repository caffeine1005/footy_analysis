"""Keep one live browser session so Sofascore's API stops answering 403.

Sofascore fronts `api/v1` with a challenge that ScraperFC's botasaurus getters
cannot get past: as of 2026-09 both the request getter and the browser getter
come back with a perfectly well-formed
``{"error": {"code": 403, "reason": "challenge"}}`` for every URL. What they are
missing is origin — the real web app calls the API as a same-origin XHR from a
loaded Sofascore page, and when it *is* challenged it fetches an `X-Captcha` JWT
from `/api/v1/token/captcha` and attaches that to everything afterwards.

So this opens a Sofascore page in a botasaurus browser and issues the scrape's
requests as same-origin `fetch` calls from inside it. Whether the challenge is
armed varies by session and by IP: unchallenged, the bare same-origin fetch is
enough; challenged, the app mints a token and a small hook installed before page
load records it so our fetches can replay it. A 403 mid-scrape reloads the page
to pick up a fresh token rather than failing the run.

The token is sniffed with an in-page `fetch` wrapper rather than with
`driver.before_request_sent`: registering a CDP request interceptor makes every
fetch *we* issue from `run_js` fail outright with "Failed to fetch", so the two
cannot coexist.

Usage — wrap whatever does the scraping, and everything underneath it (ScraperFC
and `sofascore_similarity._sofascore_get_json`) is transparently rerouted::

    import sofascore_session
    sofascore_session.run(lambda: collect(season="15/16"))

Nested calls reuse the session already running instead of opening a second
browser, so `collect` can call `collect_ages` without either of them caring
which one owns the session.
"""

from __future__ import annotations

import json
import time

# Page to warm the session on. Any Sofascore page works — this one is cheap and,
# if the challenge is armed, reliably triggers the app's own token exchange.
WARM_URL = "https://www.sofascore.com/football/tournament/england/premier-league/17"

ORIGIN = "https://www.sofascore.com"

# Installed before any page script runs. Records the captcha token off the app's
# own requests, keeping the wrapper's `toString` looking native so the usual
# "has someone patched fetch?" check doesn't trip on it.
_SNIFF_JS = """
(function () {
  if (window.__ssCaptchaHooked) { return; }
  window.__ssCaptchaHooked = true;
  window.__ssCaptcha = null;
  var native = window.fetch;
  function hooked(input, init) {
    try {
      var h = (init && init.headers) || (input && input.headers);
      if (h) {
        var v = h.get ? h.get('X-Captcha') : (h['X-Captcha'] || h['x-captcha']);
        if (v) { window.__ssCaptcha = v; }
      }
    } catch (e) { /* never let sniffing break the app */ }
    return native.apply(this, arguments);
  }
  var toString = Function.prototype.toString;
  Function.prototype.toString = function () {
    return this === hooked ? toString.call(native) : toString.call(this);
  };
  window.fetch = hooked;
})();
"""

# `run_js` wraps the body in a plain (non-async) IIFE, so this returns the
# promise rather than awaiting it, and hands back one JSON string to parse. The
# token is read from the page, so there is no token state on the Python side.
_FETCH_JS = """
var h = {};
if (window.__ssCaptcha) { h['X-Captcha'] = window.__ssCaptcha; }
return fetch(args, {headers: h})
  .then(r => r.text().then(t => JSON.stringify({status: r.status, body: t})))
  .catch(e => JSON.stringify({status: -1, body: String(e)}));
"""

_SETTLE = 6  # seconds to let the app boot after a navigation
_TOKEN_WAIT = 25  # extra seconds to wait for a token once we know we're challenged
_MAX_TRIES = 4

_active = None  # the get_json of the session currently running, if any


def _relative(url: str) -> str:
    """Rewrite an absolute Sofascore API URL to a same-origin path.

    Also drops a trailing slash from the path. `www.sofascore.com/api/v1/.../seasons/`
    301s across to `api.sofascore.com`, which answers scripted clients with a
    hang rather than a status, so the fetch dies as an opaque "Failed to fetch";
    the slashless form stays on `www` and returns 200. ScraperFC writes the
    trailing slash in `get_valid_seasons`, so this is not hypothetical.
    """
    for prefix in ("https://api.sofascore.com", "https://www.sofascore.com", ORIGIN):
        if url.startswith(prefix):
            url = url[len(prefix) :]
            break
    path, sep, query = url.partition("?")
    return path.rstrip("/") + sep + query


def _install(get_json) -> None:
    """Point ScraperFC and this repo's own getter at `get_json`."""
    import ScraperFC.sofascore as sfc_sofascore
    import ScraperFC.utils as sfc_utils

    sfc_sofascore.botasaurus_browser_get_json = get_json
    sfc_utils.botasaurus_browser_get_json = get_json
    sfc_utils.botasaurus_request_get_json = get_json

    try:
        import sofascore_similarity
    except Exception:  # noqa: BLE001 — the module is optional for a bare scrape
        pass
    else:
        sofascore_similarity._sofascore_get_json = get_json


def run(fn, headless: bool = False):
    """Run `fn()` with a warmed Sofascore session installed. Returns fn's value."""
    global _active

    if _active is not None:  # already inside a session — reuse it
        return fn()

    from botasaurus.browser import browser

    @browser(
        headless=headless,
        block_images_and_css=True,
        wait_for_complete_page_load=False,
        output=None,
        create_error_logs=False,
    )
    def _task(driver, _data):
        global _active

        try:
            # Preferred: the hook is in place before any app code runs.
            driver.run_on_new_document(_SNIFF_JS)
            preloaded = True
        except Exception:  # noqa: BLE001 — this botasaurus build rejects the CDP call
            preloaded = False

        def warm(wait_for_token: bool = False) -> bool:
            """(Re)load the app so fetches are same-origin. True if a token appeared.

            The app only mints a token when it is actually being challenged, so
            coming back empty-handed is the normal, healthy case.
            """
            driver.get(WARM_URL, timeout=90)
            time.sleep(_SETTLE)
            if not preloaded:
                # Fall back to hooking after load: misses the app's first burst
                # of requests, but it keeps making them, so a token still lands.
                driver.run_js(_SNIFF_JS + "\nreturn true")
            deadline = time.time() + (_TOKEN_WAIT if wait_for_token else 0)
            while True:
                if driver.run_js("return window.__ssCaptcha || null"):
                    return True
                if time.time() >= deadline:
                    return False
                time.sleep(1)

        warm()

        def get_json(url: str, *_args, **_kwargs) -> dict:
            """Drop-in for ScraperFC's getters: takes a URL, returns parsed JSON."""
            path = _relative(url)
            last = "no response"
            for attempt in range(_MAX_TRIES):
                raw = driver.run_js(_FETCH_JS, path)
                if raw:
                    resp = json.loads(raw)
                    if resp["status"] == 200:
                        return json.loads(resp["body"])
                    last = f"{resp['status']}: {resp['body'][:200]}"
                    if resp["status"] in (403, -1):
                        # Challenge just armed, token expired, or the page went
                        # away under us. Reload so the app can re-establish both.
                        warm(wait_for_token=resp["status"] == 403)
                        continue
                if attempt < _MAX_TRIES - 1:
                    time.sleep(2)
            raise RuntimeError(f"Sofascore refused {path} after {_MAX_TRIES} attempts — {last}")

        _install(get_json)
        _active = get_json
        try:
            return fn()
        finally:
            _active = None

    return _task(None)
