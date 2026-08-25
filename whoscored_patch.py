"""Workaround for soccerdata WhoScored returning JSON wrapped in HTML."""

from __future__ import annotations

import io
import json
from pathlib import Path

from lxml import html
from soccerdata._common import BaseReader, BaseSeleniumReader

WHO_SCORED_CACHE = Path.home() / "soccerdata/data/WhoScored"


def _extract_json_text(page_html: str) -> str | None:
    stripped = page_html.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    body_text = html.fromstring(page_html).xpath("string(//body)").strip()
    if body_text.startswith("{") or body_text.startswith("["):
        return body_text
    return None


def _normalize_json_bytes(raw: bytes) -> bytes:
    text = raw.decode("utf-8").strip()
    if not text:
        return b""

    if text.startswith("{") or text.startswith("["):
        return text.encode("utf-8")

    body_text = html.fromstring(raw).xpath("string(//body)").strip()
    return body_text.encode("utf-8")


def _is_valid_json(raw: bytes) -> bool:
    if not raw.strip():
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def repair_whoscored_cache(cache_dir: Path = WHO_SCORED_CACHE) -> int:
    """Rewrite HTML-wrapped JSON cache files as plain JSON. Returns files fixed."""
    if not cache_dir.exists():
        return 0

    fixed = 0
    for path in cache_dir.rglob("*.json"):
        raw = path.read_bytes()
        if _is_valid_json(raw):
            continue

        normalized = _normalize_json_bytes(raw)
        if _is_valid_json(normalized):
            path.write_bytes(normalized)
            fixed += 1
        else:
            path.unlink(missing_ok=True)
            fixed += 1

    return fixed


def apply_whoscored_json_patch() -> None:
    """Patch soccerdata to strip HTML wrappers from WhoScored JSON responses."""
    if not hasattr(BaseSeleniumReader, "_original_validate_page"):
        BaseSeleniumReader._original_validate_page = BaseSeleniumReader._validate_page

        def validate_json_body_page(self, url):
            page_html = self._driver.page_source
            if not page_html:
                raise Exception("Empty response.")

            json_text = _extract_json_text(page_html)
            if json_text is not None:
                return json_text

            tree = html.fromstring(page_html)
            body_text = tree.xpath("string(//body)").strip()
            if not body_text:
                raise Exception(f"Empty JSON body from {url}")

            return self._original_validate_page(url)

        BaseSeleniumReader._validate_page = validate_json_body_page

    if not hasattr(BaseReader, "_original_get"):
        BaseReader._original_get = BaseReader.get

        def get_with_json_fix(self, url, filepath=None, max_age=None, no_cache=False, var=None):
            if (
                var is None
                and filepath is not None
                and Path(filepath).suffix == ".json"
                and not no_cache
                and not self.no_cache
                and filepath.exists()
            ):
                cached = _normalize_json_bytes(filepath.read_bytes())
                if not _is_valid_json(cached):
                    filepath.unlink(missing_ok=True)
                    no_cache = True

            reader = self._original_get(url, filepath, max_age, no_cache, var)
            if var is None and filepath is not None and Path(filepath).suffix == ".json":
                normalized = _normalize_json_bytes(reader.read())
                if not _is_valid_json(normalized):
                    raise ValueError(
                        f"WhoScored returned invalid JSON for {url}. "
                        "Retry after a few seconds or delete the WhoScored cache."
                    )
                if filepath.exists() and filepath.read_bytes() != normalized:
                    filepath.write_bytes(normalized)
                return io.BytesIO(normalized)
            return reader

        BaseReader.get = get_with_json_fix

    repair_whoscored_cache()
