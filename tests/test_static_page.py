"""
Drift guards on the standalone dashboard.

`static/index.html` restates a few things the Python side owns — the component
list, the preset names — because it has to run with no server. Nothing enforces
that except these tests. The failure they exist to catch is silent: a component
added to the engine simply never appears in the dashboard's breakdown, and the
page still renders perfectly.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from squeezelab.config import WEIGHT_PRESETS
from squeezelab.engine import COMPONENTS

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html"
)


@pytest.fixture(scope="module")
def page() -> str:
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def script(page: str) -> str:
    """The page minus its embedded data blob.

    The blob is a scan payload full of field names, so searching the whole file
    for `gamma_score` would match the data and pass whatever the code does.
    """
    return re.sub(
        r'<script id="embedded-data".*?</script>', "", page, flags=re.S
    )


def test_every_engine_component_is_charted(script: str):
    listed = re.search(r"const COMPONENTS = \[(.*?)\];", script, re.S)
    assert listed, "the dashboard's COMPONENTS list moved or was renamed"
    keys = set(re.findall(r'\["(\w+)_score"', listed.group(1)))
    assert keys == set(COMPONENTS)


def test_the_embedded_payload_is_still_parseable(page: str):
    blob = re.search(
        r'<script id="embedded-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert blob, "the embedded data block is what makes the page work offline"
    payload = json.loads(blob.group(1))
    assert payload["scan"]["rows"]


def test_the_page_makes_no_external_requests(script: str):
    """Self-contained is the whole product: one file, droppable on any host."""
    for pattern in (r'src="https?://', r'href="https?://[^"]+\.css', r"@import\s"):
        assert not re.search(pattern, script), f"external asset reference: {pattern}"


def test_every_preset_the_page_offers_exists_in_the_engine(script: str):
    """A preset in the dropdown that `resolve_weights` rejects is a 500 on the
    live dashboard and a dead control on the static one."""
    offered = set(re.findall(r'data-preset="(\w+)"', script))
    assert offered <= set(WEIGHT_PRESETS)
