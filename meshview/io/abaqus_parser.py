"""Abaqus keyword parser shared by ``.inp`` and ``.blk`` files.

A ``.blk`` file is just an Abaqus model definition (NODE/ELEMENT/sets) with no
analysis steps, so it is parsed by exactly this code — a ``.blk`` is the simple
case, not a separate format. We read only what rendering needs (``*NODE``,
``*ELEMENT``, ``*NSET``, ``*ELSET``) and skip every other keyword; whatever an
unsupported keyword's data lines contain is ignored until the next keyword.

Parsing rules honoured:

* ``**`` comment lines and blank lines are skipped.
* keywords are case-insensitive; parameters are ``KEY`` or ``KEY=VALUE``.
* a data line ending in ``,`` continues onto the next line (Abaqus line
  continuation) — needed for high-order connectivity such as C3D20.
* ``*NSET``/``*ELSET`` support the ``GENERATE`` form ``start, end, inc``.

Decoding is UTF-8 with a latin-1 fallback.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.element_types import is_supported
from ..core.mesh import ElementBlock, MeshData

log = logging.getLogger("meshview.io")


def _decode(raw: bytes) -> str:
    """Decode file bytes as UTF-8, falling back to latin-1."""

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _logical_lines(text: str):
    """Yield logical lines: comments/blanks dropped, continuations joined."""

    pending = ""
    for physical in text.splitlines():
        line = physical.strip()
        if not line or line.startswith("**"):
            continue
        if pending:
            line = pending + line
            pending = ""
        if line.endswith(","):
            pending = line  # keep trailing comma; join with the next line
            continue
        yield line
    if pending:
        yield pending


def _parse_keyword(line: str) -> tuple[str, dict[str, str]]:
    """Split a ``*KEYWORD, KEY=VALUE, FLAG`` line into name + parameter dict."""

    tokens = [t.strip() for t in line.split(",")]
    name = tokens[0][1:].strip().upper()  # drop the leading '*'
    params: dict[str, str] = {}
    for tok in tokens[1:]:
        if not tok:
            continue
        if "=" in tok:
            key, value = tok.split("=", 1)
            params[key.strip().upper()] = value.strip()
        else:
            params[tok.strip().upper()] = ""
    return name, params


def _ints(line: str) -> list[int]:
    return [int(t) for t in line.split(",") if t.strip()]


def _floats(line: str) -> list[float]:
    return [float(t) for t in line.split(",") if t.strip()]


def parse_abaqus_string(text: str) -> MeshData:
    """Parse Abaqus keyword ``text`` into a :class:`MeshData`."""

    node_ids: list[int] = []
    node_coords: list[tuple[float, float, float]] = []

    # Elements grouped by exact type string, preserving insertion order.
    elem_by_type: dict[str, list[tuple[int, list[int]]]] = {}
    ignored: dict[str, int] = {}

    raw_nsets: dict[str, list[int]] = {}
    raw_elsets: dict[str, list[int]] = {}

    mode: str | None = None
    ctx: dict = {}

    for line in _logical_lines(text):
        if line.startswith("*"):
            name, params = _parse_keyword(line)
            if name == "NODE":
                mode, ctx = "NODE", {}
            elif name == "ELEMENT":
                etype = params.get("TYPE", "")
                mode, ctx = "ELEMENT", {"type": etype, "elset": params.get("ELSET")}
                if etype and params.get("ELSET"):
                    raw_elsets.setdefault(params["ELSET"], [])
            elif name == "NSET":
                key = params.get("NSET", "")
                mode = "NSET"
                ctx = {"name": key, "generate": "GENERATE" in params}
                raw_nsets.setdefault(key, [])
            elif name == "ELSET":
                key = params.get("ELSET", "")
                mode = "ELSET"
                ctx = {"name": key, "generate": "GENERATE" in params}
                raw_elsets.setdefault(key, [])
            else:
                mode, ctx = "IGNORE", {}
            continue

        # --- data line for the current section ---
        if mode == "NODE":
            vals = _floats(line)
            node_ids.append(int(vals[0]))
            x = vals[1] if len(vals) > 1 else 0.0
            y = vals[2] if len(vals) > 2 else 0.0
            z = vals[3] if len(vals) > 3 else 0.0
            node_coords.append((x, y, z))
        elif mode == "ELEMENT":
            vals = _ints(line)
            eid, conn = vals[0], vals[1:]
            elem_by_type.setdefault(ctx["type"], []).append((eid, conn))
            if ctx.get("elset"):
                raw_elsets[ctx["elset"]].append(eid)
        elif mode == "NSET":
            vals = _ints(line)
            if ctx["generate"] and len(vals) >= 2:
                start, end = vals[0], vals[1]
                inc = vals[2] if len(vals) > 2 else 1
                raw_nsets[ctx["name"]].extend(range(start, end + 1, inc))
            else:
                raw_nsets[ctx["name"]].extend(vals)
        elif mode == "ELSET":
            vals = _ints(line)
            if ctx["generate"] and len(vals) >= 2:
                start, end = vals[0], vals[1]
                inc = vals[2] if len(vals) > 2 else 1
                raw_elsets[ctx["name"]].extend(range(start, end + 1, inc))
            else:
                raw_elsets[ctx["name"]].extend(vals)
        # mode == "IGNORE": silently skip data lines

    return _assemble(node_ids, node_coords, elem_by_type, ignored, raw_nsets, raw_elsets)


def _assemble(node_ids, node_coords, elem_by_type, ignored, raw_nsets, raw_elsets) -> MeshData:
    if not node_ids:
        raise ValueError("no *NODE definitions found in input")

    nodes = np.asarray(node_coords, dtype=np.float64)
    ids = np.asarray(node_ids, dtype=np.int64)
    id_to_index = {int(nid): i for i, nid in enumerate(ids)}

    blocks: list[ElementBlock] = []
    for etype, records in elem_by_type.items():
        if not is_supported(etype):
            ignored[etype] = ignored.get(etype, 0) + len(records)
            log.warning("ignoring %d element(s) of unsupported type %r", len(records), etype)
            continue
        try:
            conn = np.asarray(
                [[id_to_index[n] for n in conn] for _eid, conn in records],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(f"element type {etype!r} references undefined node {exc}") from exc
        eids = np.asarray([eid for eid, _conn in records], dtype=np.int64)
        blocks.append(ElementBlock(elem_type=etype, connectivity=conn, elem_ids=eids))

    # Node sets resolved to node *indices*; element sets kept as original IDs.
    node_sets = {
        name: np.asarray([id_to_index[n] for n in members if n in id_to_index], dtype=np.int64)
        for name, members in raw_nsets.items()
        if name
    }
    elem_sets = {
        name: np.asarray(members, dtype=np.int64)
        for name, members in raw_elsets.items()
        if name
    }

    return MeshData(
        nodes=nodes,
        node_ids=ids,
        element_blocks=blocks,
        node_sets=node_sets,
        elem_sets=elem_sets,
        ignored_elements=ignored,
    )


def parse_abaqus_file(path) -> MeshData:
    """Read and parse an Abaqus ``.inp``/``.blk`` file at ``path``."""

    with open(path, "rb") as fh:
        text = _decode(fh.read())
    return parse_abaqus_string(text)
