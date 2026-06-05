"""Tests for the Abaqus .inp/.blk parser and the loader registry."""

from __future__ import annotations

import numpy as np
import pytest

from meshview.io.abaqus_parser import parse_abaqus_file, parse_abaqus_string
from meshview.io.loader import load_mesh, supported_extensions


def test_single_hex_nodes_and_element(fixtures_dir):
    mesh = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    assert mesh.nodes.shape == (8, 3)
    assert mesh.n_elements == 1
    assert len(mesh.element_blocks) == 1
    block = mesh.element_blocks[0]
    assert block.elem_type == "C3D8"
    assert block.connectivity.shape == (1, 8)
    # Connectivity is resolved to 0-based node indices.
    assert block.connectivity[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert block.elem_ids.tolist() == [1]


def test_nodes_coordinates_exact(fixtures_dir):
    mesh = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    assert mesh.nodes[0].tolist() == [0.0, 0.0, 0.0]
    assert mesh.nodes[6].tolist() == [1.0, 1.0, 1.0]


def test_nset_parsed_as_indices(fixtures_dir):
    mesh = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    assert "BOTTOM" in mesh.node_sets
    assert mesh.node_sets["BOTTOM"].tolist() == [0, 1, 2, 3]


def test_elset_from_element_keyword(fixtures_dir):
    mesh = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    assert "CUBE" in mesh.elem_sets
    assert mesh.elem_sets["CUBE"].tolist() == [1]


def _mesh_equal(a, b):
    assert np.array_equal(a.nodes, b.nodes)
    assert a.node_ids.tolist() == b.node_ids.tolist()
    assert len(a.element_blocks) == len(b.element_blocks)
    for ba, bb in zip(a.element_blocks, b.element_blocks, strict=True):
        assert ba.elem_type.upper() == bb.elem_type.upper()
        assert np.array_equal(ba.connectivity, bb.connectivity)


def test_blk_matches_inp(fixtures_dir):
    """.blk (model-only) and .inp must produce the same mesh."""
    inp = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    blk = parse_abaqus_file(fixtures_dir / "single_hex.blk")
    _mesh_equal(inp, blk)
    assert inp.node_sets["BOTTOM"].tolist() == blk.node_sets["BOTTOM"].tolist()


def test_step_deck_ignored_matches_model(fixtures_dir):
    """A full analysis deck yields the same mesh as the model-only file."""
    model = parse_abaqus_file(fixtures_dir / "single_hex.inp")
    with_step = parse_abaqus_file(fixtures_dir / "with_step.inp")
    _mesh_equal(model, with_step)


def test_case_insensitive_keywords():
    text = "*node\n1,0,0,0\n2,1,0,0\n3,0,1,0\n4,0,0,1\n*Element, Type=C3D4\n1,1,2,3,4\n"
    mesh = parse_abaqus_string(text)
    assert mesh.nodes.shape == (4, 3)
    assert mesh.element_blocks[0].elem_type.upper() == "C3D4"


def test_line_continuation_for_c3d20():
    """A C3D20 connectivity split across two lines is rejoined."""
    nodes = "\n".join(f"{i},{i},0,0" for i in range(1, 21))
    text = (
        "*NODE\n" + nodes + "\n"
        "*ELEMENT, TYPE=C3D20\n"
        "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,\n"
        "11, 12, 13, 14, 15, 16, 17, 18, 19, 20\n"
    )
    mesh = parse_abaqus_string(text)
    block = mesh.element_blocks[0]
    assert block.connectivity.shape == (1, 20)


def test_generate_nset():
    text = "*NODE\n" + "\n".join(f"{i},0,0,0" for i in range(1, 11))
    text += "\n*NSET, NSET=EDGE, GENERATE\n1, 9, 2\n"
    mesh = parse_abaqus_string(text)
    # node IDs 1,3,5,7,9 -> indices 0,2,4,6,8
    assert mesh.node_sets["EDGE"].tolist() == [0, 2, 4, 6, 8]


def test_unsupported_element_recorded_not_crashed():
    text = (
        "*NODE\n1,0,0,0\n2,1,0,0\n"
        "*ELEMENT, TYPE=DASHPOTA\n1, 1, 2\n"
    )
    mesh = parse_abaqus_string(text)
    assert mesh.ignored_elements.get("DASHPOTA") == 1
    assert mesh.n_elements == 0  # nothing renderable, but no crash


def test_latin1_fallback(tmp_path):
    p = tmp_path / "deg.inp"
    # 0xB0 is the degree sign in latin-1 but invalid as standalone UTF-8.
    p.write_bytes(b"** temp 20\xb0C\n*NODE\n1,0,0,0\n2,1,0,0\n3,0,1,0\n4,0,0,1\n"
                  b"*ELEMENT, TYPE=C3D4\n1,1,2,3,4\n")
    mesh = parse_abaqus_file(p)
    assert mesh.nodes.shape == (4, 3)


def test_loader_dispatches_inp_and_blk(fixtures_dir):
    assert ".inp" in supported_extensions()
    assert ".blk" in supported_extensions()
    assert load_mesh(str(fixtures_dir / "single_hex.inp")).n_nodes == 8
    assert load_mesh(str(fixtures_dir / "single_hex.blk")).n_nodes == 8


def test_loader_rejects_unknown_extension(tmp_path):
    p = tmp_path / "model.xyz"
    p.write_text("*NODE\n1,0,0,0\n")
    with pytest.raises(ValueError, match="unsupported file extension"):
        load_mesh(str(p))
