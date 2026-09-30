import re

from conftest import ROOT

from triage_graph.fake_model import FakeTriageModel
from triage_graph.graph import build_graph

EDGE = re.compile(r"^\s*(\w+)\s+-\.?-+>(?:\|[^|]*\|)?\s*(\w+)", re.MULTILINE)
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_readme_diagram_has_exactly_the_graphs_edges():
    readme = (ROOT / "README.md").read_text()
    diagram = readme.split("```mermaid", 1)[1].split("```", 1)[0]
    drawn = {(a, b) for a, b in EDGE.findall(diagram) if "start" not in (a, b) and b != "finish"}
    graph = build_graph(FakeTriageModel()).get_graph()
    actual = {
        (e.source, e.target)
        for e in graph.edges
        if e.source != "__start__" and e.target != "__end__"
    }
    assert drawn == actual


def test_relative_links_in_docs_point_at_files_that_exist():
    for doc in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        for target in LINK.findall(doc.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).exists(), f"{doc.name} links to missing {target}"
