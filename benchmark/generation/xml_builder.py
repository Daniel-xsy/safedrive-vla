"""
XML building blocks for generated route files: the pretty-printer and the
default ``<evaluation>`` element of CARLA-F routes.
"""

import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# XML pretty-print (Python 3.8+ compatible)
# ---------------------------------------------------------------------------

def _indent_xml_compat(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to an ``ElementTree`` element tree.

    Uses ``ET.indent`` when available (Python ≥ 3.9) and falls back to a
    recursive implementation for older runtimes.
    """
    if hasattr(ET, "indent"):
        ET.indent(elem, space="  ")
        return

    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml_compat(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent + "  "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


# ---------------------------------------------------------------------------
# Default evaluation element
# ---------------------------------------------------------------------------

def _build_default_evaluation() -> ET.Element:
    """Create a standard ``<evaluation>`` element with collision and compliance metrics."""
    evaluation_elem = ET.Element("evaluation")

    collision_metric = ET.SubElement(evaluation_elem, "metric", {"type": "collision_check"})
    ET.SubElement(collision_metric, "param", {"name": "expect_collision", "value": "false"})

    instruction_metric = ET.SubElement(
        evaluation_elem, "metric", {"type": "instruction_compliance"}
    )
    ET.SubElement(
        instruction_metric,
        "param",
        {"name": "compliance_threshold", "value": "0.8"},
    )

    return evaluation_elem
