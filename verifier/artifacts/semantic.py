from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from verifier.artifacts.paths import write_semantic_html_artifact


DEFAULT_ATTRIBUTE_NAMES = {
    "entity": "data-entity",
    "role": "data-role",
    "scope": "data-scope",
}


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _column_scope_for(node: Tag, column_nodes: List[Tag]) -> str:
    for index, column_node in enumerate(column_nodes, start=1):
        if column_node is node or column_node in node.parents:
            return f"column_{index}"
    return "page"


def _set_semantic_attrs(
    node: Tag,
    *,
    attr_names: Dict[str, str],
    entity: str,
    role: str,
    scope: str,
) -> Tuple[int, int]:
    added = 0
    updated = 0
    assignments = {
        attr_names["entity"]: entity,
        attr_names["role"]: role,
        attr_names["scope"]: scope,
    }
    for attr_name, attr_value in assignments.items():
        current = node.get(attr_name)
        if current is None:
            node[attr_name] = attr_value
            added += 1
        elif current != attr_value:
            node[attr_name] = attr_value
            updated += 1
    return added, updated


def apply_semantic_attribute_contract(
    html_source: str,
    contract: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    soup = BeautifulSoup(html_source, "lxml")
    attr_config = dict((contract or {}).get("attributes") or {})
    attr_names = {
        "entity": str(attr_config.get("entity") or DEFAULT_ATTRIBUTE_NAMES["entity"]),
        "role": str(attr_config.get("role") or DEFAULT_ATTRIBUTE_NAMES["role"]),
        "scope": str(attr_config.get("scope") or DEFAULT_ATTRIBUTE_NAMES["scope"]),
    }
    declared_entities_before = len(soup.select(f'[{attr_names["entity"]}]'))

    added_attr_count = 0
    updated_attr_count = 0
    touched_nodes = 0
    annotated_entities: List[str] = []

    def annotate(node: Optional[Tag], *, entity: str, role: str, scope: str) -> None:
        nonlocal added_attr_count, updated_attr_count, touched_nodes
        if node is None:
            return
        added, updated = _set_semantic_attrs(node, attr_names=attr_names, entity=entity, role=role, scope=scope)
        if added or updated:
            touched_nodes += 1
        added_attr_count += added
        updated_attr_count += updated
        if entity not in annotated_entities:
            annotated_entities.append(entity)

    page = soup.select_one(".poster") or soup.body or soup.html
    annotate(page, entity="page", role="page", scope="page")

    page_content = soup.select_one(".poster-inner")
    annotate(page_content, entity="page_content", role="page_content", scope="page")

    title_area = soup.select_one(".title-area")
    annotate(title_area, entity="title_area", role="title_area", scope="page")

    title = soup.select_one(".main-title")
    annotate(title, entity="title", role="title", scope="page")

    main_columns = soup.select_one(".columns-area")
    annotate(main_columns, entity="main_columns", role="main_columns", scope="page")

    column_nodes = main_columns.select(":scope > .col") if main_columns else soup.select(".columns-area > .col")
    for column_index, column_node in enumerate(column_nodes, start=1):
        column_scope = f"column_{column_index}"
        annotate(column_node, entity=column_scope, role="instructional_column", scope=column_scope)

        heading_area = column_node.select_one(":scope .heading-row")
        annotate(heading_area, entity=f"{column_scope}_heading_area", role="column_heading_area", scope=column_scope)

        section_heading = column_node.select_one(":scope .col-heading")
        annotate(section_heading, entity=f"{column_scope}_section_heading", role="section_heading", scope=column_scope)

        storyboard = column_node.select_one(":scope .storyboard")
        annotate(storyboard, entity=f"{column_scope}_storyboard", role="storyboard", scope=column_scope)

        for panel_index, panel_node in enumerate(column_node.select(":scope .story-panel"), start=1):
            annotate(
                panel_node,
                entity=f"{column_scope}_storyboard_panel_{panel_index}",
                role="storyboard_panel",
                scope=column_scope,
            )

        for bullet_index, bullet_item in enumerate(column_node.select(":scope .bullet-item"), start=1):
            annotate(
                bullet_item,
                entity=f"{column_scope}_bullet_item_{bullet_index}",
                role="bullet_item",
                scope=column_scope,
            )

        for marker_index, bullet_marker in enumerate(column_node.select(":scope .bullet-marker"), start=1):
            annotate(
                bullet_marker,
                entity=f"{column_scope}_bullet_marker_{marker_index}",
                role="bullet_marker",
                scope=column_scope,
            )

        for circle_index, colored_circle in enumerate(column_node.select(":scope .cc"), start=1):
            annotate(
                colored_circle,
                entity=f"{column_scope}_colored_circle_{circle_index}",
                role="colored_circle",
                scope=column_scope,
            )

        blank_rect = column_node.select_one(":scope .blank-rect")
        annotate(blank_rect, entity="blank_rectangular_frame", role="blank_rectangular_frame", scope=column_scope)

        eye_wrap = column_node.select_one(":scope .eye-wrap")
        annotate(eye_wrap, entity="eye_illustration", role="eye_illustration", scope=column_scope)

    divider_nodes = main_columns.select(":scope > .divider") if main_columns else soup.select(".columns-area > .divider")
    for divider_index, divider_node in enumerate(divider_nodes, start=1):
        annotate(
            divider_node,
            entity=f"column_{divider_index}_and_column_{divider_index + 1}_gap",
            role="column_gap",
            scope="page",
        )

    chibi_wrap = soup.select_one(".chibi-wrap")
    annotate(chibi_wrap, entity="central_child_wrap", role="central_child_wrap", scope="page")

    enriched_html = str(soup)
    declared_entities_after = enriched_html.count(f'{attr_names["entity"]}=')

    summary = {
        "changed": added_attr_count > 0 or updated_attr_count > 0,
        "attribute_names": attr_names,
        "touched_nodes": touched_nodes,
        "added_attr_count": added_attr_count,
        "updated_attr_count": updated_attr_count,
        "annotated_entities": annotated_entities,
        "declared_entities_before": declared_entities_before,
        "declared_entities_after": declared_entities_after,
    }
    return enriched_html, summary


def write_enriched_html_artifact(
    *,
    original_html_path: Path,
    sample_id: str,
    enriched_html: str,
) -> Path:
    _ = original_html_path
    return write_semantic_html_artifact(sample_id=sample_id, enriched_html=enriched_html)
