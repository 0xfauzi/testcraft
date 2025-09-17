from __future__ import annotations

import logging
from typing import Any


# Use legacy module logger name so tests patching the old path still capture warnings
logger = logging.getLogger("testcraft.application.generation.services.context_assembler")


def assemble_final_context(config: dict[str, Any], context_sections: list[list[str]]) -> str | None:
    caps_cfg = config.get("prompt_budgets", {}).get("section_caps", {})
    per_item_cap = int(config.get("prompt_budgets", {}).get("per_item_chars", 600))
    total_cap = int(config.get("prompt_budgets", {}).get("total_chars", 4000))

    def _take(items: list[str], key: str) -> list[str]:
        limit = int(caps_cfg.get(key, len(items))) if isinstance(caps_cfg, dict) else len(items)
        out: list[str] = []
        for it in items[:limit]:
            if isinstance(it, str) and it.strip():
                out.append(it[:per_item_cap])
        return out

    default_section_order = [
        "snippets",
        "neighbors",
        "test_exemplars",
        "contracts",
        "deps_config_fixtures",
        "coverage_hints",
        "callgraph",
        "error_paths",
        "usage_examples",
        "pytest_settings",
        "side_effects",
        "path_constraints",
    ]

    section_caps = caps_cfg if isinstance(caps_cfg, dict) else {}
    configured_sections = set(section_caps.keys()) if section_caps else set()
    section_keys: list[str] = []
    for section in default_section_order:
        if not configured_sections or section in configured_sections:
            section_keys.append(section)
    for section in configured_sections:
        if section not in section_keys:
            section_keys.append(section)
            logger.debug("Adding configured section '%s' to ordering", section)

    if len(context_sections) != len(section_keys):
        logger.warning(
            "Section count mismatch: expected %d sections %s, got %d context_sections. Some context may be miscategorized.",
            len(section_keys), section_keys[:5], len(context_sections)
        )

    ordered_sections: list[str] = []
    for i, section in enumerate(context_sections):
        key = section_keys[i] if i < len(section_keys) else "other"
        ordered_sections.extend(_take(section, key))

    seen: set[str] = set()
    ordered: list[str] = []
    for block in ordered_sections:
        if not isinstance(block, str):
            continue
        key = block.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(block)

    if not ordered:
        return None

    total: list[str] = []
    acc = 0
    separator = "\n\n"
    for i, block in enumerate(ordered):
        if acc >= total_cap:
            break
        sep_len = len(separator) if i > 0 else 0
        available = total_cap - acc - sep_len
        if available <= 0:
            break
        if len(block) <= available:
            piece = block
        else:
            marker = "\n# [snipped]"
            if available > len(marker):
                piece = block[: available - len(marker)] + marker
            else:
                break
        total.append(piece)
        acc += len(piece) + sep_len

    return separator.join(total) if total else None


