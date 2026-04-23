"""Procedural terrain hooks (Isaac Lab terrain generators). Implement in Phase 1+."""


def describe_default_rough() -> dict:
    """Mirror ``configs/environment.yaml`` terrain section for tooling."""
    return {
        "type": "procedural_rough",
        "size_m": (100, 100),
        "resolution": 0.1,
        "roughness": 0.05,
    }
