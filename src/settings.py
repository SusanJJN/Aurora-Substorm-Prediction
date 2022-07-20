from pathlib import Path

__all__ = [
    "resources_root",
]

resources_root = Path(__file__).parent.parent.joinpath("resources")
