import json
from pathlib import Path
from typing import Union, Optional, Set, Tuple


class LeafNode:
    def __init__(self, suit: float):
        self.suit = suit

    def is_leaf(self):
        return True


class Node:
    def __init__(
        self,
        feature: str,
        threshold: float,
        yes: Union['Node', LeafNode],
        no: Union['Node', LeafNode],
        label: Optional[str] = None,
        palette: Optional[list] = None,
        range_: Optional[list] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.yes = yes
        self.no = no
        self.label = label
        self.palette = palette
        self.range = range_

    def is_leaf(self):
        return False


def _parse_node(data: dict) -> Union[Node, LeafNode]:
    if "leaf" in data:
        return LeafNode(suit=data["suit"])

    yes = _parse_node(data["yes"])
    no = _parse_node(data["no"])

    return Node(
        feature=data["feature"],
        threshold=data["threshold"],
        yes=yes,
        no=no,
        label=data.get("label"),
        palette=data.get("palette"),
        range_=data.get("range")
    )


def load_surrogate_tree(path: Union[str, Path]) -> Node:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _parse_node(data)


def max_depth(node: Union[Node, LeafNode], depth: int = 0) -> int:
    if node.is_leaf():
        return depth
    return max(max_depth(node.yes, depth + 1), max_depth(node.no, depth + 1))


def count_leaves(node: Union[Node, LeafNode]) -> int:
    if node.is_leaf():
        return 1
    return count_leaves(node.yes) + count_leaves(node.no)


def collect_features_and_labels(
    node: Union[Node, LeafNode],
    features: Optional[Set[str]] = None,
    labels: Optional[Set[str]] = None
) -> Tuple[Set[str], Set[str]]:
    features = features or set()
    labels = labels or set()

    if node.is_leaf():
        return features, labels

    features.add(node.feature)
    if node.label:
        labels.add(node.label)

    collect_features_and_labels(node.yes, features, labels)
    collect_features_and_labels(node.no, features, labels)

    return features, labels