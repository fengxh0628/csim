from typing import Iterator, Optional, TypeVar

T = TypeVar('T', bound='TreeNode')


class TreeNode:

    def __init__(self, parent: Optional['TreeNode'] = None):
        self.parent = parent
        self.children: list = []
        if parent is not None:
            parent.children.append(self)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def preorder_iter(node: T) -> Iterator[T]:
    yield node
    for child in node.children:
        yield from preorder_iter(child)
