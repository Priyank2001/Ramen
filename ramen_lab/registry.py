"""A name -> object registry, so new behaviour is added by registering a class, never by editing an if/elif chain."""
from typing import Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps names to objects of one kind (methods, write policies, ...). Unknown names fail with the valid options."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def add(self, name: str, item: T) -> T:
        if name in self._items:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item
        return item

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: @REGISTRY.register('name')."""
        return lambda item: self.add(name, item)

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {', '.join(self.names())}")
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
