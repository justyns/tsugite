"""A bounded, thread-safe least-recently-used dict."""

import threading
from collections import OrderedDict


class LruCache:
    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self._items:
                return default
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
