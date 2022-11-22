"""Utility functions."""
from typing import List, Any


def flatten(items: Any
            ) -> List[Any]:
    """Flatten anything into a 1-D list."""
    if hasattr(items, '__iter__') and not isinstance(items, str):
        ret = []
        for item in items:
            ret += flatten(item)
    else:
        ret = [items]
    return ret


def list_len(items: Any,
             ) -> List[int]:
    """Get length of elements in first dimension."""
    if hasattr(items, '__iter__') and not isinstance(items, str):
        ret = []
        is_nested_list = False
        for item in items:
            if hasattr(items, '__iter__') and not isinstance(item, str):
                ret += [len(item)]
                is_nested_list = True
            else:
                ret += [1]
        if not is_nested_list:
            ret = [sum(ret)]
    else:
        ret = [1]
    return ret
