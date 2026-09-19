"""Shared helpers for dependency-light package exports."""

from importlib import import_module


def load_attribute(name, mapping, namespace, module_name):
    target = mapping.get(name)
    if target is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    namespace[name] = value
    return value
