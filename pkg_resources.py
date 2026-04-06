"""Minimal pkg_resources compatibility shim for mtcnn on Python 3.13.

The bundled mtcnn package only uses `resource_stream`, so this module provides
that API without pulling in the full setuptools implementation.
"""

from importlib import resources


def resource_stream(package_or_requirement, resource_name):
    package = package_or_requirement
    if not isinstance(package_or_requirement, str):
        package = package_or_requirement.__name__

    data = resources.files(package).joinpath(resource_name).read_bytes()

    from io import BytesIO

    return BytesIO(data)
