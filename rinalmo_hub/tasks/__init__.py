"""Importing this package is what makes `@register_task` fire for every shipped task.

Adding a task means adding one module here and one line below -- nothing else in the
framework changes.
"""

from . import mrl  # noqa: F401
from . import sec_struct  # noqa: F401
from . import splice_site  # noqa: F401
