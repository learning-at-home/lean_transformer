"""
Tools for defining and running block-sparse layers in LeanTransformer. This code supports a number of basic layouts and
an option to define a custom layout. Please see layout.py for details.

The current implementation supports two backends backends: native pytorch and triton, and they are **not equivalent**.
The backends require weights to be presented in different orders; when converting between layout, one will need to
 reorder blocks in the weight tensor. You can do that manually or open an issue for help and describe your case.

Native backend: supports arbitrary block sizes, but requires that the layout is balanced, i.e. has equal number of
active blocks in each row and column. Works best for large and/or uneven blocks, or when used on non-cuda devices.

Triton backend: supports more flexible layouts with unequal number of blocks per row, but the block size must be one of
(16, 32, 64 or 128). Can be significantly faster for small block sizes in relatively dense graphs.

Layout / block size restrictions can be circumvented using tiling padding with zeros. For instance, triton can treat
384x384 blocks as 9 blocks of size 128, forming a 3x3 grid, but this requires changing layout and/or weights.

"""
from .native_backend import *
from .layout import *
from .linear import *
from .triton_backend import *
