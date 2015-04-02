"""Module containing the Train class and support functionality."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2015, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from collections import namedtuple


# Lightweight container for initial YAML evaluation.
#
# This is intended as a robust, forward-compatible intermediate representation
# for either internal consumption or external consumption by another tool e.g.
# hyperopt.
#
# We've included a slot for positionals just in case, though they are
# unsupported by the instantiation mechanism as yet.
BaseProxy = namedtuple('BaseProxy', ['callable', 'positionals',
                                     'keywords', 'yaml_src'])


class Proxy(BaseProxy):
    """
    An intermediate representation between initial YAML parse and object
    instantiation.

    Parameters
    ----------
    callable : callable
        The function/class to call to instantiate this node.
    positionals : iterable
        Placeholder for future support for positional arguments (`*args`).
    keywords : dict-like
        A mapping from keywords to arguments (`**kwargs`), which may be
        `Proxy`s or `Proxy`s nested inside `dict` or `list` instances.
        Keys must be strings that are valid Python variable names.
    yaml_src : str
        The YAML source that created this node, if available.

    Notes
    -----
    This is intended as a robust, forward-compatible intermediate
    representation for either internal consumption or external consumption
    by another tool e.g. hyperopt.

    This particular class mainly exists to  override `BaseProxy`'s `__hash__`
    (to avoid hashing unhashable namedtuple elements).
    """
    __slots__ = []

    def __hash__(self):
        """
        Return a hash based on the object ID (to avoid hashing unhashable
        namedtuple elements).
        """
        return hash(id(self))
