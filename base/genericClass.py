#!/usr/bin/env python

# Contains generic classes
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

class EqualityMixin(object):
    """ Defines generic comparison operations using `__dict__`, which can
    then be mixed into any other class using multiple inheritance.

    Inspired by: https://stackoverflow.com/a/390511.
    """

    def __eq__(self, other):
        """ Check for equality of members. """
        # Check identity to avoid needing to perform the (potentially costly) dict comparison.
        if self is other:
            return True
        # Compare via the member values.
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """ Inequality check. Only needed for python 2. """
        res = (self == other)
        if res is not NotImplemented:
            return not res
        return NotImplemented

