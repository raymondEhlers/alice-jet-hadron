#!/usr/bin/env python

# Tests for jetH.base.utilClasses
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 8 May 2018

import pytest
import collections
import logging
logger = logging.getLogger(__name__)

import jetH.base.genericClass as genericClass

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
loggingLevel = logging.DEBUG

@pytest.fixture
def setupEqualityMixin():
    """ Create a basic class for tests of the equality mixin. """

    class equalityMixinTestClass(genericClass.equalityMixin):
        def __init__(self, aNumber, aString, aList, aDict):
            self.aNumber = aNumber
            self.aString = aString
            self.aList = aList
            self.aDict = aDict

    # Define some test values. We want them to be complciated enough
    # that we can test comparion of all of the relevant types.
    aNumber = 10.3
    aString = "hello world"
    aList = [1,2,3, {"hello" : "world"}],
    aDict = {"string" : "string", "list" : [1,2,3], "dict" : {"hello" : "world"}}

    testClass = equalityMixinTestClass(aNumber, aString, aList, aDict)
    expectedClass = equalityMixinTestClass(aNumber, aString, aList, aDict)

    return (testClass, expectedClass)

def testEqualityMixin(caplog, setupEqualityMixin):
    """ Test the equality mixin with the same classes. """
    caplog.set_level(loggingLevel)
    testClass, expectedClass = setupEqualityMixin

    # Check basic assertions
    assert testClass is testClass
    assert testClass == testClass
    # Check against an identical instance of the same class.
    assert testClass == expectedClass
    assert not testClass != expectedClass

    # Modify the test class to make the classes unequal.
    # (We will work through a simple shift of the elements one member forward).
    # (I would do this with a paramterization, but I don't see any straightforward
    # way to do it, so this will be fine)
    testClass.aNumber = expectedClass.aDict
    assert testClass != expectedClass
    assert not testClass == expectedClass

    testClass.aString = expectedClass.aNumber
    assert testClass != expectedClass
    assert not testClass == expectedClass

    testClass.aList = expectedClass.aString
    assert testClass != expectedClass
    assert not testClass == expectedClass

    testClass.aDict = expectedClass.aList
    assert testClass != expectedClass
    assert not testClass == expectedClass

    # Restore the changes (just to be certain)
    testClass.aNumber = expectedClass.aNumber
    testClass.aString = expectedClass.aString
    testClass.aList = expectedClass.aList
    testClass.aDict = expectedClass.aDict

def testEqualityMixinAgainstOtherClasses(caplog, setupEqualityMixin):
    """ Test the quality mixin against other classes, for which comparions are not implemented. """
    caplog.set_level(loggingLevel)
    testClass, expectedClass = setupEqualityMixin

    # Create a named tuple object to compare against.
    testNamedTuple = collections.namedtuple("testTuple", ["hello", "world"])
    anotherObject = testNamedTuple(hello = "hello", world = "world")

    # Can't catch NotImplemented, as it's a special type of raised value
    # that isn't handled the same way as other raised exceptions.
    # Instead, we just perform the assertions to cover tests against different objects.
    assert not testClass == anotherObject
    assert testClass != anotherObject
