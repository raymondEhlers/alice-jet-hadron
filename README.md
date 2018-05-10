# Tests for jet-hadron analysis code

These tests are by no means comprehensive, but they can at least help to test some more basic utilities and
test cases. These tests are developed with `pytest`. To conform to their standards, tests must be named
`test_NameOfTest.py` and each test function must start with `test`.

The tests can be executed via `pytest -l testName.py` and all tests can be run via `pytests -l tests/`. Note
that `-l` is also known as `--showlocals` and will print the local variables when a test fails.

## Developing tests

A few things to keep in mind when developing tests:

- Capture the logs written in the logging module via the caplog fixture. The logging level can be set via
`caplog.set_level(loggingLevel)`. The easiest approach is to set a global logging level for a set of tests.
Normally, this would be somewhat less than ideal, but for the tests, it seems to be fine.
- Be certain to write docs to describe the tests.
