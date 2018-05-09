# Tests for jet-hadron analysis code

These tests are by no means comprehensive, but they can at least help to test some more basic utilities and
test cases. These tests are developed with `pytest`. To conform to their standards, tests must be named
`test_NameOfTest.py` and each test function must start with `test`.

The tests can be executed via `pytest -l testName.py` and all tests can be run via `pytests -l tests/`. Note
that `-l` is also known as `--showlocals` and will print the local variables when a test fails.
