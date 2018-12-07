# Tests for jet-hadron analysis code

Tests are implemented using `pytest`. To execute the testing, I tend to use something like:

```bash
$ pytest -l --cov=jet_hadron --cov-report html --cov-branch --durations=5 tests/
```

These tests are not yet comprehensive (nor should they be for the main analysis modules), but they can at
least help to test base functionality.

## Developing tests

A few things to keep in mind when developing tests:

- You can capture the logs from the modules via the logging mixin. The default level is set to debug
  since I think it is most likely to be the right level to aid when tests fail.
- Be certain to write docs to describe the tests!
