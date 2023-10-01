"""
Tests for command-line interface.


"""

import pytest

from sgdrf import cli

from . import TestResources


class TestCli:
    def test_cli(self):
        with TestResources.capture() as capture:
            cli.info()
            assert "Processed 100 things." in capture.stdout
            assert capture.stderr.strip() == ""


if __name__ == "__main__":
    pytest.main()
