"""
Tests for command-line interface.


"""
import contextlib
import io

import pytest

from sgdrf import cli

from . import TestResources


class TestCli:
    def test_cli(self):
        with TestResources.capture() as capture:
            response = cli.info()
        assert f"Processed 100 things." in capture.stdout
        assert capture.stderr.strip() == ""


if __name__ == "__main__":
    pytest.main()
