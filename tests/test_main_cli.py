import pytest

from main import build_parser


def test_help_parser_works():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--help'])
    assert exc.value.code == 0
