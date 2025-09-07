from testcraft.adapters.llm import (
    strip_code_fences,
    balance_braces,
    try_parse_json,
    parse_json_response,
    normalize_output,
)


def test_strip_code_fences():
    s = """```json\n{"a": 1}\n```"""
    assert strip_code_fences(s) == '{"a": 1}'


def test_balance_braces():
    assert balance_braces('{"a": 1') == '{"a": 1}'


def test_try_parse_json_success():
    data, err = try_parse_json('{"x": 2}')
    assert err is None and data == {"x": 2}


def test_try_parse_json_repair():
    data, err = try_parse_json('{"x": 2')
    assert err is None and data == {"x": 2}


def test_parse_json_response_variants():
    good = parse_json_response('```json\n{"k": 3}\n```')
    assert good.success and good.data == {"k": 3}

    partial = parse_json_response('foo {"z": 9')
    assert partial.success and partial.data == {"z": 9}

    bad = parse_json_response('not json at all')
    assert not bad.success and bad.error


def test_normalize_output_quotes():
    s = normalize_output('\u201cHello\u201d')
    assert s == '"Hello"'


