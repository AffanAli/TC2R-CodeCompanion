from helper_functions import parse_days_from_context

def test_parse_days_from_context():
    assert parse_days_from_context("within 30 days") == 30
    assert parse_days_from_context("after ten days") == 10
    assert parse_days_from_context("a 14-day notice") == 14
    assert parse_days_from_context("twenty-one days remaining") == 21
    assert parse_days_from_context("response due in 1 day") == 1
    assert parse_days_from_context("submit after ninety days") == 90
    assert parse_days_from_context("") is None
    assert parse_days_from_context(None) is None
    assert parse_days_from_context("no deadline mentioned") is None