import pytest

from llmbench.config import (
    CmdParams, RangeSyntaxError, get_cmd_params_instances, parse_int_range, parse_pg_list,
)


def test_plain_value():
    assert parse_int_range("5") == [5]


def test_range_default_step():
    assert parse_int_range("1-5") == [1, 2, 3, 4, 5]


def test_range_plus_step():
    assert parse_int_range("0-10+5") == [0, 5, 10]


def test_range_mult_step_overshoots_without_landing():
    assert parse_int_range("1-100*10") == [1, 10, 100]  # 1,10,100,1000>100 stop


def test_comma_separated_multiple_ranges():
    assert parse_int_range("1,3-5,10") == [1, 3, 4, 5, 10]


def test_negative_first_requires_allow_negative():
    with pytest.raises(RangeSyntaxError):
        parse_int_range("-1")
    assert parse_int_range("-1", allow_negative=True) == [-1]


@pytest.mark.parametrize("token", ["1-10+0", "1-10*1", "1-10*0"])
def test_non_increasing_sequences_raise(token):
    with pytest.raises(RangeSyntaxError):
        parse_int_range(token)


def test_pg_list_literal_pairs_not_ranges():
    assert parse_pg_list("512,128;1024,256") == [(512, 128), (1024, 256)]


def test_cartesian_expansion_ordering_and_size():
    p = CmdParams(model=["a", "b"], ngl=[0, 1], n_prompt=[512], n_gen=[128])
    instances = get_cmd_params_instances(p)
    # 2 models * 2 ngl * (1 pp-only + 1 tg-only) = 8
    assert len(instances) == 8
    # model varies slowest (outermost)
    assert [i.model for i in instances[:4]] == ["a", "a", "a", "a"]
    assert [i.model for i in instances[4:]] == ["b", "b", "b", "b"]


def test_zero_valued_prompt_gen_skipped():
    p = CmdParams(model=["a"], n_prompt=[0], n_gen=[128])
    instances = get_cmd_params_instances(p)
    assert len(instances) == 1
    assert instances[0].n_gen == 128


def test_pg_combined_test_name():
    p = CmdParams(model=["a"], n_prompt=[], n_gen=[], pg=[(512, 128)])
    instances = get_cmd_params_instances(p)
    assert len(instances) == 1
    assert instances[0].test_name() == "pp512+tg128"


def test_depth_suffix_has_leading_space():
    p = CmdParams(model=["a"], n_prompt=[512], n_gen=[], n_depth=[64])
    instances = get_cmd_params_instances(p)
    assert instances[0].test_name() == "pp512 @ d64"


def test_attach_mode_rejects_multi_valued_group2():
    p = CmdParams(server_mode="attach", ngl=[0, 1])
    with pytest.raises(ValueError):
        p.validate_attach_mode()


def test_attach_mode_allows_single_valued_group2():
    p = CmdParams(server_mode="attach", ngl=[0])
    p.validate_attach_mode()  # no raise
