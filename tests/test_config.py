from fedcti.config import deep_merge, load_config


def test_deep_merge_keeps_unrelated_keys():
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}}


def test_load_config_merges_exp_file_and_overrides():
    cfg = load_config("exp2/data.yaml", overrides={"fl.num_clients": 10})
    assert cfg["exp"] == "exp2"
    assert cfg["lora"]["r"] == 16
    assert cfg["fl"]["num_clients"] == 10
    assert cfg["attack"]["version"] == "15.1"
