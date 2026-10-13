from fedcti.stages import is_done, run_stage


def test_stage_runs_once(tmp_path):
    calls = []

    def work(out_dir):
        calls.append(1)
        (out_dir / "x.txt").write_text("ok")
        return {"rows": 1}

    cfg = {"project": "test"}
    assert run_stage("demo", tmp_path / "s", work, cfg) == {"rows": 1}
    assert run_stage("demo", tmp_path / "s", work, cfg) == {"rows": 1}
    assert len(calls) == 1
    assert is_done(tmp_path / "s")


def test_failed_stage_is_not_marked_done(tmp_path):
    def boom(out_dir):
        raise RuntimeError("fail")

    try:
        run_stage("bad", tmp_path / "b", boom, {})
    except RuntimeError:
        pass
    assert not is_done(tmp_path / "b")
