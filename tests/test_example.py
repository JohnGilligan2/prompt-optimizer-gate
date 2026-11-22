import demo


def test_demo_runs_end_to_end(capsys):
    demo.main()
    out = capsys.readouterr().out
    assert "fixture regressed beyond tolerance: case-204" in out
    assert "staged as version 2, unlabeled" in out
    assert "production is now version 2" in out
