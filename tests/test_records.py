import json

from mathrl.records import RunRecord, format_validation_line


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_meta_line_first(tmp_path):
    rec = RunRecord("tiny", "smoke", 0, config={"lr": 3e-4}, baseline="base", root=tmp_path)
    rec.log_step(1, loss=2.5, lr=3e-4, grad_norm=0.8)
    lines = _read(rec.path)
    assert lines[0]["type"] == "meta"
    assert lines[0]["model"] == "tiny"
    assert lines[0]["training"] == "smoke"
    assert lines[0]["seed"] == 0
    assert lines[0]["config"] == {"lr": 3e-4}
    assert lines[0]["baseline"] == "base"
    assert "git_commit" in lines[0]
    assert "started" in lines[0]


def test_step_lines_flat(tmp_path):
    rec = RunRecord("tiny", "smoke", 0, config={}, root=tmp_path)
    rec.log_step(100, loss=2.4, lr=3e-4, grad_norm=0.82)
    rec.log_eval(100, val_loss=2.5)
    lines = _read(rec.path)
    step_line = lines[1]
    assert step_line["type"] == "step"
    assert step_line["step"] == 100
    assert step_line["loss"] == 2.4
    # flat: no nested dicts beyond nothing here
    assert all(not isinstance(v, dict) for v in step_line.values())
    eval_line = lines[2]
    assert eval_line["type"] == "eval"
    assert eval_line["step"] == 100
    assert eval_line["val_loss"] == 2.5


def test_append_behavior(tmp_path):
    rec = RunRecord("tiny", "smoke", 0, config={}, root=tmp_path)
    for s in range(1, 6):
        rec.log_step(s, loss=1.0 / s)
    lines = _read(rec.path)
    # 1 meta + 5 step lines
    assert len(lines) == 6
    assert [line["step"] for line in lines[1:]] == [1, 2, 3, 4, 5]


def test_validation_line_format():
    line = format_validation_line(
        epoch=2, step=1350, total=3000, elapsed_sec=553, train_loss=2.431, val_loss=2.512
    )
    assert line == "e2 | step  1350/3000 | 09:13 | loss  2.431 | val  2.512 | diff +0.081"
