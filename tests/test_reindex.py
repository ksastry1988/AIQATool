from qatool.reindex import parse_diff_output


def test_parse_diff_output_handles_add_modify_delete():
    diff_text = "A\tnew_file.py\nM\tchanged_file.py\nD\tremoved_file.py"
    changes = parse_diff_output(diff_text)

    statuses = {c.path: c.status for c in changes}
    assert statuses["new_file.py"] == "A"
    assert statuses["changed_file.py"] == "M"
    assert statuses["removed_file.py"] == "D"


def test_parse_diff_output_handles_rename():
    diff_text = "R100\told_name.py\tnew_name.py"
    changes = parse_diff_output(diff_text)

    assert len(changes) == 1
    assert changes[0].status == "R"
    assert changes[0].path == "old_name.py"
    assert changes[0].new_path == "new_name.py"
