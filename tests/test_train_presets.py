"""Splitting one data folder by preset, so a leave-one-out ablation costs one generation run.

Separate from test_train_decisions.py because that file skips without PyTorch and this logic
needs none of it.
"""

import pytest
def _rows(uid_of_kind):
    """Fake loaded rows: `group` packs the scene id into its high bits, as decisions_data does."""
    import numpy as np

    groups = np.array([(uid << 20) | 7 for uid in uid_of_kind], dtype=np.int64)
    return {"gap": {"group": groups, "x": np.arange(len(groups), dtype=np.float32).reshape(-1, 1),
                    "y": np.zeros(len(groups), np.int8)}}


def test_one_data_folder_can_be_split_by_preset():
    """The ablation rests on this: several presets generated into one folder, trained one at a time,
    instead of one generation run per arm."""
    from line2func.train_decisions import keep_presets

    scenes = [{"uid": 0, "preset": "hard2"}, {"uid": 1, "preset": "abl_jpeg"},
              {"uid": 2, "preset": "hard2"}, {"uid": 3, "preset": "abl_blur"}]
    rows = _rows([0, 1, 1, 2, 3, 3, 3])
    kept, left = keep_presets(rows, scenes, ["hard2"])
    assert [s["uid"] for s in left] == [0, 2]
    assert list(kept["gap"]["group"] >> 20) == [0, 2]
    assert list(kept["gap"]["x"].ravel()) == [0.0, 3.0]  # every column is filtered alike
    both, _ = keep_presets(rows, scenes, ["abl_jpeg", "abl_blur"])
    assert sorted(set((both["gap"]["group"] >> 20).tolist())) == [1, 3]


def test_a_preset_that_is_not_in_the_folder_is_an_error():
    """Rather than silently training on nothing, which would look like a result."""
    from line2func.train_decisions import keep_presets

    with pytest.raises(ValueError):
        keep_presets(_rows([0]), [{"uid": 0, "preset": "hard2"}], ["abl_noise"])
