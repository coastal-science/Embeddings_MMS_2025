import numpy as np

from encoder_pipeline.evaluation.linear_probe import remap_labels


def test_remap_labels_collapses_and_reindexes_to_new_sorted_classes():
    embeddings = {
        "train": (np.zeros((4, 2), np.float32), np.array([0, 1, 2, 3])),  # hw, nrkw, srkw, tkw
    }
    out, classes = remap_labels(embeddings, ["hw", "nrkw", "srkw", "tkw"], {"nrkw": "KW", "srkw": "KW", "tkw": "KW"})

    assert classes == ["KW", "hw"]  # sorted(set) of {hw, KW}
    np.testing.assert_array_equal(out["train"][1], [1, 0, 0, 0])


def test_remap_labels_passes_unmapped_names_through():
    embeddings = {"val": (np.zeros((3, 2), np.float32), np.array([0, 1, 2]))}
    out, classes = remap_labels(embeddings, ["a", "b", "c"], {"b": "a"})

    assert classes == ["a", "c"]
    np.testing.assert_array_equal(out["val"][1], [0, 0, 1])
