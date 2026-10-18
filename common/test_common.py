import numpy as np
from common import datasets, plotting


def test_make_concentric_data_shapes_and_labels():
    X, y = datasets.make_concentric_data(n=200, noise=0.0, seed=0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert set(np.unique(y)) == {0.0, 1.0}
    # 内圆半径应明显小于外环
    inner = X[y == 1]
    outer = X[y == 0]
    assert np.mean(np.linalg.norm(inner, axis=1)) < np.mean(np.linalg.norm(outer, axis=1))


def test_make_spirals_two_classes():
    X, y = datasets.make_spirals(n=200, seed=0)
    assert X.shape == (200, 2)
    assert set(np.unique(y)) == {0.0, 1.0}


def test_make_linear_data_reproducible():
    X1, y1 = datasets.make_linear_data(n=30, seed=42)
    X2, y2 = datasets.make_linear_data(n=30, seed=42)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)


def test_setup_font_returns_name():
    name = plotting.setup_cjk_font()
    assert isinstance(name, str) and len(name) > 0
