import numpy as np
from data.synthetic_generator import SyntheticMedicalGenerator

def test_generator_produces_correct_shape():
    gen = SyntheticMedicalGenerator(n_samples=500, seed=42)
    X, y = gen.generate()
    assert X.shape == (500, 13)
    assert y.shape == (500,)

def test_generator_produces_binary_labels():
    gen = SyntheticMedicalGenerator(n_samples=100, seed=42)
    X, y = gen.generate()
    assert set(np.unique(y)).issubset({0, 1})

def test_generator_is_deterministic():
    gen1 = SyntheticMedicalGenerator(n_samples=100, seed=42)
    gen2 = SyntheticMedicalGenerator(n_samples=100, seed=42)
    X1, y1 = gen1.generate()
    X2, y2 = gen2.generate()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)

def test_generator_cluster_proportions():
    gen = SyntheticMedicalGenerator(n_samples=1000, seed=42)
    X, y = gen.generate()
    healthy_ratio = np.sum(y == 0) / len(y)
    assert 0.5 < healthy_ratio < 0.8
