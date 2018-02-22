import sys

sys.path.append('..')

from discretize import entropy
from discretize import class_information_entropy
from discretize import find_best_cut
from discretize import gain
import numpy as np

def test_entropy():
    x = np.array([0, 0, 0, 1, 1, 1, 1])
    assert(entropy(x, 0, 3) == 0.0)
    assert(entropy(x, 3, 7) == 0.0)
    assert(entropy(x, 0, 7) != 0.0)

def test_class_information_entropy():
    x = np.array([0, 0, 0, 1, 1, 1, 1])
    assert(class_information_entropy(x, 0, 7, 3) == 0.0)
    assert(class_information_entropy(x, 0, 7, 2) != 0.0)
    assert(class_information_entropy(x, 0, 7, 4) != 0.0)

def test_find_best_cut():
    x = np.array([0, 0, 0, 1, 1, 1, 1])
    assert(find_best_cut(x, 0, 7) == 3)
    

def test_gain():
    x = np.array([0, 0, 0, 1, 1, 1, 1])
    assert(gain(x, 0, 7, 3) > 0.0)
