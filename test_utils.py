## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    # Cosine similarity between vector1 and vector2
    result = cosine_similarity(vector1, vector2)
    # Manually compute the expected cosine similarity
    dot_product = np.dot(vector1, vector2)  # which is 32
    norm1 = np.linalg.norm(vector1)         # sqrt(1^2 + 2^2 + 3^2)
    norm2 = np.linalg.norm(vector2)         # sqrt(4^2 + 5^2 + 6^2)
    expected_result = dot_product / (norm1 * norm2)
    # Assertion to check if the computed result is close to the manually computed result
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector = np.array([1, 2, 3])
    vectors = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    # Finding the nearest neighbor by cosine similarity
    result = nearest_neighbor(target_vector, vectors)
    # Since the first vector is identical to the target, it should be the nearest
    expected_index = 0
    # Assertion to check if the function returns the correct index
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
    