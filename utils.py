## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the scalar dot product of the two vectors.
    '''
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    '''
    # Calculate the dot product of the vectors
    dot_prod = dot_product(v1, v2)
    # Calculate the norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Calculate cosine similarity
    if norm_v1 == 0 or norm_v2 == 0:  # Avoid division by zero
        return 0
    return dot_prod / (norm_v1 * norm_v2)

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    Return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    '''
    # Initialize the index of the nearest vector and the highest similarity encountered
    best_index = -1
    highest_similarity = -np.inf  # Since cosine similarity ranges between -1 and 1

    # Iterate through each vector in the matrix
    for index, vector in enumerate(vectors):
        # Compute the cosine similarity with the target vector
        similarity = cosine_similarity(target_vector, vector)
        # Update if this vector is more similar than the previously found ones
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_index = index

    return best_index