def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    cl = len(a)
    rw = len(a[0])

    result = [[0 for _ in range(cl)] for _ in range(rw)]

    for i in range(len(a)):
        for j in range(len(a[0])):
            result[j][i] = a[i][j]


    return result