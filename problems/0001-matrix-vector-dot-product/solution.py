def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	
	a_r = len(a)
	a_c = len(a[0])

	b_r = len(b)
	b_c = len(b)

	if(a_c != b_r):
		return [-1]

	sm = 0;

	c = []
	
	for i in range(a_r):
		for j in range(b_c):
			sm += (a[i][j]*b[j])
		c.append(sm)
		sm = 0

	return c
