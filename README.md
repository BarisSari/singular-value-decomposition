# Singular Value Decomposition

SVD is a method that decomposes and rectangular matrix, e. g. an image, to three matrices
as follows:

A = U\*S\*Transpose(V), where U and V are orthogonal(U\*Transpose(U)=V\*Transpose(V)=I) and S is a diagonal matrix that has root of
the eigenvalues.

## Implementation
1. Read the image,then normalize and split it to three channels: Red, Green and Blue.
2. Use SVD built-in function to see the expected U, S, V matrices.
3. Calculate SVD by your own. In this step, I have used this approach:
	- Send the transpose of matrices because of their sizes, since in A is mxn matrix, m should
	be smaller or equal than n.
	The equation of A = U\*S\*Transpose(V) becomes A\*Transpose(A) = V\*S\*Transpose(V)
	Transpose(A)\*A = V\*S\*Transpose(U) U\*S\*Transpose(V)=V\*S^2\*Transpose(V) 
	A\*Transpose(A) = U\*S\*Transpose(V) V\*S\*Transpose(U)=U\*S^2\*Transpose(U)
	- Calculate eigenvectors and eigenvalues of these two matrices. Sort the eigenvectors
	corresponding to their eigenvalues by descending order.
	- Take the root of the first computed positive eigenvalues and put it to S. If this
	eigenvalues are not enough, add from the second computed positive eigenvalues.
4. Check the sign of eigenvectors that calculated. It makes image noisy if an eigenvector has
inverted sign than SVD built-in function’s eigenvector. Therefore, we need to compare each
computed U and V with SVD built-in function’s U and V.
5. Plot the result

The results are as follows:

![Results](output/result-for-different-ranks.jpg?raw=true)