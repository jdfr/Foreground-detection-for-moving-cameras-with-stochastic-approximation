#ifndef _BASICMATH_H

#define _BASICMATH_H
 
/* Project orthogonally the column vector Vector on the vector basis Matrix,
and store the resulting projection vector in ResultVector  */

void Project(double * const Vector,double * const Matrix,
        double * const ResultVector,
        int Dimension,int NumBasisVectors);

/* Project orthogonally the column vector Vector on the vector basis Matrix,
store the resulting projection vector in ResultVector, and
the expression of the projection vector in basis coordinates in 
 ResultVectorInBase.  */

void ProjectExtra(double * const Vector,double * const Matrix,
        double * const ResultVector,
        double * const ResultVectorInBase,
        int Dimension,int NumBasisVectors);
        
/* Find the difference vector between two vectors*/
void Difference(double * const InputVector1,double * const InputVector2,
    double * const ResultVector,int Dimension);
    
/* Find the squared Euclidean norm of a vector */
void SquaredNorm(double * const Vector,double * const Result,int Dimension);

/* Product of an scalar by a matrix. It supports Matrix==Result */
void ScalarMatrixProduct(double Escalar,double *Matrix,double *Result,
    int NumRows,int NumCols);
        
/* Matrix sum. It supports that one of the operands is also the result*/
void MatrixSum(double *A,double *B,double *Result,int NumRows,int NumCols);

/* Matrix difference */
void MatrixDifference(double *A,double *B,double *Result,int NumRows,int NumCols);

/* Matrix product */
void MatrixProduct(double *A,double *B,double *Result,int NumRowsA,
    int NumColsA,int NumColsB);  

/* Find the diagonal of the product of A and B, that is,
 Result = diag ( A * B ), where Result is a vector. It is needed that 
 the number of rows of A is the same as the number of columns of B
 */
void DiagonalMatrixProduct(double *A,double *B,double *Result,
    int NumRowsA,int NumColsA);
    
/* Traspose of a matrix*/
void Traspose(double *A,double *TrasposeA,int NumRowsA,int NumColsA);  

/* Sum a diagonal matrix with a square matrix A. If Result==NULL,
the computation is performed on A */
void SumMatrixDiagonal(double *A,double *MatrixDiagonal,double *Result,int Dimension);

/* Sum a constant to all the diagonal elements of the square matrix A. If Result==NULL,
the computation is performed on A */
void SumDiagonalConstant(double *A,double Value,double *Result,int Dimension);

/* Extract the main diagonal of the square matrix A */
void ExtractDiagonal(double *A,double *DiagonalA,int Dimension);


#endif

