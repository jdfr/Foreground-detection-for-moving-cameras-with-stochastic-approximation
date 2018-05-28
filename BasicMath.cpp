#include "BasicMath.h"
#include <float.h>
#include <stddef.h>
#include <memory.h>

 
/* Project orthogonally the column vector Vector on the vector basis Matrix,
and store the resulting projection vector in ResultVector  */

void Project(double * const Vector,double * const Matrix,
        double * const ResultVector,
        int Dimension,int NumBasisVectors)
{
    register double *Limit;
    register double *MyComponent;
    register double *MyElement;
    int ndx;
    register double Result;
    
    memset(ResultVector,0,sizeof(double)*Dimension);    
    MyElement=Matrix;
    for(ndx=0;ndx<NumBasisVectors;ndx++)
    {
        /* Find the dot product of the input vector and this basis vector */
        MyComponent=Vector;
        Limit=MyElement+Dimension;
        Result=0.0;
        while (MyElement<Limit)
        {
            Result+=(*MyElement)*(*MyComponent);
            MyElement++;
            MyComponent++;
        }    
        /* Find the contribution of this basis vector to the projection vector */
        MyComponent=ResultVector;
        MyElement-=Dimension;
        while (MyElement<Limit)
        {
            (*MyComponent)+=Result*(*MyElement);
            MyComponent++;
            MyElement++;
        }
    }          
}
  
    
/* Project orthogonally the column vector Vector on the vector basis Matrix,
store the resulting projection vector in ResultVector, and
the expression of the projection vector in basis coordinates in 
 ResultVectorInBase.  */

void ProjectExtra(double * const Vector,double * const Matrix,
        double * const ResultVector,
        double * const ResultVectorInBase,
        int Dimension,int NumBasisVectors)
{
    register double *Limit;
    register double *MyComponent;
    register double *MyElement;
    register double *MyResultEnBase;
    int ndx;
    register double Result;
    

    memset(ResultVector,0,sizeof(double)*Dimension);
    MyElement=Matrix;
    MyResultEnBase=ResultVectorInBase;
    for(ndx=0;ndx<NumBasisVectors;ndx++)
    {
        /* Find the dot product of the input vector and this basis vector */
        MyComponent=Vector;
        Limit=MyElement+Dimension;
        Result=0.0;
        while (MyElement<Limit)
        {
            Result+=(*MyElement)*(*MyComponent);
            MyElement++;
            MyComponent++;
        }    
        (*MyResultEnBase)=Result;
        MyResultEnBase++;
        /* Find the contribution of this basis vector to the projection vector */
        MyComponent=ResultVector;
        MyElement-=Dimension;
        while (MyElement<Limit)
        {
            (*MyComponent)+=Result*(*MyElement);
            MyComponent++;
            MyElement++;
        }
    }           
}  

/* Find the difference vector between two vectors*/
void Difference(double * const InputVector1,double * const InputVector2,
    double * const ResultVector,int Dimension)
{
    register double *MyComponentInput1;
    register double *MyComponentInput2;
    register double *MyComponentResult;
    register int ndx;
    
    MyComponentInput1=InputVector1;
    MyComponentInput2=InputVector2;
    MyComponentResult=ResultVector;
    for (ndx=0;ndx<Dimension;ndx++)
    {
        (*MyComponentResult)=(*MyComponentInput1)-
                (*MyComponentInput2);
        MyComponentInput1++;
        MyComponentInput2++;
        MyComponentResult++;                
    }      
}    
    
/* Find the squared Euclidean norm of a vector */
void SquaredNorm(double * const Vector,double * const Result,int Dimension)
{
    register double *MyComponent;
    register int ndx;
    register double MyResult;    
    
    MyComponent=Vector;
    MyResult=0.0;
    for (ndx=0;ndx<Dimension;ndx++)
    {
        MyResult+=(*MyComponent)*(*MyComponent);
        MyComponent++;                
    }    
    (*Result)=MyResult;  
}
    
/* Product of an scalar by a matrix. It supports Matrix==Result */
void ScalarMatrixProduct(double Escalar,double *Matrix,double *Result,
    int NumRows,int NumCols)
{
    register double Factor;
    register double *ptr;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptrres=Result;
    ptr=Matrix;
    Factor=Escalar;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=Factor*(*ptr);
        ptrres++;
        ptr++;
    }    
    
}    
/* Matrix sum. It supports that one of the operands is also the result*/
void MatrixSum(double *A,double *B,double *Result,int NumRows,int NumCols)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptra=A;
    ptrb=B;
    ptrres=Result;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=(*ptra)+(*ptrb);
        ptrres++;
        ptra++;
        ptrb++;
    }    
}

/* Matrix difference */
void MatrixDifference(double *A,double *B,double *Result,int NumRows,int NumCols)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int ndx;
    register int NumElements;
    
    ptra=A;
    ptrb=B;
    ptrres=Result;
    NumElements=NumRows*NumCols;
    for(ndx=0;ndx<NumElements;ndx++)
    {
        (*ptrres)=(*ptra)-(*ptrb);
        ptrres++;
        ptra++;
        ptrb++;
    }    
}

/* Matrix product */
void MatrixProduct(double *A,double *B,double *Result,int NumRowsA,
    int NumColsA,int NumColsB)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int i;
    register int j;
    register int k;
    register double Sum;
    
    ptrres=Result;
    for(j=0;j<NumColsB;j++)
    {
        for(i=0;i<NumRowsA;i++)
        {
            Sum=0.0;
            ptrb=B+NumColsA*j;
            ptra=A+i;
            for(k=0;k<NumColsA;k++)
            {
                Sum+=(*ptra)*(*ptrb);
                ptra+=NumRowsA;
                ptrb++;
            }    
            (*ptrres)=Sum;
            ptrres++;
        }
    }            
}   

/* Find the diagonal of the product of A and B, that is,
 Result = diag ( A * B ), where Result is a vector. It is needed that 
 the number of rows of A is the same as the number of columns of B
 */
void DiagonalMatrixProduct(double *A,double *B,double *Result,
    int NumRowsA,int NumColsA)
{
    register double *ptra;
    register double *ptrb;
    register double *ptrres;
    register int i;
    register int k;
    register double Sum;
    
    ptrres=Result;
    for(i=0;i<NumRowsA;i++)
    {
        Sum=0.0;
        ptrb=B+NumColsA*i;
        ptra=A+i;
        for(k=0;k<NumColsA;k++)
        {
            Sum+=(*ptra)*(*ptrb);
            ptra+=NumRowsA;
            ptrb++;
        }    
        (*ptrres)=Sum;
        ptrres++;
    }
         
}   

/* Traspose of a matrix*/
void Traspose(double *A,double *TrasposeA,int NumRowsA,int NumColsA)
{
    register int NdxRow;
    register int NdxCol;
    register double *ptrA;
    
    ptrA=A;
    for(NdxCol=0;NdxCol<NumColsA;NdxCol++)
    {
        for(NdxRow=0;NdxRow<NumRowsA;NdxRow++)
        {
            (*(TrasposeA+NdxRow*NumColsA+NdxCol))=(*ptrA);
            ptrA++;
        }
    }        
}    

/* Sum a diagonal matrix with a square matrix A. If Result==NULL,
the computation is performed on A */
void SumMatrixDiagonal(double *A,double *MatrixDiagonal,double *Result,int Dimension)
{
    register int NdxElement;
    register double *ptrDiagonal;
    register double *ptrResult;
    
    /* Copy the matrix A in the output, if necessary */
    if (Result!=NULL)
    {
        memcpy(Result,A,sizeof(double)*Dimension*Dimension);
    }
    else
    {
        Result=A;
    }        
    
    /* Add the diagonal matrix to the result */
    ptrDiagonal=MatrixDiagonal;
    ptrResult=Result;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)+=(*ptrDiagonal);
        ptrResult+=(Dimension+1);
        ptrDiagonal++;
    }  
}   

/* Sum a constant to all the diagonal elements of the square matrix A. If Result==NULL,
the computation is performed on A */
void SumDiagonalConstant(double *A,double Value,double *Result,int Dimension)
{
    register int NdxElement;
    register double *ptrResult;
    
   /* Copy the matrix A in the output, if necessary */
    if (Result!=NULL)
    {
        memcpy(Result,A,sizeof(double)*Dimension*Dimension);
    }
    else
    {
        Result=A;
    }
    
    /* Add the constant to the diagonal of the output */
    ptrResult=Result;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)+=Value;
        ptrResult+=(Dimension+1);
    }  
}    

/* Extract the main diagonal of the square matrix A */
void ExtractDiagonal(double *A,double *DiagonalA,int Dimension)
{
    register int NdxElement;
    register double *ptrResult;
    register double *ptrA;
    
    ptrResult=DiagonalA;
    ptrA=A;
    for(NdxElement=0;NdxElement<Dimension;NdxElement++)
    {
        (*ptrResult)=(*ptrA);
        ptrA+=(Dimension+1);
        ptrResult++;
    }  
}

