#include<iostream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<omp.h>

//Global Variables
int SIZE;
int N_THREADS;

using namespace std;

double** SetupMatrix(int n, double val)
{
    double** matrix = new double*[n];
    
    for ( int i= 0 ; i < n ; i++)
        matrix[i] = new double[n];
    
    for(int i =0 ; i < n ; i++)
        for(int j=0 ; j < n ; j++)
            matrix[i][j] = val;

    return matrix; 
}

void printMatrix(double** matrix , int n,char* name)
{
    cout << "Printing Matrix : "<< name <<endl;
    for(int i =0 ; i < n ; i++)
    {
        for(int j=0 ; j < n ; j++)
        {
            cout << matrix[i][j] <<"\t";
        }
        cout <<endl;
            
    }
        
}

void deleteMatrix(double** matrix,int size)
{
    if(matrix)
    {
        for(int i = 0; i < size; ++i) {
        delete[] matrix[i];   
        }
        //Free the array of pointers
        delete[] matrix;
    }
    
}


void AssignMatrix(double** matrix, int n, double val)
{
    for(int i =0 ; i < n ; i++)
        for(int j=0 ; j < n ; j++)
            matrix[i][j] = val;
}

void matMul_parallel(double** A, double** B , double** C)
{
    #pragma omp parallel for shared(A,  B, C) num_threads(N_THREADS)
    for (int i = 0; i < SIZE ; i++ )
		for (int j = 0; j < SIZE ; j++ )
			for (int k = 0; k < SIZE ; k++ )
				C[i][j] += A[i][k]*B[k][j];

}

void matMul_Seq(double** A, double** B , double** C)
{
    for (int i = 0; i < SIZE ; i++ )
		for (int j = 0; j < SIZE ; j++ )
			for (int k = 0; k < SIZE ; k++ )
				C[i][j] += A[i][k]*B[k][j];
				
	
}

int main(int argc,char** argv)
{
    if(argc < 3)
    {
        cout << " Enter Arguments , 1, Size of matrix, 2, num thread"<<endl;
        exit(0);
    }
    //Argument 1 -- Size of matrix
    // Argument2 -- num of thread
    SIZE = atoi(argv[1]);
    N_THREADS = atoi(argv[2]);

    // cout << "Size : " << SIZE<<endl;
    // cout << "N_THREADS : " << N_THREADS<<endl;

    //Span n threads
    omp_set_num_threads(N_THREADS);

    //Setup values for matrices
    double** A = SetupMatrix(SIZE,1.0);
    double** B = SetupMatrix(SIZE,1.0);
    double** C = SetupMatrix(SIZE,0);
    

    // MAtrix multiplication Parallel 
    double startTime = omp_get_wtime();
    matMul_parallel(A,B,C);
    double endTime = omp_get_wtime();
    double partime = endTime - startTime;
    cout << partime<<endl;

    // printMatrix(C,SIZE,(char*)"C");

    // AssignMatrix(C,SIZE,0.0);
    // startTime = omp_get_wtime();
    // matMul_Seq(A,B,C);
    // endTime = omp_get_wtime();
    // double seqtime = endTime - startTime;
    // cout << "Seq Time : " << seqtime<<endl;

    // printMatrix(C,SIZE,(char*)"C");

    


    deleteMatrix(A,SIZE);
    deleteMatrix(B,SIZE);
    deleteMatrix(C,SIZE);


    return 0;
}