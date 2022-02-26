#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;

#define SIZE 100000


void krylov_gmres(SparseMatrix<double, RowMajor> A,VectorXd Psi,MatrixXd &V,VectorXd &x,double err);



SparseMatrix<double, RowMajor> A(SIZE,SIZE);

VectorXd x(SIZE);
VectorXd b;

VectorXd x_solve;

VectorXd x_solve2;

typedef Eigen::Triplet<double> T;
std::vector<T> tripletList;




int main()
{
  for(int j,i=0;i<SIZE;i++,j++)
  {

    tripletList.push_back(T(i,j,2));
    if(i+1<SIZE) tripletList.push_back(T(i+1,j,1));
    if(j+1<SIZE) tripletList.push_back(T(i,j+1,1));
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());

  for(int i=0;i<SIZE;i++)
  {
    x(i)=i;
  }

  b=A*x;

  // cout<<b<<endl;

  // //solver 1
  // ConjugateGradient<SparseMatrix<double>, Eigen::Upper> solver;

  // std::chrono::time_point<std::chrono::system_clock> start_time1 = std::chrono::system_clock::now();
  // x_solve = solver.compute(A).solve(b);
  // std::chrono::time_point<std::chrono::system_clock> end_time1 = std::chrono::system_clock::now();

  // std::chrono::milliseconds diff1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1);




  #define M 500
  MatrixXd V(SIZE,M);

  std::chrono::time_point<std::chrono::system_clock> start_time2 = std::chrono::system_clock::now();
  krylov_gmres(A,b,V,x_solve2,0.0);
  std::chrono::time_point<std::chrono::system_clock> end_time2 = std::chrono::system_clock::now();

  std::chrono::milliseconds diff2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);

  // cout<<diff1.count()<<endl;

  //cout << "\n eigen "<<" time "<<diff1.count()<<" ms "<< endl;//输出矩阵m
  cout << x_solve2<<endl;
  cout << "\n gmres "<<" time "<<diff2.count()<<" ms "<< endl;//输出矩阵m
  
  return 0;
}




void krylov_gmres(SparseMatrix<double, RowMajor> A,VectorXd Psi,MatrixXd &V,VectorXd &x,double err)
{
  unsigned int it_max=V.cols()-1;
  

  MatrixXd H(it_max+1,it_max);
  VectorXd s(it_max);
  VectorXd c(it_max);
  VectorXd Xi(V.rows()+1);
  
  double t;
  double m=0;

  double mid_value1;
  double mid_value2;

  
  double bata=Psi.norm();
  if(bata < err) 
  {
    cout<<"\n return for the initial b is accuracy enough"<<endl;
    return;
  }

  V.col(0)=Psi.normalized();

  Xi(0)=bata;


  for(unsigned int j=0;j<it_max;j++)
  {
    VectorXd w=A*V.col(j);
    for(unsigned int i=0;i<=j;i++)
    {
      H(i,j)=w.dot(V.col(i));
      w-=H(i,j)*V.col(i);
    }
    H(j+1,j)=w.norm();

    if(H(j+1,j)<1e-14)
    {
      m=j;
      cout<<"\n break for H(j+1,j)==0"<<H(j+1,j)<<endl;
      break;
    }


    V.col(j+1)=w.normalized();

    if(j>0)
    {
      for(unsigned int i=0;i<=j-1;i++)
      {
        mid_value1=c(i)*H(i,j)+s(i)*H(i+1,j);
        mid_value2=-s(i)*H(i,j)+c(i)*H(i+1,j);
        H(i,j)=mid_value1;
        H(i+1,j)=mid_value2;
      }
    }

    

    if(abs(H(j,j))>abs(H(j+1,j))) 
    {
      t=H(j+1,j)/H(j,j);
      c(j)=1/sqrt(1+t*t);
      s(j)=c(j)*t;
    }
    else
    {
      t=H(j,j)/H(j+1,j);
      s(j)=1/sqrt(1+t*t);
      c(j)=s(j)*t;

    }

    H(j,j)=c(j)*H(j,j)+s(j)*H(j+1,j);
    H(j+1,j)=0;


    mid_value1=c(j)*Xi(j);
    mid_value2=-s(j)*Xi(j);
    Xi(j)=mid_value1;
    Xi(j+1)=mid_value2;

    //cout<<Xi<<endl;
    // while (1);
    

    if(abs(Xi(j+1))/bata<err) 
    {
      m=j;
      cout<<"\n break for get the accuracy"<<endl;
      break;
    }
    m=j;
    //cout<<j<<endl;
  }
  //cout<<V<<endl;
  if(m!=0)
  {
    cout<<"\n m is "<<m<<endl;
    VectorXd y=H.block(0,0,m+1,m+1).inverse()*Xi.head(m+1);
    x=V.block(0,0,V.rows(),m+1)*y;
  }
  else
  {
    cout<<"failed"<<endl;
  }

}