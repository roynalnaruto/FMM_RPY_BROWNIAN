#include <iostream>
using namespace std;
#define RADIUS 8
#define NPARTICLES 300
struct polynomial{
	
	long double coefficients[100];	
	polynomial(){
		for(int i=0;i<100;i++)
		 coefficients[i] = 0;
	 }
	 
	void printPolynomial(){
		for(int i=10;i>=0;i--)
			cout<<coefficients[i]<<" ";
		cout<<endl;	
	} 
	
};


polynomial mult(polynomial p1, polynomial p2){
	
	
	polynomial ret;
	for(int i=0;i<100;i++){
		long double sum = 0;
		for(int j=0;j<=i;j++)
		  sum += p1.coefficients[j] * p2.coefficients[i-j];
		ret.coefficients[i] = sum;  
	}
	return ret;
}



polynomial integrate(polynomial p1){
	polynomial ret;
	for(int i=1;i<100;i++){
			ret.coefficients[i] = p1.coefficients[i-1]/(long double)i;
	}
	return ret;
}

long double evaluate(polynomial p1, long double x){
	
	long double currX = 1;
	long double ret = 0;
	for(int i=0;i<100;i++){
		ret += currX * p1.coefficients[i];
		currX *= x;
	}
	return ret;
}


int main(){


	polynomial pI;
	long double temp = RADIUS * RADIUS * RADIUS;
	pI.coefficients[3] = 1/temp;
	temp *= RADIUS;
	pI.coefficients[4] = (-9.0/4.0)*(1/temp);
	temp *= RADIUS * RADIUS;
	pI.coefficients[6] = (15.0/32.0)*(1/temp);
	
	
	polynomial p1;
	p1.coefficients[0] = -0.6/(0.49);
	p1.coefficients[1] = 1.0/(0.49);
	
	
	polynomial p2;
	p2.coefficients[0] = 2/(0.49);
	p2.coefficients[1] = -1.0/(0.49);
	
	polynomial p3 = mult(pI, p1);
	polynomial p3I = integrate(p3);
	
	polynomial p4 = mult(pI, p2);
	polynomial p4I = integrate(p4);
	
	//p1.printPolynomial();
	//p2.printPolynomial();
	//pI.printPolynomial();
	//p3.printPolynomial();
	long double answer = evaluate(p3I, 1.3) - evaluate(p3I, 0.6);
	answer += evaluate(p4I, 2) - evaluate(p4I,1.3);
	cout<<"Expected number of Pairs :"<<(answer*NPARTICLES*(NPARTICLES-1))/2<<endl;
	
	return 0;
}
