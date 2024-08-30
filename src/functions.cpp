#include "particle.h"
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;
	constexpr double particleRadius	= 2*pow(10,-6);

std::vector<Particle> initializeParticles(){
	Point centerOfParticle1 = {0.0,0.0,0.0}; 
	Particle particle1(centerOfParticle1,particleRadius,0.0);
	vector<Particle> particles = {particle1};
	return particles;
	
}

//Writes the computed integral to a .csv file
void writeFieldToCSV(const std::vector<double>& x, 
		     const std::vector<double>& y, 
		     const std::vector<double>& z, 
		     std::vector<std::vector<std::vector<double>>>& field){

	std::ofstream outputFile("temperature.csv");
	outputFile << "x,y,z,temperature" << "\n";

	for (size_t i = 0; i < x.size()-1; i+=2) {
	    	for (size_t j = 0; j < y.size()-1; j+=2) {
	         	for (size_t k =0 ; k < z.size()-1; k+=2){
              			outputFile << x[i] << "," << y[j] << "," << z[k] << "," << field[i][j][k] << "\n";
	    	        }
	       	}
	 }
	outputFile.close();
}

//Writes the computed gradients (X and Y) to a .csv file
void writeGradToCSV(const std::vector<double>& x, 
		     const std::vector<double>& y, 
		     const std::vector<double>& z, 
		     std::vector<std::vector<std::vector<double>>>& xGrad,
			 std::vector<std::vector<std::vector<double>>>& yGrad){

	std::ofstream outputFile("gradient.csv");
	outputFile << "x,y,z,gradX,gradZ" << "\n";

	for (size_t i = 0; i < x.size()-1; i+=2) {
	    	for (size_t j = 0; j < y.size(); j++) {
	         	for (size_t k =0 ; k < z.size()-1; k+=2){
              			outputFile << x[i] << "," << y[j] << "," << z[k] << "," << xGrad[i][j][k] << "," << yGrad[i][j][k] << "\n";
	    	        }
	       	}
	 }
	outputFile.close();
}

double get_norm(std::vector<double> a){
	return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}


 std::vector<double> arange(double start, double stop, double stepSize){
	int nElements = static_cast<int>((stop-start)/stepSize)+1;
	std::vector<double> array(nElements);
	double coordinate = start;
	for (int i = 0; i< nElements; i++){
		array[i] = coordinate;
		coordinate +=stepSize;
	}
	return array;
 }
 
 std::vector<double> cross_product(std::vector<double> a,std::vector<double> b){
	std::vector<double> output = {a[1]*b[2]-a[2]*b[1],
								  a[2]*b[0]-a[0]*b[2],
								  a[0]*b[1]-a[1]*b[0]};
	
	return output;
 }
 std::vector<std::vector<double>> matrix_matrix_multiplication (std::vector<std::vector<double>> a,
 										  std::vector<std::vector<double>> b){

	std::vector<std::vector<double>> output = a;
	for(int i = 0; i<3;i++){
		for(int j = 0; j<3; j++) {
			double sum = 0.0;
			for(int k = 0; k<3; k++){
				sum += a[i][k]*b[k][j];
			}
				output[i][j] = sum;
		}
	}
	return output;					
}
 
 std::vector<double> matrix_vector_multiplication(std::vector<std::vector<double>> R, std::vector<double> x){

	std::vector<double> temp(3);
    for (int i = 0; i < 3; i++) {
        double sum = 0.0;
        for (int j = 0; j < 3; j++) {
            sum += R[i][j] * x[j];
    	}
        temp[i] = sum;
    }
	return temp;
}

std::vector<Point> update_globalDeposits(std::vector<Particle> &particles){
	
	
	int nDeposits = particles[0].deposits.size();
	int nParticles = particles.size();
	std::vector<Point> newDeposits(nDeposits*nParticles);

	for(int i = 0; i < particles.size(); i++ ){
		for(int j = 0; j<nDeposits;j++){
			int index = i*nDeposits +j;
			newDeposits[index] = particles[i].deposits[j];
		}
	} 

	return newDeposits;
}
