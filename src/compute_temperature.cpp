#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>
#include "particle.h"


	constexpr double pi	      		 	  	  = 3.14159265358979323846;
	constexpr double twoPi					  = 2*pi;
	constexpr double particleRadius   		  = 2e-6;
	constexpr double particleRadiusSquared    = particleRadius*particleRadius;
	constexpr double depositRadius	 	      = 30   *pow(10,-9);	
	constexpr double depositVolume	          = 4*pi *pow((depositRadius),3)/3;  //Spherical deposit volume
	constexpr double depositArea	 	      = 2*pi *pow(depositRadius,2);		 //Spherical deposit surface area
	constexpr double intensity		  		  = 100  *pow(10,-3);  //Watts
	constexpr double areaOfIllumination 	  = 40   *pow(10,-6);  //Meters^2  How much area the laser is distributed on.
	constexpr double I0		 				  = 2*intensity/(pow(areaOfIllumination*2,2)); //Total laser intensity Watt/meter^2
	constexpr double waterConductivity	 	  = 0.606;   //water conductivity in Watts/(meter*Kelvin)

using namespace std;
 	double bounds, lambda, stepSize, dv;
	int    nDeposits;	

int main(int argc, char** argv) {
	auto startTimer = std::chrono::high_resolution_clock::now();

	//Parse input arguments:
	bounds    = stold(argv[2])  * pow(10,-6); //size of simulation box
	stepSize  = bounds/(double)(200);       //step size based off of bounds parameter
	nDeposits = stof(argv[1]);				  //number of deposits to initialize
	lambda	  = stold(argv[3])  * pow(10,-9); //Spatial periodicity of laser 
    dv        = stepSize*stepSize*stepSize;   //volume element for integral
	

	vector<Particle> particles = initializeParticles();
	int nParticles = particles.size();
	for(int i = 0; i<nParticles; i++ ) particles[i].generateDeposits(nDeposits);

	///////////INITIALIZE LINSPACE VECTORS////////////////////////////////////////////////////////
	 vector<double> linspace = arange(-1*bounds,bounds,stepSize);

     const vector<double> z = linspace;
	 const vector<double> x = linspace;
 	 const vector<double> y = linspace; 
 	 cout<<"Finished initialization of "<< nDeposits <<" deposits."<<endl;
	 
 	//////////////////////////////////////////////////////////////////////////////////////////////
 	////////////////////////  INTEGRAL ///////////////////////////////////////////////////////////
	int nSteps = x.size();

	//Initialize space of points which will contain the temperature field
	vector<vector<vector<double>>> temperature(nSteps, vector<vector<double>>(nSteps, vector<double>(nSteps)));  

	//The following loops will call integral() for all points in 3D space, populating each point.
	for(int n = 0;n<nParticles;n++){
 	#pragma omp parallel for
    	for (size_t i = 0; i < nSteps; i++){
			double const currentLaserPower = I0 + I0*cos(twoPi*(x[i])/lambda);
			double const absorbtionTerm    = currentLaserPower*depositArea/(depositVolume);

    		for(size_t j = 0; j<y.size(); j++){
    			for(size_t k = 0; k<nSteps; k++){
					Point point = {x[i],y[j],z[k]};
					double d = particles[n].getSquaredDistance(point);
    				//Check if outside particle:
					//if (d > pow(particles[n].radius,2)){
						temperature[i][j][k] -= integral(x[i],y[j],z[k],particles[n].deposits,absorbtionTerm,dv);	
       				//}
				}
    		}
    	}
		particles[n].writeDepositToCSV();
	}

    //////////////////////WRITE TO FILE///////////////////////////////
		cout<<"Simulation finished, writing to csv..."<<endl;
		writeFieldToCSV(x,y,z,temperature);	
	///////////////////COMPUTE ELAPSED TIME///////////////////////////
   		auto endTimer = std::chrono::high_resolution_clock::now();
   		std::chrono::duration<double> duration = endTimer - startTimer;
   		double elapsed_seconds = duration.count();
  		std::cout << "Program completed after: " << elapsed_seconds << " seconds" << std::endl;
	//////////////////END PROGRAM/////////////////////////////////////
	return 0;
}	


