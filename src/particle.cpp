/*
Alex Lech 2023	

This file contains the functions related to particle dynamics and
computation of temperature increase and temperature gradient.

In functions.h the particle class is declared.

*/

#include "particle.h"

#include <cmath>
#include <vector>
#include <fstream>
#include <random>
#include <omp.h>
using namespace std;
std::random_device rd;
std::mt19937 gen(rd());

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
	constexpr double thermoDiffusion 		  = 2.8107e-9; 
	constexpr double D_T					  = 2e-13;

	//Simulation time step, at timesteps above 0.01 numerical instabilities occur.
	constexpr double dt = 0.01; 
	constexpr double brownian_term			  = sqrt(2.0*D_T*dt);
	
	

void Particle::generateDeposits(int nDeposits) {
	//Create random number generators, with certain intervals.
	//to create a janus-particle distribution, costheta parameter should be between 0 and 1 (corresponding to z axis).
	//If one wants a completely covered particle, set costheta to (-1,1).
	//To adjust the areas of initialization, play around with the phi and costheta parameters.
	//u is commonly set (0.9,1) so the deposits are near the surface.

    uniform_real_distribution<double> phi(0.0,twoPi); 
    uniform_real_distribution<double> costheta(-1.0,1.0);
    uniform_real_distribution<double> u(0.8,1);


	//Initiate deposits
	this->deposits.reserve(nDeposits);
	int i = 0;
	double _x = this->center.x; 
	double _y = this->center.y; 
	double _z = this->center.z; 
	while (i < nDeposits ){
		
		double const theta = acos(costheta(gen));
		double const r 	 = particleRadius*u(gen);

		//Convert to cartesian:
    	double  x = r*sin(theta) * cos(phi(gen)) ;
    	double  y = r*sin(theta) * sin(phi(gen)) ;
    	double  z = r*cos(theta)				 ;
   		
		 //This works for a single particle
		if( x*x + y*y + z*z < particleRadiusSquared ){

			(this->deposits).emplace_back(Point{x+_x,y+_y,z+_z});
			i++;
		}
		


		//Add to deposits list
    	
    	
    }
}

void Particle::update_position(){
	
	//Update positions of deposits and center of particle based on self propulsion
	double dx = (this->velocity)[0]*dt;
	double dy = (this->velocity)[1]*dt;
	double dz = (this->velocity)[2]*dt;
	
	for(int i = 0; i< this->deposits.size(); i++){

		this->deposits[i].x += dx;
		this->deposits[i].y += dy;
		this->deposits[i].z += dz;
	}
	this->center.x += dx;
	this->center.y += dy;
	this->center.z += dz;
}
void Particle::brownian_noise(){
	
	std::normal_distribution<double> wiener_process(0.0, 1.0);
	std::vector<double> W = {0,0,0};
	for(int i = 0; i<3; i++){
		W[i] = wiener_process(gen);
	} 
	double dx = W[0]*brownian_term;
	double dy = W[1]*brownian_term;
	double dz = W[2]*brownian_term;
	#pragma omp paralell for schedule(dynamic)
	for(int i = 0; i< this->deposits.size(); i++){
		this->deposits[i].x += dx;
		this->deposits[i].y += dy;
		this->deposits[i].z += dz;
	}
	this->center.x += dx;
	this->center.y += dy;
	this->center.z += dz;
	

}

void Particle::rotation_transform() {
	//This function extensively uses the following formula for rotation: 
	//Section 7.4.1 in:
	//Numerical Simulations of Active Brownian Particles by A. Callegari & G. Volpe
	

    double*    w    = this->selfRotation;
    double  theta = dt*sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);

	//check if theta is 0, (no rotation needed if true)a
    if (theta != 0.0) { 

		//Generate skew-symmetric theta_x matrix:
		std::vector<std::vector<double>> theta_x = {
			{    0, -w[2],  w[1] },
			{ w[2],     0, -w[0] },
			{-w[1],  w[0],     0 }
		};

		std::vector<std::vector<double>> theta_x_squared = matrix_matrix_multiplication(theta_x, theta_x);
		std::vector<std::vector<double>> R = theta_x;
		#pragma omp paralell for schedule(dynamic)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				R[i][j] = (sin(theta) / theta) * theta_x[i][j] + ((1.0 - cos(theta)) / (theta * theta)) * theta_x_squared[i][j];
				if (i == j){
					R[i][j] +=1;
				}
			}		
		}
		// Rotate the deposits, using the particle center as reference
		std::vector<double> new_values(3);
		std::vector<double> x(3);

		for (auto &p : this->deposits) {

			x = { p.x, 
				  p.y, 
				  p.z};

			new_values = matrix_vector_multiplication(R,x);
			p.x = new_values[0];
			p.y = new_values[1];
			p.z = new_values[2];
		}
		
		// Rotate the particles velocity direction
		x = { this->center.x,
			  this->center.y, 
			  this->center.z };

		std::vector<double> v(this->velocity, this->velocity +3);
		new_values = matrix_vector_multiplication(R,v);
		for(int i  = 0; i<3; i++){
			this->velocity[i] = new_values[i];
		}

		std::vector<double> p =  {this->center.x,
			  					  this->center.y, 
			                      this->center.z};

		//Rotate the particles center
		new_values = matrix_vector_multiplication(R,p);
		this->center.x = new_values[0];
		this->center.y = new_values[1];
		this->center.z = new_values[2];
	}


}

void hard_sphere_correction(std::vector<Particle> &particles){
	
	//Exclude itself
	std::vector<Particle> tempParticles = particles;
	int const nParticles = particles.size();
	
	for (int i = 0; i<nParticles; i++){
		for(int j = 0; j<nParticles; j++) {
			double centerToCenterDistance = sqrt(particles[i].getSquaredDistance(particles[j].center)); 
			double overlap =   2*particleRadius - centerToCenterDistance; //Experiment with this
			
			if (overlap > 0.0 && overlap != 0.0 && i !=j ){
				std::cout<<"Performing H-S correction"<<"\n";
				double distanceToMove = overlap/2.0; //Overlap should be slightly more than needed

				vector<double> direction = getDirection(particles[i],particles[j]);				
		

				//Move particles AND their deposits:
				tempParticles[i].center.x +=  distanceToMove*direction[0];
				tempParticles[i].center.y +=  distanceToMove*direction[1];	
				tempParticles[i].center.z +=  distanceToMove*direction[2];	

				for(int k = 0; k< tempParticles[i].deposits.size(); k++){
					tempParticles[i].deposits[k].x += distanceToMove*direction[0];
					tempParticles[i].deposits[k].y += distanceToMove*direction[1];
					tempParticles[i].deposits[k].z += distanceToMove*direction[2];

					tempParticles[j].deposits[k].x -= distanceToMove*direction[0];
					tempParticles[j].deposits[k].y -= distanceToMove*direction[1];
					tempParticles[j].deposits[k].z -= distanceToMove*direction[2];

				}

		
				tempParticles[j].center.x -=  distanceToMove*direction[0];
				tempParticles[j].center.y -=  distanceToMove*direction[1];	
				tempParticles[j].center.z -=  distanceToMove*direction[2];	
			}							
		}
	}
	particles = tempParticles;
}


double Particle::getSquaredDistance(Point r){
	return (this->center.x-r.x)*(this->center.x-r.x) + 
		   (this->center.y-r.y)*(this->center.y-r.y) +
		   (this->center.z-r.z)*(this->center.z-r.z);
}

std::vector<double> getDirection(Particle p1,Particle p2){
	std::vector<double> direction = {0,0,0};
	double norm = sqrt(p1.getSquaredDistance(p2.center));
	
	direction[0] = (p1.center.x-p2.center.x)/norm;
	direction[1] = (p1.center.y-p2.center.y)/norm;
	direction[2] = (p1.center.z-p2.center.z)/norm;
	return direction;
}

/////////////////////////////////////////////////
////////integral and central difference function
/////////////////////////////////////////////////
	double integral(double _x,double _y,double _z,std::vector<Point> deposits,double absorbtionTerm,double dv){
		//absorbtionTerm will compute the absorbed ammount of power from the laser
		//ContributionSum will sum up contributions from all deposits
		//Finally, the contributionSum is scaled with volume element dv and divided with constants												

		double contributionSum 		       = 0.0;
		
		//Since the values scale with the inverse square distance.
			for (size_t i = 0; i < deposits.size(); i++){

				double inverse_squareroot_distance = 1.0/sqrt(pow(_x-deposits[i].x,2)+
															  pow(_y-deposits[i].y,2)+
															  pow(_z-deposits[i].z,2));
				contributionSum -=  inverse_squareroot_distance;
			}
		return contributionSum*absorbtionTerm*dv/(4*pi*waterConductivity); 
	}

	double central_difference(double x1,double x2,double y1,double y2, double z1, double z2, 
						          vector<Point> deposits   ,double dl, double absorbtionTerm, double dv){

		double const back   		= integral(x1,y1,z1,deposits,absorbtionTerm,dv);
		double const forward		= integral(x2,y2,z2,deposits,absorbtionTerm,dv);
		return (forward - back)/(2*dl);
	}
/////////////////////////////////////////////////




void Particle::getKinematics(std::vector<double> linspace,
				double thickness,double dl,std::vector<Point> globalDeposits, double lambda, double dv){

	//This will compute the tangential component in a thin layer around the particle
	//And then do a surface integral to get self propulsion in X and Z direction.


	size_t counter = 1;
	vector<double> omega = {0,0,0};

	double gradientX;
	double gradientY;
	double gradientZ;
	double const Qx = this->center.x;
	double const Qy = this->center.y;
	double const Qz = this->center.z;

	vector<double> vel 		  = {0,0,0};

		#pragma omp parallel for schedule(dynamic)
		for (auto i:linspace){
			for(auto j:linspace){

				for(auto k:linspace){		
					double const currentLaserPower = I0 + I0*cos(twoPi*(sqrt(i*i + k*k))/lambda);
					double const absorbtionTerm    = currentLaserPower*depositArea/(depositVolume);
			
					
	
					//double norm = get_norm({point.x, point.y, point.z});
					//if (norm > 6*particleRadius){continue;}
					double const d = getSquaredDistance({i,j,k});


					//Compute only the points near the surface
					if (d > pow(radius,2) && d < pow(radius,2)+thickness){
							
							double u				   = i-this->center.x;
							double v 				   = j-this->center.y;
							double w 				   = k-this->center.z;


							vector<double> r		   = {u,v,w};	


							vector<double> gradient = {central_difference(i-dl,i+dl,j   ,j   ,k   ,k   ,globalDeposits,dl,absorbtionTerm,dv),
													   central_difference(i   ,i   ,j-dl,j+dl,k   ,k   ,globalDeposits,dl,absorbtionTerm,dv),
													   central_difference(i   ,i   ,j   ,j   ,k-dl,k+dl,globalDeposits,dl,absorbtionTerm,dv)};

	
							vector<double> radial     = {0,0,0};
							vector<double> tangential = {0,0,0};

							double const duvwr  = gradient[0] * u + gradient[1] * v + gradient[2] * w;
							double const theta  = atan2( sqrt((Qx-i)*(Qx-i) + (Qz-k)*(Qz-k)), sqrt((Qy-j) *(Qy-j))); //Changed order of j-Qy  k-Qz etc
							double const phi    = atan((Qz-k)/(Qx-i));
							double const sincos = sin(theta)*cos(phi);

							//Populate the cartesian vectors with their respective components:
							for(int l = 0; l<3; l++){
								radial[l]    	    = duvwr * r[l] / d;
								tangential[l]	    = gradient[l] - radial[l];
								vel[l]     		   -= tangential[l] * sincos;

							}

							vector<double> rxV = cross_product(r,vel);

							for(int l = 0; l<3; l++){
								omega[l] += rxV[l];
							}
							
						counter++;
					}
					
					
				}
			}
		}
	for(int i = 0; i<3;i++){
		velocity[i]     = D_T*vel[i]*1e-5;
		selfRotation[i] = D_T*omega[i]*1e-5;
		//cout<<selfRotation[i]<<"\n";
		//cout<<velocity[i]<<"\n";
	}
	

}


void Particle::writeDepositToCSV() {
    static bool isFirstRun = true;

    std::ofstream outputFile;

    if (isFirstRun) {
        // If it's the first run, create a new file with the header
        outputFile.open("deposits.csv");
        outputFile << "x,y,z" << "\n";
        isFirstRun = false;
    } else {
        // If it's not the first run, open the file in append mode
        outputFile.open("deposits.csv", std::ios::app);
    }

    // Write data to the file
    for (size_t i = 0; i < size(deposits); i++) {
        outputFile << (this->deposits)[i].x << "," << (this->deposits)[i].y  << "," << (this->deposits)[i].z  << "\n";
    }

    outputFile.close();
}





//Old rotation function
/*
void Particle::rotate(double angle) {
	//Rotation only works for small angle increments when updating the positions of the deposits
	//during the brownian simulation, the largest possible angle of rotation will be small either way.
	for(int l = 0; l<100;l++){
		double theta =   angle*0.01;

    	for (int i = 0; i < this->deposits.size(); i++) {
       		double distance = getSquaredDistance(deposits[i]);
        	this->deposits[i].x = (this->deposits[i].x - this->center.x) * cos(theta) - (this->deposits[i].z - this->center.z) * sin(theta) + this->center.x;
        	this->deposits[i].z = (this->deposits[i].x - this->center.x) * sin(theta) + (this->deposits[i].z - this->center.z) * cos(theta) + this->center.z;
    	}

	}

	double vx = this->velocity[0];
	double vy = this->velocity[1];
	double vz = this->velocity[2];

	double magnitude = sqrt(vx*vx + vy*vy + vz*vz);

	this->velocity[0] = magnitude*cos(angle);
	this->velocity[1] = magnitude*sin(angle);
	
}
*/



