/*
Alex Lech 2023

This header contains all declarations for the Particle class, as well as 
other functions used, such as writing to files.
*/
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector>
#include <cmath>
#include <iostream>

struct Point{
	double x;
	double y;
	double z;
};

class Particle{
	public:
		Point  center;
		double radius;
		double velocity[3];
		double selfRotation[3];

		std::vector<Point> deposits;
		Particle(Point particleCenter,double r,double vel){
			center = particleCenter;
			radius = r;
			for(int i = 0; i<3;i++){
				velocity[i] = vel;
				selfRotation[i]   = vel;
			}

		}  

		void   generateDeposits(int nDeposits);
		double getSquaredDistance(Point r);
		void   getKinematics(std::vector<double> linspace,
							       double thickness,double dl,std::vector<Point> globalDeposits, 
								  									  double lambda, double dv);
		void   update_position();
		void   rotation_transform();
		void   brownian_noise();
		//Exporting positions of deposits
		void   writeDepositToCSV();
};

void   hard_sphere_correction(std::vector<Particle> &particles);
std::vector<Point>   update_globalDeposits(std::vector<Particle> &particles);
double integral(double _x, double _y,double _z,std::vector<Point> deposits,double absorbtionTerm, double dv);


double central_difference(double x_back,double x_forward,
						  		double y_back,double y_forward, 
						  		double z_back, double z_forward,
						  		std::vector<Point> deposits,
								double dl,double absorbtionTerm, double dv);

std::vector<Particle> initializeParticles();

std::vector<double> get_new_coordinates(std::vector<double> omega, std::vector<double> x);

//Linear algebra and geometric functions
	std::vector<std::vector<double>> matrix_matrix_multiplication (std::vector<std::vector<double>> a, std::vector<std::vector<double>> b);
	std::vector<double>              matrix_vector_multiplication (std::vector<std::vector<double>> R, std::vector<double> x);
	std::vector<double>              arange(double start, double stop, double stepSize);
	std::vector<double>              cross_product(std::vector<double> a,std::vector<double> b);
	std::vector<double>              getDirection(Particle p1,Particle p2);
	double                           get_norm(std::vector<double> a);

//Printing and writing functions:
void writeFieldToCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, 
														std::vector<std::vector<std::vector<double>>>& field);

void writeGradToCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, 
					std::vector<std::vector<std::vector<double>>>& xGrad, 
					std::vector<std::vector<std::vector<double>>>& yGrad);
#endif
