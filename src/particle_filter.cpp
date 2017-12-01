/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	double std_x, std_y, std_theta; //standard deviations for x, y and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i =0; i < num_particles; i++) {
		Particle sample;
		sample.id = i;
		sample.x = dist_x(gen);
		sample.y = dist_y(gen);
		sample.theta = dist_theta(gen);
		sample.weight = 1;
		weights.push_back(sample.weight);
		particles.push_back(sample);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for (int i =0; i < num_particles; i++) {
		double new_x;
		double new_y;
		double new_theta;
		if (yaw_rate ==0){
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else {
			new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
			new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate*delta_t;			
		}
		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (int p =0; p < num_particles; p++) {
		vector<double> sense_x;
		vector<double> sense_y;
		vector<int> associations;

		vector<LandmarkObs> trans_observations;
		LandmarkObs obs;
		for (int i=0; observations.size(); i++) {
			LandmarkObs trans_obs;
			obs = observations[i];
			// Perform the space transformation from vehicle to map coordinate space
			trans_obs.x = particles[p].x + (obs.x * cos(particles[p].theta) - obs.y  * sin(particles[p].theta));
			trans_obs.y = particles[p].x + (obs.x * sin(particles[p].theta) + obs.y  * cos(particles[p].theta));
			trans_observations.push_back(trans_obs);
		}
		particles[p].weight = 1.0;

		for (int i=0; trans_observations.size(); i++) {
			double distance;
			double lowest_distance;
			lowest_distance = 9999999.0;
			double mu_x;
			double mu_y;
			int id;
			for (int l=0; map_landmarks.landmark_list.size(); l++) {
				double delta_x;
				double delta_y;
				delta_x = trans_observations[i].x - map_landmarks.landmark_list[l].x_f;
				delta_y = trans_observations[i].y - map_landmarks.landmark_list[l].y_f;
				distance = sqrt(delta_x*delta_x + delta_y*delta_y);
				if (distance < lowest_distance) {
					lowest_distance = distance;
					mu_x = map_landmarks.landmark_list[l].x_f;
					mu_y = map_landmarks.landmark_list[l].y_f;
					id = map_landmarks.landmark_list[l].id_i;
				}
			}
			double delta_x;
			double delta_y;
			double exponent;
			double gauss_norm;
			double std_x;
			double std_y;
			std_x = std_landmark[0];
			std_y = std_landmark[1];
			delta_x = trans_observations[i].x - mu_x;
			delta_y = trans_observations[id].y - mu_y;
			exponent = ((delta_x*delta_x)/(2 * std_x * std_x)  + (delta_y * delta_y)/(2 * std_y * std_y));
			gauss_norm = (1/(2 * M_PI * trans_observations[i].x * trans_observations[i].y));
			particles[p].weight *= exp(-exponent)/gauss_norm;
			associations.push_back(id);
			sense_x.push_back(mu_x);
			sense_y.push_back(mu_y);
		}
		SetAssociations(particles[p], associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for (int i=0; i<num_particles; i++) {
		resample_particles.push_back(particles[distribution(gen)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
