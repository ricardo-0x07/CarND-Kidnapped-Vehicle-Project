/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#include <map>
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
		sample.weight = 1.0 ;
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
	// cout << "prediction start" <<endl;
	default_random_engine gen;
	for (int i =0; i < num_particles; i++) {
		double new_x;
		double new_y;
		double new_theta;
		if (yaw_rate == 0 ){
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
	// cout << "prediction end" <<endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0; i < observations.size(); i++) {
		double distance;
		double lowest_distance;
		lowest_distance = 9999999.0;
		double mu_x;
		double mu_y;
		int id;
		for (int l=0; l < predicted.size(); l++) {
			double delta_x;
			double delta_y;
			delta_x = observations[i].x - predicted[l].x;
			delta_y = observations[i].y - predicted[l].y;
			distance = sqrt(delta_x*delta_x + delta_y*delta_y);
			if (distance < lowest_distance) {
				lowest_distance = distance;
				mu_x = predicted[l].x;
				mu_y = predicted[l].y;
				id = predicted[l].id;
			}
		}
		observations[i].id = id;
	}
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
	// cout << "updateWeights start" <<endl;
	for (auto& particle : particles) {
		vector<double> sense_x;
		vector<double> sense_y;
		vector<int> associations;

		vector<LandmarkObs> trans_observations;
		LandmarkObs obs;
		for (auto& obs  : observations ) {
			LandmarkObs trans_obs;
			// Perform the space transformation from vehicle to map coordinate space
			trans_obs.x = particle.x + (obs.x * cos(particle.theta)) - (obs.y  * sin(particle.theta));
			trans_obs.y = particle.y + (obs.x * sin(particle.theta)) + (obs.y  * cos(particle.theta));
			trans_observations.push_back(trans_obs);
		}

		vector<LandmarkObs> close_landmarks;
		for (auto& map_landmark : map_landmarks.landmark_list) {
			double distance;
			double mu_x;
			double mu_y;
			int id;
			LandmarkObs close_landmark;
			double delta_x;
			double delta_y;
			delta_x = particle.x - map_landmark.x_f;
			delta_y = particle.y - map_landmark.y_f;
			distance = sqrt(delta_x*delta_x + delta_y*delta_y);
			if (distance < sensor_range) {
				mu_x = map_landmark.x_f;
				mu_y = map_landmark.y_f;
				id = map_landmark.id_i;
				close_landmark.id = id;
				close_landmark.x = mu_x;
				close_landmark.y = mu_y;
				close_landmarks.push_back(close_landmark);
			}
		}

		// dataAssociation(close_landmarks, trans_observations);
		for (auto& trans_observation : trans_observations) {
			double distance;
			double lowest_distance;
			lowest_distance = 9999999.0;
			double mu_x;
			double mu_y;
			int id;
			for (auto& close_landmark : close_landmarks) {
				double delta_x;
				double delta_y;
				delta_x = trans_observation.x - close_landmark.x;
				delta_y = trans_observation.y - close_landmark.y;
				distance = sqrt(delta_x*delta_x + delta_y*delta_y);
				if (distance < lowest_distance) {
					lowest_distance = distance;
					mu_x = close_landmark.x;
					mu_y = close_landmark.y;
					id = close_landmark.id;
				}
			}
			trans_observation.id = id;
		}

		particle.weight = 1.0;
		double weight = 1.0;
		for (auto& trans_observation : trans_observations) {
			int id = trans_observation.id;
			for (auto& close_landmark : close_landmarks) {

				if(close_landmark.id == id) {
					double mu_x = 0.0;
					double mu_y = 0.0;
					double obs_x = 0.0;
					double obs_y;
					double delta_x = 0.0;
					double delta_y = 0.0;
					double exponent = 0.0;
					double gauss_norm = 0.0;
					double std_x = 0.0;
					double std_y = 0.0;
					std_x = std_landmark[0];
					std_y = std_landmark[1];
					mu_x = close_landmark.x;
					mu_y = close_landmark.y;
					obs_x = trans_observation.x;
					obs_y = trans_observation.y;
					delta_x = obs_x - mu_x;
					delta_y = obs_y - mu_y;
					exponent = ((delta_x*delta_x)/(2.0 * std_x * std_x)  + (delta_y * delta_y)/(2.0 * std_y * std_y));
					gauss_norm = (1.0/(2.0 * M_PI * std_x * std_y));
					weight = weight * (gauss_norm * exp(-exponent));
					associations.push_back(id);
					sense_x.push_back(obs_x);
					sense_y.push_back(obs_y);
				}
			}
		}
		particle.weight = weight;

		// weights.push_back(weight);
		particle = SetAssociations(particle, associations, sense_x, sense_y);
	}
	for(int i=0; i<particles.size(); i++) {
		weights[i] = particles[i].weight;
	}
	// // cout<<endl;
	// double weights_acc = 0.0;
	// weights_acc = accumulate(weights.begin(), weights.end(), 0.0);
	// for (int i=0; i < weights.size(); i++) {
	// 	weights[i] = weights[i]/weights_acc;
	// }
	// cout << "updateWeights end" <<endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// cout << "resample start weights.size()" << weights.size() <<endl;
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> distribution(weights.begin(), weights.end());
	map<int, int> m;
	vector<Particle> resample_particles;

	for (int i=0; i<num_particles; ++i) {
		int random = distribution(gen);
		// cout << " distribution(gen)" << random <<endl;
		resample_particles.push_back(particles[distribution(gen)]);
	}
	cout << "resample start resample_particles.size()" << resample_particles.size() <<endl;
	for(int i=0; i<resample_particles.size(); i++) {
		cout << "resample_particles" << resample_particles[i].id << " x " << resample_particles[i].x << " y " << resample_particles[i].y << " weight " << resample_particles[i].weight <<endl;
	}
	particles = resample_particles;
	// cout << "resample end" <<endl;
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
	return particle;
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
