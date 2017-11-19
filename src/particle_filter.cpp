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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	num_particles = 100;

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{

		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1);

		// Print your samples to the terminal.
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;


	for (int i = 0; i < num_particles; i++)
	{

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// control update using given velocity and yaw rate
		if (fabs(yaw_rate) < 0.0001)
		{
			x = x + velocity * delta_t * cos(theta);
			y = y + velocity * delta_t * sin(theta);
			// theta stays the same
		}
		else
		{
			x = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			y = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			theta = theta + yaw_rate * delta_t;
		}

		// add in Gaussian noise
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
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

	for (int i = 0; i < num_particles; i++)
	{
		particles[i].weight = 1.0;
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		for (int n = 0; n < observations.size(); n++)
		{
			//assume observations are from particle and transform coordinates to map coordinates
			double x_map = x + cos(theta) * observations[n].x - sin(theta) * observations[n].y;
			double y_map = y + sin(theta) * observations[n].x + cos(theta) * observations[n].y;

			// associate nearest neighbor landmark
			double nearest_dist = 10000.0;
			int nearest_id = -1;
			for (int m = 0; m < map_landmarks.landmark_list.size(); m++)
			{
				Map::single_landmark_s lm = map_landmarks.landmark_list[m];

				double lm_dist = dist(x_map, y_map, lm.x_f, lm.y_f);
				if (lm_dist < sensor_range)
				{
					if (lm_dist < nearest_dist)
					{
						nearest_dist = lm_dist;
						nearest_id = m; //lm.id_i;
					}
				}
			}
			// cout << "Transformed observation:" << x_map << ", " << y_map << ". Associated landmark ID: " << nearest_id << " ("
			// 	 << map_landmarks.landmark_list[nearest_id].id_i << ") " << " with coordinates "				 
			// 	 << map_landmarks.landmark_list[nearest_id].x_f << ", " << map_landmarks.landmark_list[nearest_id].y_f << endl;

			// add to particle struct
			associations.push_back(map_landmarks.landmark_list[nearest_id].id_i);
			sense_x.push_back(x_map);
			sense_y.push_back(y_map);

			// update weight of particle
			double diff_x = x_map - map_landmarks.landmark_list[nearest_id].x_f;
			double diff_y = y_map - map_landmarks.landmark_list[nearest_id].y_f;

			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			// calculate normalization term
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

			// calculate exponent
			double exponent = (diff_x * diff_x) / (2 * sig_x * sig_x) + (diff_y * diff_y) / (2 * sig_y * sig_y);

			// calculate weight using normalization terms and exponent
			particles[i].weight *= gauss_norm * exp(-exponent);
		}
		weights[i] = particles[i].weight;
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; i++)
	{

		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
