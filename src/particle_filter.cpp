#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// Initialize the number of particles
	num_particles = 100;

	// Make a Generator
	default_random_engine gen;

	// Create a normal (Gaussian) distribution for x, y, Theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Sample particles from the distribution
	for (int i = 0; i < num_particles; ++i)
	{
		// Sample from the normal distrubtions 
		// where "gen" is the random engine initialized earlier.
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1.0);
	}

	// To make sure init executes only once
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;

	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	// Sample from the normal distributions
	for (auto& particle:particles)
	{
 		double theta = particle.theta;

		// expected pose value from deterministic motion model
		if (fabs(yaw_rate) <= 0.001)
		{
			particle.x += velocity*cos(theta)*delta_t + N_x(gen);
			particle.y += velocity*sin(theta)*delta_t + N_y(gen);
		}
		else
		{
			particle.x += velocity / yaw_rate*(sin(theta + yaw_rate*delta_t) - sin(theta)) + N_x(gen);
			particle.y += velocity / yaw_rate*(-cos(theta + yaw_rate*delta_t) + cos(theta)) + N_y(gen);
		}
		particle.theta += yaw_rate*delta_t + N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); ++i)
	{
		double min = 999.9;
		int min_id = 0;
		for (int j = 0; j < predicted.size(); ++j)
		{
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if ( distance < min) {
				min = distance;
				min_id = predicted[j].id;
			}
		}
		observations[i].id = min_id;
	}
}

double prob(double x, double y, float xm, float ym, double std_landmark[])
{
	double xp = -0.5*(x - xm)*(x - xm) / (std_landmark[0] * std_landmark[0]);
	double yp = -0.5*(y - ym)*(y - ym) / (std_landmark[1] * std_landmark[1]);
	return exp(xp + yp) / (2 * M_PI*std_landmark[0] * std_landmark[1]);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	Map t_map_landmarks;
	t_map_landmarks.landmark_list.clear(); // clear the vector
	weights.clear(); // clear the previous weights vector

	for (auto& particle : particles)
	{
		double ctheta = cos(particle.theta);
		double stheta = sin(particle.theta);
		double xtrans = particle.x;
		double ytrans = particle.y;

		for (int z = 0; z < map_landmarks.landmark_list.size(); ++z)
		{
			if (dist(xtrans, ytrans, map_landmarks.landmark_list[z].x_f,
					map_landmarks.landmark_list[z].y_f) < sensor_range)
			{
				t_map_landmarks.landmark_list.push_back(map_landmarks.landmark_list[z]);
			}
		}

		particle.weight = 1.0; // set it to 1 before the next calculations begin

		LandmarkObs MapObs;
		for (auto& MapObs : observations)
		{
			double temp_x = ctheta * MapObs.x - stheta * MapObs.y + xtrans;
			double temp_y = stheta * MapObs.x + ctheta * MapObs.y + ytrans;

			//particle.sense_x[j] = temp_x; //obs rotated then translated 
			//particle.sense_y[j] = temp_y; //obs rotated then translated

			double min = 999.9;
			int min_id = 0; 
			for (int k = 0; k < t_map_landmarks.landmark_list.size(); ++k)
			{
				double temp_dist = dist(temp_x, temp_y,
						t_map_landmarks.landmark_list[k].x_f,
						t_map_landmarks.landmark_list[k].y_f);
				if (temp_dist < min)
				{
					min = temp_dist;
					min_id = t_map_landmarks.landmark_list[k].id_i;
				}
			}
			
			if (observations.size() == 0) {
				particle.weight = 0.0;
			}
			else
			{
				particle.weight *= prob(temp_x, temp_y,
						map_landmarks.landmark_list[min_id-1].x_f,
						map_landmarks.landmark_list[min_id-1].y_f, std_landmark);
			}
		}
		weights.push_back(particle.weight);
		t_map_landmarks.landmark_list.clear();
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::map<int, int> m;

	for (int n = 0; n < num_particles; ++n)
	{
		resampled_particles.push_back(particles[d(gen)]);
	}
	particles = resampled_particles; // re-sampled set of particles
	weights.clear(); // clear the previous weights vector
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations,
		std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
    s = s.substr(0, s.length()-1);
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
