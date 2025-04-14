#include "../Include/json.hpp"
using json = nlohmann::json;

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

using namespace std;

using City = pair<double, double>;
using Tour = vector<int>;

double distance(const City& a, const City& b) {
    return hypot(a.first - b.first, a.second - b.second);
}

vector<vector<double>> distance_matrix(const vector<City>& cities) {
    int n = cities.size();
    vector<vector<double>> dist(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dist[i][j] = distance(cities[i], cities[j]);
    return dist;
}

pair<Tour, double> ant_colony_tsp(
    const vector<City>& cities,
    int n_ants = 10, int n_iterations = 100,
    double alpha = 1, double beta = 5, double rho = 0.5, double Q = 100
) {
    int n = cities.size();
    auto dist = distance_matrix(cities);
    vector<vector<double>> pheromone(n, vector<double>(n, 1.0 / n));
    double best_distance = numeric_limits<double>::max();
    Tour best_tour;

    random_device rd;
    mt19937 gen(rd());

    for (int iter = 0; iter < n_iterations; ++iter) {
        vector<Tour> all_tours;
        vector<double> all_distances;

        for (int ant = 0; ant < n_ants; ++ant) {
            Tour tour;
            tour.push_back(gen() % n);
            while (tour.size() < n) {
                int i = tour.back();
                vector<pair<int, double>> probs;

                for (int j = 0; j < n; ++j) {
                    if (find(tour.begin(), tour.end(), j) == tour.end()) {
                        double tau = pow(pheromone[i][j], alpha);
                        double eta = pow(1.0 / dist[i][j], beta);
                        probs.emplace_back(j, tau * eta);
                    }
                }

                double sum = 0;
                for (auto& p : probs) sum += p.second;
                uniform_real_distribution<> dis(0.0, 1.0);
                double r = dis(gen), cumulative = 0.0;
                for (auto& [city, p] : probs) {
                    cumulative += p / sum;
                    if (r <= cumulative) {
                        tour.push_back(city);
                        break;
                    }
                }
            }

            double tour_dist = 0;
            for (int i = 0; i < n; ++i)
                tour_dist += dist[tour[i]][tour[(i + 1) % n]];

            if (tour_dist < best_distance) {
                best_distance = tour_dist;
                best_tour = tour;
            }

            all_tours.push_back(tour);
            all_distances.push_back(tour_dist);
        }

        for (auto& row : pheromone)
            for (double& val : row)
                val *= (1 - rho);

        for (int t = 0; t < all_tours.size(); ++t) {
            auto& tour = all_tours[t];
            double d = all_distances[t];
            for (int i = 0; i < n; ++i) {
                int a = tour[i], b = tour[(i + 1) % n];
                pheromone[a][b] += Q / d;
                pheromone[b][a] += Q / d;
            }
        }
        // Print after best update
        json update;
        update["iteration"] = iter;
        update["best_distance"] = best_distance;
        update["best_route"] = best_tour;
        cout << update.dump() << endl;
        cout.flush();  // important!
    }

    return {best_tour, best_distance};
}

int main() {
    // Read input
    string input((istreambuf_iterator<char>(cin)), {});
    json j = json::parse(input);

    // Parse cities
    vector<City> cities;
    for (auto& row : j["cities"]) {
        cities.emplace_back(row[0], row[1]);
    }

    // Solve TSP
    auto [tour, dist] = ant_colony_tsp(cities, j["n_ants"], j["n_iterations"], j["alpha"], j["beta"], j["rho"], j["Q"]);

    // Respond with best route and distance
    // json response;
    // response["iteration"] = 100;
    // response["best_distance"] = dist;
    // response["best_route"] = tour;

    // cout << response.dump() << endl;
}
