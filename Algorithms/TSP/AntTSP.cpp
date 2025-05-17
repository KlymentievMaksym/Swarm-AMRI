#include "../Include/json.hpp"
using json = nlohmann::json;

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <climits>
#include <random>

using namespace std;

class AntColony
{
private:
    int CitiesNum;
    vector<vector<double>> Distances, RevertDistance, Pheromone;

    const vector<vector<double>>& Cities;
    int PopSize, Iterations;
    double Alpha, Beta, Rho, Q;

    struct Ant
    {
        vector<int> path;
        vector<bool> visited;
        double tourLength;

        Ant(int n) : visited(n, false), tourLength(0.0) {}
    };

    int SendEvery;

public:
    float Distance(const vector<double>& a, const vector<double>& b)
    {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (b[i] - a[i]) * (b[i] - a[i]);
        }
        return sqrt(sum);
    }

    AntColony(const vector<vector<double>>& Cities, int PopSize, int Iterations, double Alpha, double Beta, double Rho, double Q, int SendEvery):
    Cities(Cities), PopSize(PopSize), Iterations(Iterations), Alpha(Alpha), Beta(Beta), Rho(Rho), Q(Q), SendEvery(SendEvery)
    {
        CitiesNum = Cities.size();
        
        Distances = vector<vector<double>>(CitiesNum, vector<double>(CitiesNum, 0.0));
        for (int i = 0; i < CitiesNum; i++)
        {
            for (int j = 0; j < CitiesNum; j++)
            {
                Distances[i][j] = Distance(Cities[i], Cities[j]);
            }
        }
        RevertDistance = vector<vector<double>> (CitiesNum, vector<double>(CitiesNum, 0));
        for (int i = 0; i < CitiesNum; i++)
        {
            for (int j = 0; j < CitiesNum; j++)
            {
                RevertDistance[i][j] = 1.0 / Distances[i][j];
            }
        }
        Pheromone = vector<vector<double>> (CitiesNum, vector<double>(CitiesNum, 1e-6));
    }

    void run()
    {
        srand(time(0)); // Seed random number generator
        float BestLength = numeric_limits<float>::max();
        vector<int> BestRoute;

        for (int Iteration = 0; Iteration < Iterations; Iteration++)
        {
            vector<Ant> ants;
            for (int AntIndex = 0; AntIndex < PopSize; ++AntIndex)
            {
                Ant ant(CitiesNum);
                int startCity = rand() % CitiesNum;
                ant.path.push_back(startCity);
                ant.visited[startCity] = true;
                ants.push_back(ant);
            }
            for (Ant& ant : ants)
            {
                for (int Step = 1; Step < CitiesNum; Step++)
                {
                    int currentCity = ant.path.back();
                    vector<double> probabilities(CitiesNum, 0.0);
                    double total = 0.0;

                    // Calculate probabilities for each possible next city
                    vector<int> rangeVec(CitiesNum);
                    for (int i = 0; i < CitiesNum; i++)
                    {
                        rangeVec[i] = i;
                    }
                    for (int ProbIndex = 0; ProbIndex < CitiesNum; ProbIndex++)
                    {
                        if (!ant.visited[ProbIndex])
                        {
                            double PheromoneLevel = Pheromone[currentCity][ProbIndex];
                            double RevertDistanceValue = RevertDistance[currentCity][ProbIndex];
                            probabilities[ProbIndex] = pow(PheromoneLevel, Alpha) * pow(RevertDistanceValue, Beta);
                            total += probabilities[ProbIndex];
                        }
                    }
                    for (int ProbIndex = 0; ProbIndex < CitiesNum; ProbIndex++)
                    {
                        probabilities[ProbIndex] /= total;
                    }
                    // Select one element from City based on the probabilities
                    random_device rd;
                    mt19937 gen(rd());
                    discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                
                    int NextCity = rangeVec[dist(gen)];
                    ant.path.push_back(NextCity);
                    ant.visited[NextCity] = true;
                    ant.tourLength += Distances[currentCity][NextCity];
                }
                // Complete the tour by returning to the start city
                ant.tourLength += Distances[ant.path.back()][ant.path[0]];
            }
            
            // Update pheromones
            // Evaporation
            for (int i = 0; i < CitiesNum; ++i)
            {
                for (int j = 0; j < CitiesNum; ++j)
                {
                    Pheromone[i][j] *= (1.0 - Rho);
                }
            }
            // Deposit pheromones by all ants
            for (Ant& ant : ants)
            {
                double deposit = Q / ant.tourLength;
                for (int i = 0; i < CitiesNum; ++i)
                {
                    int from = ant.path[i];
                    int to = ant.path[(i + 1) % CitiesNum];
                    Pheromone[from][to] += deposit;
                    Pheromone[to][from] += deposit;
                }
                // Update best tour
                if (ant.tourLength < BestLength)
                {
                    BestLength = ant.tourLength;
                    BestRoute = ant.path;
                }
            }
            if ((Iteration + 1) % SendEvery == 0)
            {
                // Print after best update
                json update;
                update["iteration"] = Iteration;
                update["best_distance"] = BestLength;
                update["best_route"] = BestRoute;
                cout << update.dump() << endl;
                cout.flush();
            }
            
        }
    }
};

int main()
{
    string input((istreambuf_iterator<char>(cin)), {});
    json j = json::parse(input);

    // Parse cities
    vector<vector<double>> cities;
    for (auto& row : j["cities"])
    {
        cities.emplace_back(row);
    }

    // Solve TSP
    AntColony colony(cities, j["pop_size"], j["iterations"], j["alpha"], j["beta"], j["rho"], j["Q"], j["every"]);
    colony.run();

    return 0;
}