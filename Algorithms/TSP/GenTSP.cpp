#include "../Include/json.hpp"
using json = nlohmann::json;

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <climits>
#include <random>
#include <unordered_set>
#include <stdexcept> // for invalid_argument

using namespace std;

vector<int> random_permutation(int n)
{
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0); // Fill with 0 to n-1

    random_device rd;
    mt19937 g(rd());
    shuffle(perm.begin(), perm.end(), g);

    return perm;
}

class Genetic
{
private:
    int CitiesNum;
    vector<vector<double>> Distances;

    const vector<vector<double>>& Cities;
    int PopSize, Iterations, ChildSize;
    float MutationRate;

    struct Gene
    {
        vector<int> path;
        double length;

        Gene(int n) : path(random_permutation(n)), length(0.0) {}
        Gene(int Num, bool pattern) : path(Num, 0), length(0.0) {}
        Gene(vector<int> path) : path(path), length(0.0) {}
    };

    int SendEvery;

public:
    float Distance(const vector<double>& a, const vector<double>& b)
    {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            sum += (b[i] - a[i]) * (b[i] - a[i]);
        }
        return sqrt(sum);
    }

    vector<int> unique_unsorted(const vector<int>& Input)
    {
        unordered_set<int> Seen;
        vector<int> Result;
        
        for (const auto& elem : Input)
        {
            if (Seen.insert(elem).second) // .second is true if insertion happened
            { 
                Result.push_back(elem);
            }
        }
        
        return Result;
    }

    Gene Crossover(Gene Parent1, Gene Parent2)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, Parent1.path.size());
        int cross = dis(gen);

        // json update;
        // update["Message"] = "Prepare for Parents cross";
        // cout << update.dump() << endl;
        // cout.flush();

        // update["Message"] = Parent1.path.size();
        // cout << update.dump() << endl;
        // cout.flush();
        // update["Message"] = cross;
        // cout << update.dump() << endl;
        // cout.flush();
        // update["Message"] = Parent2.path.size();
        // cout << update.dump() << endl;
        // cout.flush();
        // update["Message"] = cross;
        // cout << update.dump() << endl;
        // cout.flush();

        vector<int> P1S1(Parent1.path.begin(), Parent1.path.begin() + cross);
        vector<int> P2S2(Parent2.path.begin() + cross, Parent2.path.end());

        vector<int> P2S1(Parent2.path.begin(), Parent2.path.begin() + cross);
        vector<int> P1S2(Parent1.path.begin() + cross, Parent1.path.end());

        // update["Message"] = "Prepare for Connecting";
        // cout << update.dump() << endl;
        // cout.flush();

        vector<int> path;
        path.reserve(P1S1.size() + P2S2.size() + P2S1.size() + P1S2.size());
        path.insert(path.end(), P1S1.begin(), P1S1.end());
        path.insert(path.end(), P2S2.begin(), P2S2.end());
        path.insert(path.end(), P2S1.begin(), P2S1.end());
        path.insert(path.end(), P1S2.begin(), P1S2.end());

        // update["Message"] = "Generate Child";
        // cout << update.dump() << endl;
        // cout.flush();

        Gene Child(CitiesNum, true);
        Child.path = unique_unsorted(path);

        return Child;
    }

    void Mutation(Gene& Child)
    {
        // json update;
        // update["Message"] = "Prepare for RANDOM";
        // cout << update.dump() << endl;
        // cout.flush();
        // if (Child.path.empty()) return;
    
        random_device rd;
        mt19937 gen(rd());
        int n = Child.path.size();
        
        uniform_int_distribution<> dis(0, n - 1);
        
        // update["Message"] = "Prepare for I and J";
        // cout << update.dump() << endl;
        // cout.flush();

        int i = dis(gen);
        int j = dis(gen);

        // update["Message"] = "Prepare Check I == J";
        // cout << update.dump() << endl;
        // cout.flush();

        while (i == j && n > 2) {
            j = dis(gen);
        }
        // update["Message"] = "Prepare for Mutation SWAP";
        // cout << update.dump() << endl;
        // cout.flush();
        if (i > j) {
            swap(i, j);
        }
    
        // update["Message"] = "Prepare for Mutation MAIN";
        // cout << update.dump() << endl;
        // cout.flush();

        uniform_real_distribution<> prob(0.0, 1.0);
        if (prob(gen) < 0.5) {
            swap(Child.path[i], Child.path[j]);
        } else {
            reverse(Child.path.begin() + i, Child.path.begin() + j);
        }
        // update["Message"] = "Ended Mutation MAIN";
        // cout << update.dump() << endl;
        // cout.flush();
    }

    double Fitness(vector<int> path, const vector<vector<double>>& Cities)
    {
        vector<int> path_prev = path;
        vector<int> path_next;
    
        path_next.reserve(path_prev.size());
        path_next.insert(path_next.end(), path_prev.begin() + 1, path_prev.end());
        path_next.push_back(path_prev[0]);
    
        double total_distance = 0.0;
        
        for (size_t i = 0; i < path_prev.size(); ++i)
        {
            const auto& city_prev = Cities[path_prev[i]];
            const auto& city_next = Cities[path_next[i]];
            
            double dx = city_prev[0] - city_next[0];
            double dy = city_prev[1] - city_next[1];
            
            total_distance += sqrt(dx*dx + dy*dy);
        }
        
        return total_distance;
    }

    double Fitness(Gene gene, const vector<vector<double>>& Cities)
    {
        vector<int> path_prev = gene.path;
        vector<int> path_next;
    
        path_next.reserve(path_prev.size());
        path_next.insert(path_next.end(), path_prev.begin() + 1, path_prev.end());
        path_next.push_back(path_prev[0]);
    
        double total_distance = 0.0;
        
        for (size_t i = 0; i < path_prev.size(); ++i)
        {
            const auto& city_prev = Cities[path_prev[i]];
            const auto& city_next = Cities[path_next[i]];
            
            double dx = city_prev[0] - city_next[0];
            double dy = city_prev[1] - city_next[1];
            
            total_distance += sqrt(dx*dx + dy*dy);
        }
        
        return total_distance;
    }

    Genetic(const vector<vector<double>>& Cities, int PopSize, int Iterations, int ChildSize, float MutationRate, int SendEvery):
    Cities(Cities), PopSize(PopSize), Iterations(Iterations), ChildSize(ChildSize), MutationRate(MutationRate), SendEvery(SendEvery)
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
    }

    void run()
    {
        srand(time(0));
        float BestLength = numeric_limits<float>::max();
        vector<int> BestRoute;

        vector<Gene> Genes(PopSize + ChildSize, Gene(CitiesNum, true));
        for (int GeneIndex = 0; GeneIndex < PopSize; ++GeneIndex)
        {
            Gene gene(CitiesNum);
            Genes[GeneIndex] = gene;
        }

        // for (int GeneIndex = 0; GeneIndex < PopSize; ++GeneIndex)
        // {
        //     json update;
        //     update["Message"] = Genes[GeneIndex].path;
        //     cout << update.dump() << endl;
        //     cout.flush();
        // }
        
        // random_device rd;
        // mt19937 gen(rd());
        // uniform_real_distribution<> dis(0.0, 1.0);

        for (int Iteration = 0; Iteration < Iterations; Iteration++)
        {
            // break;
            for (int ChildIndex=PopSize; ChildIndex < PopSize + ChildSize; ChildIndex++)
            {
                Gene Child(CitiesNum, true);
                int i = rand() % PopSize;
                int j = rand() % PopSize;
                while (i == j)
                {
                    j = rand() % PopSize;
                }
                if (i > j)
                {
                    int temp = i;
                    i = j;
                    j = temp;
                }
                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<> dis(0.0, 1.0);

                // json update;
                // update["Message"] = "Prepare for Crossover";
                // cout << update.dump() << endl;
                // cout.flush();

                if (dis(gen) < 0.5)
                {
                    Child = Crossover(Genes[i], Genes[j]);
                }
                else
                {
                    Child = Crossover(Genes[j], Genes[i]);
                }
                // update["Message"] = "Ended Crossover";
                // cout << update.dump() << endl;
                // cout.flush();
                // update["Message"] = "Prepare for Mutation";
                // cout << update.dump() << endl;
                // cout.flush();
                if (dis(gen) < MutationRate)
                {
                    Mutation(Child);
                }
                // update["Message"] = "Ended Mutation";
                // cout << update.dump() << endl;
                // cout.flush();
                // update["Message"] = "Prepare for Writing";
                // cout << update.dump() << endl;
                // cout.flush();
                Genes[ChildIndex] = Child;
                // update["Message"] = Genes[ChildIndex].path;
                // cout << update.dump() << endl;
                // cout.flush();
            }

            // for (int GeneIndex = 0; GeneIndex < PopSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // break;
            for (int GeneIndex = 0; GeneIndex < PopSize + ChildSize; ++GeneIndex)
            {
                Genes[GeneIndex].length = Fitness(Genes[GeneIndex], Cities);
                Gene gene = Genes[GeneIndex];
                if (gene.length < BestLength)
                {
                    BestLength = gene.length;
                    BestRoute = gene.path;
                }
            }

            sort(Genes.begin(), Genes.end(), [](Gene const &a, Gene const &b){return a.length < b.length;});
            // for (int GeneIndex = 0; GeneIndex < PopSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // break;
            if ((Iteration + 1) % SendEvery == 0)
            {
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

    vector<vector<double>> cities;
    for (auto& row : j["cities"])
    {
        cities.emplace_back(row);
    }

    Genetic genes(cities, j["pop_size"], j["iterations"], j["child_size"], j["mutation_probability"], j["every"]);
    genes.run();

    return 0;
}