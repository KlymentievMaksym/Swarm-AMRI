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

    // vector<int> unique_keep_order(const vector<int>& input)
    // {
    //     unordered_set<int> seen;
    //     vector<int> result;
    //     for (int num : input)
    //     {
    //         if (seen.find(num) == seen.end())
    //         {
    //             seen.insert(num);
    //             result.push_back(num);
    //         }
    //     }
    //     return result;
    // }

    Gene Crossover(Gene Parent1, Gene Parent2)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, Parent1.path.size());
        int cross = dis(gen);

        vector<int> P1S1(Parent1.path.begin(), Parent1.path.begin() + cross);
        vector<int> P2S2(Parent2.path.begin() + cross, Parent2.path.end());

        vector<int> P2S1(Parent2.path.begin(), Parent2.path.begin() + cross);
        vector<int> P1S2(Parent1.path.begin() + cross, Parent1.path.end());

        vector<int> path;
        path.reserve(P1S1.size() + P2S2.size() + P2S1.size() + P1S2.size());
        path.insert(path.end(), P1S1.begin(), P1S1.end());
        path.insert(path.end(), P2S2.begin(), P2S2.end());
        path.insert(path.end(), P2S1.begin(), P2S1.end());
        path.insert(path.end(), P1S2.begin(), P1S2.end());

        Gene Child(CitiesNum, true);
        Child.path = unique_unsorted(path);

        return Child;
    }

    // Gene Crossover(Gene Parent1, Gene Parent2)
    // {
    //     random_device rd;
    //     mt19937 gen(rd());
    //     uniform_int_distribution<> dis(1, Parent1.path.size() - 1); // Correct bounds
    //     int cross = dis(gen);
    
    //     vector<int> P1S1(Parent1.path.begin(), Parent1.path.begin() + cross);
    //     vector<int> P2S2(Parent2.path.begin() + cross, Parent2.path.end());
    //     vector<int> P2S1(Parent2.path.begin(), Parent2.path.begin() + cross);
    //     vector<int> P1S2(Parent1.path.begin() + cross, Parent1.path.end());
    
    //     vector<int> combined;
    //     combined.reserve(Parent1.path.size() * 2);
    //     combined.insert(combined.end(), P1S1.begin(), P1S1.end());
    //     combined.insert(combined.end(), P2S2.begin(), P2S2.end());
    //     combined.insert(combined.end(), P2S1.begin(), P2S1.end());
    //     combined.insert(combined.end(), P1S2.begin(), P1S2.end());
    
    //     Gene Child(Parent1.path.size(), true);
    //     Child.path = unique_keep_order(combined);
        
    //     return Child;
    // }

    vector<int> InvertVector(const vector<int>& a)
    {
        return vector<int>(a.rbegin(), a.rend());
    }

    void modifySlice(vector<int>& vec, size_t start, size_t end, vector<int>& newVec)
    {
        if (start >= vec.size() || end > vec.size() || start > end || vec.size() != newVec.size()) return;
        for (int i = start; i < end; i++)
        {
            fill(vec.begin() + start, vec.begin() + end, newVec[i - start]);
        }
    }

    // void Mutation(Gene& Child)
    // {
    //     random_device rd;
    //     mt19937 gen(rd());
    //     uniform_int_distribution<> disI(0, Child.path.size());
    //     int i = disI(gen);
    //     int j = disI(gen);
    //     while (i == j)
    //     {
    //         j = disI(gen);
    //     }
    //     if (i > j)
    //     {
    //         int temp = i;
    //         i = j;
    //         j = i;
    //     }
    //     uniform_real_distribution<> dis(0.0, 1.0);
    //     if (dis(gen) < 0.5)
    //     {
    //         int temp = Child.path[i];
    //         Child.path[i] = Child.path[j];
    //         Child.path[j] = temp;
    //     }
    //     else
    //     {
    //         vector<int> Invert = InvertVector(Child.path);
    //         vector<int> path(Invert.begin() + i, Invert.begin() + j);
    //         modifySlice(Child.path, i, j, path);
    //     }
        
    // }

    void Mutation(Gene& Child)
    {
        if (Child.path.empty()) return;
    
        random_device rd;
        mt19937 gen(rd());
        int n = Child.path.size();
        
        // Generate indices between 0 and n-1
        uniform_int_distribution<> dis(0, n - 1);
        int i = dis(gen);
        int j = dis(gen);
    
        // Ensure distinct indices
        while (i == j) {
            j = dis(gen);
        }
    
        // Ensure i < j
        if (i > j) {
            swap(i, j);
        }
    
        uniform_real_distribution<> prob(0.0, 1.0);
        if (prob(gen) < 0.5) {
            // Swap mutation
            swap(Child.path[i], Child.path[j]);
        } else {
            // Reverse subroute mutation
            reverse(Child.path.begin() + i, Child.path.begin() + j);
        }
    }

    vector<vector<double>> SelectCities(const vector<int>& path, const vector<vector<double>>& Cities)
    {
        vector<vector<double>> selected;
        selected.reserve(path.size());
        
        for (int index : path)
        {
            if (index >= 0 && index < Cities.size())
            {
                selected.push_back(Cities[index]);
            }
        }
        
        return selected;
    }

    vector<double> PowVector(const vector<double>& a, double PowNumber=1)
    {
        vector<double> result;
        result.reserve(a.size());
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(pow(a[i], PowNumber));
        }
        
        return result;
    }

    vector<vector<double>> SplitVectorCol(const vector<vector<double>>& a)
    {
        int SplitCount = a[0].size();
        vector<vector<double>> result;
        for (size_t i = 0; i < a[0].size(); ++i)
        {
            vector<double> result_small;
            
            for (size_t j = 0; j < a.size(); ++j)
            {
                result_small.push_back(a[j][i]);
            }
            result.push_back(result_small);
        }
        
        return result;
    }

    vector<double> AddVectors(const vector<double>& a, const vector<double>& b, double PowNumber=1)
    {
        if (a.size() != b.size())
        {
            throw invalid_argument("Vectors must be of equal size");
        }
        
        vector<double> result;
        result.reserve(a.size());
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(pow(a[i], PowNumber) + pow(b[i], PowNumber));
        }
        
        return result;
    }

    vector<double> SubtractVectors(const vector<double>& a, const vector<double>& b)
    {
        if (a.size() != b.size())
        {
            throw invalid_argument("Vectors must be of equal size");
        }
        
        vector<double> result;
        result.reserve(a.size());
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(a[i] - b[i]);
        }
        
        return result;
    }

    vector<vector<double>> SubtractVectorsVectors(const vector<vector<double>>& a, const vector<vector<double>>& b)
    {
        if (a.size() != b.size())
        {
            throw invalid_argument("Vectors must be of equal size");
        }
        
        vector<vector<double>> result;
        result.reserve(a.size());
        
        for (size_t i = 0; i < a.size(); ++i)
        {
            result.push_back(SubtractVectors(a[i], b[i]));
        }
        
        return result;
    }

    double Fitness(Gene gene, const vector<vector<double>>& Cities) {
        vector<int> path_prev = gene.path;
        vector<int> path_next;
    
        // Create shifted path (next = prev[1:] + prev[0])
        path_next.reserve(path_prev.size());
        path_next.insert(path_next.end(), path_prev.begin() + 1, path_prev.end());
        path_next.push_back(path_prev[0]);
    
        double total_distance = 0.0;
        
        for (size_t i = 0; i < path_prev.size(); ++i) {
            // Get coordinates for both points
            const auto& city_prev = Cities[path_prev[i]];
            const auto& city_next = Cities[path_next[i]];
            
            // Calculate squared differences
            double dx = city_prev[0] - city_next[0];
            double dy = city_prev[1] - city_next[1];
            
            // Accumulate distance
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
        srand(time(0)); // Seed random number generator
        float BestLength = numeric_limits<float>::max();
        vector<int> BestRoute;

        vector<Gene> Genes(PopSize + ChildSize, Gene(CitiesNum, true));
        for (int GeneIndex = 0; GeneIndex < PopSize; ++GeneIndex)
        {
            Gene gene(CitiesNum);
            Genes[GeneIndex] = gene;
        }

        for (int Iteration = 0; Iteration < Iterations; Iteration++)
        {
            // for (int GeneIndex = 0; GeneIndex < PopSize + ChildSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // break;
            for (int ChildIndex=PopSize; ChildIndex < PopSize + ChildSize; ChildIndex++)
            {
                Gene Child(CitiesNum, true);
                int i = rand() % CitiesNum;
                int j = rand() % CitiesNum;
                while (i == j)
                {
                    j = rand() % CitiesNum;
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


                if (dis(gen) < 0.5)
                {
                    Child = Crossover(Genes[i], Genes[j]);
                }
                else
                {
                    Child = Crossover(Genes[j], Genes[i]);
                }

                // json update;
                // update["Message"] = Child.path;
                // cout << update.dump() << endl;
                // cout.flush();

                if (dis(gen) < MutationRate)
                {
                    Mutation(Child);
                }

                // update["Message"] = Child.path;
                // cout << update.dump() << endl;
                // cout.flush();

                Genes[ChildIndex] = Child;
                // break;
            }
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
            
            // for (int GeneIndex = 0; GeneIndex < PopSize + ChildSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // break;

            // for (int GeneIndex = 0; GeneIndex < PopSize + ChildSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            //     update["Message"] = Genes[GeneIndex].length;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // json update;
            // update["Message"] = "\n\n\n\n";
            // cout << update.dump() << endl;
            // cout.flush();

            sort(Genes.begin(), Genes.end(), [](Gene const &a, Gene const &b){return a.length < b.length;});
            
            // for (int GeneIndex = 0; GeneIndex < PopSize + ChildSize; ++GeneIndex)
            // {
            //     json update;
            //     update["Message"] = Genes[GeneIndex].path;
            //     cout << update.dump() << endl;
            //     cout.flush();
            //     update["Message"] = Genes[GeneIndex].length;
            //     cout << update.dump() << endl;
            //     cout.flush();
            // }
            // break;

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
    Genetic genes(cities, j["pop_size"], j["iterations"], j["child_size"], j["mutation_probability"], j["every"]);
    genes.run();

    return 0;
}