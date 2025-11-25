#include <iostream>
#include <random>

#include "utils.hpp"
#include "vector.hpp"

void printv(const VectorView& view)
{
    for (size_t i = 0; i < view.size(); i++)
        std::cout << view[i] << std::endl;
    std::cout << "----------" << std::endl;
}

int main(int argc, const char** argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> dist(0, 1);

    Vector v(10);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = dist(rng);

    VectorView view = v;
    printv(view);

    std::vector<size_t> indices = argsort(view);
    printv(view[indices]);

    Mask mask = (view[indices] <= 0);
    std::cout << "first" << std::endl;
    printv(view[indices][mask]);
    std::cout << "second" << std::endl;
    printv(view[indices][mask & (view[indices] <= -0.5)]);

    return 0;
}
