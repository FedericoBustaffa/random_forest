#ifndef BACKEND_HPP
#define BACKEND_HPP

enum class Backend
{
    Sequential,
    OpenMP,
    FastFlow,
    MPI,
    Invalid
};

#endif
