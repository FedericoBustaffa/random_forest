# Parallel Random Forest

Implementations of a **Parallel Random Forest** in C++ for my Parallel and
Distributed Systems course.

| Version      | Multi-Thread | Multi-Node |
| ------------ | ------------ | ---------- |
| Sequential   | No           | No         |
| OpenMP       | Yes          | No         |
| FastFlow     | Yes          | No         |
| OpenMP + MPI | Yes          | Yes        |

The sequential base version was inspired by another simple implentation found on
GitHub.

## Used Libraries and References

- Multi-Threading: [OpenMP](https://www.openmp.org/) and
  [FastFlow](https://github.com/fastflow/fastflow)
- Multi-Node: [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/)
