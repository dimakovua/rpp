#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

double foo(double x) {
    if (x == 0) return 1; // Handle division by zero
    return 1 / (pow(x, 5) + 1);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./montecarloopenmp <input_file> <num_processes>\n";
        return 1;
    }

    // Read input parameters from file
    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "Failed to open input file.\n";
        return 1;
    }
    
    int k = atoi(argv[2]);
    int N1;
    double x1, x2, y1, y2;
    input >> N1 >> x1 >> x2 >> y1 >> y2;

    double start_time = omp_get_wtime(); // Start measuring time

    int num = 0;
    #pragma omp parallel num_threads(k) reduction(+:num)
    {
        unsigned int seed = static_cast<unsigned int>(time(NULL) + omp_get_thread_num()); // Thread-private seed

        // Explicitly set the number of threads
        omp_set_num_threads(k);

        // Parallelize the outer loop
        #pragma omp for schedule(static)
        for (int i = 0; i < k; ++i) {
            int divided_N1 = N1 / k + ((N1 % k) > i);
            for (int j = 0; j < divided_N1; ++j) {
                double x = x1 + (x2 - x1) * rand_r(&seed) / (RAND_MAX + 1.0);
                double y = y1 + (y2 - y1) * rand_r(&seed) / (RAND_MAX + 1.0);
                double fun = foo(x);
                if (0 <= y && y <= fun) {
                    num++;
                } else if (0 >= y && y >= fun) {
                    num--;
                }
            }
        }
    }

    // Calculate integral
    double integral = 1.0 * num * (x2 - x1) * (y2 - y1) / N1;

    double end_time = omp_get_wtime(); // End measuring time
    double elapsed_time = end_time - start_time; // Calculate elapsed time

    std::cout << "Integral: " << integral << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
