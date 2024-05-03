#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <vector>
#include <mpi.h>

double foo(double x) {
    return 1 / (pow(x, 5) + 1);
}

double solve(double x1, double x2, double y1, double y2, int N1, int rank, int num_processes) {
    int num = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_x(x1, x2);
    std::uniform_real_distribution<double> dis_y(y1, y2);

    int divided_N1 = N1 / num_processes + ((N1 % num_processes) > rank);
    for (int i = 0; i < divided_N1; ++i) {
        double x = dis_x(gen);
        double y = dis_y(gen);
        double fun = foo(x);
        if ((y >= 0 && y <= fun) || (y <= 0 && y >= fun)) {
            num += 1;
        }
    }

    double integral = 1.0 * num * (x2 - x1) * (y2 - y1) / N1;
    return integral;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " input_file num_processes" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int num_requested_processes = std::stoi(argv[2]);
    if (num_processes != num_requested_processes) {
        if (rank == 0)
            std::cerr << "Number of processes provided via mpirun does not match the requested number." << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<double> input_data(5);

    if (rank == 0) {
        std::ifstream input_file(argv[1]);
        if (!input_file.is_open()) {
            std::cerr << "Error opening input file" << std::endl;
            MPI_Finalize();
            return 1;
        }

        int N1;
        double x1, x2, y1, y2;
        input_file >> N1 >> x1 >> x2 >> y1 >> y2;
        input_file.close();

        input_data[0] = x1;
        input_data[1] = x2;
        input_data[2] = y1;
        input_data[3] = y2;
        input_data[4] = static_cast<double>(N1);

        // Send input data to other processes
        for (int dest = 1; dest < num_processes; ++dest) {
            MPI_Send(input_data.data(), input_data.size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(input_data.data(), input_data.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double x1 = input_data[0];
    double x2 = input_data[1];
    double y1 = input_data[2];
    double y2 = input_data[3];
    int N1 = static_cast<int>(input_data[4]);

    auto start_time = std::chrono::high_resolution_clock::now();
    double local_integral = solve(x1, x2, y1, y2, N1, rank, num_processes);

    std::vector<double> all_integrals(num_processes);
    MPI_Gather(&local_integral, 1, MPI_DOUBLE, all_integrals.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double global_integral = 0.0;
        for (double partial_integral : all_integrals) {
            global_integral += partial_integral;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::cout << "Integral: " << global_integral << std::endl;
        std::cout << "Time taken: " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}