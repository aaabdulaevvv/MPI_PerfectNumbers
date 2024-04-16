#include <bits/stdc++.h>
#include <mpi.h>
#include <chrono>

using namespace std;

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 1)
    {
        if (rank == 0)
            cerr << "Usage: mpiexec -n number_of_proccesses " << argv[0] << " < input_file\n";
        MPI_Finalize();
        return 1;
    }

    int n;
    vector <int> perfect, gathered_perfect, gathered_perfect_sizes(size);
    int p;
    int rcounts[size];
    int rdisp[size];

    auto start = chrono::high_resolution_clock::now();

    if (rank == 0)
        scanf("%d",&n);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i=rank+2; i<=n; i+=size)
    {
        int sum = 1;
        for (int j=2; j<=i/2; j++)
            if (i%j == 0)
            sum += j;
        if (sum == i)
            perfect.push_back(i);
    }

    p = perfect.size();
    MPI_Gather(&p, 1, MPI_INT, gathered_perfect_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(gathered_perfect_sizes.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    int cnt = 0;
    for (int i=0; i<size; i++)
    {
        rdisp[i] = cnt;
        rcounts[i] = gathered_perfect_sizes[i];
        cnt += gathered_perfect_sizes[i];
    }
    gathered_perfect.resize(cnt);

    MPI_Gatherv(perfect.data(), rcounts[rank], MPI_INT, gathered_perfect.data(), rcounts, rdisp, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank==0)
    {
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Time spent: " << duration.count()/1000 << "." << duration.count()%1000 << " s\n";
        for (int i=0; i<gathered_perfect.size(); i++)
            cout << gathered_perfect[i] << " ";
    }

    MPI_Finalize();
}
