#ifdef PY_BIND
#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
namespace py = pybind11;
#endif

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <ctime>
#include <random>
#include <cassert>
using namespace std;

#define IL inline
#define len(a) int((a).size())
#define append push_back


namespace immortalco {
    int D;
    int mov;
    vector<int> g_match;
    bool debug = false;

    void set_debug(bool _debug) {
        debug = _debug;
        printf("debug mode: %d\n", debug);
        if (debug) {
            freopen("scratch/point_match_debug.txt", "w", stdout);
        }
        fflush(stdout);
    }

    struct Point {
        vector<double> d;
        int i;
        bool operator<(const Point &p) const {
            return d[D] < p.d[D];
        }
    };


    double std(const vector<double> &a) {
        double s1 = 0, s2 = 0;
        for (auto x : a) {
            s1 += x;
            s2 += x * x;
        }
        s1 /= len(a);
        s2 /= len(a);
        return sqrt(s2 - s1 * s1 + 1e-12);
    }

    void build(int K, vector<Point> &A, vector<Point> &B, int l, int r) {
        if (l == r) return;

        D = -1;
        double max_var = -1;
        for(int d = 0; d < len(A[0].d); d++) {

            double A1 = 0, A2 = 0, B1 = 0, B2 = 0;
            for (int i = l; i <= r; ++i) {
                A1 += A[i].d[d];
                A2 += A[i].d[d] * A[i].d[d];
                B1 += B[i].d[d];
                B2 += B[i].d[d] * B[i].d[d];
            }
            A1 /= r - l + 1;
            A2 /= r - l + 1;
            B1 /= r - l + 1;
            B2 /= r - l + 1;
            double A_var = A2 - A1 * A1;
            double B_var = B2 - B1 * B1;
            double var = min(A_var, B_var);
            if (var > max_var) {
                max_var = var;
                D = d;
            }
            
        }
        if (D == -1) {assert(false);}

        if (K == 2) {
            int m = (l + r) / 2;
            nth_element(A.begin() + l, A.begin() + m, A.begin() + r + 1);
            nth_element(B.begin() + l, B.begin() + m, B.begin() + r + 1);
            build(K, A, B, l, m);
            build(K, A, B, m + 1, r);
        } else {
            int each = (r - l + 1 + K - 1) / K;
            sort(A.begin() + l, A.begin() + r + 1);
            sort(B.begin() + l, B.begin() + r + 1);
            for(int i = l; i <= r; i += each) {
                int j = min(i + each - 1, r);
                build(max(2, K / 4), A, B, i, j);
            }
        }
        
    }

    vector<int> match(int k, vector<vector<double>> raw_A, vector<vector<double>> raw_B) {
        int N = len(raw_A);
        assert(N == len(raw_B));

        vector<Point> A, B;
        for (int i = 0; i < N; ++i) {
            A.append({raw_A[i], i});
            B.append({raw_B[i], i});
        }

        build(k, A, B, 0, N - 1);

        vector<int> match(N, -1);
        for (int i = 0; i < N; ++i) {
            match[A[i].i] = B[i].i;
        }
        for (int i = 0; i < N; ++i) 
            if(match[i] == -1) {assert(false);}

        return match;
    }

    vector<int> match2d(vector<vector<double>> raw_A, vector<vector<double>> raw_B) {
        return match(2, raw_A, raw_B);
    }

    void build2(vector<Point> &A, vector<Point> &B) {
        int N = len(A);

        if (debug) {
            printf("build2 N = %d\n", N);
            fflush(stdout);
        }

        assert(len(A) == len(B));
        if (N == 0) return;
        if (N == 1) {
            g_match[A[0].i] = B[0].i;
            return;
        }
        D = -1;
        double max_var = -1;

        for(int d = 0; d < len(A[0].d); d++) {
            double amean = 0, bmean = 0;
            for (int i = 0; i < N; ++i) {
                amean += A[i].d[d];
                bmean += B[i].d[d];
            }
            amean /= N;
            bmean /= N;
            for (int i = 0; i < N; ++i) {
                A[i].d[d] -= amean;
                B[i].d[d] -= bmean;
            }
        }

        for(int d = 0; d < len(A[0].d); d++) {
            double A1 = 0, A2 = 0, B1 = 0, B2 = 0;
            for (int i = 0; i < len(A); ++i) {
                A1 += A[i].d[d];
                A2 += A[i].d[d] * A[i].d[d];
                B1 += B[i].d[d];
                B2 += B[i].d[d] * B[i].d[d];
            }
            A1 /= len(A);
            A2 /= len(A);
            B1 /= len(A);
            B2 /= len(A);
            double A_var = A2 - A1 * A1;
            double B_var = B2 - B1 * B1;
            double var = min(A_var, B_var);
            if (var > max_var) {
                max_var = var;
                D = d;
            }
            
        }
        if (D == -1) {assert(false);}

        vector<Point> all;
        for (int i = 0; i < len(A); ++i) {
            all.append(A[i]);
            all.back().i += 0;
            all.append(B[i]);
            all.back().i += mov;
        }

        nth_element(all.begin(), all.begin() + len(all) / 2, all.end());
        vector<Point> Al, Ar, Bl, Br;
        for (int i = 0; i < len(all) / 2; ++i) {
            auto &p = all[i];
            if (p.i < mov) Al.append(p);
            else { p.i -= mov; Bl.append(p); }
        }
        for (int i = len(all) / 2; i < len(all); ++i) {
            auto &p = all[i];
            if (p.i < mov) Ar.append(p);
            else { p.i -= mov; Br.append(p); }
        }

        if (debug) {
            printf("build2 Al = %d Ar = %d Bl = %d Br = %d\n", len(Al), len(Ar), len(Bl), len(Br));
            fflush(stdout);
        }

        if (len(Al) != len(Bl)) {
            vector<Point> Am, Bm;
            if (len(Al) > len(Bl)) {
                int more = len(Al) - len(Bl);
                assert(more <= len(Al));
                assert(len(Al) == len(Br));
                nth_element(Al.begin(), Al.end() - more, Al.end());
                nth_element(Br.begin(), Br.begin() + more, Br.end());
                Am = vector<Point>(Al.end() - more, Al.end());
                Bm = vector<Point>(Br.begin(), Br.begin() + more);
                Al = vector<Point>(Al.begin(), Al.end() - more);
                Br = vector<Point>(Br.begin() + more, Br.end());
            } else {
                int more = len(Bl) - len(Al);
                assert(more <= len(Bl));
                assert(len(Bl) == len(Ar));
                nth_element(Bl.begin(), Bl.end() - more, Bl.end());
                nth_element(Ar.begin(), Ar.begin() + more, Ar.end());
                Bm = vector<Point>(Bl.end() - more, Bl.end());
                Am = vector<Point>(Ar.begin(), Ar.begin() + more);
                Bl = vector<Point>(Bl.begin(), Bl.end() - more);
                Ar = vector<Point>(Ar.begin() + more, Ar.end());
            }
            assert(len(Al) == len(Bl));
            assert(len(Ar) == len(Br));
            assert(len(Am) == len(Bm));
            build2(Am, Bm);
        }
        build2(Al, Bl);
        build2(Ar, Br);        
    }

    vector<int> match2(vector<vector<double>> raw_A, vector<vector<double>> raw_B) {
        int N = len(raw_A);
        mov = N;
        assert(N == len(raw_B));

        vector<Point> A, B;
        for (int i = 0; i < N; ++i) {
            A.append({raw_A[i], i});
            B.append({raw_B[i], i});
        }

        g_match = vector<int>(N, -1);
        build2(A, B);

        return g_match;
    }
}

#ifdef PY_BIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	using namespace immortalco;
	m.doc() = "Match two point clouds with K-D tree";
    m.def("set_debug", set_debug, py::arg("debug"));
	m.def("match2d", match2d, py::arg("A"), py::arg("B"));
    m.def("match", match, py::arg("k"), py::arg("A"), py::arg("B"));
    m.def("match2", match2, py::arg("A"), py::arg("B"));
}

#else

int main() {

}

#endif