// ev_sweep.cpp
// g++-14 -std=c++14 -O2 -fopenmp -o mean_host mean_host.cpp
// ./ev_sweep


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <sstream>
#include <omp.h>
#include <tuple>
#include <iomanip>

// ---------------- GLOBAL SETTINGS ----------------
static const int    NUM_TRAITS        = 21;
static const int    EVOL_STEPS        = 2000;
static const double TSPAN             = 400.0;
static const double DT                = 1.0;
static const double ALPHA_LOW         = 1.80;
static const double ALPHA_HIGH        = 2.80;
static const double DEFAULT_B         = 3.0;
static const double DEFAULT_DELTA     = 0.1;
static const double DEFAULT_GAMMA     = 1.0;
static const double DEFAULT_D         = 1.0;
static const double DEFAULT_Q         = 1.0;
static const double DEFAULT_BETA0     = 1.0;
static const double REL_EXT_THRESHOLD = 1e-9;
static const double ABS_EXT_TOL       = 1e-9;
static const double EPS               = 1e-4;
static const double TINY              = 1e-30;
static const int    MAX_ECO_STEPS     = 2000000;

// Sweep settings
static const int    RES          = 5;      // number of points per parameter
static const int    FIXED_N      = 3;      // n = 3
static const double FIXED_THETA0 = 5.0;    // theta0 = 5

using state_type = std::vector<double>;

// ----------------- Utility functions -----------------
int blockSize(int N) {
    return N*N + N;
}

std::vector<std::vector<double>> create_infection_matrix(int N, double s){
    std::vector<std::vector<double>> Q(N, std::vector<double>(N,0.0));
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i == j) Q[i][j] = (1 + (N-1)*s)/double(N);
            else       Q[i][j] = (1 - s)/double(N);
        }
    }
    return Q;
}

// ----------------- Dynamics functor -----------------
struct alleleDynamics {
    int N, numTraits;
    const std::vector<double>& alphaVec;
    double s, b, gamma_, theta0, delta, d, q, beta;

    alleleDynamics(int N_, int nt_, const std::vector<double> &aV,
                   double s_, double b_, double g_, double t_,
                   double del_, double dd_, double q_, double beta_)
      : N(N_), numTraits(nt_), alphaVec(aV),
        s(s_), b(b_), gamma_(g_), theta0(t_),
        delta(del_), d(dd_), q(q_), beta(beta_)
    {}

    void operator()(const state_type &y, state_type &dydt) const {
        dydt.assign(y.size(), 0.0);
        std::vector<double> S(N);
        for(int i=0;i<N;i++) S[i] = (y[i]<0.0? 1e-6 : y[i]);
        auto Q = create_infection_matrix(N,s);
        std::vector<double> infLoss(N,0.0), recGain(N,0.0);

        for(int tr=0; tr<numTraits; tr++){
            int off = N + tr*blockSize(N);
            double theta_k = theta0 * std::sqrt(alphaVec[tr]);
            // infected
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    int idxI = off + i*N + j;
                    double I_ij = y[idxI];
                    int idxP = off + N*N + j;
                    double P_j  = y[idxP];
                    double dI = beta*Q[i][j]*P_j*S[i]
                              - (d + gamma_ + alphaVec[tr])*I_ij;
                    dydt[idxI] = dI;
                    infLoss[i] += beta*Q[i][j]*P_j*S[i];
                    recGain[i] += gamma_*I_ij;
                }
            }
            // free parasites
            for(int j=0;j<N;j++){
                int idxP = off + N*N + j;
                double P_j = y[idxP];
                double sumI = 0.0;
                for(int i=0;i<N;i++) sumI += y[off + i*N + j];
                double inf_loss = 0.0;
                for(int i=0;i<N;i++) inf_loss += beta*Q[i][j]*S[i];
                dydt[idxP] = theta_k*sumI - delta*P_j - inf_loss*P_j;
            }
        }

        double totalPop = std::accumulate(S.begin(), S.end(), 0.0);   // S

        // add all infected I(i,j,k)
        for(int tr = 0; tr < numTraits; ++tr){
            int off = N + tr*blockSize(N);          // start of trait‑block
            for(int idx = 0; idx < N*N; ++idx)
                totalPop += y[off + idx];           // I(i,j,k)
        }
        double birthFactor = std::max(0.0, 1.0 - q*totalPop);

        for(int i=0;i<N;i++){
            double dS = b*S[i]*birthFactor - d*S[i]
                      - infLoss[i] + recGain[i];
            dydt[i] = dS;
        }
    }
};

// ---- RKCK step, all temporaries local ----
static void rkck(alleleDynamics &dyn,
                 const state_type &y,
                 const state_type &dydt_in,
                 double h,
                 state_type &yout,
                 state_type &yerr)
{
    static const double b21=0.2,
                  b31=3.0/40.0,  b32=9.0/40.0,
                  b41=0.3,       b42=-0.9,      b43=1.2,
                  b51=-11.0/54.0,b52=2.5,       b53=-70.0/27.0,b54=35.0/27.0,
                  b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
                  b64=44275.0/110592.0,b65=253.0/4096.0,
                  c1=37.0/378.0,c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
                  dc5=-277.0/14336.0,
                  dc1=c1-2825.0/27648.0,
                  dc3=c3-18575.0/48384.0,
                  dc4=c4-13525.0/55296.0,
                  dc6=c6-0.25;

    int n = (int)y.size();
    yout.assign(n, 0.0);
    yerr.assign(n, 0.0);

    state_type yt(n), k2(n), k3(n), k4(n), k5(n), k6(n);

    // k2
    for(int i=0;i<n;i++) yt[i] = y[i] + b21*h*dydt_in[i];
    dyn(yt,k2);

    // k3
    for(int i=0;i<n;i++)
        yt[i] = y[i] + h*(b31*dydt_in[i] + b32*k2[i]);
    dyn(yt,k3);

    // k4
    for(int i=0;i<n;i++)
        yt[i] = y[i] + h*(b41*dydt_in[i] + b42*k2[i] + b43*k3[i]);
    dyn(yt,k4);

    // k5
    for(int i=0;i<n;i++)
        yt[i] = y[i] + h*(b51*dydt_in[i] + b52*k2[i]
                          + b53*k3[i] + b54*k4[i]);
    dyn(yt,k5);

    // k6
    for(int i=0;i<n;i++)
        yt[i] = y[i] + h*(b61*dydt_in[i] + b62*k2[i]
                          + b63*k3[i] + b64*k4[i] + b65*k5[i]);
    dyn(yt,k6);

    for(int i=0;i<n;i++){
        yout[i] = y[i] + h*(c1*dydt_in[i]
                          + c3*k3[i] + c4*k4[i] + c6*k6[i]);
        yerr[i] = h*(dc1*dydt_in[i]
                   + dc3*k3[i] + dc4*k4[i]
                   + dc5*k5[i] + dc6*k6[i]);
    }
}

// ---- Adaptive RKQS step, all temporaries local ----
static void rkqs(alleleDynamics &dyn,
                 state_type &y,
                 state_type &dydt,
                 double &t,
                 double &h,
                 double &hnext)
{
    int n = (int)y.size();
    state_type ytemp(n), yerr(n);

    while(true){
        rkck(dyn,y,dydt,h,ytemp,yerr);
        double errmax = 0.0;
        for(int i=0;i<n;i++){
            double scale = std::fabs(y[i]) + std::fabs(h*dydt[i]) + TINY;
            double localErr = std::fabs(yerr[i]/scale);
            if(localErr > errmax) errmax = localErr;
        }
        errmax /= EPS;

        if(errmax > 1.0){
            double htemp = 0.9*h*std::pow(errmax,-0.25);
            h = std::max(htemp, 0.1*h);
            continue;
        }
        // negativity check
        for(double v : ytemp){
            if(v < -10.0){
                std::cerr<<"[Warn] compartment < -10 at t="<<t+h<<"\n";
                break;
            }
        }
        t += h;
        y = ytemp;
        if(errmax > 1.89e-4)
            hnext = 0.9*h*std::pow(errmax,-0.2);
        else
            hnext = 5.0*h;
        break;
    }
}

// ---------------- Integrator wrapper ----------------
static void integrate_ecology(alleleDynamics &dyn,
                              state_type &y,
                              double TSPAN,
                              double &dt_local)
{
    double t = 0.0;
    double h = dt_local, hnext = dt_local;
    state_type dydt(y.size());

    int stepCount = 0;
    while(t < TSPAN){
        dyn(y,dydt);
        if(t + h > TSPAN) h = TSPAN - t;
        rkqs(dyn,y,dydt,t,h,hnext);
        if(++stepCount > MAX_ECO_STEPS){
            std::cerr<<"[Warn] max eco steps reached\n";
            break;
        }
        h = hnext;
    }
    dt_local = h;
}

// ---------- one ecological trajectory, returns <mean,max,min> -------------
static std::tuple<double,double,double>
simulateHostAvailability(int N, double s,
                         double duration   = 20000.0,  // MaxTime
                         double sample_dt  = 1.0)      // 1 time‑unit steps
{
    /* parameters exactly as in the MATLAB script */
    const double b      = DEFAULT_B;       // 3
    const double d      = DEFAULT_D;       // 1
    const double q      = DEFAULT_Q;       // 1
    const double beta0  = DEFAULT_BETA0;   // 1
    const double gamma  = DEFAULT_GAMMA;   // 1
    const double delta  = DEFAULT_DELTA;   // 0.1
    const double theta0 = FIXED_THETA0;    // 5

    // ---- alpha vector (unchanged from the solver) -------------------------
    std::vector<double> alphaVec(NUM_TRAITS);
    for (int i = 0; i < NUM_TRAITS; ++i)
        alphaVec[i] = ALPHA_LOW + i*(ALPHA_HIGH - ALPHA_LOW)/(NUM_TRAITS - 1);

    // ---- initial conditions: same spirit as MATLAB (rand()/10) ------------
    std::mt19937 rng(12345 + N * 7919 + int(s * 1000));
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    const int bs   = blockSize(N);
    const int SIZE = N + NUM_TRAITS * bs;
    state_type y(SIZE, 0.0);

    for (int i = 0; i < N; ++i) y[i] = ur(rng) / 10.0;          // S
    int off0 = N;                                                // trait 0
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            y[off0 + i*N + j] = ur(rng) / 10.0;                  // I
    for (int j = 0; j < N; ++j)
        y[off0 + N*N + j] = ur(rng) / 10.0;                      // P

    // ---- simulation loop ---------------------------------------------------
    std::vector<double> series;
    auto   Q   = create_infection_matrix(N, s);
    double t   = 0.0;
    double beta = beta0 * N;                                     // scaling

    while (t < duration) {
        alleleDynamics dyn(N, NUM_TRAITS, alphaVec,
                           s, b, gamma, theta0,
                           delta, d, q, beta);

        double dt_local = sample_dt;
        integrate_ecology(dyn, y, sample_dt, dt_local);
        t += sample_dt;

        // host availability for host 1: Σ_i S_i * Q_{i1}
        double H1 = 0.0;
        for (int i = 0; i < N; ++i) H1 += y[i] * Q[i][0];
        series.push_back(H1);
    }

    // ---- stats over last 20 % of the trajectory ---------------------------
    std::size_t start = static_cast<std::size_t>(series.size() * 0.8);
    double mean = 0.0, minv = std::numeric_limits<double>::max(),
           maxv = std::numeric_limits<double>::lowest();

    for (std::size_t k = start; k < series.size(); ++k) {
        mean += series[k];
        minv  = std::min(minv, series[k]);
        maxv  = std::max(maxv, series[k]);
    }
    mean /= (series.size() - start);

    return {mean, maxv, minv};
}

int main()
{
    const std::vector<int> N_vals = {2, 3, 4, 5};
    const int              S_RES  = 21;

    std::ofstream fout("host_availability.csv");
    fout << "n,s,mean_H,max_H,min_H\n";
    fout << std::setprecision(10);

    for (int N : N_vals) {
        for (int i = 0; i < S_RES; ++i) {
            double s = i / double(S_RES - 1);      // linspace 0‑1

            auto [meanH, maxH, minH] = simulateHostAvailability(N, s);

            fout << N << ',' << s << ','
                 << meanH << ',' << maxH << ',' << minH << '\n';

            std::cout << "n=" << N << "  s=" << s
                      << "  →  mean=" << meanH << '\n';
        }
    }

    std::cout << "\nDone. Data in host_availability.csv\n";
    return 0;
}