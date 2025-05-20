/*
sensitivity_analysis.cpp 
Analyzes parameter sensitivity for main ecological parameters. Records evolved virulence levels
Compile: g++-14 -std=c++14 -O2 -fopenmp -o sensitivity sensitivity_analysis.cpp
*/

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

// Default Parameters
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

static const int    RES          = 5;
static const int    FIXED_N      = 3;
static const double FIXED_THETA0 = 5.0;

using state_type = std::vector<double>;

int blockSize(int N) {
    return N*N + N;
}

std::vector<double> logspace(double a, double b, int num){
    std::vector<double> v(num);
    double la = std::log10(a), lb = std::log10(b);
    for(int i=0; i<num; i++){
        v[i] = std::pow(10.0, la + (lb - la)*i/(num - 1));
    }
    return v;
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

void applyRelativeExtinction(state_type &y, int N, int numTraits, double relThresh){
    double sumS=0.0;
    for(int i=0; i<N; i++){
        sumS += y[i];
    }
    if(sumS>0.0){
        double cS = relThresh*sumS;
        for(int i=0; i<N; i++){
            if(y[i] < cS) y[i]=0.0;
        }
    }
    int bs=blockSize(N);
    for(int tr=0; tr<numTraits; tr++){
        int off = N + tr*bs;
        double sumI=0.0;
        for(int idx=0; idx<N*N; idx++){
            sumI += y[off+idx];
        }
        if(sumI>0.0){
            double cI=relThresh*sumI;
            for(int idx=0; idx<N*N; idx++){
                if(y[off+idx]<cI) y[off+idx]=0.0;
            }
        }
        double sumP=0.0;
        for(int j=0;j<N;j++){
            sumP += y[off + N*N + j];
        }
        if(sumP>0.0){
            double cP = relThresh*sumP;
            for(int j=0;j<N;j++){
                if(y[off + N*N + j]<cP) y[off + N*N + j]=0.0;
            }
        }
    }
}

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

        // sums over all infected and susceptible populations
        double totalPop = std::accumulate(S.begin(), S.end(), 0.0);
        for(int tr = 0; tr < numTraits; ++tr){
            int off = N + tr*blockSize(N);
            for(int idx = 0; idx < N*N; ++idx)
                totalPop += y[off + idx];
        }
        double birthFactor = std::max(0.0, 1.0 - q*totalPop);

        for(int i=0;i<N;i++){
            double dS = b*S[i]*birthFactor - d*S[i]
                      - infLoss[i] + recGain[i];
            dydt[i] = dS;
        }
    }
};

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

// ------------------------------------------------------------------
// This function runs a single simulation for parameter set
// ------------------------------------------------------------------
double runSimulation(int N, double s, double theta0,
                     double b_param, double beta0_param,
                     double gamma_param, double delta_param,
                     bool verbose=false)
{
    std::vector<double> alphaVec(NUM_TRAITS);
    for(int i=0;i<NUM_TRAITS;i++){
        alphaVec[i] = ALPHA_LOW + i*(ALPHA_HIGH-ALPHA_LOW)/(NUM_TRAITS-1);
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0,1.0);

    int bs = blockSize(N);
    int totalSize = N + NUM_TRAITS*bs;
    state_type y(totalSize,0.0);

    // init susceptibles
    for(int i=0;i<N;i++) y[i] = dist(rng);
    // init trait 0
    int off0 = N;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            y[off0 + i*N + j] = dist(rng);
    for(int j=0;j<N;j++)
        y[off0 + N*N + j] = dist(rng);

    // set parameters
    double beta  = beta0_param * N;
    double b     = b_param;
    double gamma = gamma_param;
    double delta = delta_param;
    double dd    = DEFAULT_D;
    double q     = DEFAULT_Q;

    const int LATE_STEPS = 20;
    std::vector<double> accumTraitI(NUM_TRAITS,0.0);
    int stepsCounted = 0;

    for(int step=0; step<EVOL_STEPS; step++){
        double totPara = std::accumulate(y.begin(), y.end(), 0.0);
        if(totPara < ABS_EXT_TOL) break;

        alleleDynamics dyn(N,NUM_TRAITS,alphaVec,
                           s,b,gamma,theta0,
                           delta,dd,q,beta);
        double dt_local = DT;
        integrate_ecology(dyn, y, TSPAN, dt_local);

        for(auto &v : y) if(v<ABS_EXT_TOL) v=0.0;
        applyRelativeExtinction(y, N, NUM_TRAITS, REL_EXT_THRESHOLD);

        // mutation
        {
            double mutRate = 0.01;
            std::mt19937 lrng(std::random_device{}() ^ (step*10007U));
            std::uniform_real_distribution<double> ur(0.0,1.0);

            std::vector<double> popVec;
            std::vector<std::pair<int,int>> idxVec;
            double sumPop = 0.0;

            for(int tr=0; tr<NUM_TRAITS; tr++){
                int off = N + tr*bs;
                for(int j=0;j<N;j++){
                    double pVal = y[off + N*N + j];
                    if(pVal > ABS_EXT_TOL){
                        popVec.push_back(pVal);
                        idxVec.emplace_back(tr,j);
                        sumPop += pVal;
                    }
                }
            }
            if(sumPop > 0.0){
                std::vector<double> cdf(popVec.size());
                std::partial_sum(popVec.begin(), popVec.end(), cdf.begin());
                double r = ur(lrng)*sumPop;
                auto it = std::upper_bound(cdf.begin(), cdf.end(), r);
                if(it != cdf.end()){
                    size_t iX = std::distance(cdf.begin(), it);
                    int traitC = idxVec[iX].first;
                    int jC      = idxVec[iX].second;

                    int traitM;
                    if(traitC==0) traitM=1;
                    else if(traitC==NUM_TRAITS-1) traitM=traitC-1;
                    else traitM = (ur(lrng)<0.5? traitC-1 : traitC+1);

                    int idxC = N + traitC*bs + N*N + jC;
                    int idxM = N + traitM*bs + N*N + jC;
                    double amt = y[idxC]*mutRate;
                    y[idxC] -= amt;
                    y[idxM] += amt;
                    if(y[idxC]<ABS_EXT_TOL) y[idxC]=0.0;
                    if(y[idxM]<ABS_EXT_TOL) y[idxM]=0.0;
                }
            }
        }
        applyRelativeExtinction(y, N, NUM_TRAITS, REL_EXT_THRESHOLD);

        if(step >= EVOL_STEPS - LATE_STEPS){
            for(int tr=0; tr<NUM_TRAITS; tr++){
                int off = N + tr*bs;
                double sumI = 0.0;
                for(int idx=0; idx<N*N; idx++)
                    sumI += y[off+idx];
                accumTraitI[tr] += sumI;
            }
            stepsCounted++;
        }
    }

    if(stepsCounted==0) return 0.0;
    double sumI=0.0, sumAlphaI=0.0;
    for(int tr=0; tr<NUM_TRAITS; tr++){
        double avgI = accumTraitI[tr]/double(stepsCounted);
        double alpha = ALPHA_LOW + tr*(ALPHA_HIGH-ALPHA_LOW)/(NUM_TRAITS-1);
        sumI      += avgI;
        sumAlphaI += avgI * alpha;
    }
    return (sumI>0.0 ? sumAlphaI/sumI : 0.0);
}

int main(){

    // logspace for parameter sweep
    auto B     = logspace(DEFAULT_B/2.0,      DEFAULT_B*2.0,      RES);
    auto BETA0 = logspace(DEFAULT_BETA0/10.0, DEFAULT_BETA0*10.0, RES);
    auto GAMMA = logspace(DEFAULT_GAMMA/2.0,  DEFAULT_GAMMA*2.0,  RES);
    auto DELTA = logspace(DEFAULT_DELTA/10.0, DEFAULT_DELTA*10.0, RES);

    // open csv
    std::ofstream sweepOut("sensitivity_analysis.csv");
    sweepOut<<"b,beta0,gamma,delta,mean_alpha_s1,mean_alpha_s0\n";

    struct P { double b,b0,g,d; };
    std::vector<P> combos;
    combos.reserve(RES*RES*RES*RES);
    for(double bval: B)
      for(double b0: BETA0)
        for(double g: GAMMA)
          for(double d: DELTA)
            combos.push_back({bval,b0,g,d});

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)combos.size(); ++i) {
        auto &p = combos[i];
        double a1 = runSimulation(FIXED_N, 1.0, FIXED_THETA0, p.b, p.b0, p.g, p.d, false);
        double a0 = runSimulation(FIXED_N, 0.0, FIXED_THETA0, p.b, p.b0, p.g, p.d, false);

        std::ostringstream line;
        line << p.b << ',' << p.b0 << ',' << p.g << ','
            << p.d << ',' << a1 << ',' << a0 << '\n';

        #pragma omp critical
        {
            std::ofstream fout("param_sweep_parallel_2.csv", std::ios::app); 
            fout << line.str();

            std::cout << "[T" << omp_get_thread_num() << "] "
                    << "b="   << p.b  << ",β0=" << p.b0
                    << ",γ="  << p.g  << ",δ=" << p.d
                    << " → α*(1)=" << a1 << ", α*(0)=" << a0 << '\n';
        }
    }

    sweepOut.close();
    std::cout<<"Done. Results in param_sweep_parallel.csv\n";
    return 0;
}
