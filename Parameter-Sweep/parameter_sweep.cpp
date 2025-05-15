// g++-14 -std=c++14 -O2 -I /opt/homebrew/include -fopenmp -o ev_sweep ev_sweep.cpp

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

// ---------------- GLOBAL SETTINGS (same as before) ----------------
static const int    NUM_TRAITS = 21;         
static const int    EVOL_STEPS = 2000;       
static const double TSPAN     = 400.0;       
static const double DT        = 1.0;         
static const double ALPHA_LOW  = 1.80;    
static const double ALPHA_HIGH = 2.80;    
static const double DEFAULT_B     = 3.0;    
static const double DEFAULT_DELTA = 0.1;    
static const double DEFAULT_GAMMA = 1.0;
static const double DEFAULT_D     = 1.0;
static const double DEFAULT_Q     = 1.0;
static const double DEFAULT_BETA0 = 1.0;
static const double REL_EXT_THRESHOLD = 1e-9; 
static const double ABS_EXT_TOL       = 1e-9;
static const double EPS  = 1e-4;  
static const double TINY = 1e-30; 
static const int    MAX_ECO_STEPS = 2000000; 

using state_type = std::vector<double>;

// -------------------------------------------------------------------
// Below this point is the same code you have for your model/solver,
// but wrapped in a function runSimulation(...) that returns the final 
// mean virulence. Then in main() we do a parallel sweep. 
// -------------------------------------------------------------------

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

// ---------- ODE system ----------
struct alleleDynamics {
    int N;
    int numTraits;
    const std::vector<double>& alphaVec;
    double s,b,gamma_,theta0,delta,d,q,beta;

    alleleDynamics(int N_, int nt_, const std::vector<double> &aV,
                   double s_, double b_, double g_, double t_,
                   double del_, double dd_, double q_, double beta_)
      : N(N_), numTraits(nt_), alphaVec(aV),
        s(s_), b(b_), gamma_(g_), theta0(t_),
        delta(del_), d(dd_), q(q_), beta(beta_)
    {}

    void operator()(const state_type &y, state_type &dydt) const {
        dydt.assign(y.size(), 0.0);

        // Susceptibles
        std::vector<double> S(N);
        for(int i=0;i<N;i++){
            S[i] = (y[i]<0.0? 1e-6 : y[i]);
        }

        auto Q = create_infection_matrix(N,s);
        std::vector<double> infLoss(N,0.0), recGain(N,0.0);

        // loop over traits
        for(int tr=0; tr<numTraits; tr++){
            int off = N + tr*blockSize(N);
            double theta_k = theta0*sqrt(alphaVec[tr]);

            // infected compartments
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    int idxI=off + i*N + j;
                    double I_ij=y[idxI];
                    int idxP=off + N*N + j;
                    double P_j=y[idxP];

                    double dI = beta*Q[i][j]*P_j*S[i]
                                - (d + gamma_ + alphaVec[tr])*I_ij;
                    dydt[idxI]= dI;
                    infLoss[i]+= beta*Q[i][j]*P_j*S[i];
                    recGain[i]+= gamma_*I_ij;
                }
            }
            // free parasite
            for(int j=0;j<N;j++){
                int idxP=off + N*N + j;
                double P_j=y[idxP];
                double sumI=0.0;
                for(int i=0;i<N;i++){
                    sumI += y[off + i*N + j];
                }
                double inf_loss=0.0;
                for(int i=0;i<N;i++){
                    inf_loss += beta*Q[i][j]*S[i];
                }
                dydt[idxP] = theta_k*sumI - delta*P_j - inf_loss*P_j;
            }
        }

        // update S
        double totalPop=0.0;
        for(int i=0;i<N;i++){
            totalPop += S[i];
        }
        double birthFactor=std::max(0.0,1.0 - q*totalPop);
        for(int i=0;i<N;i++){
            double dS = b*S[i]*birthFactor - d*S[i] 
                        - infLoss[i] + recGain[i];
            dydt[i]= dS;
        }
    }
};

// ---- RKCK step
static void rkck(alleleDynamics &dyn,
                 const state_type &y,
                 const state_type &dydt_in,
                 double h,
                 state_type &yout,
                 state_type &yerr)
{
    static double b21=0.2,
                  b31=3.0/40.0,  b32=9.0/40.0,
                  b41=0.3,       b42=-0.9,      b43=1.2,
                  b51=-11.0/54.0,b52=2.5,       b53=-70.0/27.0,b54=35.0/27.0,
                  b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
                  b64=44275.0/110592.0,b65=253.0/4096.0,
                  c1=37.0/378.0,c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
                  dc5=-277.0/14336.0;

    double dc1=c1-2825.0/27648.0,
           dc3=c3-18575.0/48384.0,
           dc4=c4-13525.0/55296.0,
           dc6=c6-0.25;

    int n=(int)y.size();
    yout.resize(n);
    yerr.resize(n);

    static thread_local state_type yt,k2,k3,k4,k5,k6;
    yt.resize(n); k2.resize(n); k3.resize(n);
    k4.resize(n); k5.resize(n); k6.resize(n);

    // k1 = dydt_in
    // k2
    for(int i=0;i<n;i++){
        yt[i] = y[i] + b21*h*dydt_in[i];
    }
    dyn(yt,k2);

    // k3
    for(int i=0;i<n;i++){
        yt[i] = y[i] + h*(b31*dydt_in[i] + b32*k2[i]);
    }
    dyn(yt,k3);

    // k4
    for(int i=0;i<n;i++){
        yt[i] = y[i] + h*(b41*dydt_in[i] + b42*k2[i] + b43*k3[i]);
    }
    dyn(yt,k4);

    // k5
    for(int i=0;i<n;i++){
        yt[i] = y[i] + h*(b51*dydt_in[i] + b52*k2[i] 
                          + b53*k3[i] + b54*k4[i]);
    }
    dyn(yt,k5);

    // k6
    for(int i=0;i<n;i++){
        yt[i] = y[i] + h*(b61*dydt_in[i] + b62*k2[i]
                          + b63*k3[i] + b64*k4[i] + b65*k5[i]);
    }
    dyn(yt,k6);

    for(int i=0;i<n;i++){
        double ytemp = y[i] + h*( c1*dydt_in[i] 
                                 + c3*k3[i] + c4*k4[i] + c6*k6[i] );
        double yerr_ = h*( dc1*dydt_in[i] 
                           + dc3*k3[i] + dc4*k4[i]
                           + dc5*k5[i] + dc6*k6[i] );
        yout[i]=ytemp;
        yerr[i]=yerr_;
    }
}

// ---- Adaptive step
static void rkqs(alleleDynamics &dyn,
                 state_type &y,
                 state_type &dydt,
                 double &t,
                 double &h,
                 double &hnext)
{
    static thread_local state_type ytemp,yerr;
    ytemp.resize(y.size());
    yerr.resize(y.size());

    for(;;){
        rkck(dyn,y,dydt,h,ytemp,yerr);
        double errmax=0.0;
        for(size_t i=0;i<y.size();i++){
            double scale=fabs(y[i]) + fabs(h*dydt[i]) + TINY;
            double localErr=fabs(yerr[i]/scale);
            if(localErr>errmax) errmax=localErr;
        }
        errmax/=EPS;

        if(errmax>1.0){
            double htemp=0.9*h*pow(errmax,-0.25);
            if(htemp<0.1*h) htemp=0.1*h;
            h=htemp;
            continue;
        }
        // negativity check
        bool belowMinusTen=false;
        for(double val: ytemp){
            if(val<-10.0){ belowMinusTen=true; break;}
        }
        if(belowMinusTen){
            std::cerr<<"[Warn] Some compartment < -10 @ t="<<t+h<<"\n";
        }
        t+=h; 
        y=ytemp;
        if(errmax>1.89e-4){
            hnext=0.9*h*pow(errmax,-0.2);
        }else{
            hnext=5.0*h;
        }
        break;
    }
}

// ---- Integration
static void integrate_ecology(alleleDynamics &dyn,
                              state_type &y,
                              double TSPAN,
                              double &dt_local)
{
    double t=0.0;
    double h=dt_local, hnext=dt_local;

    static thread_local state_type dydt;
    dydt.resize(y.size());

    int stepCount=0;
    while(t<TSPAN){
        dyn(y,dydt);
        if(t+h>TSPAN) h=TSPAN - t;
        rkqs(dyn,y,dydt,t,h,hnext);
        stepCount++;
        if(stepCount>MAX_ECO_STEPS){
            std::cerr<<"[Warn] max steps reached.\n";
            break;
        }
        h=hnext;
        if(t>=TSPAN) break;
    }
    dt_local=h;
}

// ------------------------------------------------------------------
// This function runs a single simulation for (N, s, theta0), then
// returns the "final mean virulence" across all infected compartments.
// ------------------------------------------------------------------
double runSimulation(int N, double s, double theta0, bool verbose=false)
{
    // 1) alpha array
    std::vector<double> alphaVec(NUM_TRAITS);
    for(int i=0; i<NUM_TRAITS; i++){
        alphaVec[i] = ALPHA_LOW 
            + i*(ALPHA_HIGH - ALPHA_LOW)/(NUM_TRAITS-1);
    }

    // 2) random init (ONLY TRAIT 0)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0,1.0);

    int bs = blockSize(N);
    int totalSize = N + NUM_TRAITS*bs;
    state_type y(totalSize,0.0);

    // Susceptible hosts (all N get random initial)
    for(int i=0; i<N; i++){
        y[i] = dist(rng);
    }

    // Only trait 0 infected compartments get random init
    // other traits remain zero
    {
        int off0 = N + 0*bs; // offset for trait 0
        // Infecteds I_{i,j} (n*n of them)
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                y[off0 + i*N + j] = dist(rng);
            }
        }
        // Free parasites P_j (n of them)
        for(int j=0; j<N; j++){
            y[off0 + N*N + j] = dist(rng);
        }
    }

    // 3) ecological params
    double beta   = DEFAULT_BETA0 * N;  // as you indicated
    double b      = DEFAULT_B;     
    double gamma_ = DEFAULT_GAMMA; 
    double delta  = DEFAULT_DELTA;
    double dd     = DEFAULT_D;
    double q      = DEFAULT_Q;

    // We'll store parasite compartments from the last 20 steps:
    const int LATE_STEPS = 20;
    // partial sums of infected compartments for each trait
    std::vector<double> accumTraitI(NUM_TRAITS, 0.0);
    int stepsCounted = 0;

    // 4) evolution
    for(int step=0; step<EVOL_STEPS; step++){
        // check parasite extinction
        double totPara=0.0;
        for(double val: y){
            totPara += val;
        }
        if(totPara < ABS_EXT_TOL){
            if(verbose) {
                std::cout<<"Parasites extinct at step "<<step<<"\n";
            }
            // We break out since no parasites remain
            break;
        }

        // integrate ecology
        alleleDynamics dyn(N,NUM_TRAITS,alphaVec,s,b,gamma_,theta0,delta,dd,q,beta);
        double dt_local=DT;
        integrate_ecology(dyn,y,TSPAN,dt_local);

        // clamp
        for(auto &val: y){
            if(val<ABS_EXT_TOL) val=0.0;
        }
        applyRelativeExtinction(y,N,NUM_TRAITS,REL_EXT_THRESHOLD);

        // mutate
        {
            double mutRate=0.01; 
            unsigned seed = std::random_device{}() ^ (step*10007U);
            std::mt19937 lrng(seed);
            std::uniform_real_distribution<double> ur(0.0,1.0);

            double sumPop=0.0;
            std::vector<double> popVec;
            std::vector<std::pair<int,int>> idxVec;
            popVec.reserve(NUM_TRAITS*N);

            for(int tr=0; tr<NUM_TRAITS; tr++){
                int off = N + tr*bs;
                for(int j=0; j<N; j++){
                    double pVal = y[off + N*N + j];
                    if(pVal>ABS_EXT_TOL){
                        popVec.push_back(pVal);
                        idxVec.push_back({tr,j});
                        sumPop += pVal;
                    }
                }
            }
            if(sumPop>0.0){
                std::vector<double> cdf(popVec.size());
                std::partial_sum(popVec.begin(), popVec.end(), cdf.begin());
                double r = ur(lrng)*sumPop;
                auto it = std::upper_bound(cdf.begin(), cdf.end(), r);
                if(it!=cdf.end()){
                    size_t iX=std::distance(cdf.begin(), it);
                    int traitC=idxVec[iX].first;
                    int jC=idxVec[iX].second;

                    // neighbor trait
                    int traitM;
                    if(traitC==0) {
                        traitM=1;
                    } else if(traitC==NUM_TRAITS-1) {
                        traitM=traitC-1;
                    } else {
                        traitM = (ur(lrng)<0.5)?(traitC-1):(traitC+1);
                    }
                    int offC = N + traitC*bs;
                    int offM = N + traitM*bs;
                    int idxC = offC + N*N + jC;
                    int idxM = offM + N*N + jC;

                    double amt = y[idxC]*mutRate;
                    y[idxC] -= amt;
                    y[idxM] += amt;
                    if(y[idxC]<ABS_EXT_TOL) y[idxC]=0.0;
                    if(y[idxM]<ABS_EXT_TOL) y[idxM]=0.0;
                }
            }
        }
        applyRelativeExtinction(y,N,NUM_TRAITS,REL_EXT_THRESHOLD);

        // ----------------------------
        // if we are within the last 20 steps, accumulate trait I
        // so we can average afterwards
        // ----------------------------
        if(step >= EVOL_STEPS - LATE_STEPS){
            // sum up infected for each trait
            for(int tr=0; tr<NUM_TRAITS; tr++){
                double traitI=0.0;
                int off = N + tr*bs;
                // infected are off..off+(n*n)-1
                for(int idx=0; idx<N*N; idx++){
                    traitI += y[off+idx];
                }
                accumTraitI[tr] += traitI;
            }
            stepsCounted++;
        }
    } // end evolutionary loop

    // 5) Compute average infected across last 20 steps => mean alpha
    if(stepsCounted == 0) {
        // Means we never got that far, or everything went extinct
        return 0.0;
    }
    // Average infected for each trait
    double sumI=0.0, sumAlphaI=0.0;
    for(int tr=0; tr<NUM_TRAITS; tr++){
        double avgTraitI = accumTraitI[tr] / double(stepsCounted);
        sumI       += avgTraitI;
        sumAlphaI  += avgTraitI * alphaVec[tr];
    }

    double meanAlpha = (sumI > 0.0 ? sumAlphaI / sumI : 0.0);
    return meanAlpha;
}

// ---------------- MAIN: parallel parameter sweep ----------------
int main(){
    // Example sets:
    std::vector<int>    N_vals = {2, 3, 4, 5};
    std::vector<double> theta_vals = {4.0,5.0,6.0};

    // s from 0 to 1 in increments of 0.05
    std::vector<double> s_vals;
    for(double ss=0.0; ss<=1.00001; ss+=0.05){
        s_vals.push_back(ss);
    }

    // open CSV
    std::ofstream sweepOut("param_sweep_parallel.csv");
    sweepOut << "N,s,theta,mean_virulence\n";

    // We'll do an integer index for the triple loop
    // to use #pragma omp parallel for. We store all combinations
    // in a vector first.
    struct ParamSet {int N; double s; double theta;};
    std::vector<ParamSet> combos;
    for(int N: N_vals){
        for(double t0: theta_vals){
            for(double s: s_vals){
                combos.push_back({N,s,t0});
            }
        }
    }

    // parallel for
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<(int)combos.size(); i++){
        int N    = combos[i].N;
        double s = combos[i].s;
        double th= combos[i].theta;
        double res = runSimulation(N, s, th, false);

        // Write result + print to console:
        // must do in a critical section to avoid mixing lines
        #pragma omp critical
        {
            sweepOut << N << "," << s << "," << th 
                     << "," << res << "\n";
            std::cout << "[Thread " << omp_get_thread_num() << "] "
                      << "Done with N=" << N 
                      << ", s=" << s 
                      << ", theta=" << th 
                      << ", meanV=" << res << "\n";
        }
    }

    sweepOut.close();
    std::cout<<"Parallel parameter sweep done. See param_sweep_parallel.csv\n";

    return 0;
}