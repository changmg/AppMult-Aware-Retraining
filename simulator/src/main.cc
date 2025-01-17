#include "cmdline.hpp"
#include "header.h"
#include "my_abc.h"
#include "simulator.h"


using namespace abc;
using namespace boost;
using namespace cmdline;
using namespace std;


parser CommPars(int argc, char* argv[]) {
    parser option;
    option.add<string>("appMult", '\0', "path to approximate multiplier (BLIF file in SOP format is recommended)", true);
    option.add<string>("outputFolder", '\0', "path to output folder", false, "./tmp/");
    option.parse_check(argc, argv);
    return option;
}


void Simulate(string& appMult, string& outputFolder) {
    NetMan circuit(appMult);
    int nPi = circuit.GetPiNum();
    assert(nPi <= 20 && nPi % 2 == 0);
    int bitWidth = nPi / 2;
    int nFrame = 1 << nPi;
    int nPo = circuit.GetPoNum();
    assert(nPi == nPo);
    Simulator simulator(circuit, 0, nFrame);
    simulator.InpEnum();
    simulator.Sim();

    int maxValuePlus1 = 1 << bitWidth;
    vector<vector<int>> lut(maxValuePlus1, vector<int>(maxValuePlus1, 0));
    for (int i = 0; i < nFrame; i++) {
        int opA = static_cast<int>(simulator.GetInp(i, 0, bitWidth - 1, false));
        int opB = static_cast<int>(simulator.GetInp(i, bitWidth, nPi - 1, false));
        lut[opA][opB] = static_cast <int>(simulator.GetOutpFast(i, false));
        // // check correctness using unsigned accurate multiplier
        // auto refRes = opA * opB;
        // if (res != refRes) {
        //     cout << "Error: " << opA << " * " << opB << " = " << res << " != " << refRes << endl;
        //     assert(0);
        // }
    }

    double sumER = 0.0, sumMED = 0.0, sumMSE = 0.0;
    double totalCount = maxValuePlus1 * maxValuePlus1;
    for (int opA = 0; opA < maxValuePlus1; opA++) {
        for (int opB = 0; opB < maxValuePlus1; opB++) {
            // cout << opA << " " << opB << " " << lut[opA][opB] << endl;
            int ref = opA * opB;
            sumER += (ref != lut[opA][opB]);
            sumMED += abs(ref - lut[opA][opB]);
            sumMSE += pow(ref - lut[opA][opB], 2);
        }
    }
    cout << "INFO: Error rate: " << sumER / totalCount << endl;
    cout << "INFO: Mean error distance: " << sumMED / totalCount << endl;
    cout << "INFO: Normalized mean error distance: " << sumMED / totalCount / (totalCount) << endl;
    cout << "INFO: Mean square error: " << sumMSE / totalCount << endl;
    cout << "LUT for approximate multiplier:" << endl;

    for (int opA = 0; opA < maxValuePlus1; opA++) {
        for (int opB = 0; opB < maxValuePlus1; opB++) {
            cout << opA << " " << opB << " " << lut[opA][opB] << endl;
            // int ref = opA * opB;
            // sumER += (ref != lut[opA][opB]);
            // sumMED += abs(ref - lut[opA][opB]);
            // sumMSE += pow(ref - lut[opA][opB], 2);
        }
    }
}


void Test() {
    AbcMan abc;
    abc.ReadNet("./app_mults/evo_selected/mul7u_003_sop.blif");
    NetMan accMult(abc.GetNet(), true);
    abc.ReadNet("./app_mults/evo_selected/mul7u_009_sop.blif");
    NetMan appMult(abc.GetNet(), true);

    assert(IsPIOSame(accMult, appMult));
    int nPi = accMult.GetPiNum();
    assert(nPi <= 20 && nPi % 2 == 0);
    int bitWidth = nPi / 2;
    int nFrame = 1 << nPi;
    int nPo = accMult.GetPoNum();
    assert(nPi == nPo);

    Simulator accSim(accMult, 0, nFrame);
    Simulator appSim(appMult, 0, nFrame);
    accSim.InpEnum();
    appSim.InpEnum();
    accSim.Sim();
    appSim.Sim();

    double errRate = accSim.GetErrRate(appSim);
    cout << "Error rate: " << errRate << endl;
    double meanErrDist = accSim.GetMeanErrDist(appSim, false);
    cout << "Mean error distance: " << meanErrDist << endl;
    cout << "Normalized mean error distance: " << meanErrDist / pow(2, bitWidth * 2) << endl;
    double meanSquareErr = accSim.GetMeanSquareErr(appSim, false);
    cout << "Mean square error: " << meanSquareErr << endl;
}


int main(int argc, char* argv[]) {
    // start abc engine
    GlobStartAbc();

    // parse options
    parser option = CommPars(argc, argv);
    string appMult = option.get<string>("appMult");
    string outputFolder = option.get<string>("outputFolder");

    // simulate approximate multiplier
    Simulate(appMult, outputFolder);

    // Test();

    // stop abc engine
    GlobStopAbc();
    return 0;
}