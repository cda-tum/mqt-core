/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "QuantumComputation.hpp"
#include "dd/NoiseFunctionality.hpp"

#include "gtest/gtest.h"
#include <random>

using namespace qc;
using namespace dd;

struct StochasticNoiseSimulatorDDPackageConfig: public dd::DDPackageConfig {
    static constexpr std::size_t STOCHASTIC_CACHE_OPS = qc::OpType::OpCount;
};

using StochasticNoiseTestPackage = dd::Package<StochasticNoiseSimulatorDDPackageConfig::UT_VEC_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::UT_VEC_INITIAL_ALLOCATION_SIZE,
                                               StochasticNoiseSimulatorDDPackageConfig::UT_MAT_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::UT_MAT_INITIAL_ALLOCATION_SIZE,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_VEC_ADD_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_ADD_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_TRANS_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_CONJ_TRANS_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_VEC_MULT_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_MAT_MULT_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_VEC_KRON_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_MAT_KRON_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_VEC_INNER_PROD_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_DM_NOISE_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::UT_DM_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::UT_DM_INITIAL_ALLOCATION_SIZE,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_DM_DM_MULT_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::CT_DM_ADD_NBUCKET,
                                               StochasticNoiseSimulatorDDPackageConfig::STOCHASTIC_CACHE_OPS>;

struct DensityMatrixSimulatorDDPackageConfig: public dd::DDPackageConfig {
    static constexpr std::size_t UT_DM_NBUCKET                 = 65536U;
    static constexpr std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 4096U;

    static constexpr std::size_t CT_DM_DM_MULT_NBUCKET = 16384U;
    static constexpr std::size_t CT_DM_ADD_NBUCKET     = 16384U;
    static constexpr std::size_t CT_DM_NOISE_NBUCKET   = 4096U;

    static constexpr std::size_t UT_MAT_NBUCKET            = 16384U;
    static constexpr std::size_t CT_MAT_ADD_NBUCKET        = 4096U;
    static constexpr std::size_t CT_VEC_ADD_NBUCKET        = 4096U;
    static constexpr std::size_t CT_MAT_TRANS_NBUCKET      = 4096U;
    static constexpr std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;

    static constexpr std::size_t CT_MAT_MAT_MULT_NBUCKET        = 1U;
    static constexpr std::size_t CT_MAT_VEC_MULT_NBUCKET        = 1U;
    static constexpr std::size_t UT_VEC_NBUCKET                 = 1U;
    static constexpr std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 1U;
    static constexpr std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 1U;
    static constexpr std::size_t CT_VEC_KRON_NBUCKET            = 1U;
    static constexpr std::size_t CT_MAT_KRON_NBUCKET            = 1U;
    static constexpr std::size_t CT_VEC_INNER_PROD_NBUCKET      = 1U;
    static constexpr std::size_t STOCHASTIC_CACHE_OPS           = 1U;
};

using DensityMatrixTestPackage = dd::Package<DensityMatrixSimulatorDDPackageConfig::UT_VEC_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::UT_VEC_INITIAL_ALLOCATION_SIZE,
                                             DensityMatrixSimulatorDDPackageConfig::UT_MAT_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::UT_MAT_INITIAL_ALLOCATION_SIZE,
                                             DensityMatrixSimulatorDDPackageConfig::CT_VEC_ADD_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_ADD_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_TRANS_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_CONJ_TRANS_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_VEC_MULT_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_MAT_MULT_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_VEC_KRON_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_MAT_KRON_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_VEC_INNER_PROD_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_DM_NOISE_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::UT_DM_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::UT_DM_INITIAL_ALLOCATION_SIZE,
                                             DensityMatrixSimulatorDDPackageConfig::CT_DM_DM_MULT_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::CT_DM_ADD_NBUCKET,
                                             DensityMatrixSimulatorDDPackageConfig::STOCHASTIC_CACHE_OPS>;

static std::unique_ptr<qc::QuantumComputation> detGetAdder4Circuit() {
    // circuit taken from https://github.com/pnnl/qasmbench
    auto quantumComputation = std::make_unique<qc::QuantumComputation>(4);
    quantumComputation->x(0);
    quantumComputation->x(1);
    quantumComputation->h(3);
    quantumComputation->x(3, 2_pc);
    quantumComputation->t(0);
    quantumComputation->t(1);
    quantumComputation->t(2);
    quantumComputation->tdag(3);
    quantumComputation->x(1, 0_pc);
    quantumComputation->x(3, 2_pc);
    quantumComputation->x(0, 3_pc);
    quantumComputation->x(2, 1_pc);
    quantumComputation->x(1, 0_pc);
    quantumComputation->x(3, 2_pc);
    quantumComputation->tdag(0);
    quantumComputation->tdag(1);
    quantumComputation->tdag(2);
    quantumComputation->t(3);
    quantumComputation->x(1, 0_pc);
    quantumComputation->x(3, 2_pc);
    quantumComputation->s(3);
    quantumComputation->x(0, 3_pc);
    quantumComputation->h(3);
    return quantumComputation;
}

TEST(DDNoiseFunctionality, DetSimulateAdder4TrackAPDApplySequential) {
    auto quantumComputation = detGetAdder4Circuit();
    auto dd                 = std::make_unique<DensityMatrixTestPackage>(quantumComputation->getNqubits());

    auto rootEdge = dd->makeZeroDensityOperator();
    dd->incRef(rootEdge);

    std::vector<dd::NoiseOperations> noiseEffects = {amplitudeDamping, phaseFlip, depolarization};

    auto deterministicNoiseFunctionality = dd::DeterministicNoiseFunctionality<DensityMatrixTestPackage>(
            dd,
            quantumComputation->getNqubits(),
            0.01,
            0.02,
            0.02,
            0.04,
            noiseEffects,
            false,
            true);

    for (auto const& op: *quantumComputation) {
        auto operation = dd::getDD(op.get(), dd);
        dd->applyOperationToDensity(rootEdge, operation, false);

        deterministicNoiseFunctionality.applyNoiseEffects(rootEdge, op);
    }

    auto m = dd->getProbVectorFromDensityMatrix(rootEdge, 0.001);

    double tolerance = 1e-10;
    EXPECT_NEAR(m.find("0000")->second, 0.0969332192741, tolerance);
    EXPECT_NEAR(m.find("0001")->second, 0.0907888041538, tolerance);
    EXPECT_NEAR(m.find("0010")->second, 0.0141409660985, tolerance);
    EXPECT_NEAR(m.find("0100")->second, 0.0238203475524, tolerance);
    EXPECT_NEAR(m.find("0101")->second, 0.0235097990017, tolerance);
    EXPECT_NEAR(m.find("0110")->second, 0.0244576087400, tolerance);
    EXPECT_NEAR(m.find("0111")->second, 0.0116282811276, tolerance);
    EXPECT_NEAR(m.find("1000")->second, 0.1731941264570, tolerance);
    EXPECT_NEAR(m.find("1001")->second, 0.4145855071998, tolerance);
    EXPECT_NEAR(m.find("1010")->second, 0.0138062113213, tolerance);
    EXPECT_NEAR(m.find("1011")->second, 0.0184033482066, tolerance);
    EXPECT_NEAR(m.find("1100")->second, 0.0242454336917, tolerance);
    EXPECT_NEAR(m.find("1101")->second, 0.0262779844799, tolerance);
    EXPECT_NEAR(m.find("1110")->second, 0.0239296920989, tolerance);
    EXPECT_NEAR(m.find("1111")->second, 0.0110373166627, tolerance);
}

TEST(DDNoiseFunctionality, DetSimulateAdder4TrackAPD) {
    auto quantumComputation = detGetAdder4Circuit();
    auto dd                 = std::make_unique<DensityMatrixTestPackage>(quantumComputation->getNqubits());

    auto rootEdge = dd->makeZeroDensityOperator();
    dd->incRef(rootEdge);

    std::vector<dd::NoiseOperations> noiseEffects = {amplitudeDamping, phaseFlip, depolarization};

    auto deterministicNoiseFunctionality = dd::DeterministicNoiseFunctionality<DensityMatrixTestPackage>(
            dd,
            quantumComputation->getNqubits(),
            0.01,
            0.02,
            0.02,
            0.04,
            noiseEffects,
            true,
            false);

    for (auto const& op: *quantumComputation) {
        auto operation = dd::getDD(op.get(), dd);
        dd->applyOperationToDensity(rootEdge, operation, true);

        deterministicNoiseFunctionality.applyNoiseEffects(rootEdge, op);
    }

    auto m = dd->getProbVectorFromDensityMatrix(rootEdge, 0.001);

    double tolerance = 1e-10;
    EXPECT_NEAR(m.find("0000")->second, 0.0969332192741, tolerance);
    EXPECT_NEAR(m.find("0001")->second, 0.0907888041538, tolerance);
    EXPECT_NEAR(m.find("0010")->second, 0.0141409660985, tolerance);
    EXPECT_NEAR(m.find("0100")->second, 0.0238203475524, tolerance);
    EXPECT_NEAR(m.find("0101")->second, 0.0235097990017, tolerance);
    EXPECT_NEAR(m.find("0110")->second, 0.0244576087400, tolerance);
    EXPECT_NEAR(m.find("0111")->second, 0.0116282811276, tolerance);
    EXPECT_NEAR(m.find("1000")->second, 0.1731941264570, tolerance);
    EXPECT_NEAR(m.find("1001")->second, 0.4145855071998, tolerance);
    EXPECT_NEAR(m.find("1010")->second, 0.0138062113213, tolerance);
    EXPECT_NEAR(m.find("1011")->second, 0.0184033482066, tolerance);
    EXPECT_NEAR(m.find("1100")->second, 0.0242454336917, tolerance);
    EXPECT_NEAR(m.find("1101")->second, 0.0262779844799, tolerance);
    EXPECT_NEAR(m.find("1110")->second, 0.0239296920989, tolerance);
    EXPECT_NEAR(m.find("1111")->second, 0.0110373166627, tolerance);
}

TEST(DDNoiseFunctionality, StochSimulateAdder4TrackAPD) {
    size_t stochRuns          = 1000;
    auto   quantumComputation = detGetAdder4Circuit();
    auto   dd                 = std::make_unique<StochasticNoiseTestPackage>(quantumComputation->getNqubits());

    std::map<std::string, double> measSummary = {
            {"0000", 0},
            {"0001", 0},
            {"0010", 0},
            {"0011", 0},
            {"0100", 0},
            {"0101", 0},
            {"0110", 0},
            {"0111", 0},
            {"1000", 0},
            {"1001", 0},
            {"1010", 0},
            {"1011", 0},
            {"1100", 0},
            {"1101", 0}};

    std::vector<dd::NoiseOperations> noiseEffects = {dd::amplitudeDamping, dd::phaseFlip, dd::depolarization};

    auto stochasticNoiseFunctionality = dd::StochasticNoiseFunctionality<StochasticNoiseTestPackage>(
            dd,
            quantumComputation->getNqubits(),
            0.01,
            0.02,
            2,
            noiseEffects);

    for (size_t i = 0; i < stochRuns; i++) {
        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
        std::random_device                                                    rd;
        std::generate(std::begin(random_data), std::end(random_data), [&rd]() { return rd(); });
        std::seed_seq   seeds(std::begin(random_data), std::end(random_data));
        std::mt19937_64 generator(seeds);

        dd::vEdge rootEdge = dd->makeZeroState(quantumComputation->getNqubits());
        dd->incRef(rootEdge);

        for (auto const& op: *quantumComputation) {
            auto        operation  = dd::getDD(op.get(), dd);
            std::vector usedQubits = op->getTargets();
            for (auto control: op->getControls()) {
                usedQubits.push_back(control.qubit);
            }
            stochasticNoiseFunctionality.applyNoiseOperation(usedQubits, operation, rootEdge, generator);
        }

        auto amplitudes = dd->getVector(rootEdge);
        for (size_t m = 0; m < amplitudes.size(); m++) {
            std::string s1                = std::bitset<4>(m).to_string();
            std::string revertedBitString = {};
            for (int n = (int)s1.length() - 1; n >= 0; n--) {
                revertedBitString.push_back(s1[n]);
            }

            auto amplitude = amplitudes[m];
            auto prob      = std::abs(std::pow(amplitude, 2));
            measSummary[revertedBitString] += (prob / (double)stochRuns);
        }
    }

    double tolerance = 0.1;
    EXPECT_NEAR(measSummary["0000"], 0.09693321927412533, tolerance);
    EXPECT_NEAR(measSummary["0001"], 0.09078880415385877, tolerance);
    EXPECT_NEAR(measSummary["0010"], 0.01414096609854787, tolerance);
    EXPECT_NEAR(measSummary["0100"], 0.02382034755245074, tolerance);
    EXPECT_NEAR(measSummary["0101"], 0.023509799001774703, tolerance);
    EXPECT_NEAR(measSummary["0110"], 0.02445760874001203, tolerance);
    EXPECT_NEAR(measSummary["0111"], 0.011628281127642115, tolerance);
    EXPECT_NEAR(measSummary["1000"], 0.1731941264570172, tolerance);
    EXPECT_NEAR(measSummary["1001"], 0.41458550719988047, tolerance);
    EXPECT_NEAR(measSummary["1010"], 0.013806211321349706, tolerance);
    EXPECT_NEAR(measSummary["1011"], 0.01840334820660922, tolerance);
    EXPECT_NEAR(measSummary["1100"], 0.024245433691737584, tolerance);
    EXPECT_NEAR(measSummary["1101"], 0.026277984479993615, tolerance);
    EXPECT_NEAR(measSummary["1110"], 0.023929692098939092, tolerance);
    EXPECT_NEAR(measSummary["1111"], 0.011037316662706232, tolerance);
}

TEST(DDNoiseFunctionality, StochSimulateAdder4IdentiyError) {
    size_t stochRuns          = 1000;
    auto   quantumComputation = detGetAdder4Circuit();
    auto   dd                 = std::make_unique<StochasticNoiseTestPackage>(quantumComputation->getNqubits());

    std::map<std::string, double> measSummary = {
            {"0000", 0},
            {"0001", 0},
            {"0010", 0},
            {"0011", 0},
            {"0100", 0},
            {"0101", 0},
            {"0110", 0},
            {"0111", 0},
            {"1000", 0},
            {"1001", 0},
            {"1010", 0},
            {"1011", 0},
            {"1100", 0},
            {"1101", 0}};

    std::vector<dd::NoiseOperations> noiseEffects = {dd::identity};

    auto stochasticNoiseFunctionality = dd::StochasticNoiseFunctionality<StochasticNoiseTestPackage>(
            dd,
            quantumComputation->getNqubits(),
            0.01,
            0.02,
            2,
            noiseEffects);

    for (size_t i = 0; i < stochRuns; i++) {
        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
        std::random_device                                                    rd;
        std::generate(std::begin(random_data), std::end(random_data), [&rd]() { return rd(); });
        std::seed_seq   seeds(std::begin(random_data), std::end(random_data));
        std::mt19937_64 generator(seeds);

        dd::vEdge rootEdge = dd->makeZeroState(quantumComputation->getNqubits());
        dd->incRef(rootEdge);

        for (auto const& op: *quantumComputation) {
            auto        operation  = dd::getDD(op.get(), dd);
            std::vector usedQubits = op->getTargets();
            for (auto control: op->getControls()) {
                usedQubits.push_back(control.qubit);
            }
            stochasticNoiseFunctionality.applyNoiseOperation(usedQubits, operation, rootEdge, generator);
        }

        auto amplitudes = dd->getVector(rootEdge);
        for (size_t m = 0; m < amplitudes.size(); m++) {
            std::string s1                = std::bitset<4>(m).to_string();
            std::string revertedBitString = {};
            for (int n = (int)s1.length() - 1; n >= 0; n--) {
                revertedBitString.push_back(s1[n]);
            }

            auto amplitude = amplitudes[m];
            auto prob      = std::abs(std::pow(amplitude, 2));
            measSummary[revertedBitString] += (prob / (double)stochRuns);
        }
    }

    double tolerance = 0.1;
    EXPECT_NEAR(measSummary["0000"], 0, tolerance);
    EXPECT_NEAR(measSummary["0001"], 0, tolerance);
    EXPECT_NEAR(measSummary["0010"], 0, tolerance);
    EXPECT_NEAR(measSummary["0100"], 0, tolerance);
    EXPECT_NEAR(measSummary["0101"], 0, tolerance);
    EXPECT_NEAR(measSummary["0110"], 0, tolerance);
    EXPECT_NEAR(measSummary["0111"], 0, tolerance);
    EXPECT_NEAR(measSummary["1000"], 0, tolerance);
    EXPECT_NEAR(measSummary["1001"], 1, tolerance);
    EXPECT_NEAR(measSummary["1010"], 0, tolerance);
    EXPECT_NEAR(measSummary["1011"], 0, tolerance);
    EXPECT_NEAR(measSummary["1100"], 0, tolerance);
    EXPECT_NEAR(measSummary["1101"], 0, tolerance);
    EXPECT_NEAR(measSummary["1110"], 0, tolerance);
    EXPECT_NEAR(measSummary["1111"], 0, tolerance);
}
