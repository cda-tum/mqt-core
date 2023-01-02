/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "QuantumComputation.hpp"
#include "dd/NoiseFunctionality.hpp"
#include "operations/NonUnitaryOperation.hpp"

#include "gtest/gtest.h"
#include <random>

using namespace qc;

struct StochasticNoiseSimulatorDDPackageConfig: public dd::DDPackageConfig {
    static constexpr std::size_t STOCHASTIC_CACHE_OPS = qc::OpType::OpCount;
};

using StochasticNoiseTestPackage = dd::Package<StochasticNoiseSimulatorDDPackageConfig>;

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

using DensityMatrixTestPackage = dd::Package<DensityMatrixSimulatorDDPackageConfig>;

class DDNoiseFunctionalityTest: public ::testing::Test {
protected:
    void SetUp() override {
        using namespace qc::literals;

        // circuit taken from https://github.com/pnnl/qasmbench
        qc.addQubitRegister(4U);
        qc.x(0);
        qc.x(1);
        qc.h(3);
        qc.x(3, 2_pc);
        qc.t(0);
        qc.t(1);
        qc.t(2);
        qc.tdag(3);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.x(0, 3_pc);
        qc.x(2, 1_pc);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.tdag(0);
        qc.tdag(1);
        qc.tdag(2);
        qc.t(3);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.s(3);
        qc.x(0, 3_pc);
        qc.h(3);
    }

    void TearDown() override {
    }

    qc::QuantumComputation qc{};
    size_t                 stochRuns = 1000U;
};

TEST_F(DDNoiseFunctionalityTest, DetSimulateAdder4TrackAPDApplySequential) {
    auto dd = std::make_unique<DensityMatrixTestPackage>(qc.getNqubits());

    auto rootEdge = dd->makeZeroDensityOperator(static_cast<dd::QubitCount>(qc.getNqubits()));
    dd->incRef(rootEdge);

    const auto noiseEffects = {dd::AmplitudeDamping, dd::PhaseFlip, dd::Depolarization, dd::Identity};

    auto deterministicNoiseFunctionality = dd::DeterministicNoiseFunctionality(
            dd,
            static_cast<dd::QubitCount>(qc.getNqubits()),
            0.01,
            0.02,
            0.02,
            0.04,
            noiseEffects,
            false,
            true);

    for (auto const& op: qc) {
        auto operation = dd::getDD(op.get(), dd);
        dd->applyOperationToDensity(rootEdge, operation, false);

        deterministicNoiseFunctionality.applyNoiseEffects(rootEdge, op);
    }

    const auto m = dd->getProbVectorFromDensityMatrix(rootEdge, 0.001);

    const double tolerance = 1e-10;
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

TEST_F(DDNoiseFunctionalityTest, DetSimulateAdder4TrackAPD) {
    auto dd = std::make_unique<DensityMatrixTestPackage>(qc.getNqubits());

    auto rootEdge = dd->makeZeroDensityOperator(static_cast<dd::QubitCount>(qc.getNqubits()));
    dd->incRef(rootEdge);

    const auto noiseEffects = {dd::AmplitudeDamping, dd::Identity, dd::PhaseFlip, dd::Depolarization};

    auto deterministicNoiseFunctionality = dd::DeterministicNoiseFunctionality(
            dd,
            static_cast<dd::QubitCount>(qc.getNqubits()),
            0.01,
            0.02,
            0.02,
            0.04,
            noiseEffects,
            true,
            false);

    for (auto const& op: qc) {
        auto operation = dd::getDD(op.get(), dd);
        dd->applyOperationToDensity(rootEdge, operation, true);

        deterministicNoiseFunctionality.applyNoiseEffects(rootEdge, op);
    }

    const auto m = dd->getProbVectorFromDensityMatrix(rootEdge, 0.001);

    const double tolerance = 1e-10;
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

TEST_F(DDNoiseFunctionalityTest, StochSimulateAdder4TrackAPD) {
    auto dd = std::make_unique<StochasticNoiseTestPackage>(qc.getNqubits());

    std::map<std::string, double, std::less<>> measSummary = {
            {"0000", 0.},
            {"0001", 0.},
            {"0010", 0.},
            {"0011", 0.},
            {"0100", 0.},
            {"0101", 0.},
            {"0110", 0.},
            {"0111", 0.},
            {"1000", 0.},
            {"1001", 0.},
            {"1010", 0.},
            {"1011", 0.},
            {"1100", 0.},
            {"1101", 0.}};

    const auto noiseEffects = {dd::AmplitudeDamping, dd::PhaseFlip, dd::Identity, dd::Depolarization};

    auto stochasticNoiseFunctionality = dd::StochasticNoiseFunctionality(
            dd,
            static_cast<dd::QubitCount>(qc.getNqubits()),
            0.01,
            0.02,
            2.,
            noiseEffects);

    for (size_t i = 0U; i < stochRuns; i++) {
        auto rootEdge = dd->makeZeroState(static_cast<dd::QubitCount>(qc.getNqubits()));
        dd->incRef(rootEdge);

        for (auto const& op: qc) {
            auto operation  = dd::getDD(op.get(), dd);
            auto usedQubits = op->getUsedQubits();
            stochasticNoiseFunctionality.applyNoiseOperation(usedQubits, operation, rootEdge, qc.getGenerator());
        }

        const auto amplitudes = dd->getVector(rootEdge);
        for (size_t m = 0U; m < amplitudes.size(); m++) {
            auto state = std::bitset<4U>(m).to_string();
            std::reverse(state.begin(), state.end());
            const auto amplitude = amplitudes[m];
            const auto prob      = std::norm(amplitude);
            measSummary[state] += prob / static_cast<double>(stochRuns);
        }
    }

    const double tolerance = 0.1;
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

TEST_F(DDNoiseFunctionalityTest, StochSimulateAdder4IdentiyError) {
    auto dd = std::make_unique<StochasticNoiseTestPackage>(qc.getNqubits());

    std::map<std::string, double, std::less<>> measSummary = {
            {"0000", 0.},
            {"0001", 0.},
            {"0010", 0.},
            {"0011", 0.},
            {"0100", 0.},
            {"0101", 0.},
            {"0110", 0.},
            {"0111", 0.},
            {"1000", 0.},
            {"1001", 0.},
            {"1010", 0.},
            {"1011", 0.},
            {"1100", 0.},
            {"1101", 0.}};

    const auto noiseEffects = {dd::Identity};

    auto stochasticNoiseFunctionality = dd::StochasticNoiseFunctionality(
            dd,
            static_cast<dd::QubitCount>(qc.getNqubits()),
            0.01,
            0.02,
            2.,
            noiseEffects);

    for (size_t i = 0U; i < stochRuns; i++) {
        auto rootEdge = dd->makeZeroState(static_cast<dd::QubitCount>(qc.getNqubits()));
        dd->incRef(rootEdge);

        for (auto const& op: qc) {
            auto operation  = dd::getDD(op.get(), dd);
            auto usedQubits = op->getUsedQubits();
            stochasticNoiseFunctionality.applyNoiseOperation(op->getUsedQubits(), operation, rootEdge, qc.getGenerator());
        }

        const auto amplitudes = dd->getVector(rootEdge);
        for (size_t m = 0U; m < amplitudes.size(); m++) {
            auto state = std::bitset<4U>(m).to_string();
            std::reverse(state.begin(), state.end());
            const auto amplitude = amplitudes[m];
            const auto prob      = std::norm(amplitude);
            measSummary[state] += prob / static_cast<double>(stochRuns);
        }
    }

    const double tolerance = 0.1;
    EXPECT_NEAR(measSummary["0000"], 0., tolerance);
    EXPECT_NEAR(measSummary["0001"], 0., tolerance);
    EXPECT_NEAR(measSummary["0010"], 0., tolerance);
    EXPECT_NEAR(measSummary["0100"], 0., tolerance);
    EXPECT_NEAR(measSummary["0101"], 0., tolerance);
    EXPECT_NEAR(measSummary["0110"], 0., tolerance);
    EXPECT_NEAR(measSummary["0111"], 0., tolerance);
    EXPECT_NEAR(measSummary["1000"], 0., tolerance);
    EXPECT_NEAR(measSummary["1001"], 1., tolerance);
    EXPECT_NEAR(measSummary["1010"], 0., tolerance);
    EXPECT_NEAR(measSummary["1011"], 0., tolerance);
    EXPECT_NEAR(measSummary["1100"], 0., tolerance);
    EXPECT_NEAR(measSummary["1101"], 0., tolerance);
    EXPECT_NEAR(measSummary["1110"], 0., tolerance);
    EXPECT_NEAR(measSummary["1111"], 0., tolerance);
}

TEST_F(DDNoiseFunctionalityTest, testingUsedQubits) {
    const std::size_t nqubits    = 1;
    auto              standardOp = StandardOperation(nqubits, 1, qc::Z);
    EXPECT_EQ(standardOp.getUsedQubits().size(), 1);
    EXPECT_TRUE(standardOp.getUsedQubits().count(1));

    auto nonUnitaryOp = NonUnitaryOperation(nqubits, 0, 0);
    EXPECT_EQ(nonUnitaryOp.getUsedQubits().size(), 1);
    EXPECT_TRUE(nonUnitaryOp.getUsedQubits().count(0) == 1U);

    auto compoundOp = qc::CompoundOperation(nqubits);
    compoundOp.emplace_back<qc::StandardOperation>(nqubits, 0, qc::Z);
    compoundOp.emplace_back<qc::StandardOperation>(nqubits, 1, qc::H);
    compoundOp.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);
    EXPECT_EQ(compoundOp.getUsedQubits().size(), 2);
    EXPECT_TRUE(compoundOp.getUsedQubits().count(0));
    EXPECT_TRUE(compoundOp.getUsedQubits().count(1));

    std::unique_ptr<qc::Operation> xOp                   = std::make_unique<qc::StandardOperation>(nqubits, 0, qc::X);
    auto                           classicalControlledOp = qc::ClassicControlledOperation(xOp, std::pair{0, nqubits}, 1U);
    EXPECT_EQ(classicalControlledOp.getUsedQubits().size(), 1);
    EXPECT_TRUE(classicalControlledOp.getUsedQubits().count(0) == 1U);
}
