/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "dd/Simulation.hpp"

namespace dd {
    template<class Config>
    std::map<std::string, std::size_t> simulate(const QuantumComputation* qc, const VectorDD& in, std::unique_ptr<dd::Package<Config>>& dd, std::size_t shots, std::size_t seed) {
        bool isDynamicCircuit = false;
        bool hasMeasurements  = false;
        bool measurementsLast = true;

        std::mt19937_64 mt{};
        if (seed != 0U) {
            mt.seed(seed);
        } else {
            // create and properly seed rng
            std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> randomData{};
            std::random_device                                                    rd;
            std::generate(std::begin(randomData), std::end(randomData), [&rd]() { return rd(); });
            std::seed_seq seeds(std::begin(randomData), std::end(randomData));
            mt.seed(seeds);
        }

        std::map<dd::Qubit, std::size_t> measurementMap{};

        // rudimentary check whether circuit is dynamic
        for (const auto& op: *qc) {
            // if it contains any dynamic circuit primitives, it certainly is dynamic
            if (op->isClassicControlledOperation() || op->getType() == qc::Reset) {
                isDynamicCircuit = true;
                break;
            }

            // once a measurement is encountered we store the corresponding mapping (qubit -> bit)
            if (op->getType() == qc::Measure) {
                const auto* measure = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                hasMeasurements     = true;

                const auto& quantum = measure->getTargets();
                const auto& classic = measure->getClassics();

                for (std::size_t i = 0; i < quantum.size(); ++i) {
                    measurementMap[static_cast<dd::Qubit>(quantum.at(i))] = classic.at(i);
                }
            }

            // if an operation happens after a measurement, the resulting circuit can only be simulated in single shots
            if (hasMeasurements && (op->isUnitary() || op->isClassicControlledOperation())) {
                measurementsLast = false;
            }
        }

        if (!measurementsLast) {
            isDynamicCircuit = true;
        }

        if (!isDynamicCircuit) {
            // if all gates are unitary (besides measurements at the end), we just simulate once and measure all qubits repeatedly
            auto permutation = qc->initialLayout;
            auto e           = in;
            dd->incRef(e);

            for (const auto& op: *qc) {
                // simply skip any non-unitary
                if (!op->isUnitary()) {
                    continue;
                }

                auto tmp = dd->multiply(getDD(op.get(), dd, permutation), e);
                dd->incRef(tmp);
                dd->decRef(e);
                e = tmp;

                dd->garbageCollect();
            }

            // correct permutation if necessary
            changePermutation(e, permutation, qc->outputPermutation, dd);
            e = dd->reduceGarbage(e, qc->garbage);

            // measure all qubits
            std::map<std::string, std::size_t> counts{};
            for (std::size_t i = 0U; i < shots; ++i) {
                // measure all returns a string of the form "q(n-1) ... q(0)"
                auto measurement = dd->measureAll(e, false, mt);
                // reverse the order of the bits so that measurements follow big-endian convention
                counts[measurement]++;
            }
            // reduce reference count of measured state
            dd->decRef(e);

            std::map<std::string, std::size_t> actualCounts{};
            for (const auto& [bitstring, count]: counts) {
                std::string measurement(qc->getNcbits(), '0');
                if (hasMeasurements) {
                    // if the circuit contains measurements, we only want to return the measured bits
                    for (const auto& [qubit, bit]: measurementMap) {
                        // measurement map specifies that the circuit `qubit` is measured into a certain `bit`
                        measurement[qc->getNcbits() - 1U - bit] = bitstring[bitstring.size() - 1U - static_cast<std::size_t>(qubit)];
                    }
                } else {
                    // otherwise, we consider the output permutation for determining where to measure the qubits to
                    for (const auto& [qubit, bit]: qc->outputPermutation) {
                        measurement[qc->getNcbits() - 1 - bit] = bitstring[bitstring.size() - 1U - static_cast<std::size_t>(qubit)];
                    }
                }
                actualCounts[measurement] += count;
            }
            return actualCounts;
        }

        std::map<std::string, std::size_t> counts{};

        for (std::size_t i = 0U; i < shots; i++) {
            std::map<std::size_t, char> measurements{};

            auto permutation = qc->initialLayout;
            auto e           = in;
            dd->incRef(e);

            for (const auto& op: *qc) {
                if (op->getType() == Measure) {
                    auto*       measure = dynamic_cast<NonUnitaryOperation*>(op.get());
                    const auto& qubits  = measure->getTargets();
                    const auto& bits    = measure->getClassics();
                    for (std::size_t j = 0U; j < qubits.size(); ++j) {
                        measurements[bits.at(j)] = dd->measureOneCollapsing(e, static_cast<dd::Qubit>(permutation.at(qubits.at(j))), true, mt);
                    }
                    continue;
                }

                if (op->getType() == Reset) {
                    auto*       reset  = dynamic_cast<NonUnitaryOperation*>(op.get());
                    const auto& qubits = reset->getTargets();
                    for (const auto& qubit: qubits) {
                        auto bit = dd->measureOneCollapsing(e, static_cast<dd::Qubit>(permutation.at(qubit)), true, mt);
                        // apply an X operation whenever the measured result is one
                        if (bit == '1') {
                            const auto x   = qc::StandardOperation(qc->getNqubits(), qubit, qc::X);
                            auto       tmp = dd->multiply(getDD(&x, dd), e);
                            dd->incRef(tmp);
                            dd->decRef(e);
                            e = tmp;
                            dd->garbageCollect();
                        }
                    }
                    continue;
                }

                if (op->getType() == ClassicControlled) {
                    auto*       classicControlled = dynamic_cast<ClassicControlledOperation*>(op.get());
                    const auto& controlRegister   = classicControlled->getControlRegister();
                    const auto& expectedValue     = classicControlled->getExpectedValue();
                    auto        actualValue       = 0ULL;
                    // determine the actual value from measurements
                    for (std::size_t j = 0; j < controlRegister.second; ++j) {
                        if (measurements[controlRegister.first + j] == '1') {
                            actualValue |= 1ULL << j;
                        }
                    }

                    // do not apply an operation if the value is not the expected one
                    if (actualValue != expectedValue) {
                        continue;
                    }
                }

                auto tmp = dd->multiply(getDD(op.get(), dd, permutation), e);
                dd->incRef(tmp);
                dd->decRef(e);
                e = tmp;

                dd->garbageCollect();
            }

            // reduce reference count of measured state
            dd->decRef(e);

            std::string shot(qc->getNcbits(), '0');
            for (const auto& [bit, value]: measurements) {
                shot[qc->getNcbits() - bit - 1U] = value;
            }
            counts[shot]++;
        }

        return counts;
    }

    template<class Config>
    void extractProbabilityVector(const QuantumComputation* qc, const VectorDD& in, ProbabilityVector& probVector, std::unique_ptr<dd::Package<Config>>& dd) {
        // ! initial layout, output permutation and garbage qubits are currently not supported here
        dd->incRef(in);
        extractProbabilityVectorRecursive(qc, in, qc->begin(), std::map<std::size_t, char>{}, 1., probVector, dd);
    }

    template<class Config>
    void extractProbabilityVectorRecursive(const QuantumComputation* qc, const VectorDD& currentState, decltype(qc->begin()) currentIt, std::map<std::size_t, char> measurements, dd::fp commonFactor, ProbabilityVector& probVector, std::unique_ptr<dd::Package<Config>>& dd) {
        auto state = currentState;
        for (auto it = currentIt; it != qc->end(); ++it) {
            const auto& op = (*it);

            // check whether a classic controlled operations can be applied
            if (op->getType() == ClassicControlled) {
                auto*       classicControlled = dynamic_cast<ClassicControlledOperation*>(op.get());
                const auto& controlRegister   = classicControlled->getControlRegister();
                const auto& expectedValue     = classicControlled->getExpectedValue();
                qc::Bit     actualValue       = 0U;
                // determine the actual value from measurements
                for (std::size_t j = 0; j < controlRegister.second; ++j) {
                    actualValue |= (measurements[controlRegister.first + j] == '1') ? (static_cast<qc::Bit>(1) << j) : 0U;
                }

                // do not apply an operation if the value is not the expected one
                if (actualValue != expectedValue) {
                    continue;
                }
            }

            if (op->getType() == Reset) {
                // a reset operation should only happen once a qubit has been measured, i.e., the qubit is in a basis state
                // thus the probabilities for 0 and 1 need to be determined
                // if p(1) ~= 1, an X operation has to be applied to the qubit
                // if p(0) ~= 1, nothing has to be done
                // if 0 < p(0), p(1) < 1, an error should be raised

                const auto& targets = op->getTargets();
                if (targets.size() != 1) {
                    throw qc::QFRException("Resets on multiple qubits are currently not supported. Please split them into multiple single resets.");
                }

                auto [pzero, pone] = dd->determineMeasurementProbabilities(state, static_cast<dd::Qubit>(targets[0]), true);

                // normalize probabilities
                const auto norm = pzero + pone;
                pzero /= norm;
                pone /= norm;

                if (dd::ComplexTable<>::Entry::approximatelyOne(pone)) {
                    qc::MatrixDD xGate      = dd->makeGateDD(dd::Xmat, static_cast<dd::QubitCount>(state.p->v + 1), static_cast<dd::Qubit>(targets[0U]));
                    qc::VectorDD resetState = dd->multiply(xGate, state);
                    dd->incRef(resetState);
                    dd->decRef(state);
                    state = resetState;
                    continue;
                }

                if (!dd::ComplexTable<>::Entry::approximatelyOne(pzero)) {
                    throw qc::QFRException("Reset on non basis state encountered. This is not supported in this method.");
                }

                continue;
            }

            // measurements form splitting points in this extraction scheme
            if (op->getType() == Measure) {
                const auto* measurement = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                const auto& targets     = measurement->getTargets();
                const auto& classics    = measurement->getClassics();
                if (targets.size() != 1U || classics.size() != 1U) {
                    throw qc::QFRException("Measurements on multiple qubits are not supported right now. Split your measurements into individual operations.");
                }

                // determine probabilities for this measurement
                auto [pzero, pone] = dd->determineMeasurementProbabilities(state, static_cast<dd::Qubit>(targets[0]), true);

                // normalize probabilities
                const auto norm = pzero + pone;
                pzero /= norm;
                pone /= norm;

                // base case -> determine the basis state from the measurement and safe the probability
                if (measurements.size() == qc->getNcbits() - 1) {
                    std::size_t idx0 = 0U;
                    std::size_t idx1 = 0U;
                    for (std::size_t i = 0U; i < qc->getNcbits(); ++i) {
                        // if this is the qubit being measured and the result is one
                        if (i == static_cast<std::size_t>(classics[0U])) {
                            idx1 |= (1ULL << i);
                        } else {
                            // sanity check
                            auto findIt = measurements.find(i);
                            if (findIt == measurements.end()) {
                                throw qc::QFRException("No information on classical bit " + std::to_string(i));
                            }
                            // if i-th bit is set increase the index appropriately
                            if (findIt->second == '1') {
                                idx0 |= (1ULL << i);
                                idx1 |= (1ULL << i);
                            }
                        }
                    }
                    const auto prob0 = commonFactor * pzero;
                    if (!dd::ComplexTable<>::Entry::approximatelyZero(prob0)) {
                        probVector[idx0] = prob0;
                    }
                    const auto prob1 = commonFactor * pone;
                    if (!dd::ComplexTable<>::Entry::approximatelyZero(prob1)) {
                        probVector[idx1] = prob1;
                    }

                    // probabilities have been written -> this path is done
                    dd->decRef(state);
                    return;
                }

                bool nonZeroP0 = !dd::ComplexTable<>::Entry::approximatelyZero(pzero);
                bool nonZeroP1 = !dd::ComplexTable<>::Entry::approximatelyZero(pone);

                // in case both outcomes are non-zero the reference count of the state has to be increased once more in order to avoid reference counting errors
                if (nonZeroP0 && nonZeroP1) {
                    dd->incRef(state);
                }

                // recursive case -- outcome 0
                if (nonZeroP0) {
                    // save measurement result
                    measurements[classics[0]] = '0';
                    // determine accumulated probability
                    auto probability = commonFactor * pzero;
                    // determine the next iteration point
                    auto nextIt = it + 1;
                    // actually collapse the state
                    const dd::GateMatrix measurementMatrix{dd::complex_one, dd::complex_zero, dd::complex_zero, dd::complex_zero};
                    qc::MatrixDD         measurementGate = dd->makeGateDD(measurementMatrix, static_cast<dd::QubitCount>(state.p->v + 1), static_cast<dd::Qubit>(targets[0]));
                    qc::VectorDD         measuredState   = dd->multiply(measurementGate, state);

                    auto c = dd->cn.getTemporary(1. / std::sqrt(pzero), 0);
                    dd::ComplexNumbers::mul(c, measuredState.w, c);
                    measuredState.w = dd->cn.lookup(c);
                    dd->incRef(measuredState);
                    dd->decRef(state);
                    // recursive call from here
                    extractProbabilityVectorRecursive(qc, measuredState, nextIt, measurements, probability, probVector, dd);
                }

                // recursive case -- outcome 1
                if (nonZeroP1) {
                    // save measurement result
                    measurements[classics[0]] = '1';
                    // determine accumulated probability
                    auto probability = commonFactor * pone;
                    // determine the next iteration point
                    auto nextIt = it + 1;
                    // actually collapse the state
                    const dd::GateMatrix measurementMatrix{dd::complex_zero, dd::complex_zero, dd::complex_zero, dd::complex_one};
                    qc::MatrixDD         measurementGate = dd->makeGateDD(measurementMatrix, static_cast<dd::QubitCount>(state.p->v + 1), static_cast<dd::Qubit>(targets[0]));
                    qc::VectorDD         measuredState   = dd->multiply(measurementGate, state);

                    auto c = dd->cn.getTemporary(1. / std::sqrt(pone), 0);
                    dd::ComplexNumbers::mul(c, measuredState.w, c);
                    measuredState.w = dd->cn.lookup(c);
                    dd->incRef(measuredState);
                    dd->decRef(state);
                    // recursive call from here
                    extractProbabilityVectorRecursive(qc, measuredState, nextIt, measurements, probability, probVector, dd);
                }

                // everything is said and done
                return;
            }

            // any standard operation or classic-controlled operation is applied here
            auto tmp = dd->multiply(getDD(op.get(), dd), state);
            dd->incRef(tmp);
            dd->decRef(state);
            state = tmp;

            dd->garbageCollect();
        }
    }

    template<class Config>
    VectorDD simulate(GoogleRandomCircuitSampling* qc, const VectorDD& in, std::unique_ptr<dd::Package<Config>>& dd, const std::optional<std::size_t> ncycles) {
        if (ncycles.has_value() && (*ncycles < qc->cycles.size() - 2U)) {
            qc->removeCycles(qc->cycles.size() - 2U - *ncycles);
        }

        Permutation permutation = qc->initialLayout;
        auto        e           = in;
        dd->incRef(e);
        for (const auto& cycle: qc->cycles) {
            for (const auto& op: cycle) {
                auto tmp = dd->multiply(getDD(op.get(), dd, permutation), e);
                dd->incRef(tmp);
                dd->decRef(e);
                e = tmp;
                dd->garbageCollect();
            }
        }
        return e;
    }

    template std::map<std::string, std::size_t> simulate<DDPackageConfig>(const QuantumComputation* qc, const VectorDD& in, std::unique_ptr<dd::Package<DDPackageConfig>>& dd, std::size_t shots, std::size_t seed);
    template void                               extractProbabilityVector<DDPackageConfig>(const QuantumComputation* qc, const VectorDD& in, ProbabilityVector& probVector, std::unique_ptr<dd::Package<DDPackageConfig>>& dd);
    template void                               extractProbabilityVectorRecursive<DDPackageConfig>(const QuantumComputation* qc, const VectorDD& in, decltype(qc->begin()) currentIt, std::map<std::size_t, char> measurements, dd::fp commonFactor, dd::ProbabilityVector& probVector, std::unique_ptr<dd::Package<DDPackageConfig>>& dd);
    template VectorDD                           simulate<DDPackageConfig>(GoogleRandomCircuitSampling* qc, const VectorDD& in, std::unique_ptr<dd::Package<DDPackageConfig>>& dd, const std::optional<std::size_t> ncycles);
} // namespace dd
