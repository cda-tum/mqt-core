/*
* This file is part of the MQT DD Package which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
*/

#ifndef DDSIM_NOISEFUNCTIONALITY_HPP
#define DDSIM_NOISEFUNCTIONALITY_HPP

#include "dd/Definitions.hpp"
#include "dd/Export.hpp"
//#include "Package.hpp"
#include <random>
#include <utility>

namespace dd {

    // noise operations available for deterministic noise aware quantum circuit simulation
    enum noiseOperations : std::uint8_t {
        amplitudeDamping,
        phaseFlip,
        depolarization
    };

    template<class DDPackage>
    class StochasticNoiseFunctionality {
    public:
        StochasticNoiseFunctionality(const std::unique_ptr<DDPackage>& package,
                                     dd::Qubit                         size,
                                     double                            gateNoiseProbability,
                                     double                            amplitudeDampingProb,
                                     double                            multiQubitGateFactor,
                                     std::vector<dd::noiseOperations>  noiseEffects,
                                     size_t                            seed):
            package(package),
            nQubits(size),
            dist(0.0, 1.0L),
            noiseProbabilityMulti(gateNoiseProbability * multiQubitGateFactor),
            noiseProbability(gateNoiseProbability),
            seed(seed),
            noiseEffects(std::move(noiseEffects)) {
            identityDD = package->makeIdent(size);
            package->incRef(identityDD);
        }

        ~StochasticNoiseFunctionality() {
            package->decRef(identityDD);
        }

    protected:
        const std::unique_ptr<DDPackage>&      package;
        dd::Qubit                              nQubits;
        std::uniform_real_distribution<dd::fp> dist{};

        double           noiseProbability;
        double           noiseProbabilityMulti;
        dd::ComplexValue sqrtAmplitudeDampingProbability;
        dd::ComplexValue oneMinusSqrtAmplitudeDampingProbability;
        dd::ComplexValue sqrtAmplitudeDampingProbabilityMulti;
        dd::ComplexValue oneMinusSqrtAmplitudeDampingProbabilityMulti;
        dd::GateMatrix   ampDampingTrue;
        dd::GateMatrix   ampDampingTrueMulti;
        dd::GateMatrix   ampDampingFalse;
        dd::GateMatrix   ampDampingFalseMulti;

        std::vector<dd::noiseOperations> noiseEffects;

        size_t seed;

        dd::mEdge identityDD;

        dd::Qubit getNumberOfQubits() { return nQubits; }

    public:
        void applyNoiseOperation(const std::vector<dd::Qubit>& usedQubits, dd::mEdge operation, dd::vEdge& state, std::mt19937_64& generator) {
            bool multiQubitOperation = usedQubits.size() > 1;

            for (auto& target: usedQubits) {
                auto stackedOperation = generateNoiseOperation(operation, target, generator, false, multiQubitOperation);
                auto tmp              = package->multiply(stackedOperation, state);

                if (dd::ComplexNumbers::mag2(tmp.w) < dist(generator)) {
                    stackedOperation = generateNoiseOperation(operation, target, generator, true, multiQubitOperation);
                    tmp              = package->multiply(stackedOperation, state);
                }

                if (tmp.w != dd::Complex::one) {
                    tmp.w = dd::Complex::one;
                }
                package->incRef(tmp);
                package->decRef(state);
                state = tmp;

                // I only need to apply the operations once
                operation = identityDD;
            }
        }

    protected:
        dd::mEdge generateNoiseOperation(dd::mEdge        dd_operation,
                                         signed char      target,
                                         std::mt19937_64& generator,
                                         bool             amplitudeDamping,
                                         bool             multiQubitOperation) {
            qc::OpType effect;
            for (const auto& noise_type: noiseEffects) {
                if (noise_type != dd::noiseOperations::amplitudeDamping) {
                    effect = ReturnNoiseOperation(noise_type, dist(generator), multiQubitOperation);
                } else {
                    if (amplitudeDamping) {
                        effect = qc::ATrue;
                    } else {
                        effect = qc::AFalse;
                    }
                }
                switch (effect) {
                    case (dd::I): {
                        continue;
                    }
                    case (qc::ATrue): {
                        auto tmp_op = package->stochasticNoiseOperationCache.lookup(multiQubitOperation ? qc::multiATrue : qc::ATrue, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = package->makeGateDD(multiQubitOperation ? ampDampingTrueMulti : ampDampingTrue, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(multiQubitOperation ? qc::multiATrue : qc::ATrue, target, tmp_op);
                        }

                        dd_operation = package->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::AFalse): {
                        auto tmp_op = package->stochasticNoiseOperationCache.lookup(multiQubitOperation ? qc::multiAFalse : qc::AFalse, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = package->makeGateDD(multiQubitOperation ? ampDampingFalseMulti : ampDampingFalse, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(multiQubitOperation ? qc::multiAFalse : qc::AFalse, target, tmp_op);
                        }
                        dd_operation = package->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::X): {
                        auto tmp_op = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = package->makeGateDD(dd::Xmat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = package->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::Y): {
                        auto tmp_op = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = package->makeGateDD(dd::Ymat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = package->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::Z): {
                        auto tmp_op = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = package->makeGateDD(dd::Zmat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = package->multiply(tmp_op, dd_operation);
                        break;
                    }
                    default: {
                        throw std::runtime_error("Unknown noise operation '" + std::to_string(effect) + "'\n");
                    }
                }
            }
            return dd_operation;
        }

        [[nodiscard]] qc::OpType ReturnNoiseOperation(dd::noiseOperations noiseOperation, double prob, bool multi_qubit_noise) const {
            switch (noiseOperation) {
                case dd::noiseOperations::depolarization: {
                    if (prob >= 3 * (multi_qubit_noise ? noiseProbabilityMulti : noiseProbability) / 4) {
                        return qc::I;
                    } else if (prob < (multi_qubit_noise ? noiseProbabilityMulti : noiseProbability) / 4) {
                        return qc::X;
                    } else if (prob < (multi_qubit_noise ? noiseProbabilityMulti : noiseProbability) / 2) {
                        return qc::Y;
                    } else {
                        return qc::Z;
                    }
                }
                case dd::noiseOperations::phaseFlip: {
                    if (prob > (multi_qubit_noise ? noiseProbabilityMulti : noiseProbability)) {
                        return dd::I;
                    } else {
                        return dd::Z;
                    }
                }
                default:
                    throw std::runtime_error(std::string{"Unknown noise effect '"} + std::to_string(noiseOperation) + "'");
            }
        }
    };

} // namespace dd

#endif //DDSIM_NOISEFUNCTIONALITY_HPP
