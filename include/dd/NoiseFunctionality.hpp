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
    enum NoiseOperations : std::uint8_t {
        amplitudeDamping,
        phaseFlip,
        depolarization,
        identity
    };

    template<class DDPackage>
    class StochasticNoiseFunctionality {
    public:
        StochasticNoiseFunctionality(const std::unique_ptr<DDPackage>& package,
                                     dd::Qubit                         nQubits,
                                     double                            gateNoiseProbability,
                                     double                            amplitudeDampingProb,
                                     double                            multiQubitGateFactor,
                                     std::vector<dd::NoiseOperations>  noiseEffects):
            package(package),
            nQubits(nQubits),
            dist(0.0, 1.0L),
            noiseProbability(gateNoiseProbability),
            noiseProbabilityMulti(gateNoiseProbability * multiQubitGateFactor),
            sqrtAmplitudeDampingProbability({std::sqrt(amplitudeDampingProb), 0}),
            oneMinusSqrtAmplitudeDampingProbability({std::sqrt(1 - amplitudeDampingProb), 0}),
            sqrtAmplitudeDampingProbabilityMulti({std::sqrt(gateNoiseProbability) * multiQubitGateFactor, 0}),
            oneMinusSqrtAmplitudeDampingProbabilityMulti({std::sqrt(1 - multiQubitGateFactor * amplitudeDampingProb), 0}),
            noiseEffects(std::move(noiseEffects)) {
            ampDampingFalse      = dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbability});
            ampDampingFalseMulti = dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbabilityMulti});
            ampDampingTrue       = dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbability, dd::complex_zero, dd::complex_zero});
            ampDampingTrueMulti  = dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbabilityMulti, dd::complex_zero, dd::complex_zero});
            identityDD           = package->makeIdent(nQubits);
            package->incRef(identityDD);
        }

        ~StochasticNoiseFunctionality() {
            package->decRef(identityDD);
        }

    protected:
        const std::unique_ptr<DDPackage>&      package;
        dd::Qubit                              nQubits;
        std::uniform_real_distribution<dd::fp> dist;

        double                           noiseProbability;
        double                           noiseProbabilityMulti;
        dd::ComplexValue                 sqrtAmplitudeDampingProbability;
        dd::ComplexValue                 oneMinusSqrtAmplitudeDampingProbability;
        dd::ComplexValue                 sqrtAmplitudeDampingProbabilityMulti;
        dd::ComplexValue                 oneMinusSqrtAmplitudeDampingProbabilityMulti;
        dd::GateMatrix                   ampDampingTrue{};
        dd::GateMatrix                   ampDampingTrueMulti{};
        dd::GateMatrix                   ampDampingFalse{};
        dd::GateMatrix                   ampDampingFalseMulti{};
        std::vector<dd::NoiseOperations> noiseEffects;

        [[nodiscard]] dd::Qubit getNumberOfQubits() const { return nQubits; }
        [[nodiscard]] double    getNoiseProbability(bool multiQubitNoiseFlag) const { return multiQubitNoiseFlag ? noiseProbabilityMulti : noiseProbability; }

        [[nodiscard]] OpType getAmplitudeDampingOperationType(bool multiQubitNoiseFlag, bool amplitudeDampingFlag) const {
            if (amplitudeDampingFlag) {
                return multiQubitNoiseFlag ? qc::MultiATrue : qc::ATrue;
            } else {
                return multiQubitNoiseFlag ? qc::MultiAFalse : qc::AFalse;
            }
        }

        [[nodiscard]] dd::GateMatrix getAmplitudeDampingOperationMatrix(bool multiQubitNoiseFlag, bool amplitudeDampingFlag) const {
            if (amplitudeDampingFlag) {
                return multiQubitNoiseFlag ? ampDampingTrueMulti : ampDampingTrue;
            } else {
                return multiQubitNoiseFlag ? ampDampingFalseMulti : ampDampingFalse;
            }
        }

    public:
        dd::mEdge identityDD;

        void setNoiseEffects(std::vector<dd::NoiseOperations> newNoiseEffects) { noiseEffects = std::move(newNoiseEffects); }

        void applyNoiseOperation(const std::vector<dd::Qubit>& targets, dd::mEdge operation, dd::vEdge& state, std::mt19937_64& generator) {
            const bool multiQubitOperation = targets.size() > 1;

            for (const auto& target: targets) {
                auto stackedOperation = generateNoiseOperation(operation, target, generator, false, multiQubitOperation);
                auto tmp              = package->multiply(stackedOperation, state);

                if (dd::ComplexNumbers::mag2(tmp.w) < dist(generator)) {
                    // The probability of amplitude damping does not only depend on the noise probability, but also the quantum state.
                    // Due to the normalization constraint of decision diagrams the probability for applying amplitude damping stands in the root edge weight,
                    // of the dd after the noise has been applied
                    stackedOperation = generateNoiseOperation(operation, target, generator, true, multiQubitOperation);
                    tmp              = package->multiply(stackedOperation, state);
                }
                tmp.w = dd::Complex::one;

                package->incRef(tmp);
                package->decRef(state);
                state = tmp;

                // I only need to apply the operations once
                operation = identityDD;
            }
        }

    protected:
        dd::mEdge generateNoiseOperation(dd::mEdge        operation,
                                         dd::Qubit        target,
                                         std::mt19937_64& generator,
                                         bool             amplitudeDamping,
                                         bool             multiQubitOperation) {
            qc::OpType effect;
            for (const auto& noiseType: noiseEffects) {
                if (noiseType != dd::amplitudeDamping) {
                    effect = ReturnNoiseOperation(noiseType, dist(generator), multiQubitOperation);
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
                        auto tmpOperation = package->stochasticNoiseOperationCache.lookup(getAmplitudeDampingOperationType(multiQubitOperation, true), target);
                        if (tmpOperation.p == nullptr) {
                            tmpOperation = package->makeGateDD(getAmplitudeDampingOperationMatrix(multiQubitOperation, true), getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(getAmplitudeDampingOperationType(multiQubitOperation, true), target, tmpOperation);
                        }
                        operation = package->multiply(tmpOperation, operation);
                        break;
                    }
                    case (qc::AFalse): {
                        auto tmpOperation = package->stochasticNoiseOperationCache.lookup(getAmplitudeDampingOperationType(multiQubitOperation, false), target);
                        if (tmpOperation.p == nullptr) {
                            tmpOperation = package->makeGateDD(getAmplitudeDampingOperationMatrix(multiQubitOperation, false), getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(getAmplitudeDampingOperationType(multiQubitOperation, false), target, tmpOperation);
                        }
                        operation = package->multiply(tmpOperation, operation);
                        break;
                    }
                    case (qc::X): {
                        auto tmpOperation = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmpOperation.p == nullptr) {
                            tmpOperation = package->makeGateDD(dd::Xmat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmpOperation);
                        }
                        operation = package->multiply(tmpOperation, operation);
                        break;
                    }
                    case (qc::Y): {
                        auto tmpOperation = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmpOperation.p == nullptr) {
                            tmpOperation = package->makeGateDD(dd::Ymat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmpOperation);
                        }
                        operation = package->multiply(tmpOperation, operation);
                        break;
                    }
                    case (qc::Z): {
                        auto tmpOperation = package->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmpOperation.p == nullptr) {
                            tmpOperation = package->makeGateDD(dd::Zmat, getNumberOfQubits(), target);
                            package->stochasticNoiseOperationCache.insert(effect, target, tmpOperation);
                        }
                        operation = package->multiply(tmpOperation, operation);
                        break;
                    }
                    default: {
                        throw std::runtime_error("Unknown noise operation '" + std::to_string(effect) + "'\n");
                    }
                }
            }
            return operation;
        }

        [[nodiscard]] qc::OpType ReturnNoiseOperation(dd::NoiseOperations noiseOperation, double prob, bool multiQubitNoiseFlag) const {
            switch (noiseOperation) {
                case dd::NoiseOperations::depolarization: {
                    if (prob >= (getNoiseProbability(multiQubitNoiseFlag) * 0.75)) {
                        // prob > prob apply qc::I, also 25 % of the time when depolarization is applied nothing happens
                        return qc::I;
                    } else if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.25)) {
                        // if 0 < prob < 0.25 (25 % of the time when applying depolarization) apply qc::X
                        return qc::X;
                    } else if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.5)) {
                        // if 0.25 < prob < 0.5 (25 % of the time when applying depolarization) apply qc::Z
                        return qc::Y;
                    } else {
                        // if 0.5 < prob < 0.75 (25 % of the time when applying depolarization) apply qc::Z
                        return qc::Z;
                    }
                }
                case dd::NoiseOperations::phaseFlip: {
                    if (prob > getNoiseProbability(multiQubitNoiseFlag)) {
                        return dd::I;
                    } else {
                        return dd::Z;
                    }
                }
                case ::dd::identity: {
                    return dd::I;
                }

                default:
                    throw std::runtime_error(std::string{"Unknown noise effect '"} + std::to_string(noiseOperation) + "'");
            }
        }
    };

} // namespace dd

#endif //DDSIM_NOISEFUNCTIONALITY_HPP
