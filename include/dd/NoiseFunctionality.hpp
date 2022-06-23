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

namespace dd {

    // noise operations available for deterministic noise aware quantum circuit simulation
    enum noiseOperations : std::uint8_t {
        amplitudeDamping,
        phaseFlip,
        depolarization
    };

    template<class customPackage>
    class StochasticNoiseFunctionality {
    public:
        StochasticNoiseFunctionality(const std::unique_ptr<customPackage>&   cLocalDD,
                                     dd::Qubit                               size,
                                     double                                  cGateNoiseProbability,
                                     double                                  amplitudeDampingProb,
                                     double                                  multiQubitGateFactor,
                                     const std::vector<dd::noiseOperations>& cGateNoiseEffects,
                                     size_t                                  cSeed):
            localDD(cLocalDD),
            seed(cSeed), dist(0.0, 1.0L) {
            nqubits = size;
            //The probability of amplitude damping (t1) often is double the probability , of phase flip, which is why I double it here
            noiseProbability                        = cGateNoiseProbability;
            sqrtAmplitudeDampingProbability         = {std::sqrt(amplitudeDampingProb), 0};
            oneMinusSqrtAmplitudeDampingProbability = {std::sqrt(1 - amplitudeDampingProb), 0};

            noiseProbabilityMulti                        = cGateNoiseProbability * multiQubitGateFactor;
            sqrtAmplitudeDampingProbabilityMulti         = {std::sqrt(noiseProbability) * multiQubitGateFactor, 0};
            oneMinusSqrtAmplitudeDampingProbabilityMulti = {std::sqrt(1 - multiQubitGateFactor * amplitudeDampingProb), 0};
            ampDampingFalse                              = dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbability});
            ampDampingFalseMulti                         = dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbabilityMulti});

            ampDampingTrue      = dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbability, dd::complex_zero, dd::complex_zero});
            ampDampingTrueMulti = dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbabilityMulti, dd::complex_zero, dd::complex_zero});

            gateNoiseEffects = cGateNoiseEffects;
            identityDD       = localDD->makeIdent(size);
            localDD->incRef(identityDD);
        }

        dd::Qubit        nqubits;
        double           noiseProbability = 0.0;
        dd::ComplexValue sqrtAmplitudeDampingProbability{};
        dd::ComplexValue oneMinusSqrtAmplitudeDampingProbability{};

        double           noiseProbabilityMulti = 0.0;
        dd::ComplexValue sqrtAmplitudeDampingProbabilityMulti{};
        dd::ComplexValue oneMinusSqrtAmplitudeDampingProbabilityMulti{};

        dd::GateMatrix ampDampingTrue      = {};
        dd::GateMatrix ampDampingTrueMulti = {};

        dd::GateMatrix ampDampingFalse      = {};
        dd::GateMatrix ampDampingFalseMulti = {};

        std::vector<dd::noiseOperations> gateNoiseEffects;

        const std::unique_ptr<customPackage>& localDD;
        size_t                                seed = 0;

        dd::mEdge identityDD;

        std::uniform_real_distribution<dd::fp> dist;

        dd::Qubit getNumberOfQubits() { return nqubits; }

        void applyNoiseOperation(const std::vector<dd::Qubit>& usedQubits, dd::mEdge dd_op,
                                 dd::vEdge& localRootEdge, std::mt19937_64& generator, const std::vector<dd::noiseOperations>& noiseOperation) {
            bool multiQubitOperation = usedQubits.size() > 1;

            for (auto& target: usedQubits) {
                auto operation = generateNoiseOperation(dd_op, target, noiseOperation, generator, false, multiQubitOperation);
                auto tmp       = localDD->multiply(operation, localRootEdge);

                if (dd::ComplexNumbers::mag2(tmp.w) < dist(generator)) {
                    operation = generateNoiseOperation(dd_op, target, noiseOperation, generator, true, multiQubitOperation);
                    tmp       = localDD->multiply(operation, localRootEdge);
                }

                if (tmp.w != dd::Complex::one) {
                    tmp.w = dd::Complex::one;
                }
                localDD->incRef(tmp);
                localDD->decRef(localRootEdge);
                localRootEdge = tmp;

                // I only need to apply the operations once
                dd_op = identityDD;
            }
        }

        dd::mEdge generateNoiseOperation(dd::mEdge   dd_operation,
                                         signed char target, const std::vector<dd::noiseOperations>& noiseOperation, std::mt19937_64& generator,
                                         bool amplitudeDamping,
                                         bool multiQubitOperation) {
            qc::OpType effect;
            for (const auto& noise_type: noiseOperation) {
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
                        auto tmp_op = localDD->stochasticNoiseOperationCache.lookup(multiQubitOperation ? qc::multiATrue : qc::ATrue, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = localDD->makeGateDD(multiQubitOperation ? ampDampingTrueMulti : ampDampingTrue, getNumberOfQubits(), target);
                            localDD->stochasticNoiseOperationCache.insert(multiQubitOperation ? qc::multiATrue : qc::ATrue, target, tmp_op);
                        }

                        dd_operation = localDD->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::AFalse): {
                        auto tmp_op = localDD->stochasticNoiseOperationCache.lookup(multiQubitOperation ? qc::multiAFalse : qc::AFalse, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = localDD->makeGateDD(multiQubitOperation ? ampDampingFalseMulti : ampDampingFalse, getNumberOfQubits(), target);
                            localDD->stochasticNoiseOperationCache.insert(multiQubitOperation ? qc::multiAFalse : qc::AFalse, target, tmp_op);
                        }
                        dd_operation = localDD->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::X): {
                        auto tmp_op = localDD->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = localDD->makeGateDD(dd::Xmat, getNumberOfQubits(), target);
                            localDD->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = localDD->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::Y): {
                        auto tmp_op = localDD->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = localDD->makeGateDD(dd::Ymat, getNumberOfQubits(), target);
                            localDD->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = localDD->multiply(tmp_op, dd_operation);
                        break;
                    }
                    case (qc::Z): {
                        auto tmp_op = localDD->stochasticNoiseOperationCache.lookup(effect, target);
                        if (tmp_op.p == nullptr) {
                            tmp_op = localDD->makeGateDD(dd::Zmat, getNumberOfQubits(), target);
                            localDD->stochasticNoiseOperationCache.insert(effect, target, tmp_op);
                        }
                        dd_operation = localDD->multiply(tmp_op, dd_operation);
                        break;
                    }
                    default: {
                        throw std::runtime_error("Unknown noise operation '" + std::to_string(effect) + "'\n");
                    }
                }
            }
            return dd_operation;
        }

        qc::OpType ReturnNoiseOperation(dd::noiseOperations noiseOperation, double prob, bool multi_qubit_noise) const {
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
