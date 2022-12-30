/*
* This file is part of the MQT DD Package which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
*/

#pragma once

#include "dd/Definitions.hpp"
#include "dd/Export.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <random>
#include <utility>

using CN           = dd::ComplexNumbers;
using ArrayOfEdges = std::array<qc::DensityMatrixDD, std::tuple_size_v<decltype(dd::dNode::e)>>;

namespace dd {

    // noise operations available for deterministic noise aware quantum circuit simulation
    enum NoiseOperations : std::uint8_t {
        AmplitudeDamping,
        PhaseFlip,
        Depolarization,
        Identity
    };

    template<class Config>
    class StochasticNoiseFunctionality {
    public:
        StochasticNoiseFunctionality(const std::unique_ptr<dd::Package<Config>>& dd,
                                     dd::QubitCount                              nq,
                                     double                                      gateNoiseProbability,
                                     double                                      amplitudeDampingProb,
                                     double                                      multiQubitGateFactor,
                                     std::vector<dd::NoiseOperations>            effects):
            package(dd),
            nQubits(nq),
            dist(0.0, 1.0L),
            noiseProbability(gateNoiseProbability),
            noiseProbabilityMulti(gateNoiseProbability * multiQubitGateFactor),
            sqrtAmplitudeDampingProbability({std::sqrt(amplitudeDampingProb), 0}),
            oneMinusSqrtAmplitudeDampingProbability({std::sqrt(1 - amplitudeDampingProb), 0}),
            sqrtAmplitudeDampingProbabilityMulti({std::sqrt(gateNoiseProbability) * multiQubitGateFactor, 0}),
            oneMinusSqrtAmplitudeDampingProbabilityMulti({std::sqrt(1 - multiQubitGateFactor * amplitudeDampingProb), 0}),
            ampDampingTrue(dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbability, dd::complex_zero, dd::complex_zero})),
            ampDampingTrueMulti(dd::GateMatrix({dd::complex_zero, sqrtAmplitudeDampingProbabilityMulti, dd::complex_zero, dd::complex_zero})),
            ampDampingFalse(dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbability})),
            ampDampingFalseMulti(dd::GateMatrix({dd::complex_one, dd::complex_zero, dd::complex_zero, oneMinusSqrtAmplitudeDampingProbabilityMulti})),
            noiseEffects(std::move(effects)), identityDD(package->makeIdent(nQubits)) {
            package->incRef(identityDD);
        }

        ~StochasticNoiseFunctionality() {
            package->decRef(identityDD);
        }

    protected:
        const std::unique_ptr<dd::Package<Config>>& package;
        dd::QubitCount                              nQubits;
        std::uniform_real_distribution<dd::fp>      dist;

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
        dd::mEdge                        identityDD;

        [[nodiscard]] dd::QubitCount getNumberOfQubits() const { return nQubits; }
        [[nodiscard]] double         getNoiseProbability(bool multiQubitNoiseFlag) const { return multiQubitNoiseFlag ? noiseProbabilityMulti : noiseProbability; }

        [[nodiscard]] qc::OpType getAmplitudeDampingOperationType(bool multiQubitNoiseFlag, bool amplitudeDampingFlag) const {
            if (amplitudeDampingFlag) {
                return multiQubitNoiseFlag ? qc::MultiATrue : qc::ATrue;
            }
            return multiQubitNoiseFlag ? qc::MultiAFalse : qc::AFalse;
        }

        [[nodiscard]] dd::GateMatrix getAmplitudeDampingOperationMatrix(bool multiQubitNoiseFlag, bool amplitudeDampingFlag) const {
            if (amplitudeDampingFlag) {
                return multiQubitNoiseFlag ? ampDampingTrueMulti : ampDampingTrue;
            }
            return multiQubitNoiseFlag ? ampDampingFalseMulti : ampDampingFalse;
        }

    public:
        [[nodiscard]] dd::mEdge getIdentityDD() const { return identityDD; }
        void                    setNoiseEffects(std::vector<dd::NoiseOperations> newNoiseEffects) { noiseEffects = std::move(newNoiseEffects); }

        void applyNoiseOperation(const std::set<qc::Qubit>& targets, dd::mEdge operation, dd::vEdge& state, std::mt19937_64& generator) {
            const bool multiQubitOperation = targets.size() > 1;

            for (const auto& target: targets) {
                auto stackedOperation = generateNoiseOperation(operation, static_cast<dd::Qubit>(target), generator, false, multiQubitOperation);
                auto tmp              = package->multiply(stackedOperation, state);

                if (dd::ComplexNumbers::mag2(tmp.w) < dist(generator)) {
                    // The probability of amplitude damping does not only depend on the noise probability, but also the quantum state.
                    // Due to the normalization constraint of decision diagrams the probability for applying amplitude damping stands in the root edge weight,
                    // of the dd after the noise has been applied
                    stackedOperation = generateNoiseOperation(operation, static_cast<dd::Qubit>(target), generator, true, multiQubitOperation);
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
        [[nodiscard]] dd::mEdge stackOperation(dd::mEdge operation, const dd::Qubit target, const qc::OpType noiseOperation, const GateMatrix matrix) {
            auto tmpOperation = package->stochasticNoiseOperationCache.lookup(noiseOperation, target);
            if (tmpOperation.p == nullptr) {
                tmpOperation = package->makeGateDD(matrix, getNumberOfQubits(), target);
                package->stochasticNoiseOperationCache.insert(noiseOperation, target, tmpOperation);
            }
            return package->multiply(tmpOperation, operation);
        }

        dd::mEdge generateNoiseOperation(dd::mEdge        operation,
                                         dd::Qubit        target,
                                         std::mt19937_64& generator,
                                         bool             amplitudeDamping,
                                         bool             multiQubitOperation) {
            for (const auto& noiseType: noiseEffects) {
                const auto effect = noiseType == dd::AmplitudeDamping ? getAmplitudeDampingOperationType(multiQubitOperation, amplitudeDamping) : returnNoiseOperation(noiseType, dist(generator), multiQubitOperation);
                switch (effect) {
                    case (qc::I): {
                        continue;
                    }
                    case (qc::MultiATrue):
                    case (qc::ATrue): {
                        const GateMatrix amplitudeDampingMatrix = getAmplitudeDampingOperationMatrix(multiQubitOperation, true);
                        operation                               = stackOperation(operation, target, effect, amplitudeDampingMatrix);
                        break;
                    }
                    case (qc::MultiAFalse):
                    case (qc::AFalse): {
                        const GateMatrix amplitudeDampingMatrix = getAmplitudeDampingOperationMatrix(multiQubitOperation, false);
                        operation                               = stackOperation(operation, target, effect, amplitudeDampingMatrix);
                        break;
                    }
                    case (qc::X): {
                        operation = stackOperation(operation, target, effect, Xmat);
                        break;
                    }
                    case (qc::Y): {
                        operation = stackOperation(operation, target, effect, Ymat);
                        break;
                    }
                    case (qc::Z): {
                        operation = stackOperation(operation, target, effect, Zmat);
                        break;
                    }
                    default: {
                        throw std::runtime_error("Unknown noise operation '" + std::to_string(effect) + "'\n");
                    }
                }
            }
            return operation;
        }

        [[nodiscard]] qc::OpType returnNoiseOperation(dd::NoiseOperations noiseOperation, double prob, bool multiQubitNoiseFlag) const {
            switch (noiseOperation) {
                case dd::NoiseOperations::Depolarization: {
                    if (prob >= (getNoiseProbability(multiQubitNoiseFlag) * 0.75)) {
                        // prob > prob apply qc::I, also 25 % of the time when depolarization is applied nothing happens
                        return qc::I;
                    }
                    if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.25)) {
                        // if 0 < prob < 0.25 (25 % of the time when applying depolarization) apply qc::X
                        return qc::X;
                    }
                    if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.5)) {
                        // if 0.25 < prob < 0.5 (25 % of the time when applying depolarization) apply qc::Y
                        return qc::Y;
                    }
                    // if 0.5 < prob < 0.75 (25 % of the time when applying depolarization) apply qc::Z
                    return qc::Z;
                }
                case dd::NoiseOperations::PhaseFlip: {
                    if (prob > getNoiseProbability(multiQubitNoiseFlag)) {
                        return qc::I;
                    }
                    return qc::Z;
                }
                case dd::NoiseOperations::Identity: {
                    return qc::I;
                }
                default:
                    throw std::runtime_error(std::string{"Unknown noise effect '"} + std::to_string(noiseOperation) + "'");
            }
        }
    };

    template<class Config>
    class DeterministicNoiseFunctionality {
    public:
        DeterministicNoiseFunctionality(const std::unique_ptr<dd::Package<Config>>& dd,
                                        dd::QubitCount                              nq,
                                        double                                      noiseProbabilitySingleQubit,
                                        double                                      noiseProbabilityMultiQubit,
                                        double                                      ampDampProbSingleQubit,
                                        double                                      ampDampProbMultiQubit,
                                        std::vector<dd::NoiseOperations>            effects,
                                        bool                                        useDensityMatType,
                                        bool                                        seqApplyNoise):
            package(dd),
            nQubits(nq),
            noiseProbSingleQubit(noiseProbabilitySingleQubit),
            noiseProbMultiQubit(noiseProbabilityMultiQubit),
            ampDampingProbSingleQubit(ampDampProbSingleQubit),
            ampDampingProbMultiQubit(ampDampProbMultiQubit),
            noiseEffects(std::move(effects)),
            useDensityMatrixType(useDensityMatType),
            sequentiallyApplyNoise(seqApplyNoise) {
        }

    protected:
        const std::unique_ptr<dd::Package<Config>>& package;
        dd::QubitCount                              nQubits;

        double noiseProbSingleQubit;
        double noiseProbMultiQubit;
        double ampDampingProbSingleQubit;
        double ampDampingProbMultiQubit;

        std::vector<dd::NoiseOperations> noiseEffects;
        bool                             useDensityMatrixType;
        bool                             sequentiallyApplyNoise;

        inline static const std::map<dd::NoiseOperations, std::size_t> SEQUENTIAL_NOISE_MAP = {
                {dd::Identity, 1},         //Identity Noise
                {dd::PhaseFlip, 2},        //Phase-flip
                {dd::AmplitudeDamping, 2}, //Amplitude Damping
                {dd::Depolarization, 4},   //Depolarisation
        };

        [[nodiscard]] dd::QubitCount getNumberOfQubits() const {
            return nQubits;
        }

    public:
        void applyNoiseEffects(qc::DensityMatrixDD& originalEdge, const std::unique_ptr<qc::Operation>& qcOperation) {
            auto usedQubits = qcOperation->getUsedQubits();

            [[maybe_unused]] const auto cacheSizeBefore = package->cn.cacheCount();

            if (sequentiallyApplyNoise) {
                applyDetNoiseSequential(originalEdge, usedQubits);
            } else {
                qc::DensityMatrixDD nodeAfterNoise = {};
                if (useDensityMatrixType) {
                    qc::DensityMatrixDD::applyDmChangesToEdge(originalEdge);
                    nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, false);
                    qc::DensityMatrixDD::revertDmChangesToEdge(originalEdge);
                } else {
                    nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, true);
                }
                if (!nodeAfterNoise.w.exactlyZero() && !nodeAfterNoise.w.exactlyOne()) {
                    const auto tmpComplexValue = package->cn.getTemporary(CTEntry::val(nodeAfterNoise.w.r), CTEntry::val(nodeAfterNoise.w.i));
                    package->cn.returnToCache(nodeAfterNoise.w);
                    nodeAfterNoise.w = package->cn.lookup(tmpComplexValue);
                }

                package->incRef(nodeAfterNoise);
                qc::DensityMatrixDD::alignDensityEdge(originalEdge);
                package->decRef(originalEdge);
                originalEdge = nodeAfterNoise;
                if (useDensityMatrixType) {
                    qc::DensityMatrixDD::setDensityMatrixTrue(originalEdge);
                }
            }
            [[maybe_unused]] const auto cacheSizeAfter = package->cn.cacheCount();
            assert(cacheSizeAfter == cacheSizeBefore);
        }

    private:
        qc::DensityMatrixDD applyNoiseEffects(qc::DensityMatrixDD& originalEdge, const std::set<qc::Qubit>& usedQubits, bool firstPathEdge) {
            if (originalEdge.p->v < static_cast<dd::Qubit>(*usedQubits.begin())) {
                qc::DensityMatrixDD tmp{};
                if (originalEdge.w.exactlyZero() || originalEdge.w.exactlyOne()) {
                    tmp.w = originalEdge.w;
                } else {
                    tmp.w = package->cn.getCached(CTEntry::val(originalEdge.w.r), CTEntry::val(originalEdge.w.i));
                }
                if (originalEdge.isTerminal()) {
                    return qc::DensityMatrixDD::terminal(tmp.w);
                }
                tmp.p = originalEdge.p;
                return tmp;
            }
            auto originalCopy = originalEdge;
            originalCopy.w    = dd::Complex::one;

            ArrayOfEdges newEdges{};
            for (size_t i = 0; i < newEdges.size(); i++) {
                if (firstPathEdge || i == 1) {
                    // If I am to the firstPathEdge I cannot minimize the necessary operations anymore
                    qc::DensityMatrixDD::applyDmChangesToEdge(originalCopy.p->e.at(i));
                    newEdges[i] = applyNoiseEffects(originalCopy.p->e.at(i), usedQubits, true);
                    qc::DensityMatrixDD::revertDmChangesToEdge(originalCopy.p->e.at(i));
                } else if (i == 2) {
                    // Since e[1] == e[2] (due to density matrix representation), I can skip calculating e[2]
                    newEdges[2].p = newEdges[1].p;
                    newEdges[2].w = package->cn.getCached(CTEntry::val(newEdges[1].w.r), CTEntry::val(newEdges[1].w.i));
                } else {
                    qc::DensityMatrixDD::applyDmChangesToEdge(originalCopy.p->e.at(i));
                    newEdges[i] = applyNoiseEffects(originalCopy.p->e.at(i), usedQubits, false);
                    qc::DensityMatrixDD::revertDmChangesToEdge(originalCopy.p->e.at(i));
                }
            }
            qc::DensityMatrixDD e = {};
            if (std::any_of(usedQubits.begin(), usedQubits.end(), [originalEdge](dd::Qubit qubit) { return originalEdge.p->v == qubit; })) {
                for (auto const& type: noiseEffects) {
                    switch (type) {
                        case dd::AmplitudeDamping:
                            applyAmplitudeDampingToEdges(newEdges, (usedQubits.size() == 1) ? ampDampingProbSingleQubit : ampDampingProbMultiQubit);
                            break;
                        case dd::PhaseFlip:
                            applyPhaseFlipToEdges(newEdges, (usedQubits.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit);
                            break;
                        case dd::Depolarization:
                            applyDepolarisationToEdges(newEdges, (usedQubits.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit);
                            break;
                        case dd::Identity:
                            continue;
                    }
                }
            }

            e = package->makeDDNode(originalCopy.p->v, newEdges, true, firstPathEdge);

            // Multiplying the old edge weight with the new one and looking up in the complex numbers table
            if (!e.w.exactlyZero()) {
                if (e.w.exactlyOne()) {
                    e.w = package->cn.getCached(CTEntry::val(originalEdge.w.r), CTEntry::val(originalEdge.w.i));
                } else {
                    CN::mul(e.w, e.w, originalEdge.w);
                }
            }
            return e;
        }

        void
        applyPhaseFlipToEdges(ArrayOfEdges& e, double probability) {
            dd::Complex complexProb = package->cn.getCached();

            //e[0] = e[0]

            //e[1] = (1-2p)*e[1]
            if (!e[1].w.approximatelyZero()) {
                complexProb.r->value = 1 - 2 * probability;
                complexProb.i->value = 0;
                CN::mul(e[1].w, complexProb, e[1].w);
            }

            //e[2] = (1-2p)*e[2]
            if (!e[2].w.approximatelyZero()) {
                if (e[1].w.approximatelyZero()) {
                    complexProb.r->value = 1 - 2 * probability;
                    complexProb.i->value = 0;
                }
                CN::mul(e[2].w, complexProb, e[2].w);
            }

            //e[3] = e[3]

            package->cn.returnToCache(complexProb);
        }

        void applyAmplitudeDampingToEdges(ArrayOfEdges& e, double probability) {
            dd::Complex         complexProb = package->cn.getCached();
            qc::DensityMatrixDD helperEdge;
            helperEdge.w = package->cn.getCached();

            // e[0] = e[0] + p*e[3]
            if (!e[3].w.exactlyZero()) {
                complexProb.r->value = probability;
                complexProb.i->value = 0;
                if (!e[0].w.exactlyZero()) {
                    CN::mul(helperEdge.w, complexProb, e[3].w);
                    helperEdge.p = e[3].p;
                    auto tmp     = package->add2(e[0], helperEdge);
                    if (!e[0].w.exactlyZero() && !e[0].w.exactlyOne()) {
                        package->cn.returnToCache(e[0].w);
                    }
                    e[0] = tmp;
                } else {
                    // e[0].w is exactly zero therefore I need to get a new cached value
                    e[0].w = package->cn.getCached();
                    CN::mul(e[0].w, complexProb, e[3].w);
                    e[0].p = e[3].p;
                }
            }

            //e[1] = sqrt(1-p)*e[1]
            if (!e[1].w.exactlyZero()) {
                complexProb.r->value = std::sqrt(1 - probability);
                complexProb.i->value = 0;
                if (e[1].w.exactlyOne()) {
                    e[1].w = package->cn.getCached(CTEntry::val(complexProb.r), CTEntry::val(complexProb.i));
                } else {
                    CN::mul(e[1].w, complexProb, e[1].w);
                }
            }

            //e[2] = sqrt(1-p)*e[2]
            if (!e[2].w.exactlyZero()) {
                if (e[1].w.exactlyZero()) {
                    complexProb.r->value = std::sqrt(1 - probability);
                    complexProb.i->value = 0;
                }
                if (e[2].w.exactlyOne()) {
                    e[2].w = package->cn.getCached(CTEntry::val(complexProb.r), CTEntry::val(complexProb.i));
                } else {
                    CN::mul(e[2].w, complexProb, e[2].w);
                }
            }

            //e[3] = (1-p)*e[3]
            if (!e[3].w.exactlyZero()) {
                complexProb.r->value = 1 - probability;
                if (e[3].w.exactlyOne()) {
                    e[3].w = package->cn.getCached(CTEntry::val(complexProb.r), CTEntry::val(complexProb.i));
                } else {
                    CN::mul(e[3].w, complexProb, e[3].w);
                }
            }
            package->cn.returnToCache(helperEdge.w);
            package->cn.returnToCache(complexProb);
        }

        void applyDepolarisationToEdges(ArrayOfEdges& e, double probability) {
            std::array<qc::DensityMatrixDD, 2> helperEdge{};
            dd::Complex                        complexProb = package->cn.getCached();
            complexProb.i->value                           = 0;

            qc::DensityMatrixDD oldE0Edge;
            oldE0Edge.w = package->cn.getCached(dd::CTEntry::val(e[0].w.r), dd::CTEntry::val(e[0].w.i));
            oldE0Edge.p = e[0].p;

            //e[0] = 0.5*((2-p)*e[0] + p*e[3])
            {
                helperEdge[0].w = dd::Complex::zero;
                helperEdge[1].w = dd::Complex::zero;

                //helperEdge[0] = 0.5*((2-p)*e[0]
                if (!e[0].w.exactlyZero()) {
                    helperEdge[0].w      = package->cn.getCached();
                    complexProb.r->value = (2 - probability) * 0.5;
                    CN::mul(helperEdge[0].w, complexProb, e[0].w);
                }
                helperEdge[0].p = e[0].p;

                //helperEdge[1] = 0.5*p*e[3]
                if (!e[3].w.exactlyZero()) {
                    helperEdge[1].w      = package->cn.getCached();
                    complexProb.r->value = probability * 0.5;
                    CN::mul(helperEdge[1].w, complexProb, e[3].w);
                }
                helperEdge[1].p = e[3].p;

                //e[0] = helperEdge[0] + helperEdge[1]
                if (!e[0].w.exactlyZero() && !e[0].w.exactlyOne()) {
                    package->cn.returnToCache(e[0].w);
                }
                e[0] = package->add2(helperEdge[0], helperEdge[1]);

                if (!helperEdge[0].w.exactlyZero() && !helperEdge[0].w.exactlyOne()) {
                    package->cn.returnToCache(helperEdge[0].w);
                }
                if (!helperEdge[1].w.exactlyZero() && !helperEdge[1].w.exactlyOne()) {
                    package->cn.returnToCache(helperEdge[1].w);
                }
            }

            //e[1]=1-p*e[1]
            if (!e[1].w.exactlyZero()) {
                complexProb.r->value = 1 - probability;
                if (e[1].w.exactlyOne()) {
                    e[1].w = package->cn.getCached(CTEntry::val(complexProb.r), CTEntry::val(complexProb.i));
                } else {
                    CN::mul(e[1].w, e[1].w, complexProb);
                }
            }
            //e[2]=1-p*e[2]
            if (!e[2].w.exactlyZero()) {
                if (e[1].w.exactlyZero()) {
                    complexProb.r->value = std::sqrt(1 - probability);
                }
                if (e[2].w.exactlyOne()) {
                    e[2].w = package->cn.getCached(CTEntry::val(complexProb.r), CTEntry::val(complexProb.i));
                } else {
                    CN::mul(e[2].w, e[2].w, complexProb);
                }
            }

            //e[3] = 0.5*((2-p)*e[3]) + 0.5*(p*e[0])
            {
                helperEdge[0].w = dd::Complex::zero;
                helperEdge[1].w = dd::Complex::zero;

                //helperEdge[0] = 0.5*((2-p)*e[3])
                if (!e[3].w.exactlyZero()) {
                    helperEdge[0].w      = package->cn.getCached();
                    complexProb.r->value = (2 - probability) * 0.5;
                    CN::mul(helperEdge[0].w, complexProb, e[3].w);
                }
                helperEdge[0].p = e[3].p;
                //helperEdge[1] = 0.5*p*e[0]
                if (!oldE0Edge.w.exactlyZero()) {
                    helperEdge[1].w      = package->cn.getCached();
                    complexProb.r->value = probability * 0.5;
                    CN::mul(helperEdge[1].w, complexProb, oldE0Edge.w);
                }
                helperEdge[1].p = oldE0Edge.p;

                //e[3] = helperEdge[0] + helperEdge[1]
                if (!e[3].w.exactlyZero() && !e[3].w.exactlyOne()) {
                    package->cn.returnToCache(e[3].w);
                }
                e[3] = package->add2(helperEdge[0], helperEdge[1]);
                if (!helperEdge[0].w.exactlyZero() && !helperEdge[0].w.exactlyOne()) {
                    package->cn.returnToCache(helperEdge[0].w);
                }
                if (!helperEdge[1].w.exactlyZero() && !helperEdge[1].w.exactlyOne()) {
                    package->cn.returnToCache(helperEdge[1].w);
                }
            }
            package->cn.returnToCache(oldE0Edge.w);
            package->cn.returnToCache(complexProb);
        }

        void applyDetNoiseSequential(qc::DensityMatrixDD& originalEdge, const std::set<qc::Qubit>& targets) {
            qc::DensityMatrixDD tmp = {};

            std::array<mEdge, std::tuple_size_v<decltype(dd::dNode::e)>> idleOperation{};

            // Iterate over qubits and check if the qubit had been used
            for (const auto targetQubit: targets) {
                for (auto const& type: noiseEffects) {
                    generateGate(idleOperation, type, static_cast<dd::Qubit>(targetQubit), getNoiseProbability(type, targets));
                    tmp.p = nullptr;
                    //Apply all noise matrices of the current noise effect
                    for (std::size_t m = 0; m < SEQUENTIAL_NOISE_MAP.find(type)->second; m++) {
                        auto tmp0 = package->conjugateTranspose(idleOperation.at(m));
                        auto tmp1 = package->multiply(originalEdge, dd::densityFromMatrixEdge(tmp0), 0, false);
                        auto tmp2 = package->multiply(dd::densityFromMatrixEdge(idleOperation.at(m)), tmp1, 0, useDensityMatrixType);
                        if (tmp.p == nullptr) {
                            tmp = tmp2;
                        } else {
                            tmp = package->add(tmp2, tmp);
                        }
                    }
                    package->incRef(tmp);
                    qc::DensityMatrixDD::alignDensityEdge(originalEdge);
                    package->decRef(originalEdge);
                    originalEdge = tmp;
                    if (useDensityMatrixType) {
                        qc::DensityMatrixDD::setDensityMatrixTrue(originalEdge);
                    }
                }
            }
        }

        void generateDepolarizationGate(std::array<mEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& pointerForMatrices, const dd::Qubit target,
                                        const double probability) {
            std::array<dd::GateMatrix, std::tuple_size_v<decltype(dd::dNode::e)>> idleNoiseGate{};
            dd::ComplexValue                                                      tmp = {};

            tmp.r = std::sqrt(1 - ((3 * probability) / 4)) * dd::complex_one.r;
            //                   (1 0)
            // sqrt(1- ((3p)/4))*(0 1)
            idleNoiseGate[0][0] = idleNoiseGate[0][3] = tmp;
            idleNoiseGate[0][1] = idleNoiseGate[0][2] = dd::complex_zero;

            pointerForMatrices[0] = package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);

            //                      (0 1)
            // sqrt(probability/4))*(1 0)
            tmp.r               = std::sqrt(probability / 4) * dd::complex_one.r;
            idleNoiseGate[1][1] = idleNoiseGate[1][2] = tmp;
            idleNoiseGate[1][0] = idleNoiseGate[1][3] = dd::complex_zero;

            pointerForMatrices[1] = package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);

            //                      (1 0)
            // sqrt(probability/4))*(0 -1)
            tmp.r               = std::sqrt(probability / 4) * dd::complex_one.r;
            idleNoiseGate[2][0] = tmp;
            tmp.r               = tmp.r * -1;
            idleNoiseGate[2][3] = tmp;
            idleNoiseGate[2][1] = idleNoiseGate[2][2] = dd::complex_zero;

            pointerForMatrices[3] = package->makeGateDD(idleNoiseGate[2], getNumberOfQubits(), target);

            //                      (0 -i)
            // sqrt(probability/4))*(i 0)
            tmp.r               = dd::complex_zero.r;
            tmp.i               = std::sqrt(probability / 4) * 1;
            idleNoiseGate[3][2] = tmp;
            tmp.i               = tmp.i * -1;
            idleNoiseGate[3][1] = tmp;
            idleNoiseGate[3][0] = idleNoiseGate[3][3] = dd::complex_zero;

            pointerForMatrices[2] = package->makeGateDD(idleNoiseGate[3], getNumberOfQubits(), target);
        }

        void generateGate(std::array<mEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& pointerForMatrices,
                          const dd::NoiseOperations                                     noiseType,
                          const dd::Qubit                                               target,
                          const double                                                  probability) {
            std::array<dd::GateMatrix, std::tuple_size_v<decltype(dd::dNode::e)>> idleNoiseGate{};
            dd::ComplexValue                                                      tmp = {};

            switch (noiseType) {
                    // identity noise (for testing)
                    //                  (1  0)
                    //                  (0  1),
                case dd::Identity: {
                    idleNoiseGate[0][0] = idleNoiseGate[0][3] = dd::complex_one;
                    idleNoiseGate[0][1] = idleNoiseGate[0][2] = dd::complex_zero;

                    pointerForMatrices[0] = package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);

                    break;
                }
                    // phase flip
                    //                          (1  0)                         (1  0)
                    //  e0= sqrt(1-probability)*(0  1), e1=  sqrt(probability)*(0 -1)
                case dd::PhaseFlip: {
                    tmp.r               = std::sqrt(1 - probability) * dd::complex_one.r;
                    idleNoiseGate[0][0] = idleNoiseGate[0][3] = tmp;
                    idleNoiseGate[0][1] = idleNoiseGate[0][2] = dd::complex_zero;
                    tmp.r                                     = std::sqrt(probability) * dd::complex_one.r;
                    idleNoiseGate[1][0]                       = tmp;
                    tmp.r *= -1;
                    idleNoiseGate[1][3] = tmp;
                    idleNoiseGate[1][1] = idleNoiseGate[1][2] = dd::complex_zero;

                    pointerForMatrices[0] = package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);
                    pointerForMatrices[1] = package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);

                    break;
                }
                    // amplitude damping
                    //      (1                  0)       (0      sqrt(probability))
                    //  e0= (0 sqrt(1-probability), e1=  (0                      0)
                case dd::AmplitudeDamping: {
                    tmp.r               = std::sqrt(1 - probability) * dd::complex_one.r;
                    idleNoiseGate[0][0] = dd::complex_one;
                    idleNoiseGate[0][1] = idleNoiseGate[0][2] = dd::complex_zero;
                    idleNoiseGate[0][3]                       = tmp;

                    tmp.r               = std::sqrt(probability) * dd::complex_one.r;
                    idleNoiseGate[1][0] = idleNoiseGate[1][3] = idleNoiseGate[1][2] = dd::complex_zero;
                    idleNoiseGate[1][1]                                             = tmp;

                    pointerForMatrices[0] = package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);
                    pointerForMatrices[1] = package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);
                    break;
                }
                    // depolarization
                case dd::Depolarization:
                    generateDepolarizationGate(pointerForMatrices, target, probability);
                    break;
                default:
                    throw std::runtime_error("Unknown noise effect received.");
            }
        }

        double getNoiseProbability(const dd::NoiseOperations type, const std::set<qc::Qubit>& targets) {
            if (type == dd::AmplitudeDamping) {
                return (targets.size() == 1) ? ampDampingProbSingleQubit : ampDampingProbMultiQubit;
            }
            return (targets.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit;
        }
    };

} // namespace dd
