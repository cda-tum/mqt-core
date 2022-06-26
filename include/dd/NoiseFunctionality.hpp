/*
* This file is part of the MQT DD Package which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
*/

#ifndef DDSIM_NOISEFUNCTIONALITY_HPP
#define DDSIM_NOISEFUNCTIONALITY_HPP

#include "dd/Definitions.hpp"
#include "dd/Export.hpp"

#include <random>
#include <utility>

using CN = dd::ComplexNumbers;

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
                                     dd::QubitCount                    nQubits,
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
        dd::QubitCount                         nQubits;
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
                    effect = returnNoiseOperation(noiseType, dist(generator), multiQubitOperation);
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

        [[nodiscard]] qc::OpType returnNoiseOperation(dd::NoiseOperations noiseOperation, double prob, bool multiQubitNoiseFlag) const {
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

    template<class DDPackage>
    class DeterministicNoiseFunctionality {
    public:
        DeterministicNoiseFunctionality(const std::unique_ptr<DDPackage>& package,
                                        dd::QubitCount                    nQubits,
                                        double                            noiseProbSingleQubit,
                                        double                            noiseProbMultiQubit,
                                        double                            ampDampingProbSingleQubit,
                                        double                            ampDampingProbMultiQubit,
                                        std::vector<dd::NoiseOperations>  noiseEffects,
                                        bool                              useDensityMatrixType,
                                        bool                              sequentiallyApplyNoise):
            package(package),
            nQubits(nQubits),
            noiseProbSingleQubit(noiseProbSingleQubit),
            noiseProbMultiQubit(noiseProbMultiQubit),
            ampDampingProbSingleQubit(ampDampingProbSingleQubit),
            ampDampingProbMultiQubit(ampDampingProbMultiQubit),
            noiseEffects(std::move(noiseEffects)),
            useDensityMatrixType(useDensityMatrixType),
            sequentiallyApplyNoise(sequentiallyApplyNoise) {
        }

    protected:
        const std::unique_ptr<DDPackage>& package;
        dd::QubitCount                    nQubits;

        double noiseProbSingleQubit;
        double noiseProbMultiQubit;
        double ampDampingProbSingleQubit;
        double ampDampingProbMultiQubit;

        std::vector<dd::NoiseOperations> noiseEffects;
        bool                             useDensityMatrixType;
        bool                             sequentiallyApplyNoise;

        const std::map<dd::NoiseOperations, int> sequentialNoiseMap = {
                {dd::phaseFlip, 2},        //Phase-flip
                {dd::amplitudeDamping, 2}, //Amplitude Damping
                {dd::depolarization, 4},   //Depolarisation
        };

        [[nodiscard]] dd::Qubit getNumberOfQubits() const {
            return nQubits;
        }

    public:
        void applyNoiseEffects(dEdge& originalEdge, const std::unique_ptr<Operation>& qcOperation) {
            auto usedQubits = qcOperation->getTargets();
            for (auto control: qcOperation->getControls()) {
                usedQubits.push_back(control.qubit);
            }

            [[maybe_unused]] auto cacheSizeBefore = package->cn.cacheCount();

            if (sequentiallyApplyNoise) {
                applyDetNoiseSequential(originalEdge, usedQubits);
            } else {
                dEdge nodeAfterNoise = {};
                sort(usedQubits.begin(), usedQubits.end(), std::greater<>());

                if (useDensityMatrixType) {
                    dEdge::applyDmChangesToEdge(&originalEdge);
                    nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, false);
                    dEdge::revertDmChangesToEdge(&originalEdge);
                } else {
                    nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, true);
                }
                package->incRef(nodeAfterNoise);
                dEdge::alignDensityEdge(&originalEdge);
                package->decRef(originalEdge);
                originalEdge = nodeAfterNoise;
                if (useDensityMatrixType) {
                    dEdge::setDensityMatrixTrue(&originalEdge);
                }
            }
            [[maybe_unused]] auto cacheSizeAfter = package->cn.cacheCount();
            assert(cacheSizeAfter == cacheSizeBefore);
        }

    private:
        dEdge applyNoiseEffects(dEdge& originalEdge, const std::vector<dd::Qubit>& usedQubits, bool firstPathEdge) {
            if (originalEdge.p->v < usedQubits.back() || originalEdge.isTerminal()) {
                dEdge tmp{};
                if (originalEdge.w.approximatelyZero()) {
                    tmp.w = dd::Complex::zero;
                } else {
                    tmp.w = originalEdge.w;
                }
                if (originalEdge.isTerminal()) {
                    return dEdge::terminal(tmp.w);
                } else {
                    tmp.p = originalEdge.p;
                    return tmp;
                }
            }
            auto originalCopy = originalEdge;
            originalCopy.w    = dd::Complex::one;

            // Check if the target of the current edge is in the "compute table".

            //    auto noiseLookUpResult = dd->densityNoise.lookup(originalCopy, usedQubits);
            //
            //    if (noiseLookUpResult.p != nullptr) {
            //        auto tmpComplexValue = dd->cn.getCached();
            //        CN::mul(tmpComplexValue, noiseLookUpResult.w, originalEdge.w);
            //        noiseLookUpResult.w = dd->cn.lookup(tmpComplexValue);
            //        dd->cn.returnToCache(tmpComplexValue);
            //        return noiseLookUpResult;
            //    }

            std::array<dEdge, std::tuple_size_v<decltype(dd::dNode::e)>> new_edges{};
            for (short i = 0; i < 4; i++) {
                if (firstPathEdge || i == 1) {
                    // If I am to the useDensityMatrix I cannot minimize the necessary operations anymore
                    dEdge::applyDmChangesToEdge(&originalCopy.p->e[i]);
                    new_edges[i] = applyNoiseEffects(originalCopy.p->e[i], usedQubits, true);
                    dEdge::revertDmChangesToEdge(&originalCopy.p->e[i]);
                } else if (i == 2) {
                    // Size e[1] == e[2] (due to density matrix representation), I can skip calculating e[2]
                    new_edges[2].p = new_edges[1].p;
                    new_edges[2].w = new_edges[1].w;
                } else {
                    dEdge::applyDmChangesToEdge(&originalCopy.p->e[i]);
                    new_edges[i] = applyNoiseEffects(originalCopy.p->e[i], usedQubits, false);
                    dEdge::revertDmChangesToEdge(&originalCopy.p->e[i]);
                }
            }
            dEdge e = {};
            if (std::count(usedQubits.begin(), usedQubits.end(), originalCopy.p->v)) {
                for (auto& new_edge: new_edges) {
                    if (new_edge.w.approximatelyZero()) {
                        new_edge.w = dd::Complex::zero;
                    } else {
                        new_edge.w = package->cn.getCached(dd::CTEntry::val(new_edge.w.r), dd::CTEntry::val(new_edge.w.i));
                    }
                }

                for (auto const& type: noiseEffects) {
                    switch (type) {
                        case dd::amplitudeDamping:
                            applyAmplitudeDampingToEdges(new_edges, (usedQubits.size() == 1) ? ampDampingProbSingleQubit : ampDampingProbMultiQubit);
                            break;
                        case dd::phaseFlip:
                            applyPhaseFlipToEdges(new_edges, (usedQubits.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit);
                            break;
                        case dd::depolarization:
                            applyDepolarisationToEdges(new_edges, (usedQubits.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit);
                            break;
                        case dd::identity:
                            continue;
                    }
                }

                for (auto& new_edge: new_edges) {
                    if (new_edge.w.approximatelyZero()) {
                        if (!new_edge.w.exactlyZero()) {
                            package->cn.returnToCache(new_edge.w);
                            new_edge.w = dd::Complex::zero;
                        }
                    } else {
                        dd::Complex c = package->cn.lookup(new_edge.w);
                        package->cn.returnToCache(new_edge.w);
                        new_edge.w = c;
                    }
                }
            }

            e = package->makeDDNode(originalCopy.p->v, new_edges, false, firstPathEdge);

            // Adding the noise operation to the cache, note that e.w is from the complex number table
            //    package->densityNoise.insert(originalCopy, e, usedQubits);

            // Multiplying the old edge weight with the new one and looking up in the complex numbers table
            if (!e.w.approximatelyZero()) {
                if (e.w.approximatelyOne()) {
                    e.w = originalEdge.w;
                } else {
                    auto tmpComplexValue = package->cn.getCached();
                    CN::mul(tmpComplexValue, e.w, originalEdge.w);
                    e.w = package->cn.lookup(tmpComplexValue);
                    package->cn.returnToCache(tmpComplexValue);
                }
            }
            return e;
        }

        void applyPhaseFlipToEdges(std::array<dEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& e, double probability) {
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

        void applyAmplitudeDampingToEdges(std::array<dEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& e, double probability) {
            dd::Complex complexProb = package->cn.getCached();
            dEdge       helperEdge;
            helperEdge.w = package->cn.getCached();

            // e[0] = e[0] + p*e[3]
            if (!e[3].w.approximatelyZero()) {
                complexProb.r->value = probability;
                complexProb.i->value = 0;
                if (!e[0].w.approximatelyZero()) {
                    CN::mul(helperEdge.w, complexProb, e[3].w);
                    helperEdge.p = e[3].p;
                    auto tmp     = package->add2(e[0], helperEdge);
                    if (!e[0].w.exactlyZero()) {
                        package->cn.returnToCache(e[0].w);
                    }
                    e[0] = tmp;
                } else {
                    e[0].w = package->cn.getCached();
                    CN::mul(e[0].w, complexProb, e[3].w);
                    e[0].p = e[3].p;
                }
            }

            //e[1] = sqrt(1-p)*e[1]
            if (!e[1].w.approximatelyZero()) {
                complexProb.r->value = std::sqrt(1 - probability);
                complexProb.i->value = 0;
                CN::mul(e[1].w, complexProb, e[1].w);
            }

            //e[2] = sqrt(1-p)*e[2]
            if (!e[2].w.approximatelyZero()) {
                if (e[1].w.approximatelyZero()) {
                    complexProb.r->value = std::sqrt(1 - probability);
                    complexProb.i->value = 0;
                }
                CN::mul(e[2].w, complexProb, e[2].w);
            }

            //e[3] = (1-p)*e[3]
            if (!e[3].w.approximatelyZero()) {
                complexProb.r->value = 1 - probability;
                CN::mul(e[3].w, complexProb, e[3].w);
            }

            package->cn.returnToCache(helperEdge.w);
            package->cn.returnToCache(complexProb);
        }

        void applyDepolarisationToEdges(std::array<dEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& e, double probability) {
            dEdge       helperEdge[2];
            dd::Complex complexProb = package->cn.getCached();
            complexProb.i->value    = 0;

            dEdge oldE0Edge;
            oldE0Edge.w = package->cn.getCached(dd::CTEntry::val(e[0].w.r), dd::CTEntry::val(e[0].w.i));
            oldE0Edge.p = e[0].p;

            //e[0] = 0.5*((2-p)*e[0] + p*e[3])
            {
                helperEdge[0].w = dd::Complex::zero;
                helperEdge[1].w = dd::Complex::zero;

                //helperEdge[0] = 0.5*((2-p)*e[0]
                if (!e[0].w.approximatelyZero()) {
                    helperEdge[0].w      = package->cn.getCached();
                    complexProb.r->value = (2 - probability) * 0.5;
                    CN::mul(helperEdge[0].w, complexProb, e[0].w);
                    helperEdge[0].p = e[0].p;
                }

                //helperEdge[1] = 0.5*p*e[3]
                if (!e[3].w.approximatelyZero()) {
                    helperEdge[1].w      = package->cn.getCached();
                    complexProb.r->value = probability * 0.5;
                    CN::mul(helperEdge[1].w, complexProb, e[3].w);
                    helperEdge[1].p = e[3].p;
                }

                //e[0] = helperEdge[0] + helperEdge[1]
                if (!e[0].w.exactlyZero()) {
                    package->cn.returnToCache(e[0].w);
                }
                e[0] = package->add2(helperEdge[0], helperEdge[1]);

                if (!helperEdge[0].w.exactlyZero()) {
                    package->cn.returnToCache(helperEdge[0].w);
                }
                if (!helperEdge[1].w.exactlyZero()) {
                    package->cn.returnToCache(helperEdge[1].w);
                }
            }

            //e[1]=1-p*e[1]
            if (!e[1].w.approximatelyZero()) {
                complexProb.r->value = 1 - probability;
                CN::mul(e[1].w, e[1].w, complexProb);
            }
            //e[2]=1-p*e[2]
            if (!e[2].w.approximatelyZero()) {
                if (e[1].w.approximatelyZero()) {
                    complexProb.r->value = 1 - probability;
                }
                CN::mul(e[2].w, e[2].w, complexProb);
            }

            //e[3] = 0.5*((2-p)*e[3]) + 0.5*(p*e[0])
            {
                helperEdge[0].w = dd::Complex::zero;
                helperEdge[1].w = dd::Complex::zero;

                //helperEdge[0] = 0.5*((2-p)*e[3])
                if (!e[3].w.approximatelyZero()) {
                    helperEdge[0].w      = package->cn.getCached();
                    complexProb.r->value = (2 - probability) * 0.5;
                    CN::mul(helperEdge[0].w, complexProb, e[3].w);
                    helperEdge[0].p = e[3].p;
                }

                //helperEdge[1] = 0.5*p*e[0]
                if (!oldE0Edge.w.approximatelyZero()) {
                    helperEdge[1].w      = package->cn.getCached();
                    complexProb.r->value = probability * 0.5;
                    CN::mul(helperEdge[1].w, complexProb, oldE0Edge.w);
                    helperEdge[1].p = oldE0Edge.p;
                }

                //e[3] = helperEdge[0] + helperEdge[1]
                if (!e[3].w.exactlyZero()) {
                    package->cn.returnToCache(e[3].w);
                }
                e[3] = package->add2(helperEdge[0], helperEdge[1]);

                if (!helperEdge[0].w.exactlyZero()) {
                    package->cn.returnToCache(helperEdge[0].w);
                }
                if (!helperEdge[1].w.exactlyZero()) {
                    package->cn.returnToCache(helperEdge[1].w);
                }
            }
            package->cn.returnToCache(oldE0Edge.w);
            package->cn.returnToCache(complexProb);
        }

        void applyDetNoiseSequential(dEdge& originalEdge, const qc::Targets& targets) {
            dd::dEdge tmp = {};

            std::array<mEdge, std::tuple_size_v<decltype(dd::dNode::e)>> idleOperation{};

            // Iterate over qubits and check if the qubit had been used
            for (auto targetQubit: targets) {
                for (auto const& type: noiseEffects) {
                    generateGate(idleOperation, type, targetQubit, getNoiseProbability(type, targets));
                    tmp.p = nullptr;
                    //Apply all noise matrices of the current noise effect
                    for (int m = 0; m < sequentialNoiseMap.find(type)->second; m++) {
                        auto tmp0 = package->conjugateTranspose(idleOperation[m]);
                        auto tmp1 = package->multiply(originalEdge, reinterpret_cast<dEdge&>(tmp0), 0, false);
                        auto tmp2 = package->multiply(reinterpret_cast<dEdge&>(idleOperation[m]), tmp1, 0, useDensityMatrixType);
                        if (tmp.p == nullptr) {
                            tmp = tmp2;
                        } else {
                            tmp = package->add(tmp2, tmp);
                        }
                    }
                    package->incRef(tmp);
                    dEdge::alignDensityEdge(&originalEdge);
                    package->decRef(originalEdge);
                    originalEdge = tmp;
                    if (useDensityMatrixType) {
                        dEdge::setDensityMatrixTrue(&originalEdge);
                    }
                }
            }
        }

        void generateGate(std::array<mEdge, std::tuple_size_v<decltype(dd::dNode::e)>>& pointerForMatrices, dd::NoiseOperations noiseType, dd::Qubit target, double probability) {
            std::array<dd::GateMatrix, std::tuple_size_v<decltype(dd::dNode::e)>> idleNoiseGate{};
            dd::ComplexValue                                                      tmp = {};

            switch (noiseType) {
                case dd::phaseFlip: {
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
                    //      (1      0           )       (0      sqrt(probability))
                    //  e0= (0      sqrt(1-probability)   ), e1=  (0      0      )
                case dd::amplitudeDamping: {
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
                case dd::depolarization: {
                    tmp.r = std::sqrt(1 - ((3 * probability) / 4)) * dd::complex_one.r;
                    //                   (1 0)
                    // sqrt(1- ((3p)/4))*(0 1)
                    idleNoiseGate[0][0] = idleNoiseGate[0][3] = tmp;
                    idleNoiseGate[0][1] = idleNoiseGate[0][2] = dd::complex_zero;

                    pointerForMatrices[0] = package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);

                    //            (0 1)
                    // sqrt(probability/4))*(1 0)
                    tmp.r               = std::sqrt(probability / 4) * dd::complex_one.r;
                    idleNoiseGate[1][1] = idleNoiseGate[1][2] = tmp;
                    idleNoiseGate[1][0] = idleNoiseGate[1][3] = dd::complex_zero;

                    pointerForMatrices[1] = package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);

                    //            (1 0)
                    // sqrt(probability/4))*(0 -1)
                    tmp.r               = std::sqrt(probability / 4) * dd::complex_one.r;
                    idleNoiseGate[2][0] = tmp;
                    tmp.r               = tmp.r * -1;
                    idleNoiseGate[2][3] = tmp;
                    idleNoiseGate[2][1] = idleNoiseGate[2][2] = dd::complex_zero;

                    pointerForMatrices[3] = package->makeGateDD(idleNoiseGate[2], getNumberOfQubits(), target);

                    //            (0 -i)
                    // sqrt(probability/4))*(i 0)
                    tmp.r               = dd::complex_zero.r;
                    tmp.i               = std::sqrt(probability / 4) * 1;
                    idleNoiseGate[3][2] = tmp;
                    tmp.i               = tmp.i * -1;
                    idleNoiseGate[3][1] = tmp;
                    idleNoiseGate[3][0] = idleNoiseGate[3][3] = dd::complex_zero;

                    pointerForMatrices[2] = package->makeGateDD(idleNoiseGate[3], getNumberOfQubits(), target);
                    break;
                }
                default:
                    throw std::runtime_error("Unknown noise effect received.");
            }
        }

        double getNoiseProbability(const dd::NoiseOperations type, const qc::Targets& targets) {
            if (type == dd::amplitudeDamping) {
                return (targets.size() == 1) ? ampDampingProbSingleQubit : ampDampingProbMultiQubit;
            } else {
                return (targets.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit;
            }
        }
    };

} // namespace dd

#endif //DDSIM_NOISEFUNCTIONALITY_HPP
