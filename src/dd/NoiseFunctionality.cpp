/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "dd/NoiseFunctionality.hpp"

namespace dd {
    template class StochasticNoiseFunctionality<DDPackageConfig>;
    template class DeterministicNoiseFunctionality<DDPackageConfig>;
} // namespace dd
