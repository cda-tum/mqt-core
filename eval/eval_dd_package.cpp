#include "CircuitOptimizer.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/Entanglement.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"
#include "algorithms/WState.hpp"
#include "dd/Benchmark.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "nlohmann/json.hpp"

#include <string>
#include <utility>

namespace dd {

static const std::string FILENAME_START = "results_";
static const std::string FILENAME_END = ".json";

static constexpr std::size_t SEED = 42U;

class BenchmarkDDPackage {
protected:
  void verifyAndSave(const std::string& name, const std::string& type,
                     qc::QuantumComputation& qc, const Experiment& exp) {

    const std::string& filename = FILENAME_START + inputFilename + FILENAME_END;

    nlohmann::json j;
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::ate);
    if (!file.is_open()) {
      std::ofstream outputFile(filename);
      outputFile << nlohmann::json();
    } else if (file.tellg() == 0) {
      file << nlohmann::json();
    }
    file.close();

    std::ifstream ifs(filename);
    ifs >> j;
    ifs.close();

    auto& entry = j[name][type][std::to_string(qc.getNqubits())];

    entry["runtime"] = exp.runtime.count();

    // collect statistics from DD package
    entry["dd"] = exp.stats;

    std::ofstream ofs(filename);
    ofs << j.dump(2U);
    ofs.close();
  }

  std::string inputFilename;

  void runGHZ() {
    const std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running GHZ Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::Entanglement(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("GHZ", "Simulation", qc, *exp);
    }
    std::cout << "Running GHZ Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::Entanglement(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("GHZ", "Functionality", qc, *exp);
    }
  }

  void runWState() {
    const std::array nqubits = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running WState Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::WState(nq);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("WState", "Simulation", qc, *exp);
    }
    std::cout << "Running WState Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::WState(nq);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("WState", "Functionality", qc, *exp);
    }
  }

  void runBV() {
    const std::array nqubits = {255U, 511U, 1023U, 2047U, 4095U};
    std::cout << "Running BV Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::BernsteinVazirani(nq);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("BV", "Simulation", qc, *exp);
    }
    std::cout << "Running BV Functionality..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = qc::BernsteinVazirani(nq);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("BV", "Functionality", qc, *exp);
    }
  }

  void runQFT() {
    const std::array nqubitsSim = {256U, 512U, 1024U, 2048U, 4096U};
    std::cout << "Running QFT Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::QFT(nq, false);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QFT", "Simulation", qc, *exp);
    }
    const std::array nqubitsFunc = {18U, 19U, 20U, 21U, 22U};
    std::cout << "Running QFT Functionality..." << '\n';
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::QFT(nq, false);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QFT", "Functionality", qc, *exp);
    }
  }

  void runGrover() {
    const std::array nqubits = {27U, 31U, 35U, 39U, 41U};
    std::cout << "Running Grover Simulation..." << '\n';
    for (const auto& nq : nqubits) {
      auto qc = std::make_unique<qc::Grover>(nq, SEED);
      auto dd = std::make_unique<dd::Package<>>(qc->getNqubits());
      const auto start = std::chrono::high_resolution_clock::now();

      // apply state preparation setup
      qc::QuantumComputation statePrep(qc->getNqubits());
      qc->setup(statePrep);
      auto s = buildFunctionality(&statePrep, *dd);
      auto e = dd->multiply(s, dd->makeZeroState(qc->getNqubits()));
      dd->incRef(e);
      dd->decRef(s);

      qc::QuantumComputation groverIteration(qc->getNqubits());
      qc->oracle(groverIteration);
      qc->diffusion(groverIteration);

      auto iter = buildFunctionalityRecursive(&groverIteration, *dd);
      std::bitset<128U> iterBits(qc->iterations);
      auto msb =
          static_cast<std::size_t>(std::floor(std::log2(qc->iterations)));
      auto f = iter;
      dd->incRef(f);
      for (std::size_t j = 0U; j <= msb; ++j) {
        if (iterBits[j]) {
          auto g = dd->multiply(f, e);
          dd->incRef(g);
          dd->decRef(e);
          e = g;
          dd->garbageCollect();
        }
        if (j < msb) {
          auto tmp = dd->multiply(f, f);
          dd->incRef(tmp);
          dd->decRef(f);
          f = tmp;
        }
      }
      dd->decRef(f);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto runtime =
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);
      std::unique_ptr<SimulationExperiment> exp =
          std::make_unique<SimulationExperiment>();
      exp->dd = std::move(dd);
      exp->sim = e;
      exp->runtime = runtime;
      exp->stats = dd::getStatistics(exp->dd.get());

      verifyAndSave("Grover", "Simulation", *qc, *exp);
    }

    std::cout << "Running Grover Functionality..." << '\n';
    for (const auto& nq : nqubits) {

      auto qc = std::make_unique<qc::Grover>(nq, SEED);
      auto exp = benchmarkFunctionalityConstruction(*qc, true);
      verifyAndSave("Grover", "Functionality", *qc, *exp);
    }
  }

  void runQPE() {
    const std::array nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running QPE Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::QPE(nq, false);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("QPE", "Simulation", qc, *exp);
    }
    std::cout << "Running QPE Functionality..." << '\n';
    const std::array nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::QPE(nq, false);
      qc::CircuitOptimizer::removeFinalMeasurements(qc);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("QPE", "Functionality", qc, *exp);
    }
  }

  void runRandomClifford() {
    const std::array<std::size_t, 5> nqubitsSim = {14U, 15U, 16U, 17U, 18U};
    std::cout << "Running RandomClifford Simulation..." << '\n';
    for (const auto& nq : nqubitsSim) {
      auto qc = qc::RandomCliffordCircuit(nq, nq * nq, SEED);
      auto exp = benchmarkSimulate(qc);
      verifyAndSave("RandomClifford", "Simulation", qc, *exp);
    }
    std::cout << "Running RandomClifford Functionality..." << '\n';
    const std::array<std::size_t, 5> nqubitsFunc = {7U, 8U, 9U, 10U, 11U};
    for (const auto& nq : nqubitsFunc) {
      auto qc = qc::RandomCliffordCircuit(nq, nq * nq, SEED);
      auto exp = benchmarkFunctionalityConstruction(qc);
      verifyAndSave("RandomClifford", "Functionality", qc, *exp);
    }
  }

public:
  explicit BenchmarkDDPackage(std::string filename)
      : inputFilename(std::move(filename)) {};

  void runAll() {
    runGHZ();
    runWState();
    runBV();
    runQFT();
    runGrover();
    runQPE();
    runRandomClifford();
  }
};

} // namespace dd

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Exactly one argument is required to name the results file."
              << '\n';
    return 1;
  }
  try {
    dd::BenchmarkDDPackage run = dd::BenchmarkDDPackage(
        argv[1]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    run.runAll();
  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Benchmarks done." << '\n';
  return 0;
}
