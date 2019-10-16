#include <random>
#include <functional>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iterator>

#include "Operations/StandardOperation.h"
#include "QuantumComputation.h"

using namespace std;
using namespace chrono;

int main() {
	/*

	std::cout << "OpenQASM Demo\n---------\n";
	qc::QuantumComputation qcQASM;
	filename = "../../test/test.qasm";
	qcQASM.import(filename, qc::OpenQASM);
	cout << qcQASM << endl;

	auto dd = make_unique<dd::Package>();
	auto op1 = qc::StandardOperation(4, 1, 0, qc::H);
	dd->export2Dot(op1.getDD(dd), "a.dot");

	auto op2 = qc::StandardOperation(4, 1, 3, qc::H);
	dd->export2Dot( op2.getDD(dd), "b.dot");

	auto op3 = qc::StandardOperation(4, 2, qc::H);
	dd->export2Dot(op3.getDD(dd), "c.dot");

	auto op4 = dd->multiply(op1.getDD(dd), op2.getDD(dd));
	dd->export2Dot(op4, "d.dot");

	auto op5 = dd->multiply(op3.getDD(dd), op4);
	dd->export2Dot(op5, "e.dot");
	dd->cn.printComplexTable();
	std::cerr << "Cache count: " << dd->cn.cacheCount << std::endl;
	*/

	/*
	std::cout << "Real Demo\n---------\n";
	qc::QuantumComputation qcReal;
	string filename = "../../test/test.real";
	qcReal.import(filename, qc::Real);
	cout << qcReal << endl;
	for (int i = 0; i <= qcReal.getNops(); ++i) {
		auto it = qcReal.begin();
		if (i > 0) {
			std::advance(it, i - 1);
			std::cout << i << ": " << *(*it) << std::endl;
		}
		auto dd = make_unique<dd::Package>();
		auto start = high_resolution_clock::now();
		auto e = qcReal.buildFunctionality(dd, i);
		auto end = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(end - start);
		std::cout << "Time: " << elapsed.count() << "ms\n";
		std::cout << "CacheCount: " << dd->cn.cacheCount << std::endl;
		if (i == qcReal.getNops()) {
			dd->statistics();
			dd->cn.statistics();
			dd->cn.printComplexTable();
			dd->export2Dot(e, "e.dot");
			return 0;
		}
		dd->garbageCollect(true);
		std::cout << "\033[1m\033[36m" << "-------------------------------------" << "\033[0m" << std::endl;

	}
	 */

	//*/
	qc::QuantumComputation inst;
	string filename = "../../test/grover_6.real";
	inst.import(filename, qc::Real);
	cout << inst << endl;

	auto dd = make_unique<dd::Package>();
	dd::Edge in = dd->makeZeroState(inst.getNqubits());
	auto start = high_resolution_clock::now();
	inst.simulate(in, dd);
	auto end = high_resolution_clock::now();
	auto elapsed = duration_cast<milliseconds>(end - start);

	std::cout << "Simulation time: " << elapsed.count() << "ms\n";
	dd->statistics();
	dd->cn.statistics();
	std::cout << "-------------------------------------" << std::endl;

	auto dd2 = make_unique<dd::Package>();
	cout << "Number of operations: " << inst.getNops() << endl;
	start = high_resolution_clock::now();
	inst.buildFunctionality(dd2);
	end = high_resolution_clock::now();
	elapsed = duration_cast<milliseconds>(end - start);
	std::cout << "Time: " << elapsed.count() << "ms\n";
	dd2->statistics();
	dd2->cn.statistics();
	//dd2->cn.printComplexTable();

	/*
	for (int i = 0; i <= inst.getNops(); ++i) {
		auto it = inst.begin();
		if (i > 0) {
			std::advance(it, i-1);
			std::cout << i << ": " << *(*it) << std::endl;
		}
		auto dd2 = make_unique<dd::Package>();
		start = high_resolution_clock::now();
		auto e = inst.buildFunctionality(dd2, i);
		end = high_resolution_clock::now();
		elapsed = duration_cast<milliseconds>(end - start);
		std::cout << "Time: " << elapsed.count() << "ms\n";
		if (i == inst.getNops()) {
			dd2->statistics();
			dd2->cn.statistics();
			//dd2->cn.printComplexTable();
			//dd2->export2Dot(e, "e.dot");
			return 0;
		}
		dd2->garbageCollect(true);
		std::cout << "\033[1m\033[36m" << "-------------------------------------" << "\033[0m" <<  std::endl;

	}

	*/
	return 0;
}
