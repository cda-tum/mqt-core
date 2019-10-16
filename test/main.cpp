#include <random>
#include <functional>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iterator>

#include "StandardOperation.h"

using namespace std;
using namespace chrono;

int main() {

	//dd->cn.printComplexTable();
	//cerr << "------------------" << endl;

	short nqubits = 16;
	int nops = 80;
	int seed = 0;

	//mt19937_64 generator(random_device{ }());
	mt19937_64 generator(seed);
	uniform_int_distribution<short> distribution(0, nqubits-1);
	auto rng = bind(distribution, ref(generator));

	vector<unique_ptr<Operation>> ops;
	for (int i = 0; i < nops; ++i) {
		vector<short> controls;
		for (int j = 0; j < rng(); ++j) {
			controls.emplace_back(rng());
		}
		short target0 = rng();
		while(std::find(controls.begin(), controls.end(), target0)!=controls.end())
			target0 = rng();

		Gate gate = (Gate) ((2*rng()) % 21);
		if (gate == Measure) {
			ops.push_back(make_unique<StandardOperation>(StandardOperation{nqubits, controls, controls}));
		} else if (gate == SWAP || gate == P || gate == Pdag) {
			short target1 = rng();
			while (target1 == target0) target1 = rng();
			ops.push_back(make_unique<StandardOperation>(StandardOperation{nqubits, controls, target0, target1, gate}));
		} else if (gate == X) {
			ops.push_back(make_unique<StandardOperation>(StandardOperation{nqubits, controls, target0}));
		} else {
			ops.push_back(make_unique<StandardOperation>(StandardOperation{nqubits, controls, {target0}, gate}));
		}
		//cout << ops[i]->getName() << ", c: ";
		//for (auto c: controls)
		//	cout << c << " ";
		//cout << ", t: " << target0;
		//cout << endl;
	}

	auto start = high_resolution_clock::now();
	auto itbegin = ops.begin();
	auto itmiddle = ops.begin();
	advance(itmiddle, ops.size() / 2);
	auto itrmiddle = ops.rbegin();
	advance(itrmiddle, ops.size()/2);
	auto itend = ops.end();

	auto dd1 = make_unique<dd::Package>();
	auto dd2 = make_unique<dd::Package>();
	auto dd3 = make_unique<dd::Package>();
	auto dd4 = make_unique<dd::Package>();
	
	dd::Edge result1 = dd1->makeIdent(0, nqubits - 1);
	dd1->incRef(result1);
	int i = 0;
	for (auto it = itbegin; it!=itmiddle; ++it) {
		//cout << "1: " << (*it)->getName() << endl;
		cout << i++ << endl;
		if (!(*it)->isMeasurement()) {
			auto tmp = dd1->multiply((*it)->getDD(dd1.get()), result1);
			dd1->incRef(tmp);
			dd1->decRef(result1);
			result1 = tmp;
		}
	}
	//dd1->export2Dot(result1, "result_1.dot");


	dd::Edge result2 = dd2->makeIdent(0, nqubits - 1);
	dd2->incRef(result2);
	for (auto it = itmiddle; it != itend; ++it) {
		//cout << "2: " << (*it)->getName() << endl;
		if (!(*it)->isMeasurement()) {
			auto tmp = dd2->multiply((*it)->getDD(dd2.get()), result2);
			dd2->incRef(tmp);
			dd2->decRef(result2);
			result2 = tmp;
			//dd2->export2Dot(result2, "result_2.dot");
		}
	}
	//dd2->export2Dot(result2, "result_2.dot");


	dd::Edge result3 = dd3->makeIdent(0, nqubits - 1);
	dd3->incRef(result3);
	for (auto it = ops.rbegin(); it != itrmiddle; ++it) {
		//cout << "3: " << (*it)->getName() << endl;
		if (!(*it)->isMeasurement()) {
			auto tmp = dd3->multiply((*it)->getInverseDD(dd3.get()), result3);
			dd3->incRef(tmp);
			dd3->decRef(result3);
			result3 = tmp;
		}
	}
	//dd3->export2Dot(result3, "result_3.dot");


	dd::Edge result4 = dd4->makeIdent(0, nqubits - 1);
	dd4->incRef(result4);
	for (auto it = itrmiddle; it != ops.rend(); ++it) {
		//cout << "4: " << (*it)->getName() << endl;
		if (!(*it)->isMeasurement()) {
			auto tmp = dd4->multiply((*it)->getInverseDD(dd4.get()), result4);
			dd4->incRef(tmp);
			dd4->decRef(result4);
			result4 = tmp;
		}
	}
	//dd4->export2Dot(result4, "result_4.dot");


	auto dd5 = make_unique<dd::Package>();
	dd::Edge result5 = dd5->makeIdent(0, nqubits - 1);
	dd5->incRef(result5);
	auto tmp = dd5->multiply(result5, result2);
	dd5->incRef(tmp);
	dd5->decRef(result5);
	result5 = tmp;
	//dd5->export2Dot(result5, "result_firstmult.dot");
	tmp = dd5->multiply(result3, result5);
	dd5->incRef(tmp);
	dd5->decRef(result5);
	result5 = tmp;
	cout << "Is identity? " << dd::Package::equals(result5, dd5->makeIdent(0, nqubits - 1)) << std::endl;
	//dd5->export2Dot(result5, "result_secondmult.dot");

	tmp = dd5->multiply(result5, result1);
	dd5->incRef(tmp);
	dd5->decRef(result5);
	result5 = tmp;
	//dd5->export2Dot(result5, "result_thirdmult.dot");
	tmp = dd5->multiply(result4, result5);
	dd5->incRef(tmp);
	dd5->decRef(result5);
	result5 = tmp;
	cout << "Is identity? " << dd::Package::equals(result5, dd5->makeIdent(0, nqubits-1)) << std::endl;
	//dd5->export2Dot(result5, "result_fourthmult.dot");



	//dd->export2Dot(result, "result.dot");
	//cout << "Is identity? " << dd::Package::equals(result, dd->makeIdent(0, nqubits-1)) << std::endl;

	auto end = high_resolution_clock::now();
	duration<double> diff = end - start;
	cout << "Iteration: " << diff.count() << endl;
	//dd1->garbageCollect(true);
	//dd2->garbageCollect(true);

	//delete dd;
	/*

	auto dd1 = make_unique<dd::Package>();
	auto op = StandardOperation(3,{2},{1},Z);
	auto e = dd1->makeIdent(0,2);
	e = dd1->multiply(op.getDD(dd1.get()), e);
	cout << dd::Package::equals(e,op.getDD(dd1.get())) << endl;
	dd1->export2Dot(e, "e.dot");
	*/
	return 0;
}
