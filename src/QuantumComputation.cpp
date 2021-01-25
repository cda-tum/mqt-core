/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

namespace qc {

	/***
     * Public Methods
     ***/
	unsigned long long QuantumComputation::getNindividualOps() const {
		unsigned long long nops = 0;
		for (const auto& op: ops) {
			if (op->isCompoundOperation()) {
				auto&& comp = dynamic_cast<CompoundOperation*>(op.get());
				nops += comp->size();
			} else {
				++nops;
			}
		}

		return nops;
	}

	void QuantumComputation::import(const std::string& filename) {
		size_t dot = filename.find_last_of('.');
		std::string extension = filename.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [] (unsigned char ch) { return ::tolower(ch); });
		if (extension == "real") {
			import(filename, Real);
		} else if (extension == "qasm") {
			import(filename, OpenQASM);
		} else if(extension == "txt") {
			import(filename, GRCS);
		} else if (extension == "tfc") {
			import(filename, TFC);
		} else if (extension == "qc") {
			import(filename, QC);
		} else {
			throw QFRException("[import] extension " + extension + " not recognized");
		}
	}

	void QuantumComputation::import(const std::string& filename, Format format) {
		size_t slash = filename.find_last_of('/');
		size_t dot = filename.find_last_of('.');
		name = filename.substr(slash+1, dot-slash-1);

		auto ifs = std::ifstream(filename);
		if (ifs.good()) {
			import(ifs, format);
		} else {
			throw QFRException("[import] Error processing input stream: " + name);
		}
	}

	void QuantumComputation::import(std::istream&& is, Format format) {
		// reset circuit before importing
		reset();

		switch (format) {
			case Real:
				importReal(is);
				break;
			case OpenQASM:
				updateMaxControls(2);
				importOpenQASM(is);
				break;
			case GRCS:
				importGRCS(is);
				break;
			case TFC:
				importTFC(is);
				break;
			case QC:
				importQC(is);
				break;
			default:
				throw QFRException("[import] Format " + std::to_string(format) + " not yet supported");
		}

		// initialize the initial layout and output permutation
		initializeIOMapping();
	}

	void QuantumComputation::initializeIOMapping() {
		// if no initial layout was found during parsing the identity mapping is assumed
		if (initialLayout.empty()) {
			for (unsigned short i = 0; i < nqubits; ++i)
				initialLayout.insert({ i, i});
		}

		// try gathering (additional) output permutation information from measurements, e.g., a measurement
		//      `measure q[i] -> c[j];`
		// implies that the j-th (logical) output is obtained from measuring the i-th physical qubit.
		bool outputPermutationFound = !outputPermutation.empty();
		for (auto opIt=ops.begin(); opIt != ops.end(); ++opIt) {
			if ((*opIt)->getType() == qc::Measure) {
				if (!isLastOperationOnQubit(opIt))
					continue;

				auto&& op = (*opIt);
				for (size_t i=0; i<op->getNcontrols(); ++i) {
					auto qubitidx = op->getControls().at(i).qubit;
					auto bitidx = op->getTargets().at(i);

					if (outputPermutationFound) {
						// output permutation was already set before -> permute existing values
						auto current = outputPermutation.at(qubitidx);
						if (qubitidx != bitidx && current != bitidx) {
							for (auto& p: outputPermutation) {
								if (p.second == bitidx) {
									p.second = current;
									break;
								}
							}
							outputPermutation.at(qubitidx) = bitidx;
						}
					} else {
						// directly set permutation if none was set beforehand
						outputPermutation[qubitidx] = bitidx;
					}
				}
			}
		}

		// if the output permutation is still empty, we assume the identity (i.e., it is equal to the initial layout)
		if (outputPermutation.empty()) {
			for (unsigned short i = 0; i < nqubits; ++i) {
				// only add to output permutation if the qubit is actually acted upon
				if (!isIdleQubit(i))
					outputPermutation.insert({ i, initialLayout.at(i)});
			}
		}

		// allow for incomplete output permutation -> mark rest as garbage
		for (const auto& in: initialLayout) {
			bool isOutput = false;
			for (const auto& out: outputPermutation) {
				if (in.second == out.second) {
					isOutput = true;
					break;
				}
			}
			if (!isOutput) {
				setLogicalQubitGarbage(in.second);
			}
		}
	}

	void QuantumComputation::addQubitRegister(unsigned short nq, const char* reg_name) {
		if (nqubits + nancillae + nq > dd::MAXN) {
			throw QFRException("[addQubitRegister] Adding additional qubits results in too many qubits " + std::to_string(nqubits + nancillae + nq) + " vs. " + std::to_string(dd::MAXN));
		}

		if (qregs.count(reg_name)) {
			auto& reg = qregs.at(reg_name);
			if(reg.first+reg.second == nqubits+nancillae) {
				reg.second+=nq;
			} else {
				throw QFRException("[addQubitRegister] Augmenting existing qubit registers is only supported for the last register in a circuit");
			}
		} else {
			qregs.insert({reg_name, {nqubits, nq}});
		}
		assert(nancillae == 0); // should only reach this point if no ancillae are present

		for (unsigned short i = 0; i < nq; ++i) {
			unsigned short j = nqubits + i;
			initialLayout.insert({ j, j});
			outputPermutation.insert({ j, j});
		}
		nqubits += nq;

		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	void QuantumComputation::addClassicalRegister(unsigned short nc, const char* reg_name) {

		if (cregs.count(reg_name)) {
			throw QFRException("[addClassicalRegister] Augmenting existing classical registers is currently not supported");
		}

		cregs.insert({reg_name, {nclassics, nc}});
		nclassics += nc;
	}

	void QuantumComputation::addAncillaryRegister(unsigned short nq, const char* reg_name) {
		if (nqubits + nancillae + nq > dd::MAXN) {
			throw QFRException("[addAncillaryQubitRegister] Adding additional qubits results in too many qubits " + std::to_string(nqubits + nancillae + nq) + " vs. " + std::to_string(dd::MAXN));
		}

		unsigned short totalqubits = nqubits + nancillae;

		if (ancregs.count(reg_name)) {
			auto& reg = ancregs.at(reg_name);
			if(reg.first+reg.second == totalqubits) {
				reg.second+=nq;
			} else {
				throw QFRException("[addAncillaryRegister] Augmenting existing ancillary registers is only supported for the last register in a circuit");
			}
		} else {
			ancregs.insert({reg_name, {totalqubits, nq}});
		}

		for (unsigned short i = 0; i < nq; ++i) {
			unsigned short j = totalqubits + i;
			initialLayout.insert({ j, j});
			outputPermutation.insert({ j, j});
			ancillary.set(j);
		}
		nancillae += nq;

		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	// removes the i-th logical qubit and returns the index j it was assigned to in the initial layout
	// i.e., initialLayout[j] = i
	std::pair<unsigned short, short> QuantumComputation::removeQubit(unsigned short logical_qubit_index) {
		#if DEBUG_MODE_QC
		std::cout << "Trying to remove logical qubit: " << logical_qubit_index << std::endl;
		#endif

		// Find index of the physical qubit i is assigned to
		unsigned short physical_qubit_index = 0;
		for (const auto& Q:initialLayout) {
			if (Q.second == logical_qubit_index)
				physical_qubit_index = Q.first;
		}

		#if DEBUG_MODE_QC
		std::cout << "Found index " << logical_qubit_index << " is assigned to: " << physical_qubit_index << std::endl;
		printRegisters(std::cout);
		#endif

		// get register and register-index of the corresponding qubit
		auto reg = getQubitRegisterAndIndex(physical_qubit_index);

		#if DEBUG_MODE_QC
		std::cout << "Found register: " << reg.first << ", and index: " << reg.second << std::endl;
		printRegisters(std::cout);
		#endif

		if (physicalQubitIsAncillary(physical_qubit_index)) {
			#if DEBUG_MODE_QC
			std::cout << physical_qubit_index << " is ancilla" << std::endl;
			#endif
			// first index
			if (reg.second == 0) {
				// last remaining qubit of register
				if (ancregs[reg.first].second == 1) {
					// delete register
					ancregs.erase(reg.first);
				}
				// first qubit of register
				else {
					ancregs[reg.first].first++;
					ancregs[reg.first].second--;
				}
			// last index
			} else if (reg.second == ancregs[reg.first].second-1) {
				// reduce count of register
				ancregs[reg.first].second--;
			} else {
				auto ancreg = ancregs.at(reg.first);
				auto low_part = reg.first + "_l";
				auto low_index = ancreg.first;
				auto low_count = reg.second;
				auto high_part = reg.first + "_h";
				auto high_index = ancreg.first + reg.second + 1;
				auto high_count = ancreg.second - reg.second - 1;

				#if DEBUG_MODE_QC
				std::cout << "Splitting register: " << reg.first << ", into:" << std::endl;
				std::cout << low_part << ": {" << low_index << ", " << low_count << "}" << std::endl;
				std::cout << high_part << ": {" << high_index << ", " << high_count << "}" << std::endl;
				#endif

				ancregs.erase(reg.first);
				ancregs.insert({low_part, {low_index, low_count}});
				ancregs.insert({high_part, {high_index, high_count}});
			}
			// reduce ancilla count
			nancillae--;
		} else {
			if (reg.second == 0) {
				// last remaining qubit of register
				if (qregs[reg.first].second == 1) {
					// delete register
					qregs.erase(reg.first);
				}
					// first qubit of register
				else {
					qregs[reg.first].first++;
					qregs[reg.first].second--;
				}
			// last index
			} else if (reg.second == qregs[reg.first].second-1) {
				// reduce count of register
				qregs[reg.first].second--;
			} else {
				auto qreg = qregs.at(reg.first);
				auto low_part = reg.first + "_l";
				auto low_index = qreg.first;
				auto low_count = reg.second;
				auto high_part = reg.first + "_h";
				auto high_index = qreg.first + reg.second + 1;
				auto high_count = qreg.second - reg.second - 1;

				#if DEBUG_MODE_QC
				std::cout << "Splitting register: " << reg.first << ", into:" << std::endl;
				std::cout << low_part << ": {" << low_index << ", " << low_count << "}" << std::endl;
				std::cout << high_part << ": {" << high_index << ", " << high_count << "}" << std::endl;
				#endif

				qregs.erase(reg.first);
				qregs.insert({low_part, {low_index, low_count}});
				qregs.insert({high_part, {high_index, high_count}});
			}
			// reduce qubit count
			nqubits--;
		}

		#if DEBUG_MODE_QC
		std::cout << "Updated registers: " << std::endl;
		printRegisters(std::cout);
		std::cout << "nqubits: " << nqubits << ", nancillae: " << nancillae << std::endl;
		#endif

		// adjust initial layout permutation
		initialLayout.erase(physical_qubit_index);

		#if DEBUG_MODE_QC
		std::cout << "Updated initial layout: " << std::endl;
		printPermutationMap(initialLayout, std::cout);
		#endif

		// remove potential output permutation entry
		short output_qubit_index = -1;
		auto it = outputPermutation.find(physical_qubit_index);
		if (it != outputPermutation.end()) {
			output_qubit_index = it->second;
			// erasing entry
			outputPermutation.erase(physical_qubit_index);
			#if DEBUG_MODE_QC
			std::cout << "Updated output permutation: " << std::endl;
			printPermutationMap(outputPermutation, std::cout);
			#endif
		}

		// update all operations
		auto totalQubits = static_cast<unsigned short>(nqubits+nancillae);
		for (auto& op:ops) {
			op->setNqubits(totalQubits);
		}

		// update ancillary and garbage tracking
		if (totalQubits < qc::MAX_QUBITS) {
			for (unsigned short i=logical_qubit_index; i<totalQubits; ++i) {
				ancillary[i] = ancillary[i+1];
				garbage[i] = garbage[i+1];
			}
			// unset last entry
			ancillary.reset(totalQubits);
			garbage.reset(totalQubits);
		}

		return { physical_qubit_index, output_qubit_index};
	}

	// adds j-th physical qubit as ancilla to the end of reg or creates the register if necessary
	void QuantumComputation::addAncillaryQubit(unsigned short physical_qubit_index, short output_qubit_index) {
		if(initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
			throw QFRException("[addAncillaryQubit] Attempting to insert physical qubit that is already assigned");
		}

		#if DEBUG_MODE_QC
		std::cout << "Trying to add physical qubit " << physical_qubit_index
				  << " as ancillary with output qubit index: " << output_qubit_index << std::endl;
		#endif

		bool fusionPossible = false;
		for( auto& ancreg : ancregs) {
			auto& anc_start_index = ancreg.second.first;
			auto& anc_count = ancreg.second.second;
			// 1st case: can append to start of existing register
			if (anc_start_index == physical_qubit_index + 1) {
				anc_start_index--;
				anc_count++;
				fusionPossible = true;
				break;
			}
			// 2nd case: can append to end of existing register
			else if (anc_start_index + anc_count == physical_qubit_index) {
				anc_count++;
				fusionPossible = true;
				break;
			}
		}

		if (ancregs.empty()) {
			ancregs.insert({DEFAULT_ANCREG, {physical_qubit_index, 1}});
		} else if(!fusionPossible) {
			auto new_reg_name = std::string(DEFAULT_ANCREG) + "_" + std::to_string(physical_qubit_index);
			ancregs.insert({new_reg_name, { physical_qubit_index, 1}});
		}

		// index of logical qubit
		unsigned short logical_qubit_index = nqubits + nancillae;

		// increase ancillae count and mark as ancillary
		nancillae++;
		ancillary.set(logical_qubit_index);

		#if DEBUG_MODE_QC
		std::cout << "Updated registers: " << std::endl;
		printRegisters(std::cout);
		std::cout << "nqubits: " << nqubits << ", nancillae: " << nancillae << std::endl;
		#endif

		// adjust initial layout
		initialLayout.insert({ physical_qubit_index, logical_qubit_index});
		#if DEBUG_MODE_QC
		std::cout << "Updated initial layout: " << std::endl;
		printPermutationMap(initialLayout, std::cout);
		#endif

		// adjust output permutation
		if (output_qubit_index >= 0) {
			outputPermutation.insert({ physical_qubit_index, output_qubit_index});
			#if DEBUG_MODE_QC
			std::cout << "Updated output permutation: " << std::endl;
			printPermutationMap(outputPermutation, std::cout);
			#endif
		}

		// update all operations
		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	void QuantumComputation::addQubit(unsigned short logical_qubit_index, unsigned short physical_qubit_index, short output_qubit_index) {
		if (initialLayout.count(physical_qubit_index) || outputPermutation.count(physical_qubit_index)) {
			std::cerr << "Attempting to insert physical qubit that is already assigned" << std::endl;
			exit(1);
		}

		if (logical_qubit_index > nqubits) {
			std::cerr << "There are currently only " << nqubits << " qubits in the circuit. Adding "
					  << logical_qubit_index << " is therefore not possible at the moment." << std::endl;
			exit(1);
			// TODO: this does not necessarily have to lead to an error. A new qubit register could be created and all ancillaries shifted
		}

		// check if qubit fits in existing register
		bool fusionPossible = false;
		for( auto& qreg : qregs) {
			auto& q_start_index = qreg.second.first;
			auto& q_count = qreg.second.second;
			// 1st case: can append to start of existing register
			if (q_start_index == physical_qubit_index + 1) {
				q_start_index--;
				q_count++;
				fusionPossible = true;
				break;
			}
			// 2nd case: can append to end of existing register
			else if (q_start_index + q_count == physical_qubit_index) {
				if (physical_qubit_index == nqubits) {
					// need to shift ancillaries
					for (auto& ancreg: ancregs) {
						ancreg.second.first++;
					}
				}
				q_count++;
				fusionPossible = true;
				break;
			}
		}

		consolidateRegister(qregs);

		if (qregs.empty()) {
			qregs.insert({DEFAULT_QREG, {physical_qubit_index, 1}});
		} else if(!fusionPossible) {
			auto new_reg_name = std::string(DEFAULT_QREG) + "_" + std::to_string(physical_qubit_index);
			qregs.insert({new_reg_name, { physical_qubit_index, 1}});
		}

		// increase qubit count
		nqubits++;
		// adjust initial layout
		initialLayout.insert({ physical_qubit_index, logical_qubit_index});
		if (output_qubit_index >= 0) {
			// adjust output permutation
			outputPermutation.insert({physical_qubit_index, output_qubit_index});
		}
		// update all operations
		for (auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}

		// update ancillary and garbage tracking
		for (unsigned short i=nqubits+nancillae-1; i>logical_qubit_index; --i) {
			ancillary[i] = ancillary[i-1];
			garbage[i] = garbage[i-1];
		}
		// unset new entry
		ancillary.reset(logical_qubit_index);
		garbage.reset(logical_qubit_index);
	}

dd::Edge QuantumComputation::reduceAncillae(dd::Edge& e, std::unique_ptr<dd::Package>& dd, bool regular) {
		// return if no more garbage left
		if (!ancillary.any() || e.p == nullptr) return e;
		unsigned short lowerbound = 0;
		for (size_t i = 0; i < ancillary.size(); ++i) {
			if (ancillary.test(i)) {
				lowerbound = i;
				break;
			}
		}
		if (e.p->v < lowerbound) return e;
		return reduceAncillaeRecursion(e, dd, lowerbound, regular);
	}

	dd::Edge QuantumComputation::reduceAncillaeRecursion(dd::Edge& e, std::unique_ptr<dd::Package>& dd, unsigned short lowerbound, bool regular) {
		if(e.p->v < lowerbound) return e;

		dd::Edge f = e;

		std::array<dd::Edge, 4> edges{ };
		std::bitset<4> handled{};
		for (int i = 0; i < 4; ++i) {
			if (!handled.test(i)) {
				if (dd->isTerminal(e.p->e[i])) {
					edges[i] = e.p->e[i];
				} else {
					edges[i] = reduceAncillaeRecursion(f.p->e[i], dd, lowerbound, regular);
					for (int j = i+1; j < 4; ++j) {
						if (e.p->e[i].p == e.p->e[j].p) {
							edges[j] = edges[i];
							handled.set(j);
						}
					}
				}
				handled.set(i);
			}
		}
		f = dd->makeNonterminal(f.p->v, edges);

		// something to reduce for this qubit
		if (f.p->v >= 0 && ancillary.test(f.p->v)) {
			if (regular) {
				if (f.p->e[1].w != CN::ZERO || f.p->e[3].w != CN::ZERO){
					f = dd->makeNonterminal(f.p->v, { f.p->e[0], dd::Package::DDzero, f.p->e[2], dd::Package::DDzero });
				}
			} else {
				if (f.p->e[2].w != CN::ZERO || f.p->e[3].w != CN::ZERO) {
					f = dd->makeNonterminal(f.p->v, { f.p->e[0], f.p->e[1], dd::Package::DDzero, dd::Package::DDzero });
				}
			}
		}

		auto c = dd->cn.mulCached(f.w, e.w);
		f.w = dd->cn.lookup(c);
		dd->cn.releaseCached(c);
		dd->incRef(f);
		return f;
	}

	dd::Edge QuantumComputation::reduceGarbage(dd::Edge& e, std::unique_ptr<dd::Package>& dd, bool regular) {
		// return if no more garbage left
		if (!garbage.any() || e.p == nullptr) return e;
		unsigned short lowerbound = 0;
		for (size_t i=0; i<garbage.size(); ++i) {
			if (garbage.test(i)) {
				lowerbound = i;
				break;
			}
		}
		if(e.p->v < lowerbound) return e;
		return reduceGarbageRecursion(e, dd, lowerbound, regular);
	}

	dd::Edge QuantumComputation::reduceGarbageRecursion(dd::Edge& e, std::unique_ptr<dd::Package>& dd, unsigned short lowerbound, bool regular) {
		if(e.p->v < lowerbound) return e;

		dd::Edge f = e;

		std::array<dd::Edge, 4> edges{ };
		std::bitset<4> handled{};
		for (int i = 0; i < 4; ++i) {
			if (!handled.test(i)) {
				if (dd->isTerminal(e.p->e[i])) {
					edges[i] = e.p->e[i];
				} else {
					edges[i] = reduceGarbageRecursion(f.p->e[i], dd, lowerbound, regular);
					for (int j = i+1; j < 4; ++j) {
						if (e.p->e[i].p == e.p->e[j].p) {
							edges[j] = edges[i];
							handled.set(j);
						}
					}
				}
				handled.set(i);
			}
		}
		f = dd->makeNonterminal(f.p->v, edges);

		// something to reduce for this qubit
		if (f.p->v >= 0 && garbage.test(f.p->v)) {
			if (regular) {
				if (f.p->e[2].w != CN::ZERO || f.p->e[3].w != CN::ZERO) {
					dd::Edge g{ };
					if (f.p->e[0].w == CN::ZERO && f.p->e[2].w != CN::ZERO) {
						g = f.p->e[2];
					} else if (f.p->e[2].w != CN::ZERO) {
						g = dd->add(f.p->e[0], f.p->e[2]);
					} else {
						g = f.p->e[0];
					}
					dd::Edge h{ };
					if (f.p->e[1].w == CN::ZERO && f.p->e[3].w != CN::ZERO) {
						h = f.p->e[3];
					} else if (f.p->e[3].w != CN::ZERO) {
						h = dd->add(f.p->e[1], f.p->e[3]);
					} else {
						h = f.p->e[1];
					}
					f = dd->makeNonterminal(e.p->v, { g, h, dd::Package::DDzero, dd::Package::DDzero });
				}
			} else {
				if (f.p->e[1].w != CN::ZERO || f.p->e[3].w != CN::ZERO) {
					dd::Edge g{ };
					if (f.p->e[0].w == CN::ZERO && f.p->e[1].w != CN::ZERO) {
						g = f.p->e[1];
					} else if (f.p->e[1].w != CN::ZERO) {
						g = dd->add(f.p->e[0], f.p->e[1]);
					} else {
						g = f.p->e[0];
					}
					dd::Edge h{ };
					if (f.p->e[2].w == CN::ZERO && f.p->e[3].w != CN::ZERO) {
						h = f.p->e[3];
					} else if (f.p->e[3].w != CN::ZERO) {
						h = dd->add(f.p->e[2], f.p->e[3]);
					} else {
						h = f.p->e[2];
					}
					f = dd->makeNonterminal(e.p->v, { g, dd::Package::DDzero, h, dd::Package::DDzero });
				}
			}
		}

		auto c = dd->cn.mulCached(f.w, e.w);
		f.w = dd->cn.lookup(c);
		dd->cn.releaseCached(c);
		// Quick-fix for normalization bug
		if (CN::mag2(f.w) > 1.0)
			f.w = CN::ONE;
		dd->incRef(f);
		return f;
	}


	dd::Edge QuantumComputation::createInitialMatrix(std::unique_ptr<dd::Package>& dd) {
		dd::Edge e = dd->makeIdent(0, short(nqubits+nancillae-1));
		dd->incRef(e);
		e = reduceAncillae(e, dd);
		return e;
	}


	dd::Edge QuantumComputation::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
		if (nqubits + nancillae == 0)
			return dd->DDone;
		
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);
		permutationMap map = initialLayout;
		dd->setMode(dd::Matrix);
		dd::Edge e = createInitialMatrix(dd);

		for (auto & op : ops) {
			auto tmp = dd->multiply(op->getDD(dd, line, map), e);

			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}
		// correct permutation if necessary
		changePermutation(e, map, outputPermutation, line, dd);
		e = reduceAncillae(e, dd);

		return e;
	}

	dd::Edge QuantumComputation::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		// measurements are currently not supported here
		std::array<short, MAX_QUBITS> line{};
		line.fill(LINE_DEFAULT);
		permutationMap map = initialLayout;
		dd->setMode(dd::Vector);
		dd::Edge e = in;
		dd->incRef(e);

		for (auto& op : ops) {
			auto tmp = dd->multiply(op->getDD(dd, line, map), e);

			dd->incRef(tmp);
			dd->decRef(e);
			e = tmp;

			dd->garbageCollect();
		}

		// correct permutation if necessary
		changePermutation(e, map, outputPermutation, line, dd);
		e = reduceAncillae(e, dd);

		return e;
	}

	void QuantumComputation::create_reg_array(const registerMap& regs, regnames_t& regnames, unsigned short defaultnumber, const char* defaultname) {
		regnames.clear();

		std::stringstream ss;
		if(!regs.empty()) {
			// sort regs by start index
			std::map<unsigned short, std::pair<std::string, reg>> sortedRegs{};
			for (const auto& reg: regs) {
				sortedRegs.insert({reg.second.first, reg});
			}

			for(const auto& reg: sortedRegs) {
				for(unsigned short i = 0; i < reg.second.second.second; i++) {
					ss << reg.second.first << "[" << i << "]";
					regnames.push_back(std::make_pair(reg.second.first, ss.str()));
					ss.str(std::string());
				}
			}
		} else {
			for(unsigned short i = 0; i < defaultnumber; i++) {
				ss << defaultname << "[" << i << "]";
				regnames.push_back(std::make_pair(defaultname, ss.str()));
				ss.str(std::string());
			}
		}
	}

	std::ostream& QuantumComputation::print(std::ostream& os) const {
		os << std::setw((int)std::log10(ops.size())+1) << "i" << ": \t\t\t";
		for (const auto& Q: initialLayout) {
			if (ancillary.test(Q.second))
				os << "\033[31m" << Q.second << "\t\033[0m";
			else
				os << Q.second << "\t";
		}
		/*for (unsigned short i = 0; i < nqubits + nancillae; ++i) {
			auto it = initialLayout.find(i);
			if(it == initialLayout.end()) {
				os << "|\t";
			} else {
				os << it->second << "\t";
			}
		}*/
		os << std::endl;
		size_t i = 0;
		for (const auto& op:ops) {
			os << std::setw((int)std::log10(ops.size())+1) << ++i << ": \t";
			op->print(os, initialLayout);
			os << std::endl;
		}
		os << std::setw((int)std::log10(ops.size())+1) << "o" << ": \t\t\t";
		for(const auto& physical_qubit: initialLayout) {
			auto it = outputPermutation.find(physical_qubit.first);
			if(it == outputPermutation.end()) {
				if (garbage.test(physical_qubit.second))
					os << "\033[31m|\t\033[0m";
				else
					os << "|\t";
			} else {
				os << it->second << "\t";
			}
		}
		os << std::endl;
		return os;
	}

	dd::Complex QuantumComputation::getEntry(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long i, unsigned long long j) {
		if (dd->isTerminal(e))
			return e.w;

		dd::Complex c = dd->cn.getTempCachedComplex(1,0);
		do {
			unsigned short row = (i >> outputPermutation.at(e.p->v)) & 1u;
			unsigned short col = (j >> initialLayout.at(e.p->v)) & 1u;
			e = e.p->e[dd::RADIX * row + col];
			CN::mul(c, c, e.w);
		} while (!dd::Package::isTerminal(e));
		return c;
	}

	std::ostream& QuantumComputation::printMatrix(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		os << "Common Factor: " << e.w << "\n";
		for (unsigned long long i = 0; i < (1ull << (unsigned int)(nqubits+nancillae)); ++i) {
			for (unsigned long long j = 0; j < (1ull << (unsigned int)(nqubits+nancillae)); ++j) {
				os << std::right << std::setw(7) << std::setfill(' ') << getEntry(dd, e, i, j) << "\t";
			}
			os << std::endl;
		}
		return os;
	}

	void QuantumComputation::printBin(unsigned long long n, std::stringstream& ss) {
		if (n > 1)
			printBin(n/2, ss);
		ss << n%2;
	}

	std::ostream& QuantumComputation::printCol(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long j, std::ostream& os) {
		os << "Common Factor: " << e.w << "\n";
		for (unsigned long long i = 0; i < (1ull << (unsigned int)(nqubits+nancillae)); ++i) {
			std::stringstream ss{};
			printBin(i, ss);
			os << std::setw(nqubits + nancillae) << ss.str() << ": " << getEntry(dd, e, i, j) << "\n";
		}
		return os;
	}

	std::ostream& QuantumComputation::printVector(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os) {
		return printCol(dd, e, 0, os);
	}

	std::ostream& QuantumComputation::printStatistics(std::ostream& os) {
		os << "QC Statistics:\n";
		os << "\tn: " << nqubits << std::endl;
		os << "\tanc: " << nancillae << std::endl;
		os << "\tm: " << ops.size() << std::endl;
		os << "--------------" << std::endl;
		return os;
	}

	void QuantumComputation::dump(const std::string& filename) {
		size_t dot = filename.find_last_of('.');
		std::string extension = filename.substr(dot + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
		if (extension == "real") {
			dump(filename, Real);
		} else if (extension == "qasm") {
			dump(filename, OpenQASM);
		} else if(extension == "py") {
			dump(filename, Qiskit);
		} else if (extension == "qc") {
			dump(filename, QC);
		} else if (extension == "tfc") {
			dump(filename, TFC);
		} else {
			throw QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
		}
	}
	
	void QuantumComputation::consolidateRegister(registerMap& regs) {
		bool finished = false;
		while (!finished) {
			for (const auto& qreg: regs) {
				finished = true;
				auto regname = qreg.first;
				// check if lower part of register
				if (regname.length() > 2 && regname.compare(regname.size() - 2, 2, "_l") == 0) {
					auto lowidx = qreg.second.first;
					auto lownum = qreg.second.second;
					// search for higher part of register
					auto highname = regname.substr(0, regname.size() - 1) + 'h';
					auto it = regs.find(highname);
					if (it != regs.end()) {
						auto highidx = it->second.first;
						auto highnum = it->second.second;
						// fusion of registers possible
						if (lowidx+lownum == highidx){
							finished = false;
							auto targetname = regname.substr(0, regname.size() - 2);
							auto targetidx = lowidx;
							auto targetnum = lownum + highnum;
							regs.insert({ targetname, { targetidx, targetnum }});
							regs.erase(regname);
							regs.erase(highname);
						}
					}
					break;
				}
			}
		}
	}
	
	void QuantumComputation::dumpOpenQASM(std::ostream& of) {
		// Add missing physical qubits
		if(!qregs.empty()) {
			for (unsigned short physical_qubit=0; physical_qubit < initialLayout.rbegin()->first; ++physical_qubit) {
				if (! initialLayout.count(physical_qubit)) {
					auto logicalQubit = getHighestLogicalQubitIndex()+1;
					addQubit(logicalQubit, physical_qubit, -1);
				}
			}
		}

		// dump initial layout and output permutation
		permutationMap inverseInitialLayout {};
		for (const auto& q: initialLayout)
			inverseInitialLayout.insert({q.second, q.first});
		of << "// i";
		for (const auto& q: inverseInitialLayout) {
			of << " " << q.second;
		}
		of << std::endl;

		permutationMap inverseOutputPermutation {};
		for (const auto& q: outputPermutation) {
			inverseOutputPermutation.insert({q.second, q.first});
		}
		of << "// o";
		for (const auto& q: inverseOutputPermutation) {
			of << " " << q.second;
		}
		of << std::endl;

		of << "OPENQASM 2.0;"                << std::endl;
		of << "include \"qelib1.inc\";"      << std::endl;
		if (!qregs.empty()){
			printSortedRegisters(qregs, "qreg", of);
		} else if (nqubits > 0) {
			of << "qreg " << DEFAULT_QREG << "[" << nqubits   << "];" << std::endl;
		}
		if(!cregs.empty()) {
			printSortedRegisters(cregs, "creg", of);
		} else if (nclassics > 0) {
			of << "creg " << DEFAULT_CREG << "[" << nclassics << "];" << std::endl;
		}
		if(!ancregs.empty()) {
			printSortedRegisters(ancregs, "qreg", of);
		} else if (nancillae > 0) {
			of << "qreg " << DEFAULT_ANCREG << "[" << nancillae << "];" << std::endl;
		}

		regnames_t qregnames{};
		regnames_t cregnames{};
		regnames_t ancregnames{};
		create_reg_array(qregs, qregnames, nqubits, DEFAULT_QREG);
		create_reg_array(cregs, cregnames, nclassics, DEFAULT_CREG);
		create_reg_array(ancregs, ancregnames, nancillae, DEFAULT_ANCREG);

		for (const auto& ancregname: ancregnames)
			qregnames.push_back(ancregname);

		for (const auto& op: ops) {
			op->dumpOpenQASM(of, qregnames, cregnames);
		}
	}

	void QuantumComputation::printSortedRegisters(const registerMap& regmap, const std::string& identifier, std::ostream& of) {
		// sort regs by start index
		std::map<unsigned short, std::pair<std::string, reg>> sortedRegs{};
		for (const auto& reg: regmap) {
			sortedRegs.insert({reg.second.first, reg});
		}

		for (const auto& reg : sortedRegs) {
			of << identifier << " " << reg.second.first << "[" << reg.second.second.second << "];" << std::endl;
		}
	}

	void QuantumComputation::dump(const std::string& filename, Format format) {
		auto of = std::ofstream(filename);
		if (!of.good()) {
			throw QFRException("[dump] Error opening file: " + filename);
		}
		dump(of, format);
	}

	void QuantumComputation::dump(std::ostream&& of, Format format) {

		switch(format) {
			case  OpenQASM:
				dumpOpenQASM(of);
				break;
			case Real:
				std::cerr << "Dumping in real format currently not supported\n";
				break;
			case GRCS:
				std::cerr << "Dumping in GRCS format currently not supported\n";
				break;
			case TFC:
				std::cerr << "Dumping in TFC format currently not supported\n";
				break;
			case QC:
				std::cerr << "Dumping in QC format currently not supported\n";
				break;
			case Qiskit:
				// TODO: improve/modernize Qiskit dump
				unsigned short totalQubits = nqubits + nancillae + (max_controls >= 2? max_controls-2: 0);
				if (totalQubits > 53) {
					std::cerr << "No more than 53 total qubits are currently supported" << std::endl;
					break;
				}

				// For the moment all registers are fused together into for simplicity
				// This may be adapted in the future
				of << "from qiskit import *" << std::endl;
				of << "from qiskit.test.mock import ";
				unsigned short narchitecture = 0;
				if (totalQubits <= 5) {
					of << "FakeBurlington";
					narchitecture = 5;
				} else if (totalQubits <= 20) {
					of << "FakeBoeblingen";
					narchitecture = 20;
				} else {
					of << "FakeRochester";
					narchitecture = 53;
				}
				of << std::endl;
				of << "from qiskit.converters import circuit_to_dag, dag_to_circuit" << std::endl;
				of << "from qiskit.transpiler.passes import *" << std::endl;
				of << "from math import pi" << std::endl << std::endl;

				of << DEFAULT_QREG << " = QuantumRegister(" << nqubits << ", '" << DEFAULT_QREG << "')" << std::endl;
				if (nclassics > 0) {
					of << DEFAULT_CREG << " = ClassicalRegister(" << nclassics << ", '" << DEFAULT_CREG << "')" << std::endl;
				}
				if (nancillae > 0) {
					of << DEFAULT_ANCREG << " = QuantumRegister(" << nancillae << ", '" << DEFAULT_ANCREG << "')" << std::endl;
				}
				if (max_controls > 2) {
					of << DEFAULT_MCTREG << " = QuantumRegister(" << max_controls - 2 << ", '"<< DEFAULT_MCTREG << "')" << std::endl;
				}
				of << "qc = QuantumCircuit(";
				of << DEFAULT_QREG;
				if (nclassics > 0) {
					of << ", " << DEFAULT_CREG;
				}
				if (nancillae > 0) {
					of << ", " << DEFAULT_ANCREG;
				}
				if(max_controls > 2) {
					of << ", " << DEFAULT_MCTREG;
				}
				of << ")" << std::endl << std::endl;

				regnames_t qregnames{};
				regnames_t cregnames{};
				regnames_t ancregnames{};
				create_reg_array({}, qregnames, nqubits, DEFAULT_QREG);
				create_reg_array({}, cregnames, nclassics, DEFAULT_CREG);
				create_reg_array({}, ancregnames, nancillae, DEFAULT_ANCREG);

				for (const auto& ancregname: ancregnames)
					qregnames.push_back(ancregname);

				for (const auto& op: ops) {
					op->dumpQiskit(of, qregnames, cregnames, DEFAULT_MCTREG);
				}
				// add measurement for determining output mapping
				of << "qc.measure_all()" << std::endl;

				of << "qc_transpiled = transpile(qc, backend=";
				if (totalQubits <= 5) {
					of << "FakeBurlington";
				} else if (totalQubits <= 20) {
					of << "FakeBoeblingen";
				} else {
					of << "FakeRochester";
				}
				of << "(), optimization_level=1)" << std::endl << std::endl;
				of << "layout = qc_transpiled._layout" << std::endl;
				of << "virtual_bits = layout.get_virtual_bits()" << std::endl;

				of << "f = open(\"circuit" << R"(_transpiled.qasm", "w"))" << std::endl;
				of << R"(f.write("// i"))" << std::endl;
				of << "for qubit in " << DEFAULT_QREG << ":" << std::endl;
				of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				if (nancillae > 0) {
					of << "for qubit in " << DEFAULT_ANCREG << ":" << std::endl;
					of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				if (max_controls > 2) {
					of << "for qubit in " << DEFAULT_MCTREG << ":" << std::endl;
					of << '\t' << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				if (totalQubits < narchitecture) {
					of << "for reg in layout.get_registers():" << std::endl;
					of << '\t' << "if reg.name is 'ancilla':" << std::endl;
					of << "\t\t" << "for qubit in reg:" << std::endl;
					of << "\t\t\t" << R"(f.write(" " + str(virtual_bits[qubit])))" << std::endl;
				}
				of << R"(f.write("\n"))" << std::endl;
				of << "dag = circuit_to_dag(qc_transpiled)" << std::endl;
				of << "out = [item for sublist in list(dag.layers())[-1]['partition'] for item in sublist]" << std::endl;
				of << R"(f.write("// o"))" << std::endl;
				of << "for qubit in out:" << std::endl;
				of << '\t' << R"(f.write(" " + str(qubit.index)))" << std::endl;
				of << R"(f.write("\n"))" << std::endl;
				// remove measurements again
				of << "qc_transpiled = dag_to_circuit(RemoveFinalMeasurements().run(dag))" << std::endl;
				of << "f.write(qc_transpiled.qasm())" << std::endl;
				of << "f.close()" << std::endl;
				break;
		}
	}

	bool QuantumComputation::isIdleQubit(unsigned short physical_qubit) {
		for(const auto& op:ops) {
			if (op->actsOn(physical_qubit))
				return false;
		}
		return true;
	}

	void QuantumComputation::stripIdleQubits(bool force, bool reduceIOpermutations) {
		auto layout_copy = initialLayout;
		for (auto physical_qubit_it = layout_copy.rbegin(); physical_qubit_it != layout_copy.rend(); ++physical_qubit_it) {
			auto physical_qubit_index = physical_qubit_it->first;
			if(isIdleQubit(physical_qubit_index)) {
				auto it = outputPermutation.find(physical_qubit_index);
				if(it != outputPermutation.end()) {
					short output_index = it->second;
					if (!force && output_index >= 0) continue;
				}

				unsigned short logical_qubit_index = initialLayout.at(physical_qubit_index);
				#if DEBUG_MODE_QC
				std::cout << "Trying to strip away idle qubit: " << physical_qubit_index
						  << ", which corresponds to logical qubit: " << logical_qubit_index << std::endl;
				print(std::cout);
				#endif
				removeQubit(logical_qubit_index);

				if (reduceIOpermutations && (logical_qubit_index < nqubits+nancillae)) {
					#if DEBUG_MODE_QC
					std::cout << "Qubit " << logical_qubit_index << " is inner qubit. Need to adjust permutations." << std::endl;
					#endif

					for (auto& q: initialLayout) {
						if (q.second > logical_qubit_index)
							q.second--;
					}

					for (auto& q: outputPermutation) {
						if (q.second > logical_qubit_index)
							q.second--;
					}

					#if DEBUG_MODE_QC
					std::cout << "Changed initial layout" << std::endl;
					printPermutationMap(initialLayout);
					std::cout << "Changed output permutation" << std::endl;
					printPermutationMap(outputPermutation);
					#endif
				}

				#if DEBUG_MODE_QC
				std::cout << "Resulting in: " << std::endl;
				print(std::cout);
				#endif
			}
		}
		for(auto& op:ops) {
			op->setNqubits(nqubits + nancillae);
		}
	}

	void QuantumComputation::changePermutation(dd::Edge& on, qc::permutationMap& from, const qc::permutationMap& to, std::array<short, qc::MAX_QUBITS>& line, std::unique_ptr<dd::Package>& dd, bool regular) {
		assert(from.size() >= to.size());

		#if DEBUG_MODE_QC
		std::cout << "Trying to change: " << std::endl;
		printPermutationMap(from);
		std::cout << "to: " << std::endl;
		printPermutationMap(to);
		#endif

		auto n = (short)(on.p->v + 1);

		// iterate over (k,v) pairs of second permutation
		for (const auto& kv: to) {
			unsigned short i = kv.first;
			unsigned short goal = kv.second;

			// search for key in the first map
			auto it = from.find(i);
			if (it == from.end()) {
				throw QFRException("[changePermutation] Key " + std::to_string(it->first) + " was not found in first permutation. This should never happen.");
			}
			unsigned short current = it->second;

			// permutations agree for this key value
			if(current == goal) continue;

			// search for goal value in first permutation
			unsigned short j = 0;
			for(const auto& pair: from) {
				unsigned short value = pair.second;
				if (value == goal) {
					j = pair.first;
					break;
				}
			}

			// swap i and j
			auto op = qc::StandardOperation(n, {i, j}, qc::SWAP);

			#if DEBUG_MODE_QC
			std::cout << "Apply SWAP: " << i << " " << j << std::endl;
			#endif

			op.setLine(line, from);
			auto saved = on;
			if (regular) {
				on = dd->multiply(op.getSWAPDD(dd, line, from), on);
			} else {
				on = dd->multiply(on, op.getSWAPDD(dd, line, from));
			}
			op.resetLine(line, from);
			dd->incRef(on);
			dd->decRef(saved);
			dd->garbageCollect();

			// update permutation
			from.at(i) = goal;
			from.at(j) = current;

			#if DEBUG_MODE_QC
			std::cout << "Changed permutation" << std::endl;
			printPermutationMap(from);
			#endif
		}

	}


	std::string QuantumComputation::getQubitRegister(unsigned short physical_qubit_index) {

		for (const auto& reg:qregs) {
			unsigned short start_idx = reg.second.first;
			unsigned short count = reg.second.second;
			if (physical_qubit_index < start_idx) continue;
			if (physical_qubit_index >= start_idx + count) continue;
			return reg.first;
		}
		for (const auto& reg:ancregs) {
			unsigned short start_idx = reg.second.first;
			unsigned short count = reg.second.second;
			if (physical_qubit_index < start_idx) continue;
			if (physical_qubit_index >= start_idx + count) continue;
			return reg.first;
		}

		throw QFRException("[getQubitRegister] Qubit index " + std::to_string(physical_qubit_index) + " not found in any register");
	}

	std::pair<std::string, unsigned short> QuantumComputation::getQubitRegisterAndIndex(unsigned short physical_qubit_index) {
		std::string reg_name = getQubitRegister(physical_qubit_index);
		unsigned short index = 0;
		auto it = qregs.find(reg_name);
		if (it != qregs.end()) {
			index = physical_qubit_index - it->second.first;
		} else {
			auto it_anc = ancregs.find(reg_name);
			if (it_anc != ancregs.end()) {
				index = physical_qubit_index - it_anc->second.first;
			}
			// no else branch needed here, since error would have already shown in getQubitRegister(physical_qubit_index)
		}
		return {reg_name, index};
	}

	std::string QuantumComputation::getClassicalRegister(unsigned short classical_index) {

		for (const auto& reg:cregs) {
			unsigned short start_idx = reg.second.first;
			unsigned short count = reg.second.second;
			if (classical_index < start_idx) continue;
			if (classical_index >= start_idx + count) continue;
			return reg.first;
		}

		throw QFRException("[getClassicalRegister] Classical index " + std::to_string(classical_index) + " not found in any register");
	}

	std::pair<std::string, unsigned short> QuantumComputation::getClassicalRegisterAndIndex(unsigned short classical_index) {
		std::string reg_name = getClassicalRegister(classical_index);
		unsigned short index = 0;
		auto it = cregs.find(reg_name);
		if (it != cregs.end()) {
			index = classical_index - it->second.first;
		} // else branch not needed since getClassicalRegister already covers this case
		return {reg_name, index};
	}

	unsigned short QuantumComputation::getIndexFromQubitRegister(const std::pair<std::string, unsigned short>& qubit) {
		// no range check is performed here!
		return static_cast<unsigned short>(qregs.at(qubit.first).first + qubit.second);
	}
	unsigned short QuantumComputation::getIndexFromClassicalRegister(const std::pair<std::string, unsigned short>& clbit) {
		// no range check is performed here!
		return static_cast<unsigned short>(cregs.at(clbit.first).first + clbit.second);
	}

	std::ostream& QuantumComputation::printPermutationMap(const permutationMap &map, std::ostream &os) {
		for(const auto& Q: map) {
			os <<"\t" << Q.first << ": " << Q.second << std::endl;
		}
		return os;
	}

	std::ostream& QuantumComputation::printRegisters(std::ostream& os) {
		os << "qregs:";
		for(const auto& qreg: qregs) {
			os << " {" << qreg.first << ", {" << qreg.second.first << ", " << qreg.second.second << "}}";
		}
		os << std::endl;
		if (!ancregs.empty()) {
			os << "ancregs:";
			for(const auto& ancreg: ancregs) {
				os << " {" << ancreg.first <<", {" << ancreg.second.first << ", " << ancreg.second.second << "}}";
			}
			os << std::endl;
		}
		os << "cregs:";
		for(const auto& creg: cregs) {
			os << " {" << creg.first <<", {" << creg.second.first << ", " << creg.second.second << "}}";
		}
		os << std::endl;
		return os;
	}

	unsigned short QuantumComputation::getHighestLogicalQubitIndex(const permutationMap& map) {
		unsigned short max_index = 0;
		for (const auto& physical_qubit: map) {
			if (physical_qubit.second > max_index) {
				max_index = physical_qubit.second;
			}
		}
		return max_index;
	}

	bool QuantumComputation::physicalQubitIsAncillary(unsigned short physical_qubit_index) {
		return std::any_of(ancregs.begin(), ancregs.end(), [&physical_qubit_index](registerMap::value_type& ancreg) { return ancreg.second.first <= physical_qubit_index && physical_qubit_index < ancreg.second.first + ancreg.second.second; });
	}

	bool QuantumComputation::isLastOperationOnQubit(decltype(ops.begin())& opIt, decltype(ops.end())& end) {
		if (opIt == end)
			return true;

		// determine which qubits the gate acts on
		std::bitset<MAX_QUBITS> actson{};
		for (unsigned short i = 0; i < MAX_QUBITS; ++i) {
			if ((*opIt)->actsOn(i))
				actson.set(i);
		}

		// iterate over remaining gates and check if any act on qubits overlapping with the target gate
		auto atEnd = opIt;
		std::advance(atEnd, 1);
		while (atEnd != end) {
			for (unsigned short i = 0; i < MAX_QUBITS; ++i) {
				if (actson[i] && (*atEnd)->actsOn(i)) return false;
			}
			++atEnd;
		}
		return true;
	}
}
