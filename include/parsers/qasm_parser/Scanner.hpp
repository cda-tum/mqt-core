/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_SCANNER_H
#define INTERMEDIATEREPRESENTATION_SCANNER_H

#include <stack>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Token.hpp"

namespace qasm {

	class Scanner {
		struct LineInfo {
			char ch;
			int line, col;

			LineInfo(char ch, int line, int col) : ch(ch), line(line), col(col) { }
		};


		std::istream&                      is;
		std::stack<std::istream *>         streams{ };
		std::map<std::string, Token::Kind> keywords{ };
		char ch  = 0;
		int line = 1;
		int col  = 0;

		void nextCh();

		void readName(Token& t);

		void readNumber(Token& t);

		void readString(Token& t);

		void skipComment();

		std::stack<LineInfo> lines{ };

	public:
		explicit Scanner(std::istream& is);

		Token next();

		void addFileInput(const std::string& filename);

		int getLine() const {
			return line;
		}
		int getCol() const {
			return col;
		}
	};
}

#endif //INTERMEDIATEREPRESENTATION_SCANNER_H
