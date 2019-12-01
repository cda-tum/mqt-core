//
// Created by Lukas Burgholzer on 22.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_SCANNER_H
#define INTERMEDIATEREPRESENTATION_SCANNER_H

#include <stack>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

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
	};
}

#endif //INTERMEDIATEREPRESENTATION_SCANNER_H
