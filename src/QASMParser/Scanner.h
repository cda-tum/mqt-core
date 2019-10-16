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

#include "Token.h"

namespace qasm {

	class Scanner {
		std::istream& is;
		std::stack<std::istream *> streams{ };
		std::map<std::string, Token::Kind> keywords{ };
		char ch = 0;
		int line = 1;
		int col = 0;

		void nextCh() {
			if (!streams.empty() && streams.top()->eof()) {
				delete streams.top();
				streams.pop();
				ch = lines.top().ch;
				line = lines.top().line;
				col = lines.top().col;
				lines.pop();
				return;
			}
			if (!streams.empty()) {
				col++;
				streams.top()->get(ch);
			} else {
				if (!is.eof()) {
					col++;
					is.get(ch);
				} else {
					ch = (char) -1;
				}
			}
			if (ch == '\n') {
				col = 0;
				line++;
			}
		}

		void readName(Token& t) {
			std::stringstream ss;
			while (isalnum(ch) || ch == '_') {
				ss << ch;
				nextCh();
			}
			t.str = ss.str();
			auto it = keywords.find(t.str);
			t.kind = (it != keywords.end()) ? it->second : Token::Kind::identifier;
		}

		void readNumber(Token& t) {
			std::stringstream ss;
			while (isdigit(ch)) {
				ss << ch;
				nextCh();
			}
			t.kind = Token::Kind::nninteger;
			t.str = ss.str();
			if (ch != '.') {
				ss >> t.val;
				return;
			}
			t.kind = Token::Kind::real;
			ss << ch;
			nextCh();
			while (isdigit(ch)) {
				ss << ch;
				nextCh();
			}
			if (ch != 'e' || ch != 'E') {
				ss >> t.valReal;
				return;
			}
			ss << ch;
			nextCh();
			if (ch == '-' || ch == '+') {
				ss << ch;
				nextCh();
			}
			while (isdigit(ch)) {
				ss << ch;
				nextCh();
			}
			ss >> t.valReal;
		}

		void readString(Token& t) {
			std::stringstream ss;
			while (ch != '"') {
				ss << ch;
				nextCh();
			}
			t.str = ss.str();
			t.kind = Token::Kind::string;
		}

		void skipComment() {
			while (ch != '\n' && ch != (char) -1) {
				nextCh();
			}
		}

		struct LineInfo {
			char ch;
			int line, col;

			LineInfo(char ch, int line, int col) : ch(ch), line(line), col(col) { }
		};

		std::stack<LineInfo> lines{ };

	public:
		explicit Scanner(std::istream& is) : is(is) {
			keywords["qreg"] = Token::Kind::qreg;
			keywords["creg"] = Token::Kind::creg;
			keywords["gate"] = Token::Kind::gate;
			keywords["measure"] = Token::Kind::measure;
			keywords["U"] = Token::Kind::ugate;
			keywords["CX"] = Token::Kind::cxgate;
			keywords["pi"] = Token::Kind::pi;
			keywords["OPENQASM"] = Token::Kind::openqasm;
			keywords["show_probabilities"] = Token::Kind::probabilities;
			keywords["sin"] = Token::Kind::sin;
			keywords["cos"] = Token::Kind::cos;
			keywords["tan"] = Token::Kind::tan;
			keywords["exp"] = Token::Kind::exp;
			keywords["ln"] = Token::Kind::ln;
			keywords["sqrt"] = Token::Kind::sqrt;
			keywords["include"] = Token::Kind::include;
			keywords["barrier"] = Token::Kind::barrier;
			keywords["opaque"] = Token::Kind::opaque;
			keywords["if"] = Token::Kind::_if;
			keywords["reset"] = Token::Kind::reset;
			keywords["snapshot"] = Token::Kind::snapshot;
			nextCh();
		}

		Token next() {
			while (iswspace(ch)) {
				nextCh();
			}

			Token t = Token(Token::Kind::none, line, col);

			switch (ch) {
				case 'a':
				case 'b':
				case 'c':
				case 'd':
				case 'e':
				case 'f':
				case 'g':
				case 'h':
				case 'i':
				case 'j':
				case 'k':
				case 'l':
				case 'm':
				case 'n':
				case 'o':
				case 'p':
				case 'q':
				case 'r':
				case 's':
				case 't':
				case 'u':
				case 'v':
				case 'w':
				case 'x':
				case 'y':
				case 'z':
				case 'A':
				case 'B':
				case 'C':
				case 'D':
				case 'E':
				case 'F':
				case 'G':
				case 'H':
				case 'I':
				case 'J':
				case 'K':
				case 'L':
				case 'M':
				case 'N':
				case 'O':
				case 'P':
				case 'Q':
				case 'R':
				case 'S':
				case 'T':
				case 'U':
				case 'V':
				case 'W':
				case 'X':
				case 'Y':
				case 'Z': readName(t);
					break;
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
				case '.': readNumber(t);
					break;
				case ';': t.kind = Token::Kind::semicolon;
					nextCh();
					break;
				case (char) -1: t.kind = Token::Kind::eof;
					break;
				case '(': t.kind = Token::Kind::lpar;
					nextCh();
					break;
				case ')': t.kind = Token::Kind::rpar;
					nextCh();
					break;
				case '[': t.kind = Token::Kind::lbrack;
					nextCh();
					break;
				case ']': t.kind = Token::Kind::rbrack;
					nextCh();
					break;
				case '{': t.kind = Token::Kind::lbrace;
					nextCh();
					break;
				case '}': t.kind = Token::Kind::rbrace;
					nextCh();
					break;
				case ',': t.kind = Token::Kind::comma;
					nextCh();
					break;
				case '+': nextCh();
					t.kind = Token::Kind::plus;
					break;
				case '-': nextCh();
					t.kind = Token::Kind::minus;
					break;
				case '*': nextCh();
					t.kind = Token::Kind::times;
					break;
				case '/': nextCh();
					if (ch == '/') {
						skipComment();
						t = next();
					} else {
						t.kind = Token::Kind::div;
					}
					break;
				case '^': nextCh();
					t.kind = Token::Kind::power;
					break;
				case '"': nextCh();
					readString(t);
					nextCh();
					break;
				case '>': nextCh();
					t.kind = Token::Kind::gt;
					break;
				case '=': nextCh();
					if (ch == '=') {
						nextCh();
						t.kind = Token::Kind::eq;
					} else {
						std::cerr << "ERROR: UNEXPECTED CHARACTER: '" << ch << "'! " << std::endl;
					}
					break;
				default: std::cerr << "ERROR: UNEXPECTED CHARACTER: '" << ch << "'! " << std::endl;
					nextCh();
			}

			return t;
		}

		void addFileInput(const std::string& filename) {
			auto in = new std::ifstream(filename, std::ifstream::in);
			if (in->fail()) {
				std::cerr << "Failes to open file '" << filename << "'!" << std::endl;
			} else {
				streams.push(in);
				lines.push(LineInfo(ch, line, col));
				line = 0;
				col = 0;
			}
			nextCh();
		}
	};
}

#endif //INTERMEDIATEREPRESENTATION_SCANNER_H
