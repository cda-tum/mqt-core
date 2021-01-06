/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "parsers/qasm_parser/Scanner.hpp"

#include <locale>

namespace qasm {

    /***
     * Private Methods
     ***/
    void Scanner::nextCh() {
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

    void Scanner::readName(Token& t) {
        std::stringstream ss;
        while (::isalnum(ch) || ch == '_') {
            ss << ch;
            nextCh();
        }
        t.str = ss.str();
        auto it = keywords.find(t.str);
        t.kind = (it != keywords.end()) ? it->second : Token::Kind::identifier;
    }

    void Scanner::readNumber(Token& t) {
        std::stringstream ss;
        while (::isdigit(ch)) {
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
        while (::isdigit(ch)) {
            ss << ch;
            nextCh();
        }
        if (ch != 'e' && ch != 'E') {
            ss >> t.valReal;
            return;
        }
        ss << ch;
        nextCh();
        if (ch == '-' || ch == '+') {
            ss << ch;
            nextCh();
        }
        while (::isdigit(ch)) {
            ss << ch;
            nextCh();
        }
        ss >> t.valReal;
    }

    void Scanner::readString(Token& t) {
        std::stringstream ss;
        while (ch != '"') {
            ss << ch;
            nextCh();
        }
        t.str = ss.str();
        t.kind = Token::Kind::string;
    }

    void Scanner::skipComment() {
        while (ch != '\n' && ch != (char) -1) {
            nextCh();
        }
    }


    /***
     * Public Methods
     ***/
    Scanner::Scanner(std::istream& is) : is(is) {
        keywords["qreg"]               = Token::Kind::qreg;
        keywords["creg"]               = Token::Kind::creg;
        keywords["gate"]               = Token::Kind::gate;
        keywords["measure"]            = Token::Kind::measure;
        keywords["U"]                  = Token::Kind::ugate;
        keywords["CX"]                 = Token::Kind::cxgate;
	    keywords["swap"]               = Token::Kind::swap;
	    keywords["mct"]                = Token::Kind::mcx_gray;
	    keywords["mcx"]                = Token::Kind::mcx_gray;
	    keywords["mcx_gray"]           = Token::Kind::mcx_gray;
	    keywords["mcx_recursive"]      = Token::Kind::mcx_recursive;
	    keywords["mcx_vchain"]         = Token::Kind::mcx_vchain;
	    keywords["pi"]                 = Token::Kind::pi;
        keywords["OPENQASM"]           = Token::Kind::openqasm;
        keywords["show_probabilities"] = Token::Kind::probabilities;
        keywords["sin"]                = Token::Kind::sin;
        keywords["cos"]                = Token::Kind::cos;
        keywords["tan"]                = Token::Kind::tan;
        keywords["exp"]                = Token::Kind::exp;
        keywords["ln"]                 = Token::Kind::ln;
        keywords["sqrt"]               = Token::Kind::sqrt;
        keywords["include"]            = Token::Kind::include;
        keywords["barrier"]            = Token::Kind::barrier;
        keywords["opaque"]             = Token::Kind::opaque;
        keywords["if"]                 = Token::Kind::_if;
        keywords["reset"]              = Token::Kind::reset;
        keywords["snapshot"]           = Token::Kind::snapshot;
        nextCh();
    }

    Token Scanner::next() {
        while (::isspace(ch)) {
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

    void Scanner::addFileInput(const std::string& filename) {
        auto in = new std::ifstream(filename, std::ifstream::in);

	    if (in->fail() && filename == "qelib1.inc") {
			// internal qelib1.inc
			// parser can also read multiple-control versions of each gate
	    	auto ss = new std::stringstream{};
		    *ss << "gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }" << std::endl;
		    *ss << "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }" << std::endl;
		    *ss << "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }" << std::endl;
		    *ss << "gate u1(lambda) q { U(0,0,lambda) q; }" << std::endl;
		    *ss << "gate p(lambda) q { U(0,0,lambda) q; }" << std::endl;
		    *ss << "gate cx c, t { CX c, t; }" << std::endl;
		    *ss << "gate id t { U(0,0,0) t; }" << std::endl;
		    *ss << "gate x t { u3(pi,0,pi) t; }" << std::endl;
	    	*ss << "gate y t { u3(pi,pi/2,pi/2) t; }" << std::endl;
		    *ss << "gate z t { u1(pi) t; }" << std::endl;
		    *ss << "gate h t { u2(0,pi) t; }" << std::endl;
		    *ss << "gate s t { u1(pi/2) t; }" << std::endl;
		    *ss << "gate sdg t { u1(-pi/2) t; }" << std::endl;
		    *ss << "gate t t { u1(pi/4) t; }" << std::endl;
		    *ss << "gate tdg t { u1(-pi/4) t; }" << std::endl;
		    *ss << "gate rx(theta) t { u3(theta,-pi/2,pi/2) t; }" << std::endl;
		    *ss << "gate ry(theta) t { u3(theta,0,0) t; }" << std::endl;
		    *ss << "gate rz(phi) t { u1(phi) t; }" << std::endl;
		    *ss << "gate sx t { sdg t; h t; sdg t; }" << std::endl;
		    *ss << "gate sxdg t { s t; h t; s t; }" << std::endl;
		    *ss << "gate rxx(theta) a, b { "
			            "u3(pi/2, theta, 0) a; h b; "
		                "cx a,b; u1(-theta) b; "
			            "cx a,b; h b; "
			            "u2(-pi, pi-theta) a; "
			        "}" << std::endl;
		    *ss << "gate rzz(theta) a, b { "
			            "cx a,b; "
			            "u1(theta) b; "
			            "cx a,b; "
		            "}" << std::endl;
		    *ss << "gate rccx a, b, c { "
                        "u2(0, pi) c; u1(pi/4) c; "
                        "cx b, c; u1(-pi/4) c; "
					    "cx a, c; u1(pi/4) c; "
                        "cx b, c; u1(-pi/4) c; "
				        "u2(0, pi) c; "
                   "}" << std::endl;
		    *ss << "gate rc3x a,b,c,d { "
			            "u2(0,pi) d; u1(pi/4) d; "
			            "cx c,d; u1(-pi/4) d; u2(0,pi) d; "
			            "cx a,d; u1(pi/4) d; "
		                "cx b,d; u1(-pi/4) d; "
				        "cx a,d; u1(pi/4) d; "
						"cx b,d; u1(-pi/4) d; "
                        "u2(0,pi) d; u1(pi/4) d; "
						"cx c,d; u1(-pi/4) d; "
	                    "u2(0,pi) d; "
					"}" << std::endl;
			*ss << "gate c3x a,b,c,d { "
	                    "h d; cu1(-pi/4) a,d; h d; "
				        "cx a,b; "
		                "h d; cu1(pi/4) b,d; h d; "
			            "cx a,b; "
						"h d; cu1(-pi/4) b,d; h d; "
                        "cx b,c; "
					    "h d; cu1(pi/4) c,d; h d; "
	                    "cx a,c; "
				        "h d; cu1(-pi/4) c,d; h d; "
		                "cx b,c; "
		                "h d; cu1(pi/4) c,d; h d; "
			            "cx a,c; "
						"h d; cu1(-pi/4) c,d; h d; "
	                "}" << std::endl;
			*ss << "gate c3sqrtx a,b,c,d { "
			            "h d; cu1(-pi/8) a,d; h d; "
			            "cx a,b; "
			            "h d; cu1(pi/8) b,d; h d; "
			            "cx a,b; "
				        "h d; cu1(-pi/8) b,d; h d; "
				        "cx b,c; "
				        "h d; cu1(pi/8) c,d; h d; "
				        "cx a,c; "
				        "h d; cu1(-pi/8) c,d; h d; "
			            "cx b,c; "
				        "h d; cu1(pi/8) c,d; h d; "
				        "cx a,c; "
				        "h d; cu1(-pi/8) c,d; h d; "
			        "}" << std::endl;
			*ss << "gate c4x a,b,c,d,e { "
			            "h e; cu1(-pi/2) d,e; h e; "
			            "c3x a,b,c,d; "
			            "h e; cu1(pi/2) d,e; h e; "
			            "c3x a,b,c,d; "
			            "c3sqrtx a,b,c,e; "
			       "}" << std::endl;

		    streams.push(ss);
		    lines.push(LineInfo(ch, line, col));
		    line = 1;
		    col = 0;
	    } else if (in->fail()) {
            std::cerr << "Failed to open file '" << filename << "'!" << std::endl;
        } else {
            streams.push(in);
            lines.push(LineInfo(ch, line, col));
            line = 1;
            col = 0;
        }
        nextCh();
    }
}
