/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#pragma once

#include "Token.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stack>

namespace qasm {

    class Scanner {
        struct LineInfo {
            char ch;
            int  line, col;

            LineInfo(char ch, int line, int col):
                ch(ch), line(line), col(col) {}
        };

        std::istream&                      is;
        std::stack<std::istream*>          streams{};
        std::map<std::string, Token::Kind> keywords{};
        char                               ch   = 0;
        int                                line = 1;
        int                                col  = 0;

        void nextCh();

        void readName(Token& t);

        void readNumber(Token& t);

        void readString(Token& t);

        void readComment(Token& t);

        std::stack<LineInfo> lines{};

    public:
        explicit Scanner(std::istream& is);

        Token next();

        void addFileInput(const std::string& filename);

        [[nodiscard]] int getLine() const {
            return line;
        }
        [[nodiscard]] int getCol() const {
            return col;
        }
    };
} // namespace qasm
