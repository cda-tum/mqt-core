#include "parsers/qasm_parser/Scanner.hpp"

#include <locale>

namespace qasm {

/***
 * Private Methods
 ***/
void Scanner::nextCh() {
  if (!streams.empty() && streams.top()->eof()) {
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
      ch = static_cast<char>(-1);
    }
  }
  if (ch == '\n') {
    col = 0;
    line++;
  }
}

void Scanner::readName(Token& t) {
  std::stringstream ss;
  while (::isalnum(ch) != 0 || ch == '_') {
    ss << ch;
    nextCh();
  }
  t.str = ss.str();
  if (const auto it = keywords.find(t.str); it != keywords.end()) {
    t.kind = it->second;
  } else {
    t.kind = Token::Kind::Identifier;
  }
}

void Scanner::readNumber(Token& t) {
  std::stringstream ss;
  while (::isdigit(ch) != 0) {
    ss << ch;
    nextCh();
  }
  t.kind = Token::Kind::Nninteger;
  t.str = ss.str();
  if (ch != '.') {
    ss >> t.val;
    return;
  }
  t.kind = Token::Kind::Real;
  ss << ch;
  nextCh();
  while (::isdigit(ch) != 0) {
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
  while (::isdigit(ch) != 0) {
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
  t.kind = Token::Kind::String;
}

void Scanner::readComment(Token& t) {
  std::stringstream ss;
  while (ch != '\n' && ch != static_cast<char>(-1)) {
    ss << ch;
    nextCh();
  }
  t.str = ss.str();
  t.kind = Token::Kind::Comment;
}

/***
 * Public Methods
 ***/
Scanner::Scanner(std::istream& in) : is(in) {
  keywords["qreg"] = Token::Kind::Qreg;
  keywords["creg"] = Token::Kind::Creg;
  keywords["gate"] = Token::Kind::Gate;
  keywords["measure"] = Token::Kind::Measure;
  keywords["mct"] = Token::Kind::McxGray;
  keywords["mcx"] = Token::Kind::McxGray;
  keywords["mcx_gray"] = Token::Kind::McxGray;
  keywords["mcx_recursive"] = Token::Kind::McxRecursive;
  keywords["mcx_vchain"] = Token::Kind::McxVchain;
  keywords["mcphase"] = Token::Kind::Mcphase;
  keywords["pi"] = Token::Kind::Pi;
  keywords["OPENQASM"] = Token::Kind::Openqasm;
  keywords["show_probabilities"] = Token::Kind::Probabilities;
  keywords["sin"] = Token::Kind::Sin;
  keywords["cos"] = Token::Kind::Cos;
  keywords["tan"] = Token::Kind::Tan;
  keywords["exp"] = Token::Kind::Exp;
  keywords["ln"] = Token::Kind::Ln;
  keywords["sqrt"] = Token::Kind::Sqrt;
  keywords["include"] = Token::Kind::Include;
  keywords["barrier"] = Token::Kind::Barrier;
  keywords["opaque"] = Token::Kind::Opaque;
  keywords["if"] = Token::Kind::If;
  keywords["reset"] = Token::Kind::Reset;
  keywords["snapshot"] = Token::Kind::Snapshot;
  nextCh();
}

Token Scanner::next() {
  // skip over any whitespace
  while (::isspace(ch) != 0) {
    nextCh();
  }

  auto t = Token(Token::Kind::None, line, col);

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
  case 'Z':
    // any name specifier [a-zA-Z]
    readName(t);
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
  case '.':
    // any number starting with [0-9] or .
    readNumber(t);
    break;
  case ';':
    t.kind = Token::Kind::Semicolon;
    nextCh();
    break;
  case static_cast<char>(-1):
    t.kind = Token::Kind::Eof;
    break;
  case '(':
    t.kind = Token::Kind::Lpar;
    nextCh();
    break;
  case ')':
    t.kind = Token::Kind::Rpar;
    nextCh();
    break;
  case '[':
    t.kind = Token::Kind::Lbrack;
    nextCh();
    break;
  case ']':
    t.kind = Token::Kind::Rbrack;
    nextCh();
    break;
  case '{':
    t.kind = Token::Kind::Lbrace;
    nextCh();
    break;
  case '}':
    t.kind = Token::Kind::Rbrace;
    nextCh();
    break;
  case ',':
    t.kind = Token::Kind::Comma;
    nextCh();
    break;
  case '+':
    nextCh();
    t.kind = Token::Kind::Plus;
    break;
  case '-':
    nextCh();
    t.kind = Token::Kind::Minus;
    break;
  case '*':
    nextCh();
    t.kind = Token::Kind::Times;
    break;
  case '/':
    // can indicate a comment or a division
    nextCh();
    if (ch == '/') {
      nextCh();
      readComment(t);
      nextCh();
    } else {
      t.kind = Token::Kind::Div;
    }
    break;
  case '^':
    nextCh();
    t.kind = Token::Kind::Power;
    break;
  case '"':
    // string literal
    nextCh();
    readString(t);
    nextCh();
    break;
  case '>':
    nextCh();
    t.kind = Token::Kind::Gt;
    break;
  case '=':
    // must be an equality operator
    nextCh();
    if (ch == '=') {
      nextCh();
      t.kind = Token::Kind::Eq;
    } else {
      std::cerr << "ERROR: UNEXPECTED CHARACTER: '" << ch << "'!\n";
    }
    break;
  default:
    // this should never be reached
    std::cerr << "ERROR: UNEXPECTED CHARACTER: '" << ch << "'!\n";
    nextCh();
  }

  return t;
}

void Scanner::addFileInput(const std::string& filename) {
  auto in = std::make_shared<std::ifstream>(filename, std::ifstream::in);

  if (in->fail() && filename == "qelib1.inc") {
    // qelib1.inc extensions
    // all other definitions are redundant as the respective gates are natively
    // supported parser can also read multiple-control versions of each gate
    auto ss = std::make_shared<std::stringstream>();
    *ss << "gate rccx a, b, c { "
           "u2(0, pi) c; u1(pi/4) c; "
           "cx b, c; u1(-pi/4) c; "
           "cx a, c; u1(pi/4) c; "
           "cx b, c; u1(-pi/4) c; "
           "u2(0, pi) c; "
           "}\n";
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
           "}\n";
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
           "}\n";
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
           "}\n";
    *ss << "gate c4x a,b,c,d,e { "
           "h e; cu1(-pi/2) d,e; h e; "
           "c3x a,b,c,d; "
           "h e; cu1(pi/2) d,e; h e; "
           "c3x a,b,c,d; "
           "c3sqrtx a,b,c,e; "
           "}\n";
    streams.push(ss);
    lines.emplace(ch, line, col);
    line = 1;
    col = 0;
  } else if (in->fail()) {
    // file could not be found and it was not the standard include file
    std::stringstream ss{};
    ss << "Failed to open file '" << filename << "'!";
    throw std::runtime_error(ss.str());
  } else {
    // file was found and opened
    streams.push(in);
    lines.emplace(ch, line, col);
    line = 1;
    col = 0;
  }
  nextCh();
}
} // namespace qasm
