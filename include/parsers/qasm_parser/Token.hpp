#pragma once

#include "Definitions.hpp"

#include <map>
#include <string>

namespace qasm {

struct Token {
  enum class Kind {
    Include,
    None,
    Identifier,
    Number,
    Plus,
    Semicolon,
    Eof,
    Lpar,
    Rpar,
    Lbrack,
    Rbrack,
    Lbrace,
    Rbrace,
    Comma,
    Minus,
    Times,
    Nninteger,
    Real,
    Qreg,
    Creg,
    Gate,
    Pi,
    Measure,
    Openqasm,
    Probabilities,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Div,
    Power,
    String,
    Gt,
    Barrier,
    Opaque,
    If,
    Eq,
    Reset,
    Snapshot,
    McxGray,
    McxRecursive,
    McxVchain,
    Mcphase,
    Comment
  };

  Kind kind = Kind::None;
  int line = 0;
  int col = 0;
  int val = 0;
  qc::fp valReal = 0.0;
  std::string str;

  Token() = default;
  Token(Kind k, int l, int c) : kind(k), line(l), col(c) {}
};

static inline const std::map<Token::Kind, std::string> KIND_NAMES{
    {Token::Kind::None, "none"},
    {Token::Kind::Include, "include"},
    {Token::Kind::Identifier, "<identifier>"},
    {Token::Kind::Number, "<number>"},
    {Token::Kind::Plus, "+"},
    {Token::Kind::Semicolon, ";"},
    {Token::Kind::Eof, "EOF"},
    {Token::Kind::Lpar, "("},
    {Token::Kind::Rpar, ")"},
    {Token::Kind::Lbrack, "["},
    {Token::Kind::Rbrack, "]"},
    {Token::Kind::Lbrace, "{"},
    {Token::Kind::Rbrace, "}"},
    {Token::Kind::Comma, ","},
    {Token::Kind::Minus, "-"},
    {Token::Kind::Times, "*"},
    {Token::Kind::Nninteger, "<nninteger>"},
    {Token::Kind::Real, "<real>"},
    {Token::Kind::Qreg, "qreg"},
    {Token::Kind::Creg, "creg"},
    {Token::Kind::Gate, "gate"},
    {Token::Kind::McxGray, "mcx_gray"},
    {Token::Kind::McxRecursive, "mcx_recursive"},
    {Token::Kind::McxVchain, "mcx_vchain"},
    {Token::Kind::Mcphase, "mcphase"},
    {Token::Kind::Pi, "pi"},
    {Token::Kind::Measure, "measure"},
    {Token::Kind::Openqasm, "openqasm"},
    {Token::Kind::Probabilities, "probabilities"},
    {Token::Kind::Opaque, "opaque"},
    {Token::Kind::Sin, "sin"},
    {Token::Kind::Cos, "cos"},
    {Token::Kind::Tan, "tan"},
    {Token::Kind::Exp, "exp"},
    {Token::Kind::Ln, "ln"},
    {Token::Kind::Sqrt, "sqrt"},
    {Token::Kind::Div, "/"},
    {Token::Kind::Power, "^"},
    {Token::Kind::String, "string"},
    {Token::Kind::Gt, ">"},
    {Token::Kind::Barrier, "barrier"},
    {Token::Kind::If, "if"},
    {Token::Kind::Eq, "=="},
    {Token::Kind::Reset, "reset"},
    {Token::Kind::Comment, "//"}};

} // namespace qasm
