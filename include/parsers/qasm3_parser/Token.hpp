#pragma once

#include "QuantumComputation.hpp"

#include <iostream>

namespace qasm3 {

struct Token {
public:
  enum class Kind {
    None,

    Comment,

    OpenQasm,
    Include,
    DefCalGrammar,
    Def,
    Cal,
    DefCal,
    Gate,
    Opaque,
    Extern,
    Box,
    Let,

    Break,
    Continue,
    If,
    Else,
    End,
    Return,
    For,
    While,
    In,

    Pragma,

    // types
    Input,
    Output,
    Const,
    ReadOnly,
    Mutable,

    Qreg,
    Qubit,

    CReg,
    Bool,
    Bit,
    Int,
    Uint,
    Float,
    Angle,
    Complex,
    Array,
    Void,

    Duration,
    Stretch,

    // builtin identifiers
    Gphase,
    Inv,
    Pow,
    Ctrl,
    NegCtrl,

    Dim,

    DurationOf,

    Delay,
    Reset,
    Measure,
    Barrier,

    True,
    False,

    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LParen,
    RParen,

    Colon,
    Semicolon,
    Eof,

    Dot,
    Comma,

    Equals,
    Arrow,
    Plus,
    DoublePlus,
    Minus,
    Asterisk,
    DoubleAsterisk,
    Slash,
    Percent,
    Pipe,
    DoublePipe,
    Ampersand,
    DoubleAmpersand,
    Caret,
    At,
    Tilde,
    ExclamationPoint,

    DoubleEquals,
    NotEquals,
    PlusEquals,
    MinusEquals,
    AsteriskEquals,
    SlashEquals,
    AmpersandEquals,
    PipeEquals,
    TildeEquals,
    CaretEquals,
    LeftShitEquals,
    RightShiftEquals,
    PercentEquals,
    DoubleAsteriskEquals,

    LessThan,
    LessThanEquals,
    GreaterThan,
    GreaterThanEquals,
    LeftShift,
    RightShift,

    Imag,

    Underscore,

    TimeUnitDt,
    TimeUnitNs,
    TimeUnitUs,
    TimeUnitMys,
    TimeUnitMs,
    // might be either TimeUnitS or the `s` gate
    S,

    DoubleQuote,
    SingleQuote,
    BackSlash,

    Identifier,

    HardwareQubit,
    StringLiteral,
    IntegerLiteral,
    FloatLiteral,

    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
  };

  Kind kind = Kind::None;
  size_t line = 0;
  size_t col = 0;
  size_t endLine = 0;
  size_t endCol = 0;
  int64_t val{};
  bool isSigned{false};
  qc::fp valReal{};
  std::string str;

  Token(const size_t l, const size_t c)
      : line(l), col(c), endLine(l), endCol(c) {}
  Token(const Kind k, const size_t l, const size_t c)
      : kind(k), line(l), col(c), endLine(l), endCol(c) {}

  static std::string kindToString(const Kind kind) {
    switch (kind) {
    case Kind::None:
      return "None";
    case Kind::Comment:
      return "Comment";
    case Kind::OpenQasm:
      return "OpenQasm";
    case Kind::Include:
      return "Include";
    case Kind::DefCalGrammar:
      return "DefCalGrammar";
    case Kind::Def:
      return "Def";
    case Kind::Cal:
      return "Cal";
    case Kind::DefCal:
      return "DefCal";
    case Kind::Gate:
      return "Gate";
    case Kind::Opaque:
      return "Opaque";
    case Kind::Extern:
      return "Extern";
    case Kind::Box:
      return "Box";
    case Kind::Let:
      return "Let";
    case Kind::Break:
      return "Break";
    case Kind::Continue:
      return "Continue";
    case Kind::If:
      return "If";
    case Kind::Else:
      return "Else";
    case Kind::End:
      return "End";
    case Kind::Return:
      return "Return";
    case Kind::For:
      return "For";
    case Kind::While:
      return "While";
    case Kind::In:
      return "In";
    case Kind::Pragma:
      return "Pragma";
    case Kind::Input:
      return "Input";
    case Kind::Output:
      return "Output";
    case Kind::Const:
      return "Const";
    case Kind::ReadOnly:
      return "ReadOnly";
    case Kind::Mutable:
      return "Mutable";
    case Kind::Qreg:
      return "Qreg";
    case Kind::Qubit:
      return "Qubit";
    case Kind::CReg:
      return "CReg";
    case Kind::Bool:
      return "Bool";
    case Kind::Bit:
      return "Bit";
    case Kind::Int:
      return "Int";
    case Kind::Uint:
      return "Uint";
    case Kind::Float:
      return "Float";
    case Kind::Angle:
      return "Angle";
    case Kind::Complex:
      return "Complex";
    case Kind::Array:
      return "Array";
    case Kind::Void:
      return "Void";
    case Kind::Duration:
      return "Duration";
    case Kind::Stretch:
      return "Stretch";
    case Kind::Gphase:
      return "Gphase";
    case Kind::Inv:
      return "Inv";
    case Kind::Pow:
      return "Pow";
    case Kind::Ctrl:
      return "Ctrl";
    case Kind::NegCtrl:
      return "NegCtrl";
    case Kind::Dim:
      return "Dim";
    case Kind::DurationOf:
      return "DurationOf";
    case Kind::Delay:
      return "Delay";
    case Kind::Reset:
      return "Reset";
    case Kind::Measure:
      return "Measure";
    case Kind::Barrier:
      return "Barrier";
    case Kind::True:
      return "True";
    case Kind::False:
      return "False";
    case Kind::LBracket:
      return "LBracket";
    case Kind::RBracket:
      return "RBracket";
    case Kind::LBrace:
      return "LBrace";
    case Kind::RBrace:
      return "RBrace";
    case Kind::LParen:
      return "LParen";
    case Kind::RParen:
      return "RParen";
    case Kind::Colon:
      return "Colon";
    case Kind::Semicolon:
      return "Semicolon";
    case Kind::Eof:
      return "Eof";
    case Kind::Dot:
      return "Dot";
    case Kind::Comma:
      return "Comma";
    case Kind::Equals:
      return "Equals";
    case Kind::Arrow:
      return "Arrow";
    case Kind::Plus:
      return "Plus";
    case Kind::DoublePlus:
      return "DoublePlus";
    case Kind::Minus:
      return "Minus";
    case Kind::Asterisk:
      return "Asterisk";
    case Kind::DoubleAsterisk:
      return "DoubleAsterisk";
    case Kind::Slash:
      return "Slash";
    case Kind::Percent:
      return "Percent";
    case Kind::Pipe:
      return "Pipe";
    case Kind::DoublePipe:
      return "DoublePipe";
    case Kind::Ampersand:
      return "Ampersand";
    case Kind::DoubleAmpersand:
      return "DoubleAmpersand";
    case Kind::Caret:
      return "Caret";
    case Kind::At:
      return "At";
    case Kind::Tilde:
      return "Tilde";
    case Kind::ExclamationPoint:
      return "ExclamationPoint";
    case Kind::DoubleEquals:
      return "DoubleEquals";
    case Kind::NotEquals:
      return "NotEquals";
    case Kind::PlusEquals:
      return "PlusEquals";
    case Kind::MinusEquals:
      return "MinusEquals";
    case Kind::AsteriskEquals:
      return "AsteriskEquals";
    case Kind::SlashEquals:
      return "SlashEquals";
    case Kind::AmpersandEquals:
      return "AmpersandEquals";
    case Kind::PipeEquals:
      return "PipeEquals";
    case Kind::TildeEquals:
      return "TildeEquals";
    case Kind::CaretEquals:
      return "CaretEquals";
    case Kind::LeftShitEquals:
      return "LeftShitEquals";
    case Kind::RightShiftEquals:
      return "RightShiftEquals";
    case Kind::PercentEquals:
      return "PercentEquals";
    case Kind::DoubleAsteriskEquals:
      return "DoubleAsteriskEquals";
    case Kind::LessThan:
      return "LessThan";
    case Kind::LessThanEquals:
      return "LessThanEquals";
    case Kind::GreaterThan:
      return "GreaterThan";
    case Kind::GreaterThanEquals:
      return "GreaterThanEquals";
    case Kind::LeftShift:
      return "LeftShift";
    case Kind::RightShift:
      return "RightShift";
    case Kind::Imag:
      return "Imag";
    case Kind::Underscore:
      return "Underscore";
    case Kind::TimeUnitDt:
      return "TimeUnitDt";
    case Kind::TimeUnitNs:
      return "TimeUnitNs";
    case Kind::TimeUnitUs:
      return "TimeUnitUs";
    case Kind::TimeUnitMys:
      return "TimeUnitMys";
    case Kind::TimeUnitMs:
      return "TimeUnitMs";
    case Kind::S:
      return "TimeUnitS";
    case Kind::DoubleQuote:
      return "DoubleQuote";
    case Kind::SingleQuote:
      return "SingleQuote";
    case Kind::BackSlash:
      return "BackSlash";
    case Kind::Identifier:
      return "Identifier";
    case Kind::HardwareQubit:
      return "HardwareQubit";
    case Kind::StringLiteral:
      return "StringLiteral";
    case Kind::IntegerLiteral:
      return "IntegerLiteral";
    case Kind::FloatLiteral:
      return "FloatLiteral";
    case Kind::Sin:
      return "Sin";
    case Kind::Cos:
      return "Cos";
    case Kind::Tan:
      return "Tan";
    case Kind::Exp:
      return "Exp";
    case Kind::Ln:
      return "Ln";
    case Kind::Sqrt:
      return "Sqrt";
    default:
      throw std::runtime_error("Unknown token kind");
    }
  }

  [[nodiscard]] std::string toString() const {
    std::stringstream ss;
    ss << kindToString(kind);
    switch (kind) {
    case Kind::Identifier:
      ss << " (" << str << ")";
      break;
    case Kind::StringLiteral:
      ss << " ('" << str << "')";
      break;
    case Kind::IntegerLiteral:
      ss << " (" << val << ")";
      break;
    case Kind::FloatLiteral:
      ss << " (" << valReal << ")";
      break;
    default:
      break;
    }
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const Kind& k) {
    os << kindToString(k);
    return os;
  }

  friend std::ostream& operator<<(std::ostream& os, const Token& t) {
    os << t.toString();
    return os;
  }
};
} // namespace qasm3
