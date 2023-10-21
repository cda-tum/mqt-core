#pragma once

#include "InstVisitor.hpp"

#include <memory>
#include <utility>

namespace qasm3 {
class Expression;

template <typename T> class Type;
using TypeExpr = Type<std::shared_ptr<Expression>>;
using ResolvedType = Type<uint64_t>;

template <typename T> class Type {
public:
  virtual ~Type() = default;

  [[nodiscard]] virtual bool operator==(const Type<T>& other) const = 0;
  [[nodiscard]] bool operator!=(const Type<T>& other) const {
    return !(*this == other);
  }
  [[nodiscard]] virtual bool allowsDesignator() const = 0;

  virtual void setDesignator(T /*designator*/) {
    throw std::runtime_error("Type does not allow designator");
  }

  [[nodiscard]] virtual T getDesignator() = 0;

  virtual std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) = 0;

  virtual bool isBool() { return false; }
  virtual bool isNumber() { return false; }
  virtual bool isFP() { return false; }
  virtual bool isUint() { return false; }
  virtual bool isBit() { return false; }

  virtual bool fits(const Type<T>& other) { return *this == other; }

  virtual std::string to_string() = 0;
};

enum DesignatedTy {
  Qubit,
  Bit,
  Int,
  Uint,
  Float,
  Angle,
};

class SizedType : public ResolvedType {
public:
  ~SizedType() override = default;

  DesignatedTy type;
  uint64_t designator;

  SizedType(DesignatedTy type, uint64_t designator)
      : type(type), designator(designator) {}

  explicit SizedType(DesignatedTy type) : type(type) {
    switch (type) {
    case DesignatedTy::Qubit:
    case DesignatedTy::Bit:
      designator = 1;
      break;
    case DesignatedTy::Int:
    case DesignatedTy::Uint:
      designator = 32;
      break;
    case DesignatedTy::Float:
    case DesignatedTy::Angle:
      designator = 64;
      break;
    }
  }

  bool operator==(const ResolvedType& other) const override {
    if (const auto* o = dynamic_cast<const SizedType*>(&other)) {
      return type == o->type && designator == o->designator;
    }
    return false;
  }

  [[nodiscard]] bool allowsDesignator() const override { return true; }

  static std::shared_ptr<ResolvedType> getQubitTy(uint64_t size = 1) {
    return std::make_shared<SizedType>(DesignatedTy::Qubit, size);
  }
  static std::shared_ptr<ResolvedType> getBitTy(uint64_t size = 1) {
    return std::make_shared<SizedType>(DesignatedTy::Bit, size);
  }
  static std::shared_ptr<ResolvedType> getIntTy(uint64_t size = 32) {
    return std::make_shared<SizedType>(DesignatedTy::Int, size);
  }
  static std::shared_ptr<ResolvedType> getUintTy(uint64_t size = 32) {
    return std::make_shared<SizedType>(DesignatedTy::Uint, size);
  }
  static std::shared_ptr<ResolvedType> getFloatTy(uint64_t size = 64) {
    return std::make_shared<SizedType>(DesignatedTy::Float, size);
  }
  static std::shared_ptr<ResolvedType> getAngleTy(uint64_t size = 64) {
    return std::make_shared<SizedType>(DesignatedTy::Angle, size);
  }

  uint64_t getDesignator() override { return designator; }

  std::shared_ptr<ResolvedType>
  accept(TypeVisitor<uint64_t>* /*visitor*/) override {
    // don't need to visit sized types
    return nullptr;
  }

  bool isNumber() override {
    return type == DesignatedTy::Int || type == DesignatedTy::Uint ||
           type == DesignatedTy::Float;
  }

  bool isUint() override { return type == DesignatedTy::Uint; }

  bool isBit() override { return type == DesignatedTy::Bit; }

  bool isFP() override { return type == DesignatedTy::Float; }

  bool fits(const ResolvedType& other) override {
    if (const auto* o = dynamic_cast<const SizedType*>(&other)) {
      bool typeFits = type == o->type;
      if (type == DesignatedTy::Int && o->type == DesignatedTy::Uint) {
        typeFits = true;
      }
      if (type == DesignatedTy::Float &&
          (o->type == DesignatedTy::Int || o->type == DesignatedTy::Uint)) {
        typeFits = true;
      }

      return typeFits && designator >= o->designator;
    }
    return false;
  }

  std::string to_string() override {
    switch (type) {
    case DesignatedTy::Qubit:
      return "qubit[" + std::to_string(designator) + "]";
    case DesignatedTy::Bit:
      return "bit[" + std::to_string(designator) + "]";
    case DesignatedTy::Int:
      return "int[" + std::to_string(designator) + "]";
    case DesignatedTy::Uint:
      return "uint[" + std::to_string(designator) + "]";
    case DesignatedTy::Float:
      return "float[" + std::to_string(designator) + "]";
    case DesignatedTy::Angle:
      return "angle[" + std::to_string(designator) + "]";
    }
  }
};

class DesignatedType : public TypeExpr {
public:
  ~DesignatedType() override = default;

  DesignatedTy type;

  std::shared_ptr<Expression> designator;

  DesignatedType(DesignatedTy type, std::shared_ptr<Expression> designator)
      : type(type), designator(std::move(designator)) {}

  bool operator==(const TypeExpr& other) const override {
    if (const auto* o = dynamic_cast<const DesignatedType*>(&other)) {
      return type == o->type && designator == o->designator;
    }
    return false;
  }

  [[nodiscard]] bool allowsDesignator() const override { return true; }

  static std::shared_ptr<TypeExpr>
  getQubitTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Qubit, designator);
  }
  static std::shared_ptr<TypeExpr>
  getBitTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Bit, designator);
  }
  static std::shared_ptr<TypeExpr>
  getIntTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Int, designator);
  }
  static std::shared_ptr<TypeExpr>
  getUintTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Uint, designator);
  }
  static std::shared_ptr<TypeExpr>
  getFloatTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Float, designator);
  }
  static std::shared_ptr<TypeExpr>
  getAngleTy(const std::shared_ptr<Expression>& designator = nullptr) {
    return std::make_shared<DesignatedType>(DesignatedTy::Angle, designator);
  }

  void setDesignator(std::shared_ptr<Expression> d) override {
    this->designator = std::move(d);
  }

  std::shared_ptr<Expression> getDesignator() override { return designator; }

  std::shared_ptr<ResolvedType>
  accept(TypeVisitor<std::shared_ptr<Expression>>* visitor) override {
    return visitor->visitDesignatedType(this);
  }

  bool isNumber() override {
    return type == DesignatedTy::Int || type == DesignatedTy::Uint ||
           type == DesignatedTy::Float;
  }

  bool isUint() override { return type == DesignatedTy::Uint; }

  bool isBit() override { return type == DesignatedTy::Bit; }

  bool isFP() override { return type == DesignatedTy::Float; }

  std::string to_string() override {
    switch (type) {
    case DesignatedTy::Qubit:
      return "qubit[expr]";
    case DesignatedTy::Bit:
      return "bit[expr]";
    case DesignatedTy::Int:
      return "int[expr]";
    case DesignatedTy::Uint:
      return "uint[expr]";
    case DesignatedTy::Float:
      return "float[expr]";
    case DesignatedTy::Angle:
      return "angle[expr]";
    }
  }
};

enum UnsizedTy { Bool, Duration };

template <typename T> class UnsizedType : public Type<T> {
public:
  ~UnsizedType() override = default;

  UnsizedTy type;

  explicit UnsizedType(UnsizedTy type) : type(type) {}

  bool operator==(const Type<T>& other) const override {
    if (const auto* o = dynamic_cast<const UnsizedType*>(&other)) {
      return type == o->type;
    }
    return false;
  }
  [[nodiscard]] bool allowsDesignator() const override { return false; }

  static std::shared_ptr<Type<T>> getBoolTy() {
    return std::make_shared<UnsizedType>(UnsizedTy::Bool);
  }
  static std::shared_ptr<Type<T>> getDurationTy() {
    return std::make_shared<UnsizedType>(UnsizedTy::Duration);
  }

  T getDesignator() override {
    throw std::runtime_error("Unsized types do not have designators");
  }

  std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) override {
    return visitor->visitUnsizedType(this);
  }

  bool isBool() override { return type == UnsizedTy::Bool; }

  std::string to_string() override {
    switch (type) {
    case UnsizedTy::Bool:
      return "bool";
    case UnsizedTy::Duration:
      return "duration";
    }
  }
};

template <typename T> class ArrayType : public Type<T> {
public:
  std::shared_ptr<Type<T>> type;
  T size;

  ArrayType(std::shared_ptr<Type<T>> type, T size)
      : type(std::move(type)), size(size) {}
  ~ArrayType() override = default;

  bool operator==(const Type<T>& other) const override {
    if (const auto* o = dynamic_cast<const ArrayType*>(&other)) {
      return *type == *o->type && size == o->size;
    }
    return false;
  }
  [[nodiscard]] bool allowsDesignator() const override { return true; }

  T getDesignator() override { return size; }

  std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) override {
    return visitor->visitArrayType(this);
  }

  bool fits(const Type<T>& other) override {
    if (const auto* o = dynamic_cast<const ArrayType*>(&other)) {
      return type->fits(*o->type) && size == o->size;
    }
    return false;
  }

  std::string to_string() override {
    return type->to_string() + "[" + std::to_string(size) + "]";
  }
};

} // namespace qasm3
