#include "dd/RealNumber.hpp"

#include <cassert>

namespace dd {

static constexpr std::size_t LSB = static_cast<std::uintptr_t>(1U);

RealNumber* RealNumber::getAlignedPointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) &
                                       ~LSB);
}

RealNumber* RealNumber::getNegativePointer(const RealNumber* e) noexcept {
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) |
                                       LSB);
}

bool RealNumber::exactlyZero(const RealNumber* e) noexcept {
  return (e == &constants::zero);
}

bool RealNumber::exactlyOne(const RealNumber* e) noexcept {
  return (e == &constants::one);
}

bool RealNumber::exactlySqrt2over2(const dd::RealNumber* e) noexcept {
  return (e == &constants::sqrt2over2);
}

RealNumber* RealNumber::flipPointerSign(const RealNumber* e) noexcept {
  if (exactlyZero(e)) {
    return &constants::zero;
  }
  return reinterpret_cast<RealNumber*>(reinterpret_cast<std::uintptr_t>(e) ^
                                       LSB);
}

bool RealNumber::isNegativePointer(const RealNumber* e) noexcept {
  return (reinterpret_cast<std::uintptr_t>(e) & LSB) != 0U;
}

fp RealNumber::val(const RealNumber* e) noexcept {
  assert(e != nullptr);
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->value;
  }
  return e->value;
}

RefCount RealNumber::refCount(const RealNumber* e) noexcept {
  assert(e != nullptr);
  if (isNegativePointer(e)) {
    return -getAlignedPointer(e)->ref;
  }
  return e->ref;
}

bool RealNumber::approximatelyEquals(const fp left, const fp right) noexcept {
  return std::abs(left - right) <= eps;
}

bool RealNumber::approximatelyEquals(const RealNumber* left,
                                     const RealNumber* right) noexcept {
  return left == right || approximatelyEquals(val(left), val(right));
}

bool RealNumber::approximatelyZero(const fp e) noexcept {
  return std::abs(e) <= eps;
}

bool RealNumber::approximatelyZero(const RealNumber* e) noexcept {
  return e == &constants::zero || approximatelyZero(val(e));
}

bool RealNumber::approximatelyOne(const fp e) noexcept {
  return approximatelyEquals(e, 1.0);
}

bool RealNumber::approximatelyOne(const RealNumber* e) noexcept {
  return e == &constants::one || approximatelyOne(val(e));
}

void RealNumber::incRef(dd::RealNumber* e) noexcept {
  auto* const ptr = getAlignedPointer(e);

  if (ptr == nullptr || constants::isStaticNumber(ptr) ||
      ptr->ref == std::numeric_limits<RefCount>::max()) {
    return;
  }

  ++ptr->ref;
}

void RealNumber::decRef(dd::RealNumber* e) noexcept {
  auto* const ptr = getAlignedPointer(e);

  if (ptr == nullptr || constants::isStaticNumber(ptr) ||
      ptr->ref == std::numeric_limits<RefCount>::max()) {
    return;
  }

  assert(ptr->ref != 0 &&
         "Reference count of RealNumber must not be zero before decrement");
  --ptr->ref;
}

void RealNumber::writeBinary(const RealNumber* e, std::ostream& os) {
  const auto temp = val(e);
  os.write(reinterpret_cast<const char*>(&temp), sizeof(decltype(temp)));
}

namespace constants {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
RealNumber zero{0., nullptr, 1U};
RealNumber one{1., nullptr, 1U};
RealNumber sqrt2over2{SQRT2_2, nullptr, 1U};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace constants
} // namespace dd
