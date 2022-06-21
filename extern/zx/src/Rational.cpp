#include "Rational.hpp"

#include "Definitions.hpp"

#include <cmath>

namespace zx {

    PiRational::PiRational(double val):
        frac() {
        if (std::abs(val) < PARAMETER_TOLERANCE)
            return;

        double mult_pi = PI / val;
        double nearest = std::round(mult_pi);
        if (std::abs(nearest - mult_pi) < PARAMETER_TOLERANCE) {
            auto denom = static_cast<int>(nearest);
            frac       = Rational(1, denom);
            modPi();
            return;
        }

        val /= PI;
        val -= 2 * static_cast<int>(val / 2);
        if (val > 1) {
            val -= 2;
        } else if (val <= -1) {
            val += 2;
        }

        frac = Rational(val * MAX_DENOM, MAX_DENOM);
        modPi();
    }

    // double PiRational::to_double() const {
    //   return zx::PI * (static_cast<float>(num)) / denom;
    // }

    PiRational& PiRational::operator+=(const PiRational& rhs) {
        frac += rhs.frac;
        modPi();
        return *this;
    }
    PiRational& PiRational::operator+=(const int64_t rhs) {
        frac += rhs;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator-=(const PiRational& rhs) {
        frac -= rhs.frac;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator-=(const int64_t rhs) {
        frac -= rhs;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator*=(const PiRational& rhs) {
        frac *= rhs.frac;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator*=(const int64_t rhs) {
        frac *= rhs;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator/=(const PiRational& rhs) {
        frac /= rhs.frac;
        modPi();
        return *this;
    }

    PiRational& PiRational::operator/=(const int64_t rhs) {
        frac /= rhs;
        modPi();
        return *this;
    }

    void PiRational::modPi() {
        if (*this > 1) {
            frac = Rational(getNum() - 2 * getDenom(), getDenom());
        } else if (*this <= -1) {
            frac = Rational(getNum() + 2 * getDenom(), getDenom());
        }
        if (getNum() == 0) {
            setDenom(1);
            return;
        }
    }
} // namespace zx
