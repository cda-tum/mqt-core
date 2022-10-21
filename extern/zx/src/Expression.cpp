#include "Expression.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

namespace sym {
    std::unordered_map<std::string, std::size_t> Variable::registered = std::unordered_map<std::string, std::size_t>();
    std::unordered_map<std::size_t, std::string> Variable::names      = std::unordered_map<std::size_t, std::string>();
    std::size_t                                  Variable::nextId;

    Variable::Variable(const std::string& name) {
        const auto it = registered.find(name);
        if (it != registered.end()) {
            id = it->second;
        } else {
            registered[name] = nextId;
            names[nextId]    = name;
            id               = nextId;
            ++nextId;
        }
    }

    std::string Variable::getName() const {
        return names[id];
    }

    std::ostream& operator<<(std::ostream& os, const Variable& var) {
        os << var.getName();
        return os;
    }
} // namespace sym
