#pragma once

#include <source_location>
#include <string>

namespace nn {

[[noreturn]] void throw_invalid_argument(
    const std::string& message,
    const std::source_location& location = std::source_location::current());

[[noreturn]] void throw_logic_error(
    const std::string& message,
    const std::source_location& location = std::source_location::current());

inline void require(bool condition, const std::string& message,
                    const std::source_location& location = std::source_location::current()) {
    if (!condition)
        throw_invalid_argument(message, location);
}

inline void ensure(bool condition, const std::string& message,
                   const std::source_location& location = std::source_location::current()) {
    if (!condition)
        throw_logic_error(message, location);
}

}  // namespace nn

#define NN_REQUIRE(condition, message) \
    ::nn::require((condition), (message), std::source_location::current())

#define NN_ASSERT(condition, message) \
    ::nn::ensure((condition), (message), std::source_location::current())
