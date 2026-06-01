#include "app/except.h"
#include <sstream>
#include <stdexcept>

namespace nn {

namespace {

std::string format_message(const std::string& message, const std::source_location& location) {
    std::ostringstream oss;
    oss << message << " [" << location.file_name() << ":" << location.line() << "]";
    return oss.str();
}

}  // namespace

[[noreturn]] void throw_invalid_argument(const std::string& message,
                                         const std::source_location& location) {
    throw std::invalid_argument(format_message(message, location));
}

[[noreturn]] void throw_logic_error(const std::string& message,
                                    const std::source_location& location) {
    throw std::logic_error(format_message(message, location));
}

}  // namespace nn
