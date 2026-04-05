#include "except.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace nn {

void react() {
    try {
        throw;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::exit(1);
    } catch (...) {
        std::cerr << "error: unknown exception\n";
        std::exit(1);
    }
}

}  // namespace nn
