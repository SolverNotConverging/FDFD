#include "tui.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

[[nodiscard]] bool streamIsTerminal(FILE* stream) {
#ifdef _WIN32
    return ::_isatty(::_fileno(stream)) != 0;
#else
    return ::isatty(::fileno(stream)) != 0;
#endif
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [--help | --version]\n\n"
        << "Launches the interactive FTXUI transmission-line calculator.\n"
           "A real input/output terminal is required.\n\n"
           "Keyboard shortcuts:\n"
           "  F5       Calculate the selected geometry\n"
           "  F6       Double mesh density and calculate\n"
           "  F1       Open the in-app help\n"
           "  Ctrl+R   Restore audited defaults\n"
           "  Ctrl+Q   Quit\n";
}

[[nodiscard]] bool isRemovedBatchOption(const std::string_view option) {
    return option == "--type" || option == "--refine" ||
           option == "--benchmark" || option == "--batch";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            const std::string option = argv[1];
            if (option == "--help" || option == "-h") {
                printUsage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (option == "--version") {
                std::cout << "Transmission Line Calculator "
                          << TL_CALCULATOR_VERSION << '\n';
                return EXIT_SUCCESS;
            }
            if (option == "--smoke-test") {
                return tl::tui::smokeTest() ? EXIT_SUCCESS : EXIT_FAILURE;
            }
            if (isRemovedBatchOption(option)) {
                throw std::invalid_argument(
                    "legacy batch flags were removed; launch with no arguments "
                    "and use the interactive controls");
            }
            throw std::invalid_argument("unknown option: " + option);
        }
        if (argc != 1) {
            if (isRemovedBatchOption(argv[1])) {
                throw std::invalid_argument(
                    "legacy batch flags were removed; launch with no arguments "
                    "and use the interactive controls");
            }
            throw std::invalid_argument(
                "the interactive calculator does not accept positional arguments");
        }
        if (!streamIsTerminal(stdin) || !streamIsTerminal(stdout)) {
            std::cerr
                << "error: the FTXUI calculator requires an interactive terminal; "
                   "run it directly instead of redirecting input or output\n";
            return EXIT_FAILURE;
        }
        return tl::tui::run();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
