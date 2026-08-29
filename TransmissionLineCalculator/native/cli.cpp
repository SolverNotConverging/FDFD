#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::string_view lineName(const tl::LineType type) {
    switch (type) {
        case tl::LineType::Coaxial:
            return "coaxial";
        case tl::LineType::Microstrip:
            return "microstrip";
        case tl::LineType::Stripline:
            return "stripline";
        case tl::LineType::CoplanarWaveguide:
            return "cpw";
    }
    return "unknown";
}

[[nodiscard]] tl::LineType parseLineType(const std::string_view value) {
    if (value == "coaxial" || value == "coax") {
        return tl::LineType::Coaxial;
    }
    if (value == "microstrip") {
        return tl::LineType::Microstrip;
    }
    if (value == "stripline") {
        return tl::LineType::Stripline;
    }
    if (value == "cpw" || value == "coplanar_waveguide") {
        return tl::LineType::CoplanarWaveguide;
    }
    throw std::invalid_argument(
        "--type must be coaxial, microstrip, stripline, or cpw");
}

[[nodiscard]] double parsePositiveDouble(const std::string& value,
                                         const char* const option) {
    std::size_t consumed{};
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
        throw std::invalid_argument(std::string(option) +
                                    " requires a finite positive number");
    }
    return parsed;
}

[[nodiscard]] int parsePositiveInteger(const std::string& value,
                                       const char* const option) {
    std::size_t consumed{};
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed <= 0L || parsed > 1000L) {
        throw std::invalid_argument(std::string(option) +
                                    " requires an integer from 1 to 1000");
    }
    return static_cast<int>(parsed);
}

void printUsage(const char* const executable) {
    std::cout
        << "Usage: " << executable
        << " [--type coaxial|microstrip|stripline|cpw] [--refine N]"
           " [--benchmark N]\n\n"
           "With no arguments, solves all four transmission-line defaults.\n"
           "--refine N multiplies mesh density (N=2 halves the target edge).\n"
           "--benchmark N repeats the complete mesh-and-solve path and reports"
           " its median.\n";
}

void printComplex(const std::string_view label, const std::complex<double> value,
                  const std::string_view unit = {}) {
    std::cout << "  " << std::left << std::setw(11) << label << std::right
              << std::scientific << std::setprecision(9) << value.real()
              << (value.imag() < 0.0 ? " - j" : " + j") << std::abs(value.imag());
    if (!unit.empty()) {
        std::cout << ' ' << unit;
    }
    std::cout << '\n';
}

void printReal(const std::string_view label, const double value,
               const std::string_view unit) {
    std::cout << "  " << std::left << std::setw(11) << label << std::right
              << std::scientific << std::setprecision(9) << value;
    if (!unit.empty()) {
        std::cout << ' ' << unit;
    }
    std::cout << '\n';
}

void printResult(const tl::Result& result, const double medianMilliseconds,
                 const int repetitions, const bool benchmarkRequested) {
    std::cout << '\n' << lineName(result.parameters.type) << '\n';
    std::cout << "  nodes      " << result.mesh.nodes.size() << '\n';
    std::cout << "  triangles  " << result.mesh.triangles.size() << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "  mesh       " << result.meshMilliseconds << " ms\n"
              << "  assembly   " << result.assemblyMilliseconds << " ms\n"
              << "  factorize  " << result.factorizationMilliseconds << " ms\n"
              << "  solve      " << result.solveMilliseconds << " ms\n";
    if (benchmarkRequested) {
        std::cout << "  benchmark  " << medianMilliseconds << " ms median ("
                  << repetitions << " complete runs)\n";
    }
    printComplex("n_eff", result.neff);
    printComplex("beta", result.beta, "1/m");
    printComplex("Zc", result.characteristicImpedance, "ohm");
    printComplex("Zwave", result.waveImpedance, "ohm");
    printReal("R'", result.resistancePerLength, "ohm/m");
    printReal("L'", result.inductancePerLength, "H/m");
    printReal("G'", result.conductancePerLength, "S/m");
    printComplex("C'", result.capacitancePerLength, "F/m");
    printReal("C0'", result.vacuumCapacitancePerLength, "F/m");
    printComplex("power", result.power, "W");
    if (result.parameters.metalConductivity.has_value()) {
        printReal("Rs", result.surfaceResistance, "ohm");
        printReal("geometry", result.conductorGeometryFactorPerLength, "1/m");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::optional<tl::LineType> requestedType;
        double refinementFactor = 1.0;
        int repetitions = 1;
        bool benchmarkRequested = false;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--help" || option == "-h") {
                printUsage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (option == "--type" || option == "--refine" ||
                option == "--benchmark") {
                if (index + 1 >= argc) {
                    throw std::invalid_argument(option + " requires a value");
                }
                const std::string value = argv[++index];
                if (option == "--type") {
                    requestedType = parseLineType(value);
                } else if (option == "--refine") {
                    refinementFactor = parsePositiveDouble(value, "--refine");
                } else {
                    repetitions = parsePositiveInteger(value, "--benchmark");
                    benchmarkRequested = true;
                }
                continue;
            }
            throw std::invalid_argument("unknown option: " + option);
        }

        std::vector<tl::LineType> types;
        if (requestedType.has_value()) {
            types.push_back(*requestedType);
        } else {
            types = {tl::LineType::Coaxial, tl::LineType::Microstrip,
                     tl::LineType::Stripline,
                     tl::LineType::CoplanarWaveguide};
        }
        for (const tl::LineType type : types) {
            tl::Parameters parameters = tl::defaultParameters(type);
            parameters.refinementFactor = refinementFactor;
            std::vector<double> timings;
            timings.reserve(static_cast<std::size_t>(repetitions));
            std::optional<tl::Result> result;
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                const auto begin = Clock::now();
                result = tl::solve(parameters);
                const auto end = Clock::now();
                timings.push_back(
                    std::chrono::duration<double, std::milli>(end - begin).count());
            }
            std::sort(timings.begin(), timings.end());
            const std::size_t middle = timings.size() / 2U;
            const double median =
                timings.size() % 2U == 0U
                    ? 0.5 * (timings[middle - 1U] + timings[middle])
                    : timings[middle];
            printResult(*result, median, repetitions, benchmarkRequested);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
