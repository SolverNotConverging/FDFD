#include "h5_reader.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

std::size_t parseIndex(std::string_view value, std::string_view name) {
    try {
        std::size_t consumed{};
        const auto result = std::stoull(std::string(value), &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::size_t>(result);
    } catch (const std::exception&) {
        throw std::runtime_error(std::format("{} must be a nonnegative integer.", name));
    }
}

std::string asciiArgument(const std::filesystem::path& value) {
#if defined(_WIN32)
    std::string result;
    for (const auto character : value.native()) {
        if (static_cast<unsigned long>(character) > 0x7fUL) {
            throw std::runtime_error("Inspector options and indices must use ASCII characters.");
        }
        result.push_back(static_cast<char>(character));
    }
    return result;
#else
    return value.native();
#endif
}

int runInspector(const std::vector<std::filesystem::path>& arguments) {
    if (arguments.size() < 2 || arguments.size() > 5) {
        std::cerr << "usage: fem-periodic-mode-inspect FILE [CASE_INDEX] [MODE_INDEX] [--coefficients]\n";
        return EXIT_FAILURE;
    }
    try {
        const auto path = arguments[1];
        const auto caseArgument = arguments.size() >= 3 ? asciiArgument(arguments[2]) : "";
        const auto modeArgument = arguments.size() >= 4 ? asciiArgument(arguments[3]) : "";
        const auto caseIndex = arguments.size() >= 3 && caseArgument != "--coefficients"
            ? parseIndex(caseArgument, "case index") : 0U;
        const auto modeIndex = arguments.size() >= 4 && modeArgument != "--coefficients"
            ? parseIndex(modeArgument, "mode index") : 0U;
        bool loadCoefficients = false;
        for (std::size_t index = 2; index < arguments.size(); ++index) {
            loadCoefficients = loadCoefficients
                || asciiArgument(arguments[index]) == "--coefficients";
        }

        const auto started = std::chrono::steady_clock::now();
        const auto archive = femperiodic::H5Reader::loadIndex(path);
        const auto indexed = std::chrono::steady_clock::now();
        if (caseIndex >= archive->cases.size()) {
            throw std::runtime_error("Requested case index is outside the archive.");
        }
        const auto& selected = archive->cases[caseIndex];
        if (modeIndex >= selected.modeCount) {
            throw std::runtime_error("Requested mode index is outside the selected case.");
        }
        const auto mesh = femperiodic::H5Reader::loadMesh(*archive, selected.meshIndex);
        const auto material = femperiodic::H5Reader::loadMaterialState(
            *archive, selected.materialStateIndex);
        const auto fields = femperiodic::H5Reader::loadModeFields(
            *archive, caseIndex, modeIndex);
        if (material->meshIndex != mesh->index
            || material->epsilonR.size() != mesh->cells.size()) {
            throw std::runtime_error("Mesh/material-state references are inconsistent.");
        }
        if (fields->electric.size() != mesh->samplePoints.size()) {
            throw std::runtime_error("Mesh/visualization sample counts are inconsistent.");
        }
        std::size_t coefficientCount{};
        if (loadCoefficients) {
            coefficientCount = femperiodic::H5Reader::loadModeCoefficients(
                *archive, caseIndex, modeIndex).values.size();
        }
        const auto loaded = std::chrono::steady_clock::now();
        const auto indexMs = std::chrono::duration<double, std::milli>(indexed - started).count();
        const auto loadMs = std::chrono::duration<double, std::milli>(loaded - indexed).count();

        std::cout << std::format(
            "format=fem-periodic-modes\n"
            "schema={}.{}\nkind={}\ncases={}\ntotal_modes={}\n"
            "selected_case={}\nselected_mode={}\nfrequency_hz={:.12g}\n"
            "dimension={}\ntopology={}\nmesh_points={}\nmesh_cells={}\n"
            "material_cells={}\nloaded_field_samples={}\nloaded_coefficients={}\n"
            "index_ms={:.3f}\nload_ms={:.3f}\n",
            archive->schemaMajor, archive->schemaMinor, archive->kind,
            archive->cases.size(), archive->modes.size(), caseIndex, modeIndex,
            selected.frequencyHz, mesh->dimension, mesh->topology, mesh->points.size(),
            mesh->cells.size(), material->epsilonR.size(), fields->electric.size(),
            coefficientCount, indexMs, loadMs);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "fem-periodic-mode-inspect: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace

#if defined(_WIN32)
int main() {
    int argumentCount{};
    auto** wideArguments = CommandLineToArgvW(GetCommandLineW(), &argumentCount);
    if (wideArguments == nullptr) {
        std::cerr << "fem-periodic-mode-inspect: could not decode the Windows command line.\n";
        return EXIT_FAILURE;
    }
    std::vector<std::filesystem::path> arguments;
    arguments.reserve(static_cast<std::size_t>(argumentCount));
    for (int index = 0; index < argumentCount; ++index) {
        arguments.emplace_back(wideArguments[index]);
    }
    LocalFree(wideArguments);
    return runInspector(arguments);
}
#else
int main(int argc, char* argv[]) {
    std::vector<std::filesystem::path> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }
    return runInspector(arguments);
}
#endif
