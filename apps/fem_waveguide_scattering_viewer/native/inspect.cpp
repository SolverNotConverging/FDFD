#include "h5_reader.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: fem-waveguide-scattering-viewer-inspect FILE [RESULT_INDEX]\n";
        return 2;
    }
    try {
        const auto resultIndex = argc == 3 ? static_cast<std::size_t>(std::stoull(argv[2])) : 0U;
        const auto started = std::chrono::steady_clock::now();
        const auto index = fem_waveguide_scattering::H5Reader::loadIndex(std::filesystem::path(argv[1]));
        const auto indexed = std::chrono::steady_clock::now();
        const auto result = fem_waveguide_scattering::H5Reader::loadResult(*index, resultIndex);
        const auto loaded = std::chrono::steady_clock::now();
        const auto indexMs = std::chrono::duration<double, std::milli>(indexed - started).count();
        const auto resultMs = std::chrono::duration<double, std::milli>(loaded - indexed).count();
        std::cout << std::format(
            "kind={} results={} selected={} frequency_hz={:.12g}\n"
            "samples={} modes={} s_parameters={} scene_triangles={}\n"
            "index_ms={:.3f} result_ms={:.3f} total_ms={:.3f}\n",
            index->kind, index->frequenciesHz.size(), resultIndex, result->frequencyHz,
            result->coordinates.columns, result->modes.size(), result->sParameters.size(),
            result->scene ? result->scene->triangles.columns : 0U,
            indexMs, resultMs, indexMs + resultMs);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
