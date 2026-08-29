#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace wavefem {

using Complex = std::complex<double>;

struct ComplexMatrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<Complex> values;

    [[nodiscard]] Complex at(std::size_t row, std::size_t column) const {
        return values.at(row * columns + column);
    }
};

struct RealMatrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t row, std::size_t column) const {
        return values.at(row * columns + column);
    }
};

struct IndexMatrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<std::int64_t> values;

    [[nodiscard]] std::int64_t at(std::size_t row, std::size_t column) const {
        return values.at(row * columns + column);
    }
};

struct SParameter {
    std::string side;
    std::int64_t outMode{};
    std::int64_t inMode{};
    Complex value{};
};

struct ModeData {
    std::vector<double> x;
    ComplexMatrix electric;
    ComplexMatrix magnetic;
    std::string label;
};

struct SceneLine {
    std::string kind;
    std::string label;
    std::array<double, 4> endpoints{}; // x0, z0, x1, z1
};

struct SceneData {
    RealMatrix points;       // (2, N), rows x and z
    IndexMatrix triangles;   // (3, M)
    std::vector<Complex> epsR;
    std::array<double, 2> xSpan{};
    std::array<double, 2> zSpan{};
    std::vector<SceneLine> lines;
};

enum class FieldPart { Total, Incident, Scattered };
enum class FieldName { Electric, Magnetic };
enum class ScalarQuantity { Absolute, Real, Imaginary };

struct ResultData {
    double frequencyHz{};
    std::optional<double> ky;
    RealMatrix coordinates;
    std::array<ComplexMatrix, 6> fields;
    std::vector<SParameter> sParameters;
    std::vector<ModeData> modes;
    std::optional<SceneData> scene;

    [[nodiscard]] const ComplexMatrix& field(FieldName name, FieldPart part) const {
        const auto base = name == FieldName::Electric ? 0U : 3U;
        const auto offset = part == FieldPart::Incident ? 0U
            : (part == FieldPart::Scattered ? 1U : 2U);
        return fields.at(base + offset);
    }
};

struct FileIndex {
    std::filesystem::path path;
    std::string kind;
    std::vector<double> frequenciesHz;
    std::vector<std::vector<SParameter>> sParameters;
};

using ResultPtr = std::shared_ptr<const ResultData>;
using FileIndexPtr = std::shared_ptr<const FileIndex>;

} // namespace wavefem
