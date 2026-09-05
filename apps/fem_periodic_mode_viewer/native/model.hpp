#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace femperiodic {

using Complex = std::complex<double>;

enum class FieldFamily { Electric, Magnetic };
enum class ScalarQuantity { Magnitude, Real, Imaginary, Phase };

struct ModeSummary {
    Complex gammaPerM{};
    Complex neff{};
    Complex neffFolded{};
    Complex blochMultiplier{};
    double alphaPerM{};
    double betaPerM{};
    double betaFoldedPerM{};
    double residual{};
    std::optional<double> gaussResidual;
    double pmlFraction{};
    std::string polarization;
    std::string direction;
    std::string normalization;
};

struct CaseIndex {
    double frequencyHz{};
    std::size_t modeBegin{};
    std::size_t modeCount{};
    std::size_t meshIndex{};
    std::size_t materialStateIndex{};
};

struct FileIndex {
    std::filesystem::path path;
    std::string kind;
    std::string producer;
    std::string producerVersion;
    std::string timeConvention;
    std::string fieldRepresentation;
    std::int64_t schemaMajor{};
    std::int64_t schemaMinor{};
    std::vector<CaseIndex> cases;
    std::vector<ModeSummary> modes;
};

struct MeshData {
    std::size_t index{};
    int dimension{};
    std::string topology;
    std::string periodicAxis;
    double periodM{};
    double referenceZM{};
    std::vector<std::array<double, 3>> points;
    std::vector<std::vector<std::int64_t>> cells;
    std::vector<std::int64_t> cellRegionIds;
    std::vector<std::array<double, 3>> samplePoints;
    std::vector<std::int64_t> sampleOwnerCells;
    std::vector<std::vector<std::int64_t>> boundaryFacets;
    std::vector<std::int64_t> boundaryTags;
    std::vector<std::array<std::int64_t, 2>> periodicNodePairs;
    std::array<double, 16> periodicAffine{};
    bool hasPeriodicAffine{false};
    std::vector<std::array<std::int64_t, 2>> edgeNodes;
    std::vector<std::array<std::int64_t, 6>> cellEdges;
    std::vector<std::array<std::int64_t, 6>> cellEdgeSigns;
    std::vector<std::array<std::int64_t, 2>> periodicEdgePairs;
    std::vector<std::int64_t> periodicEdgeSigns;
};

struct MaterialState {
    std::size_t index{};
    std::size_t meshIndex{};
    std::vector<std::array<Complex, 3>> epsilonR;
    std::vector<std::array<Complex, 3>> muR;
    std::vector<double> pmlFraction;
};

struct ModeFields {
    std::size_t caseIndex{};
    std::size_t localModeIndex{};
    std::vector<std::array<Complex, 3>> electric;
    std::vector<std::array<Complex, 3>> magnetic;
};

struct ModeCoefficients {
    std::size_t caseIndex{};
    std::size_t localModeIndex{};
    std::string space;
    std::string primaryUnknown;
    std::vector<Complex> values;
};

using FileIndexPtr = std::shared_ptr<const FileIndex>;
using MeshPtr = std::shared_ptr<const MeshData>;
using MaterialStatePtr = std::shared_ptr<const MaterialState>;
using ModeFieldsPtr = std::shared_ptr<const ModeFields>;

inline double scalarValue(Complex value, ScalarQuantity quantity) {
    switch (quantity) {
    case ScalarQuantity::Magnitude:
        return std::abs(value);
    case ScalarQuantity::Real:
        return value.real();
    case ScalarQuantity::Imaginary:
        return value.imag();
    case ScalarQuantity::Phase:
        return std::arg(value);
    }
    return 0.0;
}

} // namespace femperiodic
