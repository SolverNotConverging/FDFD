#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <optional>
#include <vector>

namespace tl {

enum class LineType {
    Coaxial,
    Microstrip,
    Stripline,
    CoplanarWaveguide,
};

struct Vec2 {
    double x{};
    double y{};
};

struct Parameters {
    LineType type{LineType::Microstrip};
    double frequencyHz{10.0e9};
    double maxElementSize{1.0e-3};
    double refinementFactor{1.0};
    int maxRefinements{2};
    double adaptiveTolerance{0.05};

    double innerRadius{0.50e-3};
    double outerRadius{1.67e-3};
    double outerConductorThickness{0.15e-3};

    double traceWidth{3.00e-3};
    double substrateHeight{1.524e-3};
    double conductorThickness{35.0e-6};
    double groundSpacing{1.524e-3};

    double centerWidth{0.60e-3};
    double gap{0.25e-3};
    double groundWidth{1.50e-3};

    double epsilonR{3.55};
    double lossTangent{2.7e-3};
    double domainPaddingFactor{3.0};
    std::optional<double> metalConductivity{};
};

struct Triangle {
    std::array<int, 3> nodes{};
    std::complex<double> relativePermittivity{1.0, 0.0};
};

struct Mesh {
    std::vector<Vec2> nodes;
    std::vector<Triangle> triangles;
};

struct FieldVector {
    std::complex<double> x{};
    std::complex<double> y{};
};

struct FieldSample {
    Vec2 position{};
    FieldVector electric{};
    FieldVector magnetic{};
    double area{};
    std::complex<double> relativePermittivity{1.0, 0.0};
};

struct Result {
    Parameters parameters{};
    Mesh mesh{};
    std::vector<FieldSample> samples;
    std::vector<std::complex<double>> electricPotential;
    std::vector<std::complex<double>> vacuumPotential;

    std::complex<double> neff{};
    std::complex<double> characteristicImpedance{};
    std::complex<double> waveImpedance{};
    std::complex<double> capacitancePerLength{};
    std::complex<double> beta{};
    std::complex<double> voltage{1.0, 0.0};
    std::complex<double> current{};
    std::complex<double> power{};

    double vacuumCapacitancePerLength{};
    double inductancePerLength{};
    double externalInductancePerLength{};
    double resistancePerLength{};
    double conductancePerLength{};
    double surfaceResistance{};

    double meshMilliseconds{};
    double solveMilliseconds{};

    // Additional diagnostics used by the CLI and regression benchmarks.
    double assemblyMilliseconds{};
    double factorizationMilliseconds{};
    double conductorGeometryFactorPerLength{};
    double materialResidual{};
    double vacuumResidual{};
    // Each entry is {element count, normalized flux-jump residual}.
    std::vector<std::array<double, 2>> adaptiveHistory;
    bool adaptiveConverged{};
};

}  // namespace tl
