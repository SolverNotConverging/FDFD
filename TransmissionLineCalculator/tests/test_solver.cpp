#include "solver.hpp"

#include <gmsh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr double epsilon0 = 8.8541878128e-12;
constexpr double mu0 = 1.25663706212e-6;

class TestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

bool isFinite(const std::complex<double> value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

double relativeError(const double actual, const double expected) {
    return std::abs(actual - expected) / std::abs(expected);
}

double relativeError(
    const std::complex<double> actual,
    const std::complex<double> expected
) {
    return std::abs(actual - expected) / std::abs(expected);
}

double median(std::vector<double> values) {
    check(!values.empty(), "median requires at least one value");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U == 0U) {
        return 0.5 * (values[middle - 1U] + values[middle]);
    }
    return values[middle];
}

double maximumEdge(const tl::Triangle& triangle, const tl::Mesh& mesh) {
    double result{};
    for (const auto pair : {std::array{0U, 1U}, std::array{1U, 2U},
                            std::array{2U, 0U}}) {
        const tl::Vec2 first =
            mesh.nodes[static_cast<std::size_t>(triangle.nodes[pair[0]])];
        const tl::Vec2 second =
            mesh.nodes[static_cast<std::size_t>(triangle.nodes[pair[1]])];
        result = std::max(result,
                          std::hypot(second.x - first.x, second.y - first.y));
    }
    return result;
}

void checkPhysicalResult(const tl::Result& result) {
    check(!result.mesh.nodes.empty(), "the mesh must contain nodes");
    check(!result.mesh.triangles.empty(), "the mesh must contain triangles");
    check(!result.samples.empty(), "the result must contain field samples");
    check(
        result.electricPotential.size() == result.mesh.nodes.size(),
        "the electric potential must have one value per node"
    );
    check(
        result.vacuumPotential.size() == result.mesh.nodes.size(),
        "the vacuum potential must have one value per node"
    );

    for (const auto& triangle : result.mesh.triangles) {
        for (const int node : triangle.nodes) {
            check(node >= 0, "triangle node indices must be nonnegative");
            check(
                static_cast<std::size_t>(node) < result.mesh.nodes.size(),
                "triangle node indices must address the node array"
            );
        }
        check(
            isFinite(triangle.relativePermittivity),
            "triangle permittivity must be finite"
        );
    }
    for (const auto& sample : result.samples) {
        check(std::isfinite(sample.position.x), "sample x must be finite");
        check(std::isfinite(sample.position.y), "sample y must be finite");
        check(sample.area > 0.0 && std::isfinite(sample.area), "sample area must be positive");
        check(isFinite(sample.electric.x), "sample Ex must be finite");
        check(isFinite(sample.electric.y), "sample Ey must be finite");
        check(isFinite(sample.magnetic.x), "sample Hx must be finite");
        check(isFinite(sample.magnetic.y), "sample Hy must be finite");
    }

    check(isFinite(result.neff) && result.neff.real() > 0.0, "neff must use the positive-real branch");
    check(isFinite(result.beta) && result.beta.real() > 0.0, "beta must use the forward branch");
    check(
        result.beta.imag() <= 1.0e-10 * std::max(1.0, std::abs(result.beta)),
        "beta must use the passive exp(+jwt-jbetaz) branch"
    );
    check(
        isFinite(result.characteristicImpedance)
            && result.characteristicImpedance.real() > 0.0,
        "characteristic impedance must be finite and positive-real"
    );
    check(
        isFinite(result.waveImpedance) && result.waveImpedance.real() > 0.0,
        "wave impedance must be finite and positive-real"
    );
    check(
        isFinite(result.capacitancePerLength)
            && result.capacitancePerLength.real() > 0.0,
        "capacitance must be finite and positive-real"
    );
    check(
        result.vacuumCapacitancePerLength > 0.0
            && std::isfinite(result.vacuumCapacitancePerLength),
        "vacuum capacitance must be finite and positive"
    );
    check(
        result.inductancePerLength > 0.0
            && std::isfinite(result.inductancePerLength),
        "inductance must be finite and positive"
    );
    check(
        result.externalInductancePerLength > 0.0
            && result.externalInductancePerLength <= result.inductancePerLength,
        "external inductance must be positive and no larger than total inductance"
    );
    check(
        result.resistancePerLength >= 0.0
            && std::isfinite(result.resistancePerLength),
        "resistance must be finite and nonnegative"
    );
    check(
        result.conductancePerLength >= 0.0
            && std::isfinite(result.conductancePerLength),
        "conductance must be finite and nonnegative"
    );
    check(isFinite(result.voltage), "voltage must be finite");
    check(isFinite(result.current), "current must be finite");
    check(isFinite(result.power), "power must be finite");
    check(
        result.meshMilliseconds >= 0.0 && std::isfinite(result.meshMilliseconds),
        "mesh timing must be finite and nonnegative"
    );
    check(
        result.solveMilliseconds >= 0.0 && std::isfinite(result.solveMilliseconds),
        "solve timing must be finite and nonnegative"
    );
}

void testAllDefaults() {
    struct DefaultCase {
        tl::LineType type;
        std::string_view name;
    };
    constexpr std::array cases {
        DefaultCase{tl::LineType::Coaxial, "coaxial"},
        DefaultCase{tl::LineType::Microstrip, "microstrip"},
        DefaultCase{tl::LineType::Stripline, "stripline"},
        DefaultCase{tl::LineType::CoplanarWaveguide, "coplanar waveguide"},
    };
    for (const auto& testCase : cases) {
        try {
            const auto result = tl::solve(tl::defaultParameters(testCase.type));
            checkPhysicalResult(result);
        } catch (const std::exception& error) {
            throw TestFailure(
                std::string(testCase.name) + " default: " + error.what()
            );
        }
    }
}

void testConductorMeshUsesDistanceGrading() {
    auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
    parameters.maxElementSize = 0.60e-3;
    const auto result = tl::solve(parameters);
    std::vector<double> nearConductor;
    std::vector<double> bulk;
    for (const tl::Triangle& triangle : result.mesh.triangles) {
        tl::Vec2 center{};
        for (const int node : triangle.nodes) {
            center.x += result.mesh.nodes[static_cast<std::size_t>(node)].x / 3.0;
            center.y += result.mesh.nodes[static_cast<std::size_t>(node)].y / 3.0;
        }
        const double radius = std::hypot(center.x, center.y);
        const double wallDistance =
            std::min(radius - parameters.innerRadius,
                     parameters.outerRadius - radius);
        const double edge = maximumEdge(triangle, result.mesh);
        if (wallDistance < 0.10e-3) {
            nearConductor.push_back(edge);
        } else if (wallDistance > 0.48e-3) {
            bulk.push_back(edge);
        }
    }
    check(!nearConductor.empty(), "coax mesh needs near-conductor triangles");
    check(!bulk.empty(), "coax mesh needs bulk triangles");
    check(median(nearConductor) < 0.45 * median(bulk),
          "coax conductor mesh must grade smoothly from fine walls to coarse bulk");
}

void testPythonParityAtReferenceInputs() {
    struct PythonAnchor {
        tl::LineType type;
        std::string_view name;
        double neffReal;
        double characteristicImpedanceReal;
        double capacitanceReal;
        double vacuumCapacitance;
        double inductance;
    };
    constexpr std::array anchors {
        PythonAnchor{
            tl::LineType::Coaxial,
            "coaxial",
            1.4491376818646304,
            49.89499373726891,
            9.687951712078439e-11,
            4.613310339084980e-11,
            2.4118257265872673e-7,
        },
        PythonAnchor{
            tl::LineType::Microstrip,
            "microstrip",
            1.6145925210747611,
            50.224847610686766,
            1.0717799402956350e-10,
            4.1101036868150024e-11,
            2.7071094571724350e-7,
        },
        PythonAnchor{
            tl::LineType::Stripline,
            "stripline",
            1.8841460850643263,
            48.20796978135287,
            1.3005455602802767e-10,
            3.6635086205077896e-11,
            3.0371159762670270e-7,
        },
        PythonAnchor{
            tl::LineType::CoplanarWaveguide,
            "coplanar waveguide",
            1.4426621111188755,
            69.13442155624863,
            6.9446691052664610e-11,
            3.3367496137117440e-11,
            3.3345326586130190e-7,
        },
    };
    constexpr double tolerance = 0.04;

    for (const auto& anchor : anchors) {
        auto parameters = tl::defaultParameters(anchor.type);
        if (anchor.type != tl::LineType::Coaxial) {
            parameters.domainPaddingFactor = 1.0;
        }
        check(
            parameters.maxElementSize == 1.0e-3,
            "Python parity anchors require the 1 mm reference mesh size"
        );
        const auto result = tl::solve(parameters);

        const auto checkParity = [&](const double actual,
                                     const double expected,
                                     const std::string_view quantity) {
            const double error = relativeError(actual, expected);
            const std::string message =
                std::string(anchor.name) + " " + std::string(quantity)
                + " differs from the Python 1 mm anchor by "
                + std::to_string(100.0 * error) + "% (limit 4%)";
            check(error <= tolerance, message);
        };
        checkParity(result.neff.real(), anchor.neffReal, "neff.real");
        checkParity(
            result.characteristicImpedance.real(),
            anchor.characteristicImpedanceReal,
            "Zc.real"
        );
        checkParity(
            result.capacitancePerLength.real(),
            anchor.capacitanceReal,
            "C.real"
        );
        checkParity(
            result.vacuumCapacitancePerLength,
            anchor.vacuumCapacitance,
            "C0"
        );
        checkParity(result.inductancePerLength, anchor.inductance, "L");
    }
}

void testOpenDomainDefaultPaddingIsConverged() {
    struct OpenCase {
        tl::LineType type;
        std::string_view name;
    };
    constexpr std::array cases{
        OpenCase{tl::LineType::Microstrip, "microstrip"},
        OpenCase{tl::LineType::Stripline, "stripline"},
        OpenCase{tl::LineType::CoplanarWaveguide, "coplanar waveguide"},
    };
    constexpr double residualTolerance = 0.015;

    for (const auto& testCase : cases) {
        auto parameters = tl::defaultParameters(testCase.type);
        check(
            parameters.domainPaddingFactor == 3.0,
            std::string(testCase.name) + " must use the converged padding-3 default"
        );
        const auto defaultResult = tl::solve(parameters);

        parameters.domainPaddingFactor = 4.0;
        const auto expandedResult = tl::solve(parameters);
        const auto checkStable = [&](const double defaultValue,
                                     const double expandedValue,
                                     const std::string_view quantity) {
            check(
                relativeError(defaultValue, expandedValue) <= residualTolerance,
                std::string(testCase.name) + " " + std::string(quantity)
                    + " changes by more than 1.5% from padding 3 to 4"
            );
        };
        checkStable(
            defaultResult.neff.real(), expandedResult.neff.real(), "neff.real"
        );
        checkStable(
            defaultResult.characteristicImpedance.real(),
            expandedResult.characteristicImpedance.real(),
            "Zc.real"
        );
        checkStable(
            defaultResult.vacuumCapacitancePerLength,
            expandedResult.vacuumCapacitancePerLength,
            "C0"
        );
        checkStable(
            defaultResult.inductancePerLength,
            expandedResult.inductancePerLength,
            "L"
        );
    }
}

void testIdealCoaxAgainstExactTem() {
    auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
    parameters.maxElementSize = 0.30e-3;
    parameters.lossTangent = 0.0;
    parameters.metalConductivity.reset();
    const auto result = tl::solve(parameters);

    const double logarithm = std::log(parameters.outerRadius / parameters.innerRadius);
    const double expectedCapacitance =
        2.0 * std::numbers::pi * epsilon0 * parameters.epsilonR / logarithm;
    const double expectedInductance = mu0 * logarithm / (2.0 * std::numbers::pi);
    const double expectedNeff = std::sqrt(parameters.epsilonR);
    const double expectedImpedance = std::sqrt(expectedInductance / expectedCapacitance);

    check(
        relativeError(result.capacitancePerLength.real(), expectedCapacitance) < 0.03,
        "ideal coax capacitance must agree with the exact TEM result within 3%"
    );
    check(
        relativeError(result.inductancePerLength, expectedInductance) < 0.03,
        "ideal coax inductance must agree with the exact TEM result within 3%"
    );
    check(
        relativeError(result.neff.real(), expectedNeff) < 0.03,
        "ideal coax neff must agree with sqrt(epsilon_r) within 3%"
    );
    check(
        relativeError(result.characteristicImpedance.real(), expectedImpedance) < 0.03,
        "ideal coax Zc must agree with the exact TEM result within 3%"
    );
    check(result.resistancePerLength == 0.0, "PEC coax must have zero resistance");
    check(result.conductancePerLength == 0.0, "lossless coax must have zero conductance");
}

void testLossyPassiveBranch() {
    auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
    parameters.maxElementSize = 0.40e-3;
    parameters.lossTangent = 0.02;
    parameters.metalConductivity.reset();
    const auto result = tl::solve(parameters);

    check(result.neff.imag() < 0.0, "lossy passive neff must have negative imaginary part");
    check(result.beta.imag() < 0.0, "lossy passive beta must have negative imaginary part");
    check(
        result.characteristicImpedance.imag() > 0.0,
        "lossy dielectric coax Zc must have positive imaginary part"
    );
    check(
        result.waveImpedance.imag() > 0.0,
        "lossy dielectric coax Zwave must have positive imaginary part"
    );
    check(result.current.imag() < 0.0, "unit-voltage lossy coax current must have negative imaginary part");
    check(result.power.imag() > 0.0, "lossy coax complex field power must have positive imaginary part");
    check(result.conductancePerLength > 0.0, "loss tangent must produce positive shunt conductance");
}

void testFiniteConductivity() {
    auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
    parameters.maxElementSize = 0.30e-3;
    parameters.lossTangent = 0.0;
    parameters.metalConductivity = 5.96e7;
    const auto result = tl::solve(parameters);

    check(result.surfaceResistance > 0.0, "finite conductivity must produce positive surface resistance");
    check(result.resistancePerLength > 0.0, "finite conductivity must produce positive line resistance");
    check(
        result.inductancePerLength > result.externalInductancePerLength,
        "surface impedance must add positive internal inductance"
    );
    check(result.beta.imag() < 0.0, "conductor loss must produce positive attenuation");

    const double omega = 2.0 * std::numbers::pi * parameters.frequencyHz;
    const double expectedTotal =
        result.externalInductancePerLength + result.resistancePerLength / omega;
    check(
        relativeError(result.inductancePerLength, expectedTotal) < 1.0e-10,
        "good-conductor SIBC must add R/omega to external inductance"
    );
}

void testMicrostripProjectedConductorLoss() {
    auto parameters = tl::defaultParameters(tl::LineType::Microstrip);
    parameters.domainPaddingFactor = 1.0;
    parameters.metalConductivity = 59.6e6;
    const auto result = tl::solve(parameters);

    check(
        std::isfinite(result.conductorGeometryFactorPerLength)
            && result.conductorGeometryFactorPerLength > 0.0,
        "conductive microstrip must have a finite positive conductor geometry factor"
    );
    check(
        std::isfinite(result.resistancePerLength)
            && result.resistancePerLength > 0.0,
        "conductive microstrip must have finite positive resistance"
    );
    check(
        result.inductancePerLength > result.externalInductancePerLength,
        "conductive microstrip must include positive surface internal inductance"
    );
    check(
        result.beta.real() > 0.0 && result.beta.imag() < 0.0,
        "conductive microstrip must select the passive forward beta branch"
    );

    constexpr double pythonGeometryFactor = 391.31539749;
    constexpr double pythonResistance = 10.071256926;
    check(
        relativeError(
            result.conductorGeometryFactorPerLength,
            pythonGeometryFactor
        ) <= 0.10,
        "microstrip conductor geometry factor must remain within 10% of the Python projection"
    );
    check(
        relativeError(result.resistancePerLength, pythonResistance) <= 0.10,
        "microstrip resistance must remain within 10% of the Python projection"
    );
}

void testRefinementIncreasesElementsAndIsStable() {
    auto coarseParameters = tl::defaultParameters(tl::LineType::Stripline);
    coarseParameters.maxElementSize = 1.0e-3;
    coarseParameters.lossTangent = 0.0;
    coarseParameters.refinementFactor = 1.0;
    const auto coarse = tl::solve(coarseParameters);

    auto refinedParameters = coarseParameters;
    refinedParameters.refinementFactor = 2.0;
    const auto refined = tl::solve(refinedParameters);

    check(
        refined.mesh.triangles.size() > coarse.mesh.triangles.size(),
        "Refine x2 must increase the triangle count"
    );
    checkPhysicalResult(refined);

    const double exactNeff = std::sqrt(refinedParameters.epsilonR);
    check(
        relativeError(coarse.neff.real(), exactNeff) < 0.03
            && relativeError(refined.neff.real(), exactNeff) < 0.03,
        "homogeneous stripline neff must remain converged under refinement"
    );
    check(
        relativeError(refined.characteristicImpedance, coarse.characteristicImpedance) < 0.10,
        "Refine x2 must keep stripline Zc within the converged range"
    );
}

void expectInvalid(tl::Parameters parameters, const std::string_view label) {
    try {
        static_cast<void>(tl::solve(parameters));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw TestFailure(std::string(label) + " must throw std::invalid_argument");
}

void testInvalidParametersThrow() {
    {
        auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
        parameters.frequencyHz = 0.0;
        expectInvalid(parameters, "zero frequency");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Microstrip);
        parameters.maxElementSize = 0.0;
        expectInvalid(parameters, "zero mesh size");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Stripline);
        parameters.refinementFactor = 0.0;
        expectInvalid(parameters, "zero refinement factor");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Microstrip);
        parameters.epsilonR = 0.0;
        expectInvalid(parameters, "zero relative permittivity");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::CoplanarWaveguide);
        parameters.lossTangent = -1.0e-3;
        expectInvalid(parameters, "negative loss tangent");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
        parameters.metalConductivity = 0.0;
        expectInvalid(parameters, "zero metal conductivity");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
        parameters.outerRadius = parameters.innerRadius;
        expectInvalid(parameters, "coax outer radius not greater than inner radius");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Stripline);
        parameters.conductorThickness = parameters.groundSpacing;
        expectInvalid(parameters, "stripline conductor thickness at least its ground spacing");
    }
    {
        auto parameters = tl::defaultParameters(tl::LineType::Microstrip);
        parameters.frequencyHz = std::numeric_limits<double>::quiet_NaN();
        expectInvalid(parameters, "non-finite frequency");
    }
}

void testRestoresExternalGmshState() {
    gmsh::initialize();
    try {
        constexpr double hostMeshSizeMax = 0.123456;
        constexpr double hostAlgorithm = 5.0;
        gmsh::model::add("tl_host_state_test");
        gmsh::option::setNumber("General.Terminal", 1.0);
        gmsh::option::setNumber("Mesh.MeshSizeMax", hostMeshSizeMax);
        gmsh::option::setNumber("Mesh.Algorithm", hostAlgorithm);

        auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
        parameters.maxElementSize = 2.0e-3;
        static_cast<void>(tl::solve(parameters));

        std::string currentModel;
        double terminal{};
        double meshSizeMax{};
        double algorithm{};
        gmsh::model::getCurrent(currentModel);
        gmsh::option::getNumber("General.Terminal", terminal);
        gmsh::option::getNumber("Mesh.MeshSizeMax", meshSizeMax);
        gmsh::option::getNumber("Mesh.Algorithm", algorithm);
        check(currentModel == "tl_host_state_test",
              "solve must restore an embedding host's current Gmsh model");
        check(terminal == 1.0,
              "solve must restore an embedding host's Gmsh terminal option");
        check(meshSizeMax == hostMeshSizeMax,
              "solve must restore an embedding host's Gmsh mesh-size option");
        check(algorithm == hostAlgorithm,
              "solve must restore an embedding host's Gmsh algorithm option");
        gmsh::finalize();
    } catch (...) {
        if (gmsh::isInitialized() != 0) {
            gmsh::finalize();
        }
        throw;
    }
}

using Test = std::pair<std::string_view, std::function<void()>>;

} // namespace

int main() {
    const std::vector<Test> tests {
        {"all four defaults", testAllDefaults},
        {"distance-graded conductor mesh", testConductorMeshUsesDistanceGrading},
        {"Python padding-1 numerical parity", testPythonParityAtReferenceInputs},
        {"open-domain default padding convergence", testOpenDomainDefaultPaddingIsConverged},
        {"ideal coax exact TEM", testIdealCoaxAgainstExactTem},
        {"lossy passive branch", testLossyPassiveBranch},
        {"finite conductor loss", testFiniteConductivity},
        {"microstrip projected conductor loss", testMicrostripProjectedConductorLoss},
        {"mesh refinement", testRefinementIncreasesElementsAndIsStable},
        {"invalid parameters", testInvalidParametersThrow},
        {"external Gmsh state restoration", testRestoresExternalGmshState},
    };

    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " solver test groups passed\n";
    return 0;
}
