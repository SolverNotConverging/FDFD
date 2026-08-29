#include "solver.hpp"

#include <gmsh.h>

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tl {
namespace {

using Complex = std::complex<double>;
using Clock = std::chrono::steady_clock;
using SparseComplex = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<Complex, int>;
using VectorComplex = Eigen::VectorXcd;
using DimTag = std::pair<int, int>;

constexpr double kSpeedOfLight = 299'792'458.0;
constexpr double kEpsilon0 = 8.8541878188e-12;
constexpr double kMu0 = 1.25663706127e-6;
constexpr double kPi = std::numbers::pi_v<double>;
constexpr std::size_t kMaximumTriangles = 4'000'000;

std::mutex gGmshMutex;
std::uint64_t gModelSequence{};

enum class BoundaryRole {
    Outer,
    Signal,
    Reference,
};

struct BoundaryEdge {
    int first{};
    int second{};
    BoundaryRole role{BoundaryRole::Outer};
    int conductor{-1};
};

struct GeneratedMesh {
    Mesh mesh;
    std::vector<BoundaryEdge> boundaryEdges;
};

struct Rectangle {
    double xmin{};
    double xmax{};
    double ymin{};
    double ymax{};
};

struct RectConductor : Rectangle {
    BoundaryRole role{BoundaryRole::Reference};
    int id{};
};

struct MaterialRegion : Rectangle {
    Complex relativePermittivity{1.0, 0.0};
};

struct RectangularDefinition {
    double xmin{};
    double xmax{};
    double ymin{};
    double ymax{};
    Complex backgroundPermittivity{1.0, 0.0};
    std::vector<MaterialRegion> materials;
    std::vector<RectConductor> conductors;
    std::vector<Rectangle> refinementRegions;
    double localTarget{};
    double transitionWidth{};
};

struct CoordinateTransform {
    double xmin{};
    double ymin{};
    double scale{};

    [[nodiscard]] double x(const double physical) const {
        return (physical - xmin) * scale;
    }

    [[nodiscard]] double y(const double physical) const {
        return (physical - ymin) * scale;
    }

    [[nodiscard]] Vec2 physical(const double normalizedX,
                                const double normalizedY) const {
        return {normalizedX / scale + xmin, normalizedY / scale + ymin};
    }
};

struct DirichletSolution {
    VectorComplex potential;
    VectorComplex reaction;
    double residual{};
    double factorizationMilliseconds{};
};

class GmshSession {
public:
    GmshSession() {
        owned_ = gmsh::isInitialized() == 0;
        if (owned_) {
            gmsh::initialize();
        } else {
            gmsh::model::getCurrent(previousModel_);
            for (const char* const option : changedOptions_) {
                double value{};
                gmsh::option::getNumber(option, value);
                previousOptions_.emplace_back(option, value);
            }
        }
        modelName_ = "tl_quasi_tem_" + std::to_string(++gModelSequence);
        gmsh::model::add(modelName_);
        active_ = true;
        gmsh::option::setNumber("General.Terminal", 0.0);
    }

    GmshSession(const GmshSession&) = delete;
    GmshSession& operator=(const GmshSession&) = delete;

    ~GmshSession() noexcept {
        if (owned_) {
            try {
                if (gmsh::isInitialized() != 0) {
                    gmsh::finalize();
                }
            } catch (...) {
                // Destructors must not mask the numerical or geometry error.
            }
            return;
        }
        if (gmsh::isInitialized() == 0) {
            return;
        }
        if (active_) {
            try {
                gmsh::model::setCurrent(modelName_);
                gmsh::model::remove();
            } catch (...) {
                // Continue restoring the caller's global Gmsh state.
            }
        }
        for (const auto& [option, value] : previousOptions_) {
            try {
                gmsh::option::setNumber(option, value);
            } catch (...) {
                // Restore as much state as possible from a noexcept destructor.
            }
        }
        if (!previousModel_.empty()) {
            try {
                gmsh::model::setCurrent(previousModel_);
            } catch (...) {
                // The host may have removed its model while the solve ran.
            }
        }
    }

private:
    static constexpr std::array changedOptions_ {
        "General.Terminal",
        "Mesh.MeshSizeMax",
        "Mesh.MeshSizeMin",
        "Mesh.MeshSizeExtendFromBoundary",
        "Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature",
        "Mesh.ElementOrder",
        "Mesh.RecombineAll",
        "Mesh.Algorithm",
    };

    bool owned_{};
    bool active_{};
    std::string modelName_;
    std::string previousModel_;
    std::vector<std::pair<std::string, double>> previousOptions_;
};

[[nodiscard]] double elapsedMilliseconds(const Clock::time_point begin,
                                         const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

[[nodiscard]] bool finite(const Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void requirePositive(const double value, const char* const name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void validate(const Parameters& p) {
    requirePositive(p.frequencyHz, "frequencyHz");
    requirePositive(p.maxElementSize, "maxElementSize");
    requirePositive(p.refinementFactor, "refinementFactor");
    requirePositive(p.innerRadius, "innerRadius");
    requirePositive(p.outerRadius, "outerRadius");
    requirePositive(p.outerConductorThickness, "outerConductorThickness");
    requirePositive(p.traceWidth, "traceWidth");
    requirePositive(p.substrateHeight, "substrateHeight");
    requirePositive(p.conductorThickness, "conductorThickness");
    requirePositive(p.groundSpacing, "groundSpacing");
    requirePositive(p.centerWidth, "centerWidth");
    requirePositive(p.gap, "gap");
    requirePositive(p.groundWidth, "groundWidth");
    requirePositive(p.epsilonR, "epsilonR");
    requirePositive(p.domainPaddingFactor, "domainPaddingFactor");
    if (!std::isfinite(p.lossTangent) || p.lossTangent < 0.0) {
        throw std::invalid_argument("lossTangent must be finite and nonnegative");
    }
    if (p.metalConductivity.has_value()) {
        requirePositive(*p.metalConductivity, "metalConductivity");
    }
    if (p.type == LineType::Coaxial && p.outerRadius <= p.innerRadius) {
        throw std::invalid_argument("outerRadius must be greater than innerRadius");
    }
    if (p.type == LineType::Stripline && p.conductorThickness >= p.groundSpacing) {
        throw std::invalid_argument(
            "conductorThickness must be smaller than groundSpacing for stripline");
    }
}

[[nodiscard]] Complex relativePermittivity(const Parameters& p) {
    return {p.epsilonR, -p.epsilonR * p.lossTangent};
}

[[nodiscard]] bool edgeOnRectangle(const Vec2 first, const Vec2 second,
                                   const Rectangle& rectangle,
                                   const double tolerance) {
    const auto inX = [&rectangle, tolerance](const double value) {
        return value >= rectangle.xmin - tolerance && value <= rectangle.xmax + tolerance;
    };
    const auto inY = [&rectangle, tolerance](const double value) {
        return value >= rectangle.ymin - tolerance && value <= rectangle.ymax + tolerance;
    };
    return (std::abs(first.x - rectangle.xmin) <= tolerance &&
            std::abs(second.x - rectangle.xmin) <= tolerance && inY(first.y) &&
            inY(second.y)) ||
           (std::abs(first.x - rectangle.xmax) <= tolerance &&
            std::abs(second.x - rectangle.xmax) <= tolerance && inY(first.y) &&
            inY(second.y)) ||
           (std::abs(first.y - rectangle.ymin) <= tolerance &&
            std::abs(second.y - rectangle.ymin) <= tolerance && inX(first.x) &&
            inX(second.x)) ||
           (std::abs(first.y - rectangle.ymax) <= tolerance &&
            std::abs(second.y - rectangle.ymax) <= tolerance && inX(first.x) &&
            inX(second.x));
}

[[nodiscard]] std::uint64_t edgeKey(const int first, const int second) {
    const auto low = static_cast<std::uint32_t>(std::min(first, second));
    const auto high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
}

[[nodiscard]] RectangularDefinition rectangularDefinition(const Parameters& p) {
    const double baseTarget = p.maxElementSize / p.refinementFactor;
    const Complex dielectric = relativePermittivity(p);
    RectangularDefinition definition;
    if (p.type == LineType::Microstrip) {
        const double sideClearance = p.domainPaddingFactor *
                                     std::max(1.5 * p.traceWidth,
                                              3.0 * p.substrateHeight);
        const double halfWidth = 0.5 * p.traceWidth + sideClearance;
        const double airHeight = p.domainPaddingFactor *
                                 std::max(3.0 * p.substrateHeight,
                                          1.5 * p.traceWidth);
        definition.xmin = -halfWidth;
        definition.xmax = halfWidth;
        definition.ymin = -p.conductorThickness;
        definition.ymax = p.substrateHeight + p.conductorThickness + airHeight;
        definition.backgroundPermittivity = {1.0, 0.0};
        definition.materials.push_back(
            {{-halfWidth, halfWidth, 0.0, p.substrateHeight}, dielectric});
        definition.conductors.push_back(
            {{-0.5 * p.traceWidth,
              0.5 * p.traceWidth,
              p.substrateHeight,
              p.substrateHeight + p.conductorThickness},
             BoundaryRole::Signal,
             0});
        definition.conductors.push_back(
            {{-halfWidth, halfWidth, -p.conductorThickness, 0.0},
             BoundaryRole::Reference,
             1});
        definition.localTarget =
            std::min(baseTarget, std::min(p.traceWidth, p.substrateHeight) /
                                     (10.0 * p.refinementFactor));
        definition.transitionWidth =
            std::max(0.5 * p.substrateHeight, 0.25 * p.traceWidth);
        const double fringeX =
            std::max(0.75 * p.traceWidth, 0.75 * p.substrateHeight);
        const double fringeY = std::min(0.5 * p.substrateHeight, p.traceWidth);
        definition.refinementRegions.push_back(
            {-0.5 * p.traceWidth - fringeX,
             0.5 * p.traceWidth + fringeX,
             p.substrateHeight - fringeY,
             p.substrateHeight + p.conductorThickness + fringeY});
        return definition;
    }
    if (p.type == LineType::Stripline) {
        const double halfSpacing = 0.5 * p.groundSpacing;
        const double signalHalfHeight = 0.5 * p.conductorThickness;
        const double dielectricGap = halfSpacing - signalHalfHeight;
        requirePositive(dielectricGap, "stripline dielectric gap");
        const double sideClearance = p.domainPaddingFactor *
                                     std::max(3.0 * p.groundSpacing,
                                              2.0 * p.traceWidth);
        const double halfWidth = 0.5 * p.traceWidth + sideClearance;
        definition.xmin = -halfWidth;
        definition.xmax = halfWidth;
        definition.ymin = -halfSpacing - p.conductorThickness;
        definition.ymax = halfSpacing + p.conductorThickness;
        definition.backgroundPermittivity = dielectric;
        definition.conductors.push_back(
            {{-0.5 * p.traceWidth,
              0.5 * p.traceWidth,
              -signalHalfHeight,
              signalHalfHeight},
             BoundaryRole::Signal,
             0});
        definition.conductors.push_back(
            {{-halfWidth,
              halfWidth,
              -halfSpacing - p.conductorThickness,
              -halfSpacing},
             BoundaryRole::Reference,
             1});
        definition.conductors.push_back(
            {{-halfWidth,
              halfWidth,
              halfSpacing,
              halfSpacing + p.conductorThickness},
             BoundaryRole::Reference,
             2});
        definition.localTarget =
            std::min(baseTarget, std::min(p.traceWidth, dielectricGap) /
                                     (10.0 * p.refinementFactor));
        definition.transitionWidth = 0.5 * dielectricGap;
        definition.refinementRegions.push_back(
            {-1.25 * p.traceWidth, 1.25 * p.traceWidth,
             -0.45 * dielectricGap, 0.45 * dielectricGap});
        return definition;
    }
    if (p.type == LineType::CoplanarWaveguide) {
        const double signalEdge = 0.5 * p.centerWidth;
        const double groundInnerEdge = signalEdge + p.gap;
        const double metalHalfWidth = groundInnerEdge + p.groundWidth;
        const double sideClearance = p.domainPaddingFactor *
                                     std::max(2.0 * p.substrateHeight,
                                              0.75 * metalHalfWidth);
        const double halfWidth = metalHalfWidth + sideClearance;
        const double verticalClearance = p.domainPaddingFactor *
                                         std::max(2.0 * p.substrateHeight,
                                                  metalHalfWidth);
        definition.xmin = -halfWidth;
        definition.xmax = halfWidth;
        definition.ymin = -p.substrateHeight - verticalClearance;
        definition.ymax = p.conductorThickness + verticalClearance;
        definition.backgroundPermittivity = {1.0, 0.0};
        definition.materials.push_back(
            {{-halfWidth, halfWidth, -p.substrateHeight, 0.0}, dielectric});
        definition.conductors.push_back(
            {{-signalEdge, signalEdge, 0.0, p.conductorThickness},
             BoundaryRole::Signal,
             0});
        definition.conductors.push_back(
            {{-metalHalfWidth,
              -groundInnerEdge,
              0.0,
              p.conductorThickness},
             BoundaryRole::Reference,
             1});
        definition.conductors.push_back(
            {{groundInnerEdge,
              metalHalfWidth,
              0.0,
              p.conductorThickness},
             BoundaryRole::Reference,
             2});
        definition.localTarget =
            std::min(baseTarget,
                     std::min({p.centerWidth, p.gap, p.groundWidth,
                               p.substrateHeight}) /
                         (8.0 * p.refinementFactor));
        definition.transitionWidth =
            std::max(2.0 * p.gap, 0.25 * p.substrateHeight);
        const double refinementDepth =
            std::min(0.5 * p.substrateHeight, 2.0 * p.gap);
        definition.refinementRegions.push_back(
            {-metalHalfWidth - p.gap, metalHalfWidth + p.gap,
             -refinementDepth,
             p.conductorThickness + refinementDepth});
        return definition;
    }
    throw std::invalid_argument("rectangular geometry requested for a coaxial line");
}

[[nodiscard]] std::vector<double> numericTags(const std::vector<int>& tags) {
    std::vector<double> values;
    values.reserve(tags.size());
    for (const int tag : tags) {
        values.push_back(static_cast<double>(tag));
    }
    return values;
}

void addDistanceField(const std::vector<int>& curves, const double minimumSize,
                      const double maximumSize, const double transition,
                      const double scale, std::vector<int>& fields,
                      double& smallestSize) {
    if (curves.empty()) {
        return;
    }
    const int distance = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(distance, "CurvesList", numericTags(curves));
    const int threshold = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(threshold, "InField",
                                        static_cast<double>(distance));
    gmsh::model::mesh::field::setNumber(threshold, "SizeMin", minimumSize * scale);
    gmsh::model::mesh::field::setNumber(threshold, "SizeMax", maximumSize * scale);
    gmsh::model::mesh::field::setNumber(threshold, "DistMin", 0.0);
    gmsh::model::mesh::field::setNumber(threshold, "DistMax", transition * scale);
    fields.push_back(threshold);
    smallestSize = std::min(smallestSize, minimumSize);
}

void applyMeshFields(const std::vector<int>& physicalCurves,
                     const std::vector<int>& outerCurves,
                     const std::vector<int>& refinementSurfaces,
                     const std::vector<int>& refinementCurves,
                     const std::unordered_map<int, Complex>& surfaceMaterials,
                     const Complex background, const double localTarget,
                     const double transition, const double baseTarget,
                     const double scale) {
    std::vector<int> fields;
    double smallestSize = baseTarget;
    addDistanceField(physicalCurves, localTarget, baseTarget,
                     std::max(transition, 2.0 * localTarget), scale, fields,
                     smallestSize);
    const double outerTarget = 0.4 * baseTarget;
    addDistanceField(outerCurves, outerTarget, baseTarget, 3.0 * outerTarget, scale,
                     fields, smallestSize);

    if (!refinementSurfaces.empty()) {
        const int constant = gmsh::model::mesh::field::add("Constant");
        gmsh::model::mesh::field::setNumber(constant, "VIn", localTarget * scale);
        gmsh::model::mesh::field::setNumber(constant, "VOut", baseTarget * scale);
        gmsh::model::mesh::field::setNumbers(
            constant, "SurfacesList", numericTags(refinementSurfaces));
        fields.push_back(constant);
        smallestSize = std::min(smallestSize, localTarget);
        addDistanceField(refinementCurves, localTarget, baseTarget,
                         std::max(transition, 2.0 * localTarget), scale, fields,
                         smallestSize);
    }

    struct MaterialGroup {
        Complex material{};
        std::vector<int> surfaces;
    };
    std::vector<MaterialGroup> groupedMaterials;
    for (const auto& [surface, material] : surfaceMaterials) {
        if (std::abs(material - background) >
            64.0 * std::numeric_limits<double>::epsilon()) {
            const auto existing = std::find_if(
                groupedMaterials.begin(), groupedMaterials.end(),
                [material](const MaterialGroup& group) {
                    return group.material == material;
                });
            if (existing == groupedMaterials.end()) {
                groupedMaterials.push_back({material, {surface}});
            } else {
                existing->surfaces.push_back(surface);
            }
        }
    }
    for (const MaterialGroup& group : groupedMaterials) {
        const Complex material = group.material;
        const std::vector<int>& surfaces = group.surfaces;
        const double materialTarget =
            std::min(baseTarget, baseTarget / std::sqrt(std::abs(material)));
        if (materialTarget >=
            baseTarget * (1.0 - 64.0 * std::numeric_limits<double>::epsilon())) {
            continue;
        }
        const int constant = gmsh::model::mesh::field::add("Constant");
        gmsh::model::mesh::field::setNumber(constant, "VIn", materialTarget * scale);
        gmsh::model::mesh::field::setNumber(constant, "VOut", baseTarget * scale);
        gmsh::model::mesh::field::setNumbers(constant, "SurfacesList",
                                             numericTags(surfaces));
        fields.push_back(constant);
        smallestSize = std::min(smallestSize, materialTarget);
    }

    if (fields.size() == 1U) {
        gmsh::model::mesh::field::setAsBackgroundMesh(fields.front());
    } else if (!fields.empty()) {
        const int combined = gmsh::model::mesh::field::add("Min");
        gmsh::model::mesh::field::setNumbers(combined, "FieldsList",
                                             numericTags(fields));
        gmsh::model::mesh::field::setAsBackgroundMesh(combined);
    }
    gmsh::option::setNumber("Mesh.MeshSizeMax", baseTarget * scale);
    gmsh::option::setNumber("Mesh.MeshSizeMin",
                            std::max(0.05 * smallestSize * scale, 1.0e-12));
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0.0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0.0);
    gmsh::option::setNumber("Mesh.ElementOrder", 1.0);
    gmsh::option::setNumber("Mesh.RecombineAll", 0.0);
    gmsh::option::setNumber("Mesh.Algorithm", 6.0);
}

[[nodiscard]] std::vector<int> boundaryCurves(
    const std::unordered_map<int, Complex>& surfaceMaterials) {
    std::unordered_set<int> solveSurfaces;
    solveSurfaces.reserve(surfaceMaterials.size());
    for (const auto& [surface, material] : surfaceMaterials) {
        (void)material;
        solveSurfaces.insert(surface);
    }
    std::vector<DimTag> curveEntities;
    gmsh::model::getEntities(curveEntities, 1);
    std::vector<int> result;
    for (const auto& [dimension, curve] : curveEntities) {
        (void)dimension;
        std::vector<int> upward;
        std::vector<int> downward;
        gmsh::model::getAdjacencies(1, curve, upward, downward);
        (void)downward;
        const auto adjacent = static_cast<int>(std::count_if(
            upward.begin(), upward.end(), [&solveSurfaces](const int surface) {
                return solveSurfaces.contains(surface);
            }));
        if (adjacent == 1) {
            result.push_back(curve);
        }
    }
    return result;
}

[[nodiscard]] std::vector<int> selectedSurfaceBoundaryCurves(
    const std::unordered_set<int>& selectedSurfaces,
    const std::unordered_map<int, Complex>& surfaceMaterials) {
    if (selectedSurfaces.empty()) {
        return {};
    }
    std::vector<DimTag> curveEntities;
    gmsh::model::getEntities(curveEntities, 1);
    std::vector<int> result;
    for (const auto& [dimension, curve] : curveEntities) {
        (void)dimension;
        std::vector<int> upward;
        std::vector<int> downward;
        gmsh::model::getAdjacencies(1, curve, upward, downward);
        (void)downward;
        bool hasSelected{};
        bool hasUnselected{};
        int adjacentSolveSurfaces{};
        for (const int surface : upward) {
            if (!surfaceMaterials.contains(surface)) {
                continue;
            }
            ++adjacentSolveSurfaces;
            if (selectedSurfaces.contains(surface)) {
                hasSelected = true;
            } else {
                hasUnselected = true;
            }
        }
        if (hasSelected && (hasUnselected || adjacentSolveSurfaces == 1)) {
            result.push_back(curve);
        }
    }
    return result;
}

[[nodiscard]] bool curveOnConductor(const int curve,
                                    const CoordinateTransform& transform,
                                    const std::vector<RectConductor>& conductors,
                                    const double tolerance) {
    double xmin{};
    double ymin{};
    double zmin{};
    double xmax{};
    double ymax{};
    double zmax{};
    gmsh::model::getBoundingBox(1, curve, xmin, ymin, zmin, xmax, ymax, zmax);
    (void)zmin;
    (void)zmax;
    const Vec2 first = transform.physical(xmin, ymin);
    const Vec2 second = transform.physical(xmax, ymax);
    return std::any_of(conductors.begin(), conductors.end(),
                       [first, second, tolerance](const RectConductor& conductor) {
                           return edgeOnRectangle(first, second, conductor, tolerance);
                       });
}

[[nodiscard]] std::unordered_map<std::size_t, int> extractNodes(
    const CoordinateTransform& transform, Mesh& mesh) {
    std::vector<std::size_t> nodeTags;
    std::vector<double> coordinates;
    std::vector<double> parametric;
    gmsh::model::mesh::getNodes(nodeTags, coordinates, parametric);
    if (nodeTags.empty() || coordinates.size() != nodeTags.size() * 3U) {
        throw std::runtime_error("Gmsh returned an invalid node array");
    }
    if (nodeTags.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Gmsh mesh exceeds the supported node index range");
    }
    std::unordered_map<std::size_t, int> lookup;
    lookup.reserve(nodeTags.size());
    mesh.nodes.reserve(nodeTags.size());
    for (std::size_t index = 0; index < nodeTags.size(); ++index) {
        lookup.emplace(nodeTags[index], static_cast<int>(index));
        mesh.nodes.push_back(
            transform.physical(coordinates[3U * index], coordinates[3U * index + 1U]));
    }
    return lookup;
}

void extractTriangles(const std::unordered_map<int, Complex>& surfaceMaterials,
                      const std::unordered_map<std::size_t, int>& nodeLookup,
                      Mesh& mesh) {
    for (const auto& [surface, material] : surfaceMaterials) {
        std::vector<std::size_t> elementTags;
        std::vector<std::size_t> connectivity;
        gmsh::model::mesh::getElementsByType(2, elementTags, connectivity, surface);
        if (connectivity.size() != elementTags.size() * 3U) {
            throw std::runtime_error("Gmsh returned invalid triangle connectivity");
        }
        for (std::size_t element = 0; element < elementTags.size(); ++element) {
            Triangle triangle;
            triangle.relativePermittivity = material;
            for (std::size_t local = 0; local < 3U; ++local) {
                const auto found = nodeLookup.find(connectivity[3U * element + local]);
                if (found == nodeLookup.end()) {
                    throw std::runtime_error("Gmsh triangle references an unknown node");
                }
                triangle.nodes[local] = found->second;
            }
            const Vec2 first = mesh.nodes[static_cast<std::size_t>(triangle.nodes[0])];
            const Vec2 second = mesh.nodes[static_cast<std::size_t>(triangle.nodes[1])];
            const Vec2 third = mesh.nodes[static_cast<std::size_t>(triangle.nodes[2])];
            const double orientation = (second.x - first.x) * (third.y - first.y) -
                                       (second.y - first.y) * (third.x - first.x);
            if (orientation < 0.0) {
                std::swap(triangle.nodes[1], triangle.nodes[2]);
            }
            mesh.triangles.push_back(triangle);
            if (mesh.triangles.size() > kMaximumTriangles) {
                throw std::invalid_argument("Gmsh mesh exceeds the triangle safety limit");
            }
        }
    }
    if (mesh.triangles.empty()) {
        throw std::runtime_error("Gmsh generated no first-order triangles");
    }
}

template <typename Classifier>
void extractBoundaryEdges(const std::vector<int>& curves,
                          const std::unordered_map<std::size_t, int>& nodeLookup,
                          Mesh& mesh, std::vector<BoundaryEdge>& boundaryEdges,
                          Classifier&& classify) {
    std::unordered_set<std::uint64_t> seen;
    for (const int curve : curves) {
        std::vector<std::size_t> elementTags;
        std::vector<std::size_t> connectivity;
        gmsh::model::mesh::getElementsByType(1, elementTags, connectivity, curve);
        if (connectivity.size() != elementTags.size() * 2U) {
            throw std::runtime_error("Gmsh returned invalid boundary-line connectivity");
        }
        for (std::size_t element = 0; element < elementTags.size(); ++element) {
            const auto firstFound = nodeLookup.find(connectivity[2U * element]);
            const auto secondFound = nodeLookup.find(connectivity[2U * element + 1U]);
            if (firstFound == nodeLookup.end() || secondFound == nodeLookup.end()) {
                throw std::runtime_error("Gmsh boundary line references an unknown node");
            }
            const int first = firstFound->second;
            const int second = secondFound->second;
            if (!seen.insert(edgeKey(first, second)).second) {
                continue;
            }
            boundaryEdges.push_back(classify(first, second, mesh));
        }
    }
    if (boundaryEdges.empty()) {
        throw std::runtime_error("Gmsh generated no boundary edges");
    }
}

[[nodiscard]] GeneratedMesh generateRectangularGmsh(const Parameters& p) {
    const RectangularDefinition definition = rectangularDefinition(p);
    const double width = definition.xmax - definition.xmin;
    const double height = definition.ymax - definition.ymin;
    requirePositive(width, "geometry width");
    requirePositive(height, "geometry height");
    const CoordinateTransform transform{definition.xmin, definition.ymin,
                                        1.0 / std::max(width, height)};
    const auto addRectangle = [&transform](const Rectangle& rectangle) {
        return gmsh::model::occ::addRectangle(
            transform.x(rectangle.xmin), transform.y(rectangle.ymin), 0.0,
            (rectangle.xmax - rectangle.xmin) * transform.scale,
            (rectangle.ymax - rectangle.ymin) * transform.scale);
    };
    const Rectangle domain{definition.xmin, definition.xmax, definition.ymin,
                           definition.ymax};
    const int domainTag = addRectangle(domain);
    std::vector<DimTag> tools;
    tools.reserve(definition.materials.size() + definition.conductors.size() +
                  definition.refinementRegions.size());
    for (const MaterialRegion& material : definition.materials) {
        tools.emplace_back(2, addRectangle(material));
    }
    for (const RectConductor& conductor : definition.conductors) {
        tools.emplace_back(2, addRectangle(conductor));
    }
    for (const Rectangle& refinement : definition.refinementRegions) {
        tools.emplace_back(2, addRectangle(refinement));
    }
    std::vector<std::unordered_set<int>> materialSurfaceTags(
        definition.materials.size());
    std::unordered_set<int> conductorSurfaceTags;
    std::unordered_set<int> refinementSurfaceTags;
    if (!tools.empty()) {
        std::vector<DimTag> fragments;
        std::vector<std::vector<DimTag>> fragmentMap;
        gmsh::model::occ::fragment({{2, domainTag}}, tools, fragments, fragmentMap, -1,
                                   true, true);
        const std::size_t expectedMapSize = 1U + tools.size();
        if (fragmentMap.size() != expectedMapSize) {
            throw std::runtime_error(
                "Gmsh returned incomplete rectangular fragment provenance");
        }
        for (std::size_t material = 0; material < definition.materials.size();
             ++material) {
            for (const auto& [dimension, surface] : fragmentMap[1U + material]) {
                if (dimension == 2) {
                    materialSurfaceTags[material].insert(surface);
                }
            }
        }
        const std::size_t firstConductor = 1U + definition.materials.size();
        for (std::size_t conductor = 0; conductor < definition.conductors.size();
             ++conductor) {
            for (const auto& [dimension, surface] :
                 fragmentMap[firstConductor + conductor]) {
                if (dimension == 2) {
                    conductorSurfaceTags.insert(surface);
                }
            }
        }
        const std::size_t firstRefinement =
            firstConductor + definition.conductors.size();
        for (std::size_t refinement = 0;
             refinement < definition.refinementRegions.size(); ++refinement) {
            for (const auto& [dimension, surface] :
                 fragmentMap[firstRefinement + refinement]) {
                if (dimension == 2) {
                    refinementSurfaceTags.insert(surface);
                }
            }
        }
    }
    gmsh::model::occ::synchronize();

    std::vector<DimTag> excluded;
    excluded.reserve(conductorSurfaceTags.size());
    for (const int surface : conductorSurfaceTags) {
        excluded.emplace_back(2, surface);
    }
    if (!excluded.empty()) {
        gmsh::model::occ::remove(excluded, true);
        gmsh::model::occ::synchronize();
    }

    std::vector<DimTag> surfaces;
    gmsh::model::getEntities(surfaces, 2);
    std::unordered_map<int, Complex> surfaceMaterials;
    for (const auto& [dimension, surface] : surfaces) {
        (void)dimension;
        Complex material = definition.backgroundPermittivity;
        for (std::size_t region = 0; region < definition.materials.size(); ++region) {
            if (materialSurfaceTags[region].contains(surface)) {
                material = definition.materials[region].relativePermittivity;
            }
        }
        surfaceMaterials.emplace(surface, material);
    }
    if (surfaceMaterials.empty()) {
        throw std::runtime_error("Gmsh Boolean geometry has no dielectric surfaces");
    }

    const std::vector<int> curves = boundaryCurves(surfaceMaterials);
    std::vector<int> refinementSurfaces;
    refinementSurfaces.reserve(refinementSurfaceTags.size());
    for (const int surface : refinementSurfaceTags) {
        if (surfaceMaterials.contains(surface)) {
            refinementSurfaces.push_back(surface);
        }
    }
    const std::vector<int> refinementCurves =
        selectedSurfaceBoundaryCurves(refinementSurfaceTags, surfaceMaterials);
    const double tolerance = 1.0e-7 * std::max(width, height);
    std::vector<int> physicalCurves;
    std::vector<int> outerCurves;
    for (const int curve : curves) {
        if (curveOnConductor(curve, transform, definition.conductors, tolerance)) {
            physicalCurves.push_back(curve);
        } else {
            outerCurves.push_back(curve);
        }
    }
    const double baseTarget = p.maxElementSize / p.refinementFactor;
    applyMeshFields(physicalCurves, outerCurves, refinementSurfaces,
                    refinementCurves, surfaceMaterials,
                    definition.backgroundPermittivity, definition.localTarget,
                    definition.transitionWidth, baseTarget, transform.scale);
    gmsh::model::mesh::generate(2);

    GeneratedMesh generated;
    const auto nodeLookup = extractNodes(transform, generated.mesh);
    extractTriangles(surfaceMaterials, nodeLookup, generated.mesh);
    extractBoundaryEdges(
        curves, nodeLookup, generated.mesh, generated.boundaryEdges,
        [&definition, tolerance](const int first, const int second, const Mesh& mesh) {
            const Vec2 firstPoint = mesh.nodes[static_cast<std::size_t>(first)];
            const Vec2 secondPoint = mesh.nodes[static_cast<std::size_t>(second)];
            for (const RectConductor& conductor : definition.conductors) {
                if (edgeOnRectangle(firstPoint, secondPoint, conductor, tolerance)) {
                    return BoundaryEdge{first, second, conductor.role, conductor.id};
                }
            }
            return BoundaryEdge{first, second, BoundaryRole::Outer, -1};
        });
    return generated;
}

[[nodiscard]] GeneratedMesh generateCoaxialGmsh(const Parameters& p) {
    const double extent = p.outerRadius;
    const CoordinateTransform transform{-extent, -extent, 1.0 / (2.0 * extent)};
    const double centerX = transform.x(0.0);
    const double centerY = transform.y(0.0);
    const int outer = gmsh::model::occ::addDisk(
        centerX, centerY, 0.0, p.outerRadius * transform.scale,
        p.outerRadius * transform.scale);
    const int inner = gmsh::model::occ::addDisk(
        centerX, centerY, 0.0, p.innerRadius * transform.scale,
        p.innerRadius * transform.scale);
    std::vector<DimTag> cutSurfaces;
    std::vector<std::vector<DimTag>> cutMap;
    gmsh::model::occ::cut({{2, outer}}, {{2, inner}}, cutSurfaces, cutMap, -1, true,
                          true);
    gmsh::model::occ::synchronize();
    std::vector<DimTag> surfaces;
    gmsh::model::getEntities(surfaces, 2);
    std::unordered_map<int, Complex> surfaceMaterials;
    for (const auto& [dimension, surface] : surfaces) {
        (void)dimension;
        surfaceMaterials.emplace(surface, relativePermittivity(p));
    }
    if (surfaceMaterials.size() != 1U) {
        throw std::runtime_error("Gmsh coaxial Boolean cut did not produce one annulus");
    }
    const std::vector<int> curves = boundaryCurves(surfaceMaterials);
    const double baseTarget = p.maxElementSize / p.refinementFactor;
    const double localTarget =
        std::min(baseTarget,
                 std::min(p.innerRadius, p.outerRadius - p.innerRadius) /
                     (8.0 * p.refinementFactor));
    std::vector<int> refinementSurfaces;
    refinementSurfaces.reserve(surfaceMaterials.size());
    for (const auto& [surface, material] : surfaceMaterials) {
        (void)material;
        refinementSurfaces.push_back(surface);
    }
    applyMeshFields(curves, {}, refinementSurfaces, curves, surfaceMaterials,
                    relativePermittivity(p), localTarget,
                    0.35 * (p.outerRadius - p.innerRadius), baseTarget,
                    transform.scale);
    gmsh::model::mesh::generate(2);

    GeneratedMesh generated;
    const auto nodeLookup = extractNodes(transform, generated.mesh);
    extractTriangles(surfaceMaterials, nodeLookup, generated.mesh);
    extractBoundaryEdges(
        curves, nodeLookup, generated.mesh, generated.boundaryEdges,
        [&p](const int first, const int second, const Mesh& mesh) {
            const Vec2 firstPoint = mesh.nodes[static_cast<std::size_t>(first)];
            const Vec2 secondPoint = mesh.nodes[static_cast<std::size_t>(second)];
            const double radius = 0.5 * (std::hypot(firstPoint.x, firstPoint.y) +
                                         std::hypot(secondPoint.x, secondPoint.y));
            if (std::abs(radius - p.innerRadius) <
                std::abs(radius - p.outerRadius)) {
                return BoundaryEdge{first, second, BoundaryRole::Signal, 0};
            }
            return BoundaryEdge{first, second, BoundaryRole::Reference, 1};
        });
    return generated;
}

[[nodiscard]] GeneratedMesh generateMeshWithGmsh(const Parameters& p) {
    std::scoped_lock lock(gGmshMutex);
    try {
        GmshSession session;
        if (p.type == LineType::Coaxial) {
            return generateCoaxialGmsh(p);
        }
        return generateRectangularGmsh(p);
    } catch (const std::string& message) {
        throw std::runtime_error("Gmsh failed: " + message);
    } catch (const char* const message) {
        throw std::runtime_error(std::string("Gmsh failed: ") + message);
    }
}

[[nodiscard]] double twiceSignedArea(const Vec2 first, const Vec2 second,
                                     const Vec2 third) {
    return (second.x - first.x) * (third.y - first.y) -
           (second.y - first.y) * (third.x - first.x);
}

[[nodiscard]] std::array<Vec2, 3> basisGradients(const Vec2 first, const Vec2 second,
                                                 const Vec2 third,
                                                 const double twiceArea) {
    return {{{(second.y - third.y) / twiceArea,
              (third.x - second.x) / twiceArea},
             {(third.y - first.y) / twiceArea,
              (first.x - third.x) / twiceArea},
             {(first.y - second.y) / twiceArea,
              (second.x - first.x) / twiceArea}}};
}

struct AssembledSystems {
    SparseComplex dielectric;
    SparseComplex vacuum;
};

[[nodiscard]] AssembledSystems assembleSystems(const Mesh& mesh) {
    const int count = static_cast<int>(mesh.nodes.size());
    std::vector<Triplet> dielectricTriplets;
    std::vector<Triplet> vacuumTriplets;
    dielectricTriplets.reserve(mesh.triangles.size() * 9U);
    vacuumTriplets.reserve(mesh.triangles.size() * 9U);
    for (const Triangle& triangle : mesh.triangles) {
        const Vec2 first = mesh.nodes[static_cast<std::size_t>(triangle.nodes[0])];
        const Vec2 second = mesh.nodes[static_cast<std::size_t>(triangle.nodes[1])];
        const Vec2 third = mesh.nodes[static_cast<std::size_t>(triangle.nodes[2])];
        const double twiceArea = twiceSignedArea(first, second, third);
        if (!std::isfinite(twiceArea) || twiceArea <= 0.0) {
            throw std::runtime_error("mesh contains a non-positive triangle");
        }
        if (!finite(triangle.relativePermittivity) ||
            triangle.relativePermittivity.real() <= 0.0) {
            throw std::runtime_error("mesh contains invalid relative permittivity");
        }
        const double area = 0.5 * twiceArea;
        const auto gradients = basisGradients(first, second, third, twiceArea);
        for (std::size_t row = 0; row < 3U; ++row) {
            for (std::size_t column = 0; column < 3U; ++column) {
                const double dot = gradients[row].x * gradients[column].x +
                                   gradients[row].y * gradients[column].y;
                const double vacuumValue = kEpsilon0 * area * dot;
                dielectricTriplets.emplace_back(
                    triangle.nodes[row], triangle.nodes[column],
                    triangle.relativePermittivity * vacuumValue);
                vacuumTriplets.emplace_back(triangle.nodes[row], triangle.nodes[column],
                                             Complex{vacuumValue, 0.0});
            }
        }
    }
    AssembledSystems systems{SparseComplex(count, count), SparseComplex(count, count)};
    systems.dielectric.setFromTriplets(dielectricTriplets.begin(),
                                       dielectricTriplets.end());
    systems.vacuum.setFromTriplets(vacuumTriplets.begin(), vacuumTriplets.end());
    systems.dielectric.makeCompressed();
    systems.vacuum.makeCompressed();
    return systems;
}

struct BoundaryNodes {
    std::vector<int> signal;
    std::vector<int> zero;
    std::vector<bool> isSignal;
    std::vector<bool> isZero;
};

[[nodiscard]] BoundaryNodes collectBoundaryNodes(
    const std::size_t nodeCount, const std::vector<BoundaryEdge>& boundaryEdges) {
    BoundaryNodes nodes;
    nodes.isSignal.assign(nodeCount, false);
    nodes.isZero.assign(nodeCount, false);
    for (const BoundaryEdge& edge : boundaryEdges) {
        for (const int endpoint : {edge.first, edge.second}) {
            if (endpoint < 0 || static_cast<std::size_t>(endpoint) >= nodeCount) {
                throw std::runtime_error("boundary edge references a node outside the mesh");
            }
            if (edge.role == BoundaryRole::Signal) {
                nodes.isSignal[static_cast<std::size_t>(endpoint)] = true;
            } else {
                nodes.isZero[static_cast<std::size_t>(endpoint)] = true;
            }
        }
    }
    for (std::size_t index = 0; index < nodeCount; ++index) {
        if (nodes.isSignal[index] && nodes.isZero[index]) {
            throw std::invalid_argument(
                "signal and zero-volt boundaries share a mesh node");
        }
        if (nodes.isSignal[index]) {
            nodes.signal.push_back(static_cast<int>(index));
        }
        if (nodes.isZero[index]) {
            nodes.zero.push_back(static_cast<int>(index));
        }
    }
    if (nodes.signal.empty() || nodes.zero.empty()) {
        throw std::invalid_argument(
            "electrostatic solve requires nonempty signal and reference boundaries");
    }
    return nodes;
}

[[nodiscard]] DirichletSolution solveDirichlet(const SparseComplex& stiffness,
                                               const BoundaryNodes& boundaries) {
    if (stiffness.rows() > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("stiffness matrix exceeds the supported index range");
    }
    const int count = static_cast<int>(stiffness.rows());
    std::vector<int> freeMap(static_cast<std::size_t>(count), -1);
    int freeCount = 0;
    VectorComplex potential = VectorComplex::Zero(count);
    for (int index = 0; index < count; ++index) {
        const auto offset = static_cast<std::size_t>(index);
        if (boundaries.isSignal[offset]) {
            potential[index] = Complex{1.0, 0.0};
        } else if (!boundaries.isZero[offset]) {
            freeMap[offset] = freeCount++;
        }
    }
    if (freeCount == 0) {
        throw std::invalid_argument("conductor constraints leave no free mesh nodes");
    }
    std::vector<Triplet> reducedTriplets;
    reducedTriplets.reserve(static_cast<std::size_t>(stiffness.nonZeros()));
    VectorComplex rhs = VectorComplex::Zero(freeCount);
    for (int column = 0; column < stiffness.outerSize(); ++column) {
        for (SparseComplex::InnerIterator item(stiffness, column); item; ++item) {
            const int reducedRow = freeMap[static_cast<std::size_t>(item.row())];
            if (reducedRow < 0) {
                continue;
            }
            const int reducedColumn = freeMap[static_cast<std::size_t>(item.col())];
            if (reducedColumn >= 0) {
                reducedTriplets.emplace_back(reducedRow, reducedColumn, item.value());
            } else {
                rhs[reducedRow] -= item.value() * potential[item.col()];
            }
        }
    }
    SparseComplex reduced(freeCount, freeCount);
    reduced.setFromTriplets(reducedTriplets.begin(), reducedTriplets.end());
    reduced.makeCompressed();
    const auto factorBegin = Clock::now();
    Eigen::SparseLU<SparseComplex, Eigen::COLAMDOrdering<int>> factorization;
    factorization.analyzePattern(reduced);
    factorization.factorize(reduced);
    if (factorization.info() != Eigen::Success) {
        throw std::runtime_error(
            "scalar electrostatic stiffness factorization is singular");
    }
    const VectorComplex freePotential = factorization.solve(rhs);
    const auto factorEnd = Clock::now();
    if (factorization.info() != Eigen::Success || !freePotential.allFinite()) {
        throw std::runtime_error("scalar electrostatic solve failed");
    }
    for (int index = 0; index < count; ++index) {
        const int reducedIndex = freeMap[static_cast<std::size_t>(index)];
        if (reducedIndex >= 0) {
            potential[index] = freePotential[reducedIndex];
        }
    }
    const VectorComplex reaction = stiffness * potential;
    double freeNormSquared = 0.0;
    double fixedNormSquared = 0.0;
    for (int index = 0; index < count; ++index) {
        const double magnitudeSquared = std::norm(reaction[index]);
        if (freeMap[static_cast<std::size_t>(index)] >= 0) {
            freeNormSquared += magnitudeSquared;
        } else {
            fixedNormSquared += magnitudeSquared;
        }
    }
    const double residual =
        std::sqrt(freeNormSquared) /
        std::max(std::sqrt(fixedNormSquared), std::numeric_limits<double>::min());
    if (!std::isfinite(residual) || residual > 1.0e-7) {
        throw std::runtime_error("scalar electrostatic residual is too large");
    }
    return {potential, reaction, residual,
            elapsedMilliseconds(factorBegin, factorEnd)};
}

[[nodiscard]] Complex sumAt(const VectorComplex& vector,
                            const std::vector<int>& indices) {
    Complex result{};
    for (const int index : indices) {
        result += vector[index];
    }
    return result;
}

[[nodiscard]] Complex positiveRealRoot(const Complex value, const char* const name) {
    Complex root = std::sqrt(value);
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                             std::max(1.0, std::abs(root));
    if (root.real() < -tolerance ||
        (std::abs(root.real()) <= tolerance && root.imag() > tolerance)) {
        root = -root;
    }
    if (!finite(root) || root.real() <= 0.0) {
        throw std::runtime_error(std::string(name) +
                                 " has no physical positive-real branch");
    }
    return root;
}

[[nodiscard]] double conductorGeometryFactor(
    const Mesh& mesh, const std::vector<BoundaryEdge>& edges,
    const VectorComplex& vacuumReaction, const double vacuumCapacitance,
    double& factorizationMilliseconds) {
    int maximumId = -1;
    for (const BoundaryEdge& edge : edges) {
        maximumId = std::max(maximumId, edge.conductor);
    }
    double total = 0.0;
    for (int conductor = 0; conductor <= maximumId; ++conductor) {
        std::vector<const BoundaryEdge*> selected;
        for (const BoundaryEdge& edge : edges) {
            if (edge.conductor == conductor && edge.role != BoundaryRole::Outer) {
                selected.push_back(&edge);
            }
        }
        if (selected.empty()) {
            continue;
        }
        std::vector<int> localMap(mesh.nodes.size(), -1);
        std::vector<int> globalNodes;
        for (const BoundaryEdge* edge : selected) {
            for (const int node : {edge->first, edge->second}) {
                int& mapped = localMap[static_cast<std::size_t>(node)];
                if (mapped < 0) {
                    mapped = static_cast<int>(globalNodes.size());
                    globalNodes.push_back(node);
                }
            }
        }
        const int count = static_cast<int>(globalNodes.size());
        std::vector<Triplet> massTriplets;
        massTriplets.reserve(selected.size() * 4U);
        for (const BoundaryEdge* edge : selected) {
            const Vec2 first = mesh.nodes[static_cast<std::size_t>(edge->first)];
            const Vec2 second = mesh.nodes[static_cast<std::size_t>(edge->second)];
            const double length = std::hypot(second.x - first.x, second.y - first.y);
            requirePositive(length, "conductor boundary edge length");
            const int firstLocal = localMap[static_cast<std::size_t>(edge->first)];
            const int secondLocal = localMap[static_cast<std::size_t>(edge->second)];
            massTriplets.emplace_back(firstLocal, firstLocal,
                                      Complex{length / 3.0, 0.0});
            massTriplets.emplace_back(secondLocal, secondLocal,
                                      Complex{length / 3.0, 0.0});
            massTriplets.emplace_back(firstLocal, secondLocal,
                                      Complex{length / 6.0, 0.0});
            massTriplets.emplace_back(secondLocal, firstLocal,
                                      Complex{length / 6.0, 0.0});
        }
        SparseComplex mass(count, count);
        mass.setFromTriplets(massTriplets.begin(), massTriplets.end());
        mass.makeCompressed();
        VectorComplex rhs(count);
        for (int local = 0; local < count; ++local) {
            rhs[local] = vacuumReaction[globalNodes[static_cast<std::size_t>(local)]];
        }
        const auto factorBegin = Clock::now();
        Eigen::SparseLU<SparseComplex, Eigen::COLAMDOrdering<int>> factorization;
        factorization.compute(mass);
        if (factorization.info() != Eigen::Success) {
            throw std::runtime_error("conductor boundary mass projection is singular");
        }
        const VectorComplex density = factorization.solve(rhs);
        const auto factorEnd = Clock::now();
        factorizationMilliseconds += elapsedMilliseconds(factorBegin, factorEnd);
        if (factorization.info() != Eigen::Success || !density.allFinite()) {
            throw std::runtime_error("conductor boundary mass projection failed");
        }
        const Complex norm = density.dot(mass * density);
        const double imaginaryTolerance =
            512.0 * std::numeric_limits<double>::epsilon() *
            std::max(std::abs(norm), std::numeric_limits<double>::min());
        if (!finite(norm) || norm.real() <= 0.0 ||
            std::abs(norm.imag()) > imaginaryTolerance) {
            throw std::runtime_error("projected conductor current norm is invalid");
        }
        total += norm.real() / (vacuumCapacitance * vacuumCapacitance);
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error("total conductor geometry factor is not positive");
    }
    return total;
}

[[nodiscard]] std::vector<FieldSample> reconstructFields(
    const Mesh& mesh, const VectorComplex& electricPotential,
    const VectorComplex& vacuumPotential, std::vector<FieldVector>& vacuumElectric) {
    std::vector<FieldSample> samples;
    samples.reserve(mesh.triangles.size());
    vacuumElectric.reserve(mesh.triangles.size());
    for (const Triangle& triangle : mesh.triangles) {
        const Vec2 first = mesh.nodes[static_cast<std::size_t>(triangle.nodes[0])];
        const Vec2 second = mesh.nodes[static_cast<std::size_t>(triangle.nodes[1])];
        const Vec2 third = mesh.nodes[static_cast<std::size_t>(triangle.nodes[2])];
        const double twiceArea = twiceSignedArea(first, second, third);
        if (!(twiceArea > 0.0)) {
            throw std::runtime_error("field reconstruction found a non-positive triangle");
        }
        const auto gradients = basisGradients(first, second, third, twiceArea);
        FieldVector electric{};
        FieldVector vacuum{};
        for (std::size_t local = 0; local < 3U; ++local) {
            const int node = triangle.nodes[local];
            electric.x -= electricPotential[node] * gradients[local].x;
            electric.y -= electricPotential[node] * gradients[local].y;
            vacuum.x -= vacuumPotential[node] * gradients[local].x;
            vacuum.y -= vacuumPotential[node] * gradients[local].y;
        }
        if (!finite(electric.x) || !finite(electric.y) || !finite(vacuum.x) ||
            !finite(vacuum.y)) {
            throw std::runtime_error("field reconstruction produced non-finite values");
        }
        samples.push_back({{(first.x + second.x + third.x) / 3.0,
                            (first.y + second.y + third.y) / 3.0},
                           electric,
                           {},
                           0.5 * twiceArea,
                           triangle.relativePermittivity});
        vacuumElectric.push_back(vacuum);
    }
    return samples;
}

[[nodiscard]] std::vector<Complex> toStandardVector(const VectorComplex& vector) {
    return {vector.data(), vector.data() + vector.size()};
}

}  // namespace

Parameters defaultParameters(const LineType type) {
    Parameters parameters;
    parameters.type = type;
    parameters.frequencyHz = 10.0e9;
    parameters.maxElementSize = 1.0e-3;
    parameters.refinementFactor = 1.0;
    parameters.metalConductivity.reset();
    switch (type) {
        case LineType::Coaxial:
            parameters.innerRadius = 0.50e-3;
            parameters.outerRadius = 1.67e-3;
            parameters.outerConductorThickness = 0.15e-3;
            parameters.epsilonR = 2.10;
            parameters.lossTangent = 2.0e-4;
            break;
        case LineType::Microstrip:
            parameters.traceWidth = 3.00e-3;
            parameters.substrateHeight = 1.524e-3;
            parameters.conductorThickness = 35.0e-6;
            parameters.epsilonR = 3.55;
            parameters.lossTangent = 2.7e-3;
            parameters.domainPaddingFactor = 1.0;
            break;
        case LineType::Stripline:
            parameters.traceWidth = 0.80e-3;
            parameters.groundSpacing = 1.524e-3;
            parameters.conductorThickness = 35.0e-6;
            parameters.epsilonR = 3.55;
            parameters.lossTangent = 2.7e-3;
            parameters.domainPaddingFactor = 1.0;
            break;
        case LineType::CoplanarWaveguide:
            parameters.centerWidth = 0.60e-3;
            parameters.gap = 0.25e-3;
            parameters.groundWidth = 1.50e-3;
            parameters.substrateHeight = 0.80e-3;
            parameters.conductorThickness = 35.0e-6;
            parameters.epsilonR = 3.55;
            parameters.lossTangent = 2.7e-3;
            parameters.domainPaddingFactor = 1.0;
            break;
    }
    return parameters;
}

Result solve(const Parameters& parameters) {
    validate(parameters);
    Result result;
    result.parameters = parameters;
    const auto meshBegin = Clock::now();
    GeneratedMesh generated = generateMeshWithGmsh(parameters);
    const auto meshEnd = Clock::now();
    result.meshMilliseconds = elapsedMilliseconds(meshBegin, meshEnd);
    result.mesh = std::move(generated.mesh);

    const auto solveBegin = Clock::now();
    const auto assemblyBegin = Clock::now();
    AssembledSystems systems = assembleSystems(result.mesh);
    const auto assemblyEnd = Clock::now();
    result.assemblyMilliseconds = elapsedMilliseconds(assemblyBegin, assemblyEnd);
    const BoundaryNodes boundaries =
        collectBoundaryNodes(result.mesh.nodes.size(), generated.boundaryEdges);
    const DirichletSolution electric = solveDirichlet(systems.dielectric, boundaries);
    const DirichletSolution vacuum = solveDirichlet(systems.vacuum, boundaries);
    result.factorizationMilliseconds = electric.factorizationMilliseconds +
                                       vacuum.factorizationMilliseconds;
    result.materialResidual = electric.residual;
    result.vacuumResidual = vacuum.residual;

    const Complex capacitance = sumAt(electric.reaction, boundaries.signal);
    const Complex vacuumCapacitanceComplex = sumAt(vacuum.reaction, boundaries.signal);
    if (!finite(capacitance) || capacitance.real() <= 0.0) {
        throw std::runtime_error("extracted capacitance is not positive-real");
    }
    if (!finite(vacuumCapacitanceComplex) ||
        vacuumCapacitanceComplex.real() <= 0.0) {
        throw std::runtime_error("extracted vacuum capacitance is not positive");
    }
    const double vacuumCapacitance = vacuumCapacitanceComplex.real();
    result.capacitancePerLength = capacitance;
    result.vacuumCapacitancePerLength = vacuumCapacitance;
    result.externalInductancePerLength =
        1.0 / (kSpeedOfLight * kSpeedOfLight * vacuumCapacitance);

    const double omega = 2.0 * kPi * parameters.frequencyHz;
    const double rawConductance = -omega * capacitance.imag();
    const double conductanceTolerance =
        256.0 * std::numeric_limits<double>::epsilon() * omega *
        std::max(std::abs(capacitance), std::numeric_limits<double>::min());
    if (rawConductance < -conductanceTolerance) {
        throw std::runtime_error("dielectric capacitance implies active shunt loss");
    }
    result.conductancePerLength = std::max(0.0, rawConductance);
    result.resistancePerLength = 0.0;
    result.surfaceResistance = 0.0;
    result.conductorGeometryFactorPerLength = 0.0;
    if (parameters.metalConductivity.has_value()) {
        if (parameters.type == LineType::Coaxial) {
            result.conductorGeometryFactorPerLength =
                (1.0 / (2.0 * kPi)) *
                (1.0 / parameters.innerRadius + 1.0 / parameters.outerRadius);
        } else {
            result.conductorGeometryFactorPerLength = conductorGeometryFactor(
                result.mesh, generated.boundaryEdges, vacuum.reaction,
                vacuumCapacitance, result.factorizationMilliseconds);
        }
        result.surfaceResistance = std::sqrt(
            kPi * parameters.frequencyHz * kMu0 / *parameters.metalConductivity);
        result.resistancePerLength =
            result.surfaceResistance * result.conductorGeometryFactorPerLength;
    }
    result.inductancePerLength = result.externalInductancePerLength +
                                 result.resistancePerLength / omega;

    const Complex seriesImpedance{result.resistancePerLength,
                                  omega * result.inductancePerLength};
    const Complex shuntAdmittance{result.conductancePerLength,
                                  omega * capacitance.real()};
    if (parameters.metalConductivity.has_value()) {
        result.beta = positiveRealRoot(-seriesImpedance * shuntAdmittance,
                                       "complex phase constant");
        result.neff = result.beta * kSpeedOfLight / omega;
        result.characteristicImpedance = positiveRealRoot(
            seriesImpedance / shuntAdmittance, "characteristic impedance");
    } else {
        result.neff = positiveRealRoot(capacitance / vacuumCapacitance,
                                       "effective index");
        result.characteristicImpedance = positiveRealRoot(
            result.inductancePerLength / capacitance,
            "characteristic impedance");
        result.beta = omega * result.neff / kSpeedOfLight;
    }
    const double passiveTolerance =
        512.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(result.beta));
    if (result.beta.imag() > passiveTolerance ||
        result.neff.imag() > passiveTolerance) {
        throw std::runtime_error("RLGC extraction selected an active propagation branch");
    }

    result.voltage = {1.0, 0.0};
    result.current = result.voltage / result.characteristicImpedance;
    std::vector<FieldVector> vacuumElectric;
    result.samples = reconstructFields(result.mesh, electric.potential,
                                       vacuum.potential, vacuumElectric);
    const Complex magneticScale = result.current * kEpsilon0 / vacuumCapacitance;
    Complex poyntingIntegral{};
    double magneticNormIntegral = 0.0;
    for (std::size_t index = 0; index < result.samples.size(); ++index) {
        FieldSample& sample = result.samples[index];
        sample.magnetic.x = -magneticScale * vacuumElectric[index].y;
        sample.magnetic.y = magneticScale * vacuumElectric[index].x;
        const Complex poynting = sample.electric.x * std::conj(sample.magnetic.y) -
                                 sample.electric.y * std::conj(sample.magnetic.x);
        poyntingIntegral += sample.area * poynting;
        magneticNormIntegral +=
            sample.area *
            (std::norm(sample.magnetic.x) + std::norm(sample.magnetic.y));
    }
    if (!std::isfinite(magneticNormIntegral) || magneticNormIntegral <= 0.0) {
        throw std::runtime_error("reconstructed magnetic field has zero norm");
    }
    result.power = 0.5 * poyntingIntegral;
    result.waveImpedance = poyntingIntegral / magneticNormIntegral;
    if (!finite(result.power) || !finite(result.waveImpedance) ||
        result.waveImpedance.real() <= 0.0) {
        throw std::runtime_error("field power or integrated wave impedance is invalid");
    }
    result.electricPotential = toStandardVector(electric.potential);
    result.vacuumPotential = toStandardVector(vacuum.potential);
    const auto solveEnd = Clock::now();
    result.solveMilliseconds = elapsedMilliseconds(solveBegin, solveEnd);
    return result;
}

}  // namespace tl
