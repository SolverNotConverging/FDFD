#include "field_view.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tl {
namespace {

[[nodiscard]] bool validBounds(const FieldViewBounds& bounds) {
    return std::isfinite(bounds.xMin) && std::isfinite(bounds.xMax)
        && std::isfinite(bounds.yMin) && std::isfinite(bounds.yMax)
        && bounds.xMax > bounds.xMin && bounds.yMax > bounds.yMin;
}

[[nodiscard]] double focusedRatio(const Parameters& parameters) {
    const auto padding = parameters.domainPaddingFactor;
    if (!(padding > 0.0) || !std::isfinite(padding)) {
        return 1.0;
    }
    return std::min(padding, 1.0) / padding;
}

[[nodiscard]] double scaleRemoteBoundary(
    const double fullBoundary,
    const double geometryBoundary,
    const double ratio
) {
    return geometryBoundary + ratio * (fullBoundary - geometryBoundary);
}

[[nodiscard]] FieldViewBounds focusedBounds(
    const Result& result,
    const FieldViewBounds& full
) {
    const auto& parameters = result.parameters;
    const auto ratio = focusedRatio(parameters);
    if (ratio >= 1.0 || parameters.type == LineType::Coaxial) {
        return full;
    }

    auto focused = full;
    switch (parameters.type) {
    case LineType::Coaxial:
        return full;
    case LineType::Microstrip: {
        const auto signalEdge = 0.5 * parameters.traceWidth;
        const auto topOfSignal =
            parameters.substrateHeight + parameters.conductorThickness;
        focused.xMin = scaleRemoteBoundary(full.xMin, -signalEdge, ratio);
        focused.xMax = scaleRemoteBoundary(full.xMax, signalEdge, ratio);
        focused.yMax = scaleRemoteBoundary(full.yMax, topOfSignal, ratio);
        break;
    }
    case LineType::Stripline: {
        const auto signalEdge = 0.5 * parameters.traceWidth;
        focused.xMin = scaleRemoteBoundary(full.xMin, -signalEdge, ratio);
        focused.xMax = scaleRemoteBoundary(full.xMax, signalEdge, ratio);
        break;
    }
    case LineType::CoplanarWaveguide: {
        const auto signalEdge = 0.5 * parameters.centerWidth;
        const auto groundInnerEdge = signalEdge + parameters.gap;
        const auto metalEdge = groundInnerEdge + parameters.groundWidth;
        focused.xMin = scaleRemoteBoundary(full.xMin, -metalEdge, ratio);
        focused.xMax = scaleRemoteBoundary(full.xMax, metalEdge, ratio);
        focused.yMin = scaleRemoteBoundary(
            full.yMin, -parameters.substrateHeight, ratio
        );
        focused.yMax = scaleRemoteBoundary(
            full.yMax, parameters.conductorThickness, ratio
        );
        break;
    }
    }

    focused.xMin = std::clamp(focused.xMin, full.xMin, full.xMax);
    focused.xMax = std::clamp(focused.xMax, full.xMin, full.xMax);
    focused.yMin = std::clamp(focused.yMin, full.yMin, full.yMax);
    focused.yMax = std::clamp(focused.yMax, full.yMin, full.yMax);
    return validBounds(focused) ? focused : full;
}

} // namespace

std::optional<FieldViewBounds> fieldMeshBounds(const Result& result) {
    if (result.mesh.nodes.empty()) {
        return std::nullopt;
    }
    FieldViewBounds bounds{
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const auto& node : result.mesh.nodes) {
        bounds.xMin = std::min(bounds.xMin, node.x);
        bounds.xMax = std::max(bounds.xMax, node.x);
        bounds.yMin = std::min(bounds.yMin, node.y);
        bounds.yMax = std::max(bounds.yMax, node.y);
    }
    if (!std::isfinite(bounds.xMin) || !std::isfinite(bounds.xMax)
        || !std::isfinite(bounds.yMin) || !std::isfinite(bounds.yMax)) {
        return std::nullopt;
    }
    const auto xScale = std::max({std::abs(bounds.xMin), std::abs(bounds.xMax), 1.0e-12});
    const auto yScale = std::max({std::abs(bounds.yMin), std::abs(bounds.yMax), 1.0e-12});
    if (bounds.xMax <= bounds.xMin) {
        bounds.xMin -= 0.5 * xScale;
        bounds.xMax += 0.5 * xScale;
    }
    if (bounds.yMax <= bounds.yMin) {
        bounds.yMin -= 0.5 * yScale;
        bounds.yMax += 0.5 * yScale;
    }
    return bounds;
}

std::optional<FieldViewBounds> fieldDisplayBounds(
    const Result& result,
    const FieldViewMode mode
) {
    const auto full = fieldMeshBounds(result);
    if (!full || mode == FieldViewMode::FullDomain) {
        return full;
    }
    return focusedBounds(result, *full);
}

bool fieldViewIsCropped(
    const FieldViewBounds& fullBounds,
    const FieldViewBounds& displayBounds
) {
    const auto scale = std::max({
        std::abs(fullBounds.xMin), std::abs(fullBounds.xMax),
        std::abs(fullBounds.yMin), std::abs(fullBounds.yMax), 1.0e-12,
    });
    const auto tolerance = 1.0e-12 * scale;
    return std::abs(fullBounds.xMin - displayBounds.xMin) > tolerance
        || std::abs(fullBounds.xMax - displayBounds.xMax) > tolerance
        || std::abs(fullBounds.yMin - displayBounds.yMin) > tolerance
        || std::abs(fullBounds.yMax - displayBounds.yMax) > tolerance;
}

bool fieldViewContains(const FieldViewBounds& bounds, const Vec2& point) {
    return point.x >= bounds.xMin && point.x <= bounds.xMax
        && point.y >= bounds.yMin && point.y <= bounds.yMax;
}

std::vector<std::size_t> visibleFieldSampleIndices(
    const Result& result,
    const FieldViewBounds& bounds
) {
    std::vector<std::size_t> indices;
    indices.reserve(result.samples.size());
    for (std::size_t index = 0; index < result.samples.size(); ++index) {
        if (fieldViewContains(bounds, result.samples[index].position)) {
            indices.push_back(index);
        }
    }
    return indices;
}

} // namespace tl
