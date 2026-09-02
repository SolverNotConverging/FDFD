#pragma once

#include "model.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace tl {

enum class FieldViewMode { Focused, FullDomain };

struct FieldViewBounds {
    double xMin{};
    double xMax{};
    double yMin{};
    double yMax{};
};

[[nodiscard]] std::optional<FieldViewBounds> fieldMeshBounds(const Result& result);

[[nodiscard]] std::optional<FieldViewBounds> fieldDisplayBounds(
    const Result& result,
    FieldViewMode mode
);

[[nodiscard]] bool fieldViewIsCropped(
    const FieldViewBounds& fullBounds,
    const FieldViewBounds& displayBounds
);

[[nodiscard]] bool fieldViewContains(
    const FieldViewBounds& bounds,
    const Vec2& point
);

[[nodiscard]] FieldViewBounds zoomFieldView(
    const FieldViewBounds& bounds,
    const FieldViewBounds& limits,
    const Vec2& anchor,
    double scale
);

[[nodiscard]] FieldViewBounds panFieldView(
    const FieldViewBounds& bounds,
    const FieldViewBounds& limits,
    double xOffset,
    double yOffset
);

[[nodiscard]] std::vector<std::size_t> visibleFieldSampleIndices(
    const Result& result,
    const FieldViewBounds& bounds
);

} // namespace tl
