#include "field_view.hpp"
#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        throw TestFailure(std::string(message));
    }
}

void checkNear(
    const double actual,
    const double expected,
    const std::string_view message
) {
    const auto scale = std::max({std::abs(actual), std::abs(expected), 1.0e-12});
    check(std::abs(actual - expected) <= 1.0e-12 * scale, message);
}

tl::FieldViewBounds rectangularBounds(const tl::Parameters& parameters) {
    switch (parameters.type) {
    case tl::LineType::Microstrip: {
        const auto clearance = parameters.domainPaddingFactor
            * std::max(1.5 * parameters.traceWidth,
                       3.0 * parameters.substrateHeight);
        const auto halfWidth = 0.5 * parameters.traceWidth + clearance;
        const auto airHeight = parameters.domainPaddingFactor
            * std::max(3.0 * parameters.substrateHeight,
                       1.5 * parameters.traceWidth);
        return {
            -halfWidth,
            halfWidth,
            -parameters.conductorThickness,
            parameters.substrateHeight + parameters.conductorThickness
                + airHeight,
        };
    }
    case tl::LineType::Stripline: {
        const auto clearance = parameters.domainPaddingFactor
            * std::max(3.0 * parameters.groundSpacing,
                       2.0 * parameters.traceWidth);
        const auto halfWidth = 0.5 * parameters.traceWidth + clearance;
        const auto halfSpacing = 0.5 * parameters.groundSpacing;
        return {
            -halfWidth,
            halfWidth,
            -halfSpacing - parameters.conductorThickness,
            halfSpacing + parameters.conductorThickness,
        };
    }
    case tl::LineType::CoplanarWaveguide: {
        const auto metalEdge = 0.5 * parameters.centerWidth + parameters.gap
            + parameters.groundWidth;
        const auto sideClearance = parameters.domainPaddingFactor
            * std::max(2.0 * parameters.substrateHeight, 0.75 * metalEdge);
        const auto verticalClearance = parameters.domainPaddingFactor
            * std::max(2.0 * parameters.substrateHeight, metalEdge);
        return {
            -metalEdge - sideClearance,
            metalEdge + sideClearance,
            -parameters.substrateHeight - verticalClearance,
            parameters.conductorThickness + verticalClearance,
        };
    }
    case tl::LineType::Coaxial: {
        const auto radius = parameters.outerRadius
            + parameters.outerConductorThickness;
        return {-radius, radius, -radius, radius};
    }
    }
    throw TestFailure("unsupported line type");
}

tl::Result makeResult(const tl::Parameters& parameters) {
    tl::Result result;
    result.parameters = parameters;
    const auto bounds = rectangularBounds(parameters);
    result.mesh.nodes = {
        {bounds.xMin, bounds.yMin},
        {bounds.xMax, bounds.yMin},
        {bounds.xMax, bounds.yMax},
        {bounds.xMin, bounds.yMax},
    };
    result.mesh.triangles = {{{0, 1, 2}}, {{0, 2, 3}}};
    return result;
}

tl::FieldViewBounds requireBounds(
    const tl::Result& result,
    const tl::FieldViewMode mode
) {
    const auto bounds = tl::fieldDisplayBounds(result, mode);
    check(bounds.has_value(), "field view bounds must exist");
    return *bounds;
}

void testDefaultFocusedBounds() {
    {
        const auto parameters = tl::defaultParameters(tl::LineType::Microstrip);
        const auto result = makeResult(parameters);
        const auto focused = requireBounds(result, tl::FieldViewMode::Focused);
        checkNear(focused.xMin, -6.072e-3, "microstrip focused x minimum");
        checkNear(focused.xMax, 6.072e-3, "microstrip focused x maximum");
        checkNear(focused.yMin, -35.0e-6, "microstrip focused y minimum");
        checkNear(focused.yMax, 6.131e-3, "microstrip focused y maximum");
    }
    {
        const auto parameters = tl::defaultParameters(tl::LineType::Stripline);
        const auto result = makeResult(parameters);
        const auto full = requireBounds(result, tl::FieldViewMode::FullDomain);
        const auto focused = requireBounds(result, tl::FieldViewMode::Focused);
        checkNear(focused.xMin, -4.972e-3, "stripline focused x minimum");
        checkNear(focused.xMax, 4.972e-3, "stripline focused x maximum");
        checkNear(focused.yMin, full.yMin, "stripline must retain lower ground");
        checkNear(focused.yMax, full.yMax, "stripline must retain upper ground");
        check(tl::fieldViewIsCropped(full, focused),
              "default stripline view must be cropped");
    }
    {
        const auto parameters =
            tl::defaultParameters(tl::LineType::CoplanarWaveguide);
        const auto result = makeResult(parameters);
        const auto focused = requireBounds(result, tl::FieldViewMode::Focused);
        checkNear(focused.xMin, -3.65e-3, "CPW focused x minimum");
        checkNear(focused.xMax, 3.65e-3, "CPW focused x maximum");
        checkNear(focused.yMin, -2.85e-3, "CPW focused y minimum");
        checkNear(focused.yMax, 2.085e-3, "CPW focused y maximum");
    }
}

void testPaddingPolicy() {
    auto parameters = tl::defaultParameters(tl::LineType::Stripline);
    parameters.domainPaddingFactor = 4.0;
    const auto expanded = makeResult(parameters);
    const auto expandedFull = requireBounds(expanded, tl::FieldViewMode::FullDomain);
    const auto expandedFocus = requireBounds(expanded, tl::FieldViewMode::Focused);
    checkNear(expandedFocus.xMin, -4.972e-3,
              "padding above one must keep the padding-one focus");
    check(expandedFull.xMin < expandedFocus.xMin,
          "full-domain bounds must expand with numerical padding");

    parameters.domainPaddingFactor = 0.75;
    const auto compact = makeResult(parameters);
    const auto compactFull = requireBounds(compact, tl::FieldViewMode::FullDomain);
    const auto compactFocus = requireBounds(compact, tl::FieldViewMode::Focused);
    check(!tl::fieldViewIsCropped(compactFull, compactFocus),
          "padding at or below one must not be cropped");
}

void testCoaxialAlwaysUsesFullDomain() {
    const auto parameters = tl::defaultParameters(tl::LineType::Coaxial);
    const auto result = makeResult(parameters);
    const auto full = requireBounds(result, tl::FieldViewMode::FullDomain);
    const auto focused = requireBounds(result, tl::FieldViewMode::Focused);
    check(!tl::fieldViewIsCropped(full, focused),
          "coaxial focused mode must still show the full domain");
}

void testVisibleSampleSelection() {
    auto parameters = tl::defaultParameters(tl::LineType::Stripline);
    auto result = makeResult(parameters);
    result.samples = {
        {{0.0, 0.0}},
        {{-10.0e-3, 0.0}},
        {{4.0e-3, 0.5e-3}},
        {{0.0, 2.0e-3}},
    };
    const auto focused = requireBounds(result, tl::FieldViewMode::Focused);
    const auto visible = tl::visibleFieldSampleIndices(result, focused);
    check(visible == std::vector<std::size_t>{0, 2},
          "arrow candidates must include only samples inside the viewport");
}

void testCursorAnchoredZoomAndLimits() {
    const tl::FieldViewBounds limits{-10.0, 10.0, -5.0, 5.0};
    const tl::FieldViewBounds initial{-4.0, 4.0, -2.0, 2.0};
    const tl::Vec2 anchor{2.0, 1.0};
    const auto zoomed = tl::zoomFieldView(initial, limits, anchor, 0.5);
    checkNear(zoomed.xMin, -1.0, "zoom must preserve cursor-relative x minimum");
    checkNear(zoomed.xMax, 3.0, "zoom must preserve cursor-relative x maximum");
    checkNear(zoomed.yMin, -0.5, "zoom must preserve cursor-relative y minimum");
    checkNear(zoomed.yMax, 1.5, "zoom must preserve cursor-relative y maximum");

    const auto zoomedOut = tl::zoomFieldView(initial, limits, anchor, 100.0);
    checkNear(zoomedOut.xMin, limits.xMin,
              "maximum zoom-out must reach full x domain");
    checkNear(zoomedOut.xMax, limits.xMax,
              "maximum zoom-out must reach full x domain");
    checkNear(zoomedOut.yMin, limits.yMin,
              "maximum zoom-out must reach full y domain");
    checkNear(zoomedOut.yMax, limits.yMax,
              "maximum zoom-out must reach full y domain");
}

void testPanClampsToMeshDomain() {
    const tl::FieldViewBounds limits{-10.0, 10.0, -5.0, 5.0};
    const tl::FieldViewBounds initial{-4.0, 4.0, -2.0, 2.0};
    const auto panned = tl::panFieldView(initial, limits, 20.0, -20.0);
    checkNear(panned.xMin, 2.0, "pan must clamp at right domain edge");
    checkNear(panned.xMax, 10.0, "pan must clamp at right domain edge");
    checkNear(panned.yMin, -5.0, "pan must clamp at lower domain edge");
    checkNear(panned.yMax, -1.0, "pan must preserve height when clamped");

    const tl::FieldViewBounds fullHeight{-4.0, 4.0, -5.0, 5.0};
    const auto horizontalPan = tl::panFieldView(
        fullHeight, limits, 20.0, 2.0
    );
    checkNear(horizontalPan.xMin, 2.0,
              "full-height view must still pan horizontally");
    checkNear(horizontalPan.xMax, 10.0,
              "full-height horizontal pan must clamp at domain edge");
    checkNear(horizontalPan.yMin, -5.0,
              "full-height pan must retain lower y limit");
    checkNear(horizontalPan.yMax, 5.0,
              "full-height pan must retain upper y limit");
}

} // namespace

int main() {
    try {
        testDefaultFocusedBounds();
        testPaddingPolicy();
        testCoaxialAlwaysUsesFullDomain();
        testVisibleSampleSelection();
        testCursorAnchoredZoomAndLimits();
        testPanClampsToMeshDomain();
        std::cout << "6 field-view test groups passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
