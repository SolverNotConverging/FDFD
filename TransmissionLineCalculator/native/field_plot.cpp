#include "field_plot.hpp"

#include <QLinearGradient>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace tl {
namespace {

[[nodiscard]] QRectF aspectFit(
    const QRectF& available,
    const FieldViewBounds& bounds
) {
    const auto dataWidth = bounds.xMax - bounds.xMin;
    const auto dataHeight = bounds.yMax - bounds.yMin;
    const auto dataAspect = dataWidth / dataHeight;
    const auto availableAspect = available.width() / available.height();
    if (availableAspect > dataAspect) {
        const auto width = available.height() * dataAspect;
        return {available.center().x() - 0.5 * width, available.top(), width, available.height()};
    }
    const auto height = available.width() / dataAspect;
    return {available.left(), available.center().y() - 0.5 * height,
            available.width(), height};
}

[[nodiscard]] QPointF mapPoint(const Vec2& point, const FieldViewBounds& bounds,
                               const QRectF& area) {
    const auto x = (point.x - bounds.xMin) / (bounds.xMax - bounds.xMin);
    const auto y = (point.y - bounds.yMin) / (bounds.yMax - bounds.yMin);
    return {area.left() + x * area.width(), area.bottom() - y * area.height()};
}

[[nodiscard]] QColor interpolateColour(const std::array<QColor, 5>& colours,
                                       double value) {
    const auto scaled = std::clamp(value, 0.0, 1.0) * (colours.size() - 1);
    const auto first = std::min<std::size_t>(static_cast<std::size_t>(scaled),
                                             colours.size() - 2);
    const auto fraction = scaled - static_cast<double>(first);
    const auto& low = colours[first];
    const auto& high = colours[first + 1];
    const auto channel = [fraction](int a, int b) {
        return static_cast<int>(std::lround(a + fraction * (b - a)));
    };
    return {channel(low.red(), high.red()), channel(low.green(), high.green()),
            channel(low.blue(), high.blue())};
}

[[nodiscard]] QColor fieldColour(FieldFamily family, double fraction) {
    static const std::array<QColor, 5> viridis{
        QColor(68, 1, 84), QColor(59, 82, 139), QColor(33, 145, 140),
        QColor(94, 201, 98), QColor(253, 231, 37),
    };
    static const std::array<QColor, 5> magma{
        QColor(0, 0, 4), QColor(78, 18, 123), QColor(182, 54, 121),
        QColor(251, 136, 97), QColor(252, 253, 191),
    };
    // The power transform keeps the large low-field portion of a FEM mesh legible.
    const auto mapped = std::pow(std::clamp(fraction, 0.0, 1.0), 0.55);
    return interpolateColour(family == FieldFamily::Electric ? viridis : magma, mapped);
}

[[nodiscard]] double fieldMagnitude(const FieldSample& sample, FieldFamily family) {
    const auto& field = family == FieldFamily::Electric ? sample.electric : sample.magnetic;
    return std::sqrt(std::norm(field.x) + std::norm(field.y));
}

void drawArrow(QPainter& painter, const QPointF& centre, double x, double y) {
    const auto norm = std::hypot(x, y);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        return;
    }
    // Screen y grows downwards, while the field's y component grows upwards.
    const QPointF direction{x / norm, -y / norm};
    constexpr double length = 15.0;
    constexpr double headLength = 4.5;
    constexpr double headWidth = 2.9;
    const auto start = centre - 0.5 * length * direction;
    const auto end = centre + 0.5 * length * direction;
    const QPointF normal{-direction.y(), direction.x()};
    QPainterPath path;
    path.moveTo(start);
    path.lineTo(end);
    path.moveTo(end);
    path.lineTo(end - headLength * direction + headWidth * normal);
    path.moveTo(end);
    path.lineTo(end - headLength * direction - headWidth * normal);
    painter.setPen(QPen(QColor(0, 0, 0, 175), 2.5, Qt::SolidLine,
                        Qt::RoundCap, Qt::RoundJoin));
    painter.drawPath(path);
    painter.setPen(QPen(QColor(255, 255, 255, 235), 1.0, Qt::SolidLine,
                        Qt::RoundCap, Qt::RoundJoin));
    painter.drawPath(path);
}

[[nodiscard]] QString scientific(double value) {
    return QString::number(value, 'g', 4);
}

} // namespace

FieldPlot::FieldPlot(FieldFamily family, QWidget* parent)
    : QWidget(parent), family_(family),
      emptyMessage_(family == FieldFamily::Electric
                        ? QStringLiteral("E-field appears after calculation")
                        : QStringLiteral("H-field appears after calculation")) {
    setAutoFillBackground(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void FieldPlot::setResult(std::shared_ptr<const Result> result) {
    result_ = std::move(result);
    update();
}

void FieldPlot::setMeshVisible(bool visible) {
    if (meshVisible_ == visible) {
        return;
    }
    meshVisible_ = visible;
    update();
}

void FieldPlot::setViewMode(const FieldViewMode mode) {
    if (viewMode_ == mode) {
        return;
    }
    viewMode_ = mode;
    update();
}

void FieldPlot::setEmptyMessage(QString message) {
    result_.reset();
    emptyMessage_ = std::move(message);
    update();
}

QSize FieldPlot::minimumSizeHint() const {
    return {380, 360};
}

void FieldPlot::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), palette().brush(QPalette::Base));

    const auto familyName = family_ == FieldFamily::Electric
        ? QStringLiteral("E") : QStringLiteral("H");
    const auto unit = family_ == FieldFamily::Electric
        ? QStringLiteral("V/m") : QStringLiteral("A/m");
    painter.setPen(palette().color(QPalette::Text));
    const QFont bodyFont = painter.font();
    QFont titleFont = bodyFont;
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.drawText(QRectF(8.0, 8.0, width() - 16.0, 28.0), Qt::AlignCenter,
                     QStringLiteral("|%1ₜ| colour; direction of Re(%1ₜ), φ = 0")
                         .arg(familyName));
    painter.setFont(bodyFont);

    if (!result_ || result_->mesh.nodes.empty() || result_->mesh.triangles.empty()) {
        painter.setPen(palette().color(QPalette::PlaceholderText));
        painter.drawText(rect().adjusted(24, 42, -24, -24), Qt::AlignCenter,
                         emptyMessage_);
        return;
    }

    const auto fullBounds = fieldMeshBounds(*result_);
    const auto bounds = fieldDisplayBounds(*result_, viewMode_);
    if (!fullBounds || !bounds) {
        painter.setPen(QColor(170, 30, 45));
        painter.drawText(rect(), Qt::AlignCenter,
                         QStringLiteral("Field mesh contains invalid coordinates."));
        return;
    }
    const QRectF available = QRectF(rect()).adjusted(58.0, 46.0, -72.0, -47.0);
    if (available.width() < 20.0 || available.height() < 20.0) {
        return;
    }
    const auto plotArea = aspectFit(available, *bounds);
    const auto cropped = fieldViewIsCropped(*fullBounds, *bounds);

    std::vector<double> magnitudes;
    magnitudes.reserve(result_->samples.size());
    double maximum = 0.0;
    for (const auto& sample : result_->samples) {
        const auto magnitude = fieldMagnitude(sample, family_);
        magnitudes.push_back(std::isfinite(magnitude) ? magnitude : 0.0);
        maximum = std::max(maximum, magnitudes.back());
    }
    const auto colourScale = maximum > std::numeric_limits<double>::min() ? maximum : 1.0;

    painter.save();
    painter.setClipRect(plotArea);

    const auto triangleCount = result_->mesh.triangles.size();
    for (std::size_t index = 0; index < triangleCount; ++index) {
        const auto& triangle = result_->mesh.triangles[index];
        QPolygonF polygon;
        bool valid = true;
        for (int corner = 0; corner < 3; ++corner) {
            const auto nodeIndex = static_cast<std::size_t>(triangle.nodes[corner]);
            if (nodeIndex >= result_->mesh.nodes.size()) {
                valid = false;
                break;
            }
            polygon << mapPoint(result_->mesh.nodes[nodeIndex], *bounds, plotArea);
        }
        if (!valid) {
            continue;
        }
        const auto magnitude = index < magnitudes.size() ? magnitudes[index] : 0.0;
        painter.setBrush(fieldColour(family_, magnitude / colourScale));
        painter.setPen(meshVisible_ ? QPen(QColor(0, 0, 0, 72), 0.55) : Qt::NoPen);
        painter.drawPolygon(polygon);
    }

    const auto visibleSamples = visibleFieldSampleIndices(*result_, *bounds);
    const auto maximumArrows = std::max<std::size_t>(1,
        static_cast<std::size_t>(plotArea.width() * plotArea.height() / 2000.0));
    const auto stride = std::max<std::size_t>(1,
        (visibleSamples.size() + maximumArrows - 1) / maximumArrows);
    for (std::size_t visibleIndex = 0;
         visibleIndex < visibleSamples.size();
         visibleIndex += stride) {
        const auto index = visibleSamples[visibleIndex];
        const auto& sample = result_->samples[index];
        const auto& field = family_ == FieldFamily::Electric
            ? sample.electric : sample.magnetic;
        const auto x = std::real(field.x);
        const auto y = std::real(field.y);
        const auto instantaneous = std::hypot(x, y);
        const auto magnitude = index < magnitudes.size() ? magnitudes[index] : 0.0;
        if (!(magnitude > 0.0) || instantaneous <= 1.0e-3 * magnitude) {
            continue;
        }
        drawArrow(painter, mapPoint(sample.position, *bounds, plotArea), x, y);
    }
    painter.restore();

    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(palette().color(QPalette::Text), 1.0));
    painter.drawRect(plotArea);
    painter.drawText(QRectF(plotArea.left(), plotArea.bottom() + 25.0,
                            plotArea.width(), 22.0),
                     Qt::AlignCenter, QStringLiteral("x (m)"));
    painter.save();
    painter.translate(15.0, plotArea.center().y());
    painter.rotate(-90.0);
    painter.drawText(QRectF(-0.5 * plotArea.height(), -10.0,
                            plotArea.height(), 20.0),
                     Qt::AlignCenter, QStringLiteral("y (m)"));
    painter.restore();
    painter.setPen(palette().color(QPalette::Text));
    painter.drawText(QRectF(plotArea.left() - 48.0, plotArea.bottom() - 8.0,
                            44.0, 16.0), Qt::AlignRight | Qt::AlignVCenter,
                     scientific(bounds->yMin));
    painter.drawText(QRectF(plotArea.left() - 48.0, plotArea.top() - 8.0,
                            44.0, 16.0), Qt::AlignRight | Qt::AlignVCenter,
                     scientific(bounds->yMax));
    painter.drawText(QRectF(plotArea.left() - 30.0, plotArea.bottom() + 1.0,
                            60.0, 16.0), Qt::AlignLeft | Qt::AlignVCenter,
                     scientific(bounds->xMin));
    painter.drawText(QRectF(plotArea.right() - 30.0, plotArea.bottom() + 1.0,
                            60.0, 16.0), Qt::AlignRight | Qt::AlignVCenter,
                     scientific(bounds->xMax));

    if (cropped) {
        QFont badgeFont = bodyFont;
        badgeFont.setPointSizeF(std::max(7.0, bodyFont.pointSizeF() - 1.0));
        painter.setFont(badgeFont);
        const auto badgeText = QStringLiteral("Focused view — display only");
        const auto textBounds = painter.fontMetrics().boundingRect(badgeText);
        const QRectF badge(
            plotArea.left() + 6.0,
            plotArea.top() + 6.0,
            textBounds.width() + 12.0,
            textBounds.height() + 6.0
        );
        auto badgeBackground = palette().color(QPalette::Base);
        badgeBackground.setAlpha(210);
        painter.setBrush(badgeBackground);
        painter.setPen(QPen(palette().color(QPalette::Mid), 0.8));
        painter.drawRoundedRect(badge, 3.0, 3.0);
        painter.setPen(palette().color(QPalette::Text));
        painter.drawText(badge, Qt::AlignCenter, badgeText);
        painter.setFont(bodyFont);
    }

    const QRectF colourBar(width() - 46.0, plotArea.top(), 13.0, plotArea.height());
    QLinearGradient gradient(colourBar.bottomLeft(), colourBar.topLeft());
    for (int index = 0; index <= 16; ++index) {
        const auto fraction = static_cast<double>(index) / 16.0;
        gradient.setColorAt(fraction, fieldColour(family_, fraction));
    }
    painter.fillRect(colourBar, gradient);
    painter.setPen(QPen(palette().color(QPalette::Text), 0.8));
    painter.drawRect(colourBar);
    painter.drawText(QRectF(colourBar.right() + 4.0, colourBar.top() - 8.0,
                            38.0, 18.0), Qt::AlignLeft | Qt::AlignVCenter,
                     scientific(maximum));
    painter.drawText(QRectF(colourBar.right() + 4.0, colourBar.bottom() - 9.0,
                            38.0, 18.0), Qt::AlignLeft | Qt::AlignVCenter,
                     QStringLiteral("0"));
    painter.save();
    painter.translate(width() - 8.0, colourBar.center().y());
    painter.rotate(-90.0);
    painter.drawText(QRectF(-0.5 * colourBar.height(), -10.0,
                            colourBar.height(), 20.0), Qt::AlignCenter,
                     QStringLiteral("|%1ₜ| (%2)").arg(familyName, unit));
    painter.restore();
}

} // namespace tl
