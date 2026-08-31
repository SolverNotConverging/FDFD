#include "field_plot_2d.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace femperiodic {
namespace {

QColor viridis(double value) {
    constexpr std::array<std::array<double, 3>, 6> stops{{
        {{68, 1, 84}}, {{59, 82, 139}}, {{33, 145, 140}},
        {{94, 201, 98}}, {{180, 222, 44}}, {{253, 231, 37}},
    }};
    value = std::clamp(value, 0.0, 1.0);
    const auto position = value * static_cast<double>(stops.size() - 1);
    const auto lower = std::min<std::size_t>(static_cast<std::size_t>(position), stops.size() - 2);
    const auto fraction = position - static_cast<double>(lower);
    const auto mix = [fraction](double a, double b) {
        return static_cast<int>(std::lround(a + fraction * (b - a)));
    };
    return {mix(stops[lower][0], stops[lower + 1][0]),
            mix(stops[lower][1], stops[lower + 1][1]),
            mix(stops[lower][2], stops[lower + 1][2])};
}

QColor coolwarm(double value) {
    constexpr std::array<std::array<double, 3>, 3> stops{{
        {{59, 76, 192}}, {{221, 221, 221}}, {{180, 4, 38}},
    }};
    value = std::clamp(value, 0.0, 1.0);
    const auto position = value * static_cast<double>(stops.size() - 1);
    const auto lower = std::min<std::size_t>(static_cast<std::size_t>(position), stops.size() - 2);
    const auto fraction = position - static_cast<double>(lower);
    const auto mix = [fraction](double a, double b) {
        return static_cast<int>(std::lround(a + fraction * (b - a)));
    };
    return {mix(stops[lower][0], stops[lower + 1][0]),
            mix(stops[lower][1], stops[lower + 1][1]),
            mix(stops[lower][2], stops[lower + 1][2])};
}

QString componentLabel(FieldFamily family, int component) {
    const auto prefix = family == FieldFamily::Electric ? QStringLiteral("E") : QStringLiteral("H");
    constexpr std::array<const char*, 3> suffix{"x", "y", "z"};
    return prefix + QString::fromLatin1(suffix.at(static_cast<std::size_t>(component)));
}

QString quantityLabel(ScalarQuantity quantity) {
    switch (quantity) {
    case ScalarQuantity::Magnitude: return QStringLiteral("magnitude");
    case ScalarQuantity::Real: return QStringLiteral("real");
    case ScalarQuantity::Imaginary: return QStringLiteral("imaginary");
    case ScalarQuantity::Phase: return QStringLiteral("phase (rad)");
    }
    return {};
}

} // namespace

FieldPlot2D::FieldPlot2D(QWidget* parent) : QWidget(parent) {
    setMinimumSize(640, 460);
    setMouseTracking(true);
    setAutoFillBackground(true);
}

void FieldPlot2D::setData(MeshPtr mesh, MaterialStatePtr material, ModeFieldsPtr fields) {
    mesh_ = std::move(mesh);
    material_ = std::move(material);
    fields_ = std::move(fields);
    if (!mesh_ || mesh_->dimension != 2 || !fields_) {
        clearData(QStringLiteral("The selected result does not contain a 2D field."));
        return;
    }
    auto xMin = std::numeric_limits<double>::infinity();
    auto xMax = -xMin;
    auto zMin = xMin;
    auto zMax = -xMin;
    for (const auto& point : mesh_->points) {
        xMin = std::min(xMin, point[0]);
        xMax = std::max(xMax, point[0]);
        zMin = std::min(zMin, point[2]);
        zMax = std::max(zMax, point[2]);
    }
    if (!(xMax > xMin) || !(zMax > zMin)) {
        clearData(QStringLiteral("The 2D mesh has a degenerate x-z extent."));
        return;
    }
    dataRange_ = {xMin, xMax, zMin, zMax};
    message_.clear();
    rebuildValues();
    resetView();
}

void FieldPlot2D::setSelection(
    FieldFamily family, int component, ScalarQuantity quantity) {
    family_ = family;
    component_ = std::clamp(component, 0, 2);
    quantity_ = quantity;
    rebuildValues();
    update();
}

void FieldPlot2D::setMaterialOnly(bool enabled) {
    materialOnly_ = enabled;
    rebuildValues();
    update();
}

void FieldPlot2D::clearData(const QString& message) {
    mesh_.reset();
    material_.reset();
    fields_.reset();
    cellValues_.clear();
    message_ = message;
    update();
}

void FieldPlot2D::resetView() {
    viewRange_ = dataRange_;
    update();
}

QRectF FieldPlot2D::plotRect() const {
    const QRectF available(
        74.0, 48.0, std::max(1, width() - 174), std::max(1, height() - 116));
    const auto xExtent = viewRange_[1] - viewRange_[0];
    const auto zExtent = viewRange_[3] - viewRange_[2];
    if (!(xExtent > 0.0) || !(zExtent > 0.0)) {
        return available;
    }
    const auto dataAspect = zExtent / xExtent;
    const auto availableAspect = available.width() / available.height();
    if (availableAspect > dataAspect) {
        const auto physicalWidth = available.height() * dataAspect;
        return {available.center().x() - 0.5 * physicalWidth, available.top(),
                physicalWidth, available.height()};
    }
    const auto physicalHeight = available.width() / dataAspect;
    return {available.left(), available.center().y() - 0.5 * physicalHeight,
            available.width(), physicalHeight};
}

QPointF FieldPlot2D::mapPoint(double x, double z) const {
    const auto area = plotRect();
    const auto horizontal = area.left()
        + (z - viewRange_[2]) / (viewRange_[3] - viewRange_[2]) * area.width();
    const auto vertical = area.bottom()
        - (x - viewRange_[0]) / (viewRange_[1] - viewRange_[0]) * area.height();
    return {horizontal, vertical};
}

std::array<double, 2> FieldPlot2D::unmapPoint(const QPointF& point) const {
    const auto area = plotRect();
    const auto z = viewRange_[2] + (point.x() - area.left()) / area.width()
        * (viewRange_[3] - viewRange_[2]);
    const auto x = viewRange_[0] + (area.bottom() - point.y()) / area.height()
        * (viewRange_[1] - viewRange_[0]);
    return {x, z};
}

void FieldPlot2D::rebuildValues() {
    cellValues_.clear();
    if (!mesh_ || !fields_) {
        return;
    }
    if (materialOnly_) {
        if (!material_ || material_->epsilonR.size() != mesh_->cells.size()
            || material_->muR.size() != mesh_->cells.size()) {
            message_ = QStringLiteral("Material/cell sizes are inconsistent.");
            return;
        }
        cellValues_.resize(mesh_->cells.size());
        for (std::size_t cell = 0; cell < mesh_->cells.size(); ++cell) {
            const auto& epsilon = material_->epsilonR[cell];
            const auto& mu = material_->muR[cell];
            cellValues_[cell] = std::sqrt(std::max(
                {std::abs(epsilon[0] * mu[0]), std::abs(epsilon[1] * mu[1]),
                 std::abs(epsilon[2] * mu[2])}));
        }
    } else {
    const auto& source = family_ == FieldFamily::Electric
        ? fields_->electric : fields_->magnetic;
    if (source.size() != mesh_->sampleOwnerCells.size()) {
        message_ = QStringLiteral("Field/sample sizes are inconsistent.");
        return;
    }
    cellValues_.assign(mesh_->cells.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::size_t> counts(mesh_->cells.size(), 0);
    for (std::size_t sample = 0; sample < source.size(); ++sample) {
        const auto owner = static_cast<std::size_t>(mesh_->sampleOwnerCells[sample]);
        const auto value = scalarValue(source[sample][static_cast<std::size_t>(component_)], quantity_);
        if (counts[owner] == 0) {
            cellValues_[owner] = value;
        } else {
            cellValues_[owner] += value;
        }
        ++counts[owner];
    }
    for (std::size_t cell = 0; cell < cellValues_.size(); ++cell) {
        if (counts[cell] > 1) {
            cellValues_[cell] /= static_cast<double>(counts[cell]);
        }
    }
    }
    valueMin_ = std::numeric_limits<double>::infinity();
    valueMax_ = -valueMin_;
    for (const auto value : cellValues_) {
        if (std::isfinite(value)) {
            valueMin_ = std::min(valueMin_, value);
            valueMax_ = std::max(valueMax_, value);
        }
    }
    if (!std::isfinite(valueMin_) || !std::isfinite(valueMax_)) {
        valueMin_ = 0.0;
        valueMax_ = 1.0;
    } else if (!(valueMax_ > valueMin_)) {
        const auto padding = std::max(1.0, std::abs(valueMin_)) * 0.5;
        valueMin_ -= padding;
        valueMax_ += padding;
    }
}

void FieldPlot2D::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), palette().base());
    if (!mesh_ || !fields_ || cellValues_.empty()) {
        painter.setPen(palette().text().color());
        painter.drawText(rect(), Qt::AlignCenter, message_);
        return;
    }
    const auto area = plotRect();
    painter.save();
    painter.setClipRect(area);
    for (std::size_t cellIndex = 0; cellIndex < mesh_->cells.size(); ++cellIndex) {
        const auto& cell = mesh_->cells[cellIndex];
        if (cell.size() != 3) {
            continue;
        }
        QPolygonF polygon;
        for (const auto vertex : cell) {
            const auto& point = mesh_->points[static_cast<std::size_t>(vertex)];
            polygon << mapPoint(point[0], point[2]);
        }
        const auto value = cellValues_[cellIndex];
        const auto normalized = std::isfinite(value)
            ? (value - valueMin_) / (valueMax_ - valueMin_) : 0.0;
        painter.setPen(QPen(QColor(25, 25, 25, 80), 0.45));
        painter.setBrush(materialOnly_ ? coolwarm(normalized) : viridis(normalized));
        painter.drawPolygon(polygon);
        if (material_ && cellIndex < material_->pmlFraction.size()
            && material_->pmlFraction[cellIndex] > 0.0) {
            painter.setBrush(QColor(255, 255, 255, 45));
            painter.setPen(QPen(QColor(240, 240, 240, 125), 0.7, Qt::DashLine));
            painter.drawPolygon(polygon);
        }
    }
    constexpr std::array<QColor, 5> boundaryColors{
        QColor(255, 211, 67), QColor(74, 144, 226), QColor(46, 204, 113),
        QColor(231, 76, 60), QColor(189, 195, 199)};
    for (std::size_t facetIndex = 0; facetIndex < mesh_->boundaryFacets.size(); ++facetIndex) {
        const auto& facet = mesh_->boundaryFacets[facetIndex];
        if (facet.size() != 2) {
            continue;
        }
        const auto tag = facetIndex < mesh_->boundaryTags.size()
            ? std::abs(mesh_->boundaryTags[facetIndex]) : 0;
        painter.setPen(QPen(boundaryColors[static_cast<std::size_t>(tag) % boundaryColors.size()],
                            2.0));
        const auto& first = mesh_->points[static_cast<std::size_t>(facet[0])];
        const auto& second = mesh_->points[static_cast<std::size_t>(facet[1])];
        painter.drawLine(mapPoint(first[0], first[2]), mapPoint(second[0], second[2]));
    }
    painter.restore();

    painter.setPen(palette().text().color());
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(area);
    const auto title = materialOnly_
        ? QStringLiteral("Material · |n_eff|")
        : componentLabel(family_, component_) + QStringLiteral(" · ")
              + quantityLabel(quantity_);
    painter.drawText(QRectF(area.left(), 8.0, area.width(), 30.0), Qt::AlignCenter, title);
    painter.drawText(QRectF(area.left(), area.bottom() + 30.0, area.width(), 24.0),
                     Qt::AlignCenter, QStringLiteral("z (m)"));
    painter.save();
    painter.translate(18.0, area.center().y());
    painter.rotate(-90.0);
    painter.drawText(QRectF(-area.height() / 2.0, -16.0, area.height(), 24.0),
                     Qt::AlignCenter, QStringLiteral("x (m)"));
    painter.restore();

    const QRectF colorbar(area.right() + 28.0, area.top(), 18.0, area.height());
    for (int pixel = 0; pixel < static_cast<int>(colorbar.height()); ++pixel) {
        const auto t = 1.0 - static_cast<double>(pixel) / std::max(1.0, colorbar.height() - 1.0);
        painter.setPen(materialOnly_ ? coolwarm(t) : viridis(t));
        painter.drawLine(QPointF(colorbar.left(), colorbar.top() + pixel),
                         QPointF(colorbar.right(), colorbar.top() + pixel));
    }
    painter.setPen(palette().text().color());
    painter.drawRect(colorbar);
    painter.drawText(QRectF(colorbar.right() + 6.0, colorbar.top() - 10.0, 70.0, 22.0),
                     Qt::AlignLeft | Qt::AlignVCenter, QString::number(valueMax_, 'g', 4));
    painter.drawText(QRectF(colorbar.right() + 6.0, colorbar.bottom() - 11.0, 70.0, 22.0),
                     Qt::AlignLeft | Qt::AlignVCenter, QString::number(valueMin_, 'g', 4));
}

void FieldPlot2D::wheelEvent(QWheelEvent* event) {
    if (!mesh_ || !plotRect().contains(event->position())) {
        event->ignore();
        return;
    }
    const auto anchor = unmapPoint(event->position());
    const auto factor = event->angleDelta().y() > 0 ? 0.82 : 1.22;
    viewRange_[0] = anchor[0] + (viewRange_[0] - anchor[0]) * factor;
    viewRange_[1] = anchor[0] + (viewRange_[1] - anchor[0]) * factor;
    viewRange_[2] = anchor[1] + (viewRange_[2] - anchor[1]) * factor;
    viewRange_[3] = anchor[1] + (viewRange_[3] - anchor[1]) * factor;
    update();
    event->accept();
}

void FieldPlot2D::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && plotRect().contains(event->position())) {
        dragging_ = true;
        dragStart_ = event->position().toPoint();
        dragStartRange_ = viewRange_;
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void FieldPlot2D::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) {
        QWidget::mouseMoveEvent(event);
        return;
    }
    const auto area = plotRect();
    const auto dx = static_cast<double>(event->position().x() - dragStart_.x()) / area.width()
        * (dragStartRange_[3] - dragStartRange_[2]);
    const auto dy = static_cast<double>(event->position().y() - dragStart_.y()) / area.height()
        * (dragStartRange_[1] - dragStartRange_[0]);
    viewRange_ = {dragStartRange_[0] + dy, dragStartRange_[1] + dy,
                  dragStartRange_[2] - dx, dragStartRange_[3] - dx};
    update();
}

void FieldPlot2D::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && dragging_) {
        dragging_ = false;
        unsetCursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void FieldPlot2D::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        resetView();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

} // namespace femperiodic
