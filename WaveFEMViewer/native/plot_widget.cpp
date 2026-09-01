#include "plot_widget.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <tuple>

namespace wavefem {
namespace {

constexpr std::array<QColor, 8> seriesColors{
    QColor(31, 119, 180), QColor(214, 39, 40), QColor(44, 160, 44),
    QColor(148, 103, 189), QColor(255, 127, 14), QColor(23, 190, 207),
    QColor(227, 119, 194), QColor(127, 127, 127)};

QColor viridis(double normalized) {
    constexpr std::array<std::array<double, 3>, 6> colors{{
        {68, 1, 84}, {59, 82, 139}, {33, 145, 140},
        {94, 201, 98}, {253, 231, 37}, {253, 231, 37}}};
    const auto value = std::clamp(normalized, 0.0, 1.0) * 4.0;
    const auto index = std::min<std::size_t>(static_cast<std::size_t>(value), 3);
    const auto fraction = value - static_cast<double>(index);
    const auto interpolate = [&](std::size_t channel) {
        return static_cast<int>(std::lround(
            colors[index][channel] * (1.0 - fraction)
            + colors[index + 1][channel] * fraction));
    };
    return {interpolate(0), interpolate(1), interpolate(2)};
}

QString numberLabel(double value) {
    return QString::number(value, 'g', 4);
}

double scalarValue(Complex value, ScalarQuantity quantity) {
    if (quantity == ScalarQuantity::Absolute) {
        return std::abs(value);
    }
    return quantity == ScalarQuantity::Real ? value.real() : value.imag();
}

} // namespace

PlotWidget::PlotWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(360, 260);
    setMouseTracking(true);
    setAutoFillBackground(true);
    setEmpty(QStringLiteral("WaveFEM"), QStringLiteral("Open an HDF5 result file"));
}

void PlotWidget::setEmpty(QString title, QString message) {
    kind_ = PlotKind::Empty;
    title_ = std::move(title);
    message_ = std::move(message);
    series_.clear();
    arrows_.clear();
    result_.reset();
    invalidateSceneCache();
    update();
}

void PlotWidget::setDataRange(double xMin, double xMax, double yMin, double yMax) {
    if (!(xMin < xMax)) {
        const auto center = 0.5 * (xMin + xMax);
        const auto scale = std::max(1.0, std::abs(center));
        xMin = center - 0.05 * scale;
        xMax = center + 0.05 * scale;
    }
    if (!(yMin < yMax)) {
        const auto scale = std::max(1.0, std::abs(yMin));
        yMin -= 0.05 * scale;
        yMax += 0.05 * scale;
    }
    const auto xPadding = 0.03 * (xMax - xMin);
    const auto yPadding = 0.06 * (yMax - yMin);
    dataRange_ = {xMin - xPadding, xMax + xPadding,
                  yMin - yPadding, yMax + yPadding};
    viewRange_ = dataRange_;
    invalidateSceneCache();
}

void PlotWidget::setLines(std::vector<PlotSeries> series, QString title, QString xLabel,
                          QString yLabel, std::optional<double> selectedX) {
    kind_ = PlotKind::Lines;
    series_ = std::move(series);
    title_ = std::move(title);
    xLabel_ = std::move(xLabel);
    yLabel_ = std::move(yLabel);
    selectedX_ = selectedX;
    result_.reset();
    arrows_.clear();
    double xMin = std::numeric_limits<double>::infinity();
    double xMax = -xMin;
    double yMin = xMin;
    double yMax = -xMin;
    for (const auto& seriesItem : series_) {
        for (const auto value : seriesItem.x) {
            if (std::isfinite(value)) {
                xMin = std::min(xMin, value);
                xMax = std::max(xMax, value);
            }
        }
        for (const auto value : seriesItem.y) {
            if (std::isfinite(value)) {
                yMin = std::min(yMin, value);
                yMax = std::max(yMax, value);
            }
        }
    }
    if (!std::isfinite(xMin) || !std::isfinite(yMin)) {
        setEmpty(std::move(title_), QStringLiteral("No plottable data"));
        return;
    }
    setDataRange(xMin, xMax, yMin, yMax);
    update();
}

void PlotWidget::setModal(const ModeData& mode, FieldName field, int component,
                          ScalarQuantity quantity) {
    const auto& matrix = field == FieldName::Electric ? mode.electric : mode.magnetic;
    PlotSeries series;
    series.label = field == FieldName::Electric ? QStringLiteral("E") : QStringLiteral("H");
    series.x = mode.x;
    series.y.reserve(mode.x.size());
    for (std::size_t index = 0; index < mode.x.size(); ++index) {
        if (component >= 0) {
            series.y.push_back(scalarValue(matrix.at(static_cast<std::size_t>(component), index),
                                           quantity));
        } else {
            double sum = 0.0;
            for (std::size_t row = 0; row < 3; ++row) {
                const auto value = scalarValue(matrix.at(row, index), quantity);
                sum += value * value;
            }
            series.y.push_back(std::sqrt(sum));
        }
    }
    const auto quantityName = quantity == ScalarQuantity::Absolute ? QStringLiteral("abs")
        : (quantity == ScalarQuantity::Real ? QStringLiteral("real")
                                             : QStringLiteral("imag"));
    const auto componentName = component < 0 ? QStringLiteral("norm")
        : QString(QChar(QStringLiteral("xyz").at(component)));
    const auto fieldName = field == FieldName::Electric ? QStringLiteral("E")
                                                         : QStringLiteral("H");
    setLines({std::move(series)},
             QStringLiteral("%1 · Modal %2").arg(QString::fromStdString(mode.label), fieldName),
             QStringLiteral("x (m)"),
             QStringLiteral("%1(%2_%3)").arg(quantityName, fieldName, componentName));
}

void PlotWidget::setVector(ResultPtr result, FieldName field, FieldPart part,
                           ScalarQuantity quantity, std::size_t maxArrows) {
    result_ = std::move(result);
    kind_ = PlotKind::Vector;
    series_.clear();
    selectedX_.reset();
    arrows_.clear();
    if (!result_ || result_->coordinates.columns == 0) {
        setEmpty(QStringLiteral("2D vector field"), QStringLiteral("No field samples"));
        return;
    }
    const auto& values = result_->field(field, part);
    struct Sample {
        double x;
        double z;
        double horizontal;
        double vertical;
    };
    std::vector<Sample> samples;
    samples.reserve(result_->coordinates.columns);
    for (std::size_t index = 0; index < result_->coordinates.columns; ++index) {
        samples.push_back({result_->coordinates.at(0, index), result_->coordinates.at(1, index),
                           scalarValue(values.at(2, index), quantity),
                           scalarValue(values.at(0, index), quantity)});
    }
    std::sort(samples.begin(), samples.end(), [](const Sample& left, const Sample& right) {
        return std::tie(left.x, left.z) < std::tie(right.x, right.z);
    });
    std::vector<Arrow> unique;
    unique.reserve(samples.size());
    for (std::size_t begin = 0; begin < samples.size();) {
        std::size_t end = begin + 1;
        double horizontal = samples[begin].horizontal;
        double vertical = samples[begin].vertical;
        while (end < samples.size() && samples[end].x == samples[begin].x
               && samples[end].z == samples[begin].z) {
            horizontal += samples[end].horizontal;
            vertical += samples[end].vertical;
            ++end;
        }
        const auto count = static_cast<double>(end - begin);
        horizontal /= count;
        vertical /= count;
        const auto magnitude = std::hypot(horizontal, vertical);
        if (magnitude > 0.0) {
            horizontal /= magnitude;
            vertical /= magnitude;
        }
        unique.push_back({samples[begin].x, samples[begin].z,
                          horizontal, vertical, magnitude});
        begin = end;
    }
    const auto outputCount = std::min(maxArrows, unique.size());
    arrows_.reserve(outputCount);
    for (std::size_t index = 0; index < outputCount; ++index) {
        const auto source = outputCount == 1 ? 0U
            : static_cast<std::size_t>(std::llround(
                static_cast<double>(index) * static_cast<double>(unique.size() - 1)
                / static_cast<double>(outputCount - 1)));
        arrows_.push_back(unique[source]);
    }
    magnitudeMin_ = std::numeric_limits<double>::infinity();
    magnitudeMax_ = 0.0;
    for (const auto& arrow : arrows_) {
        magnitudeMin_ = std::min(magnitudeMin_, arrow.magnitude);
        magnitudeMax_ = std::max(magnitudeMax_, arrow.magnitude);
    }
    if (!std::isfinite(magnitudeMin_)) {
        magnitudeMin_ = 0.0;
    }
    const auto fieldName = field == FieldName::Electric ? QStringLiteral("E")
                                                         : QStringLiteral("H");
    const auto partName = part == FieldPart::Total ? QStringLiteral("total")
        : (part == FieldPart::Incident ? QStringLiteral("incident")
                                       : QStringLiteral("scattered"));
    const auto quantityName = quantity == ScalarQuantity::Real ? QStringLiteral("real")
                                                                : QStringLiteral("imag");
    title_ = QStringLiteral("%1 %2 · %3 — direction; colour shows magnitude")
                 .arg(partName, fieldName, quantityName);
    xLabel_ = QStringLiteral("z (m)");
    yLabel_ = QStringLiteral("x (m)");
    if (result_->scene) {
        setDataRange(result_->scene->zSpan[0], result_->scene->zSpan[1],
                     result_->scene->xSpan[0], result_->scene->xSpan[1]);
    } else {
        const auto [xMinIt, xMaxIt] = std::minmax_element(
            result_->coordinates.values.begin(),
            result_->coordinates.values.begin() + static_cast<std::ptrdiff_t>(
                result_->coordinates.columns));
        const auto zBegin = result_->coordinates.values.begin()
            + static_cast<std::ptrdiff_t>(result_->coordinates.columns);
        const auto [zMinIt, zMaxIt] = std::minmax_element(zBegin,
            zBegin + static_cast<std::ptrdiff_t>(result_->coordinates.columns));
        setDataRange(*zMinIt, *zMaxIt, *xMinIt, *xMaxIt);
    }
    update();
}

void PlotWidget::setMesh(ResultPtr result) {
    result_ = std::move(result);
    kind_ = PlotKind::Mesh;
    series_.clear();
    selectedX_.reset();
    arrows_.clear();
    if (!result_ || !result_->scene) {
        setEmpty(QStringLiteral("Mesh"), QStringLiteral("No saved mesh"));
        return;
    }
    const auto& scene = *result_->scene;
    title_ = QStringLiteral("Mesh · %1 triangles").arg(scene.triangles.columns);
    xLabel_ = QStringLiteral("z (m)");
    yLabel_ = QStringLiteral("x (m)");
    setDataRange(scene.zSpan[0], scene.zSpan[1], scene.xSpan[0], scene.xSpan[1]);
    update();
}

QRectF PlotWidget::plotRect() const {
    const auto rightMargin = kind_ == PlotKind::Vector ? 110.0 : 34.0;
    return QRectF(72.0, 42.0, std::max(10.0, width() - 72.0 - rightMargin),
                  std::max(10.0, height() - 42.0 - 62.0));
}

QPointF PlotWidget::mapPoint(double horizontal, double vertical) const {
    const auto area = plotRect();
    const auto sx = area.left() + (horizontal - viewRange_[0])
        / (viewRange_[1] - viewRange_[0]) * area.width();
    const auto sy = area.bottom() - (vertical - viewRange_[2])
        / (viewRange_[3] - viewRange_[2]) * area.height();
    return {sx, sy};
}

QPointF PlotWidget::unmapPoint(const QPointF& point) const {
    const auto area = plotRect();
    return {viewRange_[0] + (point.x() - area.left()) / area.width()
                * (viewRange_[1] - viewRange_[0]),
            viewRange_[2] + (area.bottom() - point.y()) / area.height()
                * (viewRange_[3] - viewRange_[2])};
}

void PlotWidget::drawAxes(QPainter& painter, const QRectF& area, bool) const {
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(QColor(220, 220, 220), 1));
    for (int tick = 0; tick <= 5; ++tick) {
        const auto fraction = static_cast<double>(tick) / 5.0;
        const auto x = area.left() + fraction * area.width();
        const auto y = area.bottom() - fraction * area.height();
        painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()));
        painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y));
    }
    painter.setPen(QPen(QColor(55, 55, 55), 1));
    painter.drawRect(area);
    const auto metrics = painter.fontMetrics();
    for (int tick = 0; tick <= 5; ++tick) {
        const auto fraction = static_cast<double>(tick) / 5.0;
        const auto xValue = viewRange_[0] + fraction * (viewRange_[1] - viewRange_[0]);
        const auto yValue = viewRange_[2] + fraction * (viewRange_[3] - viewRange_[2]);
        const auto x = area.left() + fraction * area.width();
        const auto y = area.bottom() - fraction * area.height();
        const auto xText = numberLabel(xValue);
        const auto yText = numberLabel(yValue);
        painter.drawText(QPointF(x - 0.5 * metrics.horizontalAdvance(xText),
                                 area.bottom() + 20.0), xText);
        painter.drawText(QPointF(area.left() - metrics.horizontalAdvance(yText) - 8.0,
                                 y + 0.35 * metrics.height()), yText);
    }
    painter.drawText(QRectF(area.left(), height() - 32.0, area.width(), 24.0),
                     Qt::AlignCenter, xLabel_);
    painter.save();
    painter.translate(18.0, area.center().y());
    painter.rotate(-90.0);
    painter.drawText(QRectF(-0.5 * area.height(), -12.0, area.height(), 24.0),
                     Qt::AlignCenter, yLabel_);
    painter.restore();
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    painter.setFont(titleFont);
    painter.drawText(QRectF(area.left(), 8.0, area.width(), 28.0), Qt::AlignCenter, title_);
    painter.restore();
}

void PlotWidget::drawLines(QPainter& painter, const QRectF& area) const {
    painter.save();
    painter.setClipRect(area.adjusted(-1, -1, 1, 1));
    painter.setRenderHint(QPainter::Antialiasing);
    if (selectedX_) {
        const auto point = mapPoint(*selectedX_, viewRange_[2]);
        painter.setPen(QPen(QColor(90, 90, 90), 1, Qt::DashLine));
        painter.drawLine(QPointF(point.x(), area.top()), QPointF(point.x(), area.bottom()));
    }
    for (std::size_t seriesIndex = 0; seriesIndex < series_.size(); ++seriesIndex) {
        const auto& seriesItem = series_[seriesIndex];
        QPainterPath path;
        std::vector<QPointF> isolatedPoints;
        bool started = false;
        const auto count = std::min(seriesItem.x.size(), seriesItem.y.size());
        const auto isFiniteAt = [&](std::size_t index) {
            return index < count && std::isfinite(seriesItem.x[index])
                && std::isfinite(seriesItem.y[index]);
        };
        for (std::size_t index = 0; index < count; ++index) {
            if (!isFiniteAt(index)) {
                started = false;
                continue;
            }
            const auto point = mapPoint(seriesItem.x[index], seriesItem.y[index]);
            const auto hasPrevious = index > 0 && isFiniteAt(index - 1);
            const auto hasNext = index + 1 < count && isFiniteAt(index + 1);
            if (!hasPrevious && !hasNext) {
                isolatedPoints.push_back(point);
            }
            if (!started) {
                path.moveTo(point);
                started = true;
            } else {
                path.lineTo(point);
            }
        }
        const auto color = seriesColors[seriesIndex % seriesColors.size()];
        painter.setPen(QPen(color, 2));
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
        painter.setPen(Qt::NoPen);
        painter.setBrush(color);
        for (const auto& point : isolatedPoints) {
            painter.drawEllipse(point, 3.5, 3.5);
        }
    }
    painter.restore();

    double legendY = area.top() + 8.0;
    for (std::size_t index = 0; index < series_.size(); ++index) {
        painter.setPen(QPen(seriesColors[index % seriesColors.size()], 3));
        painter.drawLine(QPointF(area.left() + 10.0, legendY + 5.0),
                         QPointF(area.left() + 30.0, legendY + 5.0));
        painter.setPen(QColor(35, 35, 35));
        painter.drawText(QPointF(area.left() + 36.0, legendY + 10.0), series_[index].label);
        legendY += 17.0;
    }
}

void PlotWidget::rebuildSceneCache(const QRectF& area) {
    sceneCache_ = QImage(size() * devicePixelRatioF(), QImage::Format_ARGB32_Premultiplied);
    sceneCache_.setDevicePixelRatio(devicePixelRatioF());
    sceneCache_.fill(Qt::transparent);
    sceneCacheSize_ = size();
    sceneCacheRange_ = viewRange_;
    if (!result_ || !result_->scene) {
        return;
    }
    QPainter painter(&sceneCache_);
    painter.setClipRect(area);
    painter.setRenderHint(QPainter::Antialiasing, false);
    const auto& scene = *result_->scene;
    double epsMaximum = 1.0;
    for (const auto value : scene.epsR) {
        epsMaximum = std::max(epsMaximum, std::abs(value.real()));
    }
    if (kind_ == PlotKind::Mesh) {
        QPen meshPen(QColor(95, 95, 95, 165), 0.7);
        meshPen.setCosmetic(true);
        painter.setPen(meshPen);
    } else {
        painter.setPen(Qt::NoPen);
    }
    for (std::size_t triangle = 0; triangle < scene.triangles.columns; ++triangle) {
        QPolygonF polygon;
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            const auto pointIndex = static_cast<std::size_t>(scene.triangles.at(vertex, triangle));
            if (pointIndex >= scene.points.columns) {
                continue;
            }
            polygon << mapPoint(scene.points.at(1, pointIndex), scene.points.at(0, pointIndex));
        }
        const auto eps = std::abs(scene.epsR[triangle].real());
        const auto normalized = std::clamp((eps - 1.0) / std::max(1e-12, epsMaximum - 1.0),
                                           0.0, 1.0);
        const auto grey = static_cast<int>(std::lround(244.0 - 70.0 * normalized));
        painter.setBrush(QColor(grey, grey, grey));
        painter.drawPolygon(polygon);
    }
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const auto& line : scene.lines) {
        QColor color;
        Qt::PenStyle style = Qt::SolidLine;
        if (line.kind == "pec") {
            color = QColor(242, 201, 76);
        } else if (line.kind == "pmc") {
            color = QColor(47, 128, 237);
        } else if (line.kind == "wave_port") {
            color = QColor(229, 57, 53);
        } else {
            color = QColor(39, 174, 96);
            style = Qt::DashLine;
        }
        painter.setPen(QPen(color, 2.2, style));
        painter.drawLine(mapPoint(line.endpoints[1], line.endpoints[0]),
                         mapPoint(line.endpoints[3], line.endpoints[2]));
    }
}

void PlotWidget::drawVector(QPainter& painter, const QRectF& area) {
    if (sceneCache_.isNull() || sceneCacheSize_ != size() || sceneCacheRange_ != viewRange_) {
        rebuildSceneCache(area);
    }
    painter.drawImage(QPointF(0, 0), sceneCache_);
    painter.save();
    painter.setClipRect(area);
    painter.setRenderHint(QPainter::Antialiasing);
    constexpr double arrowLength = 16.0;
    constexpr double headLength = 5.0;
    for (const auto& arrow : arrows_) {
        const auto center = mapPoint(arrow.z, arrow.x);
        const QPointF direction(arrow.horizontal, -arrow.vertical);
        const auto start = center - 0.5 * arrowLength * direction;
        const auto end = center + 0.5 * arrowLength * direction;
        const auto denominator = magnitudeMax_ - magnitudeMin_;
        const auto normalized = denominator > 0.0
            ? (arrow.magnitude - magnitudeMin_) / denominator : 0.5;
        painter.setPen(QPen(viridis(normalized), 1.4));
        painter.drawLine(start, end);
        if (arrow.magnitude > 0.0) {
            const QPointF normal(-direction.y(), direction.x());
            painter.drawLine(end, end - headLength * direction + 0.55 * headLength * normal);
            painter.drawLine(end, end - headLength * direction - 0.55 * headLength * normal);
        }
    }
    painter.restore();

    const QRectF bar(area.right() + 32.0, area.top() + 15.0, 16.0,
                     std::max(40.0, area.height() - 30.0));
    QLinearGradient gradient(bar.bottomLeft(), bar.topLeft());
    for (int step = 0; step <= 20; ++step) {
        const auto value = static_cast<double>(step) / 20.0;
        gradient.setColorAt(value, viridis(value));
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(45, 45, 45));
    painter.drawRect(bar);
    painter.drawText(QPointF(bar.right() + 6.0, bar.top() + 5.0), numberLabel(magnitudeMax_));
    painter.drawText(QPointF(bar.right() + 6.0, bar.bottom()), numberLabel(magnitudeMin_));
    painter.save();
    painter.translate(bar.right() + 54.0, bar.center().y());
    painter.rotate(-90.0);
    painter.drawText(QRectF(-0.5 * bar.height(), -10.0, bar.height(), 20.0),
                     Qt::AlignCenter, QStringLiteral("in-plane magnitude"));
    painter.restore();
}

void PlotWidget::drawMesh(QPainter& painter, const QRectF& area) {
    if (sceneCache_.isNull() || sceneCacheSize_ != size() || sceneCacheRange_ != viewRange_) {
        rebuildSceneCache(area);
    }
    painter.drawImage(QPointF(0, 0), sceneCache_);
}

void PlotWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), palette().base());
    if (kind_ == PlotKind::Empty) {
        QFont titleFont = painter.font();
        titleFont.setBold(true);
        painter.setFont(titleFont);
        painter.drawText(QRectF(0, 15, width(), 30), Qt::AlignCenter, title_);
        painter.setFont(font());
        painter.setPen(palette().mid().color());
        painter.drawText(rect(), Qt::AlignCenter, message_);
        return;
    }
    const auto area = plotRect();
    drawAxes(painter, area, kind_ == PlotKind::Vector);
    if (kind_ == PlotKind::Lines) {
        drawLines(painter, area);
    } else if (kind_ == PlotKind::Vector) {
        drawVector(painter, area);
    } else {
        drawMesh(painter, area);
    }
}

void PlotWidget::resizeEvent(QResizeEvent* event) {
    invalidateSceneCache();
    QWidget::resizeEvent(event);
}

void PlotWidget::wheelEvent(QWheelEvent* event) {
    if (!plotRect().contains(event->position()) || kind_ == PlotKind::Empty) {
        QWidget::wheelEvent(event);
        return;
    }
    const auto anchor = unmapPoint(event->position());
    const auto factor = event->angleDelta().y() > 0 ? 0.82 : 1.22;
    viewRange_[0] = anchor.x() + (viewRange_[0] - anchor.x()) * factor;
    viewRange_[1] = anchor.x() + (viewRange_[1] - anchor.x()) * factor;
    viewRange_[2] = anchor.y() + (viewRange_[2] - anchor.y()) * factor;
    viewRange_[3] = anchor.y() + (viewRange_[3] - anchor.y()) * factor;
    invalidateSceneCache();
    update();
    event->accept();
}

void PlotWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && plotRect().contains(event->position())) {
        dragging_ = true;
        dragStart_ = event->position().toPoint();
        dragStartRange_ = viewRange_;
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    }
}

void PlotWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) {
        return;
    }
    const auto area = plotRect();
    const auto delta = event->position() - QPointF(dragStart_);
    const auto dx = -delta.x() / area.width() * (dragStartRange_[1] - dragStartRange_[0]);
    const auto dy = delta.y() / area.height() * (dragStartRange_[3] - dragStartRange_[2]);
    viewRange_ = {dragStartRange_[0] + dx, dragStartRange_[1] + dx,
                  dragStartRange_[2] + dy, dragStartRange_[3] + dy};
    invalidateSceneCache();
    update();
}

void PlotWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && dragging_) {
        dragging_ = false;
        unsetCursor();
        event->accept();
    }
}

void PlotWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        resetView();
        event->accept();
    }
}

void PlotWidget::resetView() {
    viewRange_ = dataRange_;
    invalidateSceneCache();
    update();
}

void PlotWidget::invalidateSceneCache() {
    sceneCache_ = QImage();
    sceneCacheSize_ = {};
}

} // namespace wavefem
