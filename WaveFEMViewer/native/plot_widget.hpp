#pragma once

#include "model.hpp"

#include <QImage>
#include <QPoint>
#include <QWidget>

#include <optional>
#include <vector>

namespace wavefem {

struct PlotSeries {
    QString label;
    std::vector<double> x;
    std::vector<double> y;
};

class PlotWidget final : public QWidget {
public:
    explicit PlotWidget(QWidget* parent = nullptr);

    void setEmpty(QString title, QString message);
    void setLines(std::vector<PlotSeries> series, QString title, QString xLabel,
                  QString yLabel, std::optional<double> selectedX = std::nullopt);
    void setModal(const ModeData& mode, FieldName field, int component,
                  ScalarQuantity quantity);
    void setVector(ResultPtr result, FieldName field, FieldPart part,
                   ScalarQuantity quantity, std::size_t maxArrows = 1200);
    void setMesh(ResultPtr result);
    void resetView();

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    enum class PlotKind { Empty, Lines, Vector, Mesh };

    struct Arrow {
        double x{};
        double z{};
        double horizontal{};
        double vertical{};
        double magnitude{};
    };

    [[nodiscard]] QRectF plotRect() const;
    [[nodiscard]] QPointF mapPoint(double horizontal, double vertical) const;
    [[nodiscard]] QPointF unmapPoint(const QPointF& point) const;
    void setDataRange(double xMin, double xMax, double yMin, double yMax);
    void drawAxes(QPainter& painter, const QRectF& area, bool reserveColorbar) const;
    void drawLines(QPainter& painter, const QRectF& area) const;
    void drawVector(QPainter& painter, const QRectF& area);
    void drawMesh(QPainter& painter, const QRectF& area);
    void rebuildSceneCache(const QRectF& area);
    void invalidateSceneCache();

    PlotKind kind_{PlotKind::Empty};
    QString title_;
    QString message_;
    QString xLabel_;
    QString yLabel_;
    std::vector<PlotSeries> series_;
    std::optional<double> selectedX_;
    ResultPtr result_;
    std::vector<Arrow> arrows_;
    double magnitudeMin_{};
    double magnitudeMax_{};
    std::array<double, 4> dataRange_{0.0, 1.0, 0.0, 1.0};
    std::array<double, 4> viewRange_{0.0, 1.0, 0.0, 1.0};
    QImage sceneCache_;
    QSize sceneCacheSize_;
    std::array<double, 4> sceneCacheRange_{};
    bool dragging_{false};
    QPoint dragStart_;
    std::array<double, 4> dragStartRange_{};
};

} // namespace wavefem
