#pragma once

#include "model.hpp"

#include <QPoint>
#include <QRectF>
#include <QString>
#include <QWidget>

#include <memory>
#include <vector>

namespace femperiodic {

class FieldPlot2D final : public QWidget {
public:
    explicit FieldPlot2D(QWidget* parent = nullptr);

    void setData(MeshPtr mesh, MaterialStatePtr material, ModeFieldsPtr fields);
    void setSelection(FieldFamily family, int component, ScalarQuantity quantity);
    void setMaterialOnly(bool enabled);
    void clearData(const QString& message);
    void resetView();

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    [[nodiscard]] QRectF plotRect() const;
    [[nodiscard]] QPointF mapPoint(double x, double z) const;
    [[nodiscard]] std::array<double, 2> unmapPoint(const QPointF& point) const;
    void rebuildValues();

    MeshPtr mesh_;
    MaterialStatePtr material_;
    ModeFieldsPtr fields_;
    FieldFamily family_{FieldFamily::Electric};
    int component_{0};
    ScalarQuantity quantity_{ScalarQuantity::Magnitude};
    bool materialOnly_{false};
    std::vector<double> cellValues_;
    double valueMin_{0.0};
    double valueMax_{1.0};
    std::array<double, 4> dataRange_{0.0, 1.0, 0.0, 1.0}; // x min/max, z min/max
    std::array<double, 4> viewRange_{dataRange_};
    QString message_{QStringLiteral("Open a FEM periodic HDF5 result")};
    bool dragging_{false};
    QPoint dragStart_;
    std::array<double, 4> dragStartRange_{};
};

} // namespace femperiodic
