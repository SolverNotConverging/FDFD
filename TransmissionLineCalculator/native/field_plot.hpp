#pragma once

#include "field_view.hpp"
#include "model.hpp"

#include <QString>
#include <QWidget>

#include <memory>
#include <optional>

namespace tl {

enum class FieldFamily { Electric, Magnetic };

class FieldPlot final : public QWidget {
public:
    explicit FieldPlot(FieldFamily family, QWidget* parent = nullptr);

    void setResult(std::shared_ptr<const Result> result);
    void setMeshVisible(bool visible);
    void setViewMode(FieldViewMode mode);
    void setEmptyMessage(QString message);

    [[nodiscard]] QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    [[nodiscard]] std::optional<FieldViewBounds> defaultBounds() const;
    [[nodiscard]] std::optional<FieldViewBounds> activeBounds() const;
    [[nodiscard]] QRectF plotArea(const FieldViewBounds& bounds) const;
    void resetView();

    FieldFamily family_;
    std::shared_ptr<const Result> result_;
    QString emptyMessage_;
    bool meshVisible_{};
    FieldViewMode viewMode_{FieldViewMode::Focused};
    std::optional<FieldViewBounds> userBounds_;
    QPointF lastMousePosition_;
    bool dragging_{};
};

} // namespace tl
