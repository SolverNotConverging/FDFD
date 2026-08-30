#pragma once

#include "field_view.hpp"
#include "model.hpp"

#include <QString>
#include <QWidget>

#include <memory>

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

private:
    FieldFamily family_;
    std::shared_ptr<const Result> result_;
    QString emptyMessage_;
    bool meshVisible_{};
    FieldViewMode viewMode_{FieldViewMode::Focused};
};

} // namespace tl
