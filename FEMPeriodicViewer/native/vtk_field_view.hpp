#pragma once

#include "model.hpp"

#include <QString>
#include <QWidget>

#include <cstddef>
#include <memory>

namespace femperiodic {

class VtkFieldView final : public QWidget {
public:
    explicit VtkFieldView(QWidget* parent = nullptr);
    ~VtkFieldView() override;

    void setData(MeshPtr mesh, MaterialStatePtr material, ModeFieldsPtr fields);
    void setSelection(FieldFamily family, int component, ScalarQuantity quantity);
    void setSlice(int axis, double fraction);
    void setSliceEnabled(bool enabled);
    void setMaterialOnly(bool enabled);
    void clearData(const QString& message);
    [[nodiscard]] std::size_t glyphCount() const;
    [[nodiscard]] bool sliceIsHeatMapOnly() const;
    [[nodiscard]] bool annotationsVisible() const;

    [[nodiscard]] static bool available();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace femperiodic
