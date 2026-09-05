#include "vtk_field_view.hpp"

#include <QLabel>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <string>
#include <vector>

#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
#include <QVTKOpenGLNativeWidget.h>
#include <vtkActor.h>
#include <vtkArrowSource.h>
#include <vtkAxesActor.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCutter.h>
#include <vtkDataSetMapper.h>
#include <vtkDoubleArray.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGlyph3D.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVersion.h>
#endif

namespace femperiodic {

#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
namespace {

void configureLookupTable(vtkLookupTable* lookup, bool materialOnly) {
    lookup->SetNumberOfTableValues(256);
    if (!materialOnly) {
        lookup->SetHueRange(0.72, 0.15);
        lookup->SetSaturationRange(0.9, 0.9);
        lookup->SetValueRange(0.75, 1.0);
        lookup->Build();
        return;
    }
    lookup->SetSaturationRange(0.0, 0.0);
    lookup->SetValueRange(1.0, 1.0);
    lookup->Build();
    constexpr std::array<std::array<double, 3>, 3> stops{{
        {{59.0 / 255.0, 76.0 / 255.0, 192.0 / 255.0}},
        {{221.0 / 255.0, 221.0 / 255.0, 221.0 / 255.0}},
        {{180.0 / 255.0, 4.0 / 255.0, 38.0 / 255.0}},
    }};
    for (int index = 0; index < 256; ++index) {
        const auto position = 2.0 * static_cast<double>(index) / 255.0;
        const auto lower = std::min<std::size_t>(static_cast<std::size_t>(position), 1U);
        const auto fraction = position - static_cast<double>(lower);
        std::array<double, 3> color{};
        for (std::size_t channel = 0; channel < color.size(); ++channel) {
            color[channel] = stops[lower][channel]
                + fraction * (stops[lower + 1][channel] - stops[lower][channel]);
        }
        lookup->SetTableValue(index, color[0], color[1], color[2], 1.0);
    }
}

} // namespace
#endif

class VtkFieldView::Impl {
public:
    explicit Impl(VtkFieldView* owner) : owner_(owner) {
        auto* layout = new QVBoxLayout(owner_);
        layout->setContentsMargins(0, 0, 0, 0);
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        widget_ = new QVTKOpenGLNativeWidget(owner_);
        renderWindow_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
        renderer_ = vtkSmartPointer<vtkRenderer>::New();
        renderWindow_->AddRenderer(renderer_);
        widget_->setRenderWindow(renderWindow_);
        renderer_->SetBackground(0.08, 0.09, 0.12);

        axes_ = vtkSmartPointer<vtkAxesActor>::New();
        axes_->SetShaftTypeToCylinder();
        axes_->SetNormalizedShaftLength(0.70, 0.70, 0.70);
        axes_->SetNormalizedTipLength(0.30, 0.30, 0.30);
        orientationMarker_ = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
        orientationMarker_->SetOrientationMarker(axes_);
        orientationMarker_->SetInteractor(widget_->interactor());
        orientationMarker_->SetViewport(0.0, 0.0, 0.18, 0.18);
        orientationMarker_->SetEnabled(1);
        orientationMarker_->InteractiveOff();

        scalarBar_ = vtkSmartPointer<vtkScalarBarActor>::New();
        scalarBar_->SetNumberOfLabels(5);
        scalarBar_->SetMaximumNumberOfColors(256);
        scalarBar_->SetPosition(0.84, 0.12);
        scalarBar_->SetWidth(0.12);
        scalarBar_->SetHeight(0.72);
        scalarBar_->GetLabelTextProperty()->SetColor(0.95, 0.95, 0.95);
        scalarBar_->GetTitleTextProperty()->SetColor(0.95, 0.95, 0.95);
        scalarBar_->SetVisibility(false);
        renderer_->AddViewProp(scalarBar_);
        layout->addWidget(widget_);
#else
        unavailable_ = new QLabel(
            QStringLiteral("This build has no VTK support. Reconfigure with "
                           "FEM_PERIODIC_MODE_VIEWER_WITH_VTK=ON to view tetrahedral fields."),
            owner_);
        unavailable_->setWordWrap(true);
        unavailable_->setAlignment(Qt::AlignCenter);
        layout->addWidget(unavailable_);
#endif
    }

    void setData(MeshPtr mesh, MaterialStatePtr material, ModeFieldsPtr fields) {
        mesh_ = std::move(mesh);
        material_ = std::move(material);
        fields_ = std::move(fields);
        rebuild();
    }

    void setSelection(FieldFamily family, int component, ScalarQuantity quantity) {
        family_ = family;
        component_ = std::clamp(component, 0, 2);
        quantity_ = quantity;
        rebuildScalars();
    }

    void setSlice(int axis, double fraction) {
        sliceAxis_ = std::clamp(axis, 0, 2);
        sliceFraction_ = std::clamp(fraction, 0.0, 1.0);
        updatePlane();
        if (sliceEnabled_ && !materialOnly_) {
            rebuildScalars();
        }
    }

    void setSliceEnabled(bool enabled) {
        sliceEnabled_ = enabled;
        rebuildScalars();
        updatePlane();
    }

    void setMaterialOnly(bool enabled) {
        materialOnly_ = enabled;
        rebuildScalars();
        updatePlane();
    }

    void clearData(const QString& message) {
        mesh_.reset();
        material_.reset();
        fields_.reset();
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        (void)message;
        renderer_->RemoveAllViewProps();
        if (widget_->isValid()) {
            renderWindow_->Render();
        }
#else
        unavailable_->setText(message);
#endif
    }

    [[nodiscard]] std::size_t glyphCount() const {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        return glyphData_ == nullptr
            ? 0U : static_cast<std::size_t>(glyphData_->GetNumberOfPoints());
#else
        return 0U;
#endif
    }

    [[nodiscard]] bool sliceIsHeatMapOnly() const {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        if (!sliceEnabled_ || !cutter_ || !cutterActor_ || !surfaceActor_
            || !glyphActor_) {
            return false;
        }
        cutter_->Update();
        return cutter_->GetOutput()->GetNumberOfCells() > 0
            && cutterActor_->GetVisibility() != 0
            && surfaceActor_->GetVisibility() == 0
            && glyphActor_->GetVisibility() == 0;
#else
        return false;
#endif
    }

    [[nodiscard]] bool annotationsVisible() const {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        return axes_ && orientationMarker_ && scalarBar_
            && orientationMarker_->GetEnabled() != 0
            && scalarBar_->GetVisibility() != 0;
#else
        return false;
#endif
    }

private:
    void rebuild() {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        renderer_->RemoveAllViewProps();
        scalarBar_->SetVisibility(false);
        renderer_->AddViewProp(scalarBar_);
        grid_ = nullptr;
        if (!mesh_ || mesh_->dimension != 3 || mesh_->topology != "tetra4" || !fields_) {
            renderWindow_->Render();
            return;
        }
        grid_ = vtkSmartPointer<vtkUnstructuredGrid>::New();
        vtkNew<vtkPoints> points;
        points->SetNumberOfPoints(static_cast<vtkIdType>(mesh_->points.size()));
        for (std::size_t index = 0; index < mesh_->points.size(); ++index) {
            const auto& point = mesh_->points[index];
            points->SetPoint(static_cast<vtkIdType>(index), point[0], point[1], point[2]);
        }
        grid_->SetPoints(points);
        vtkNew<vtkCellArray> cells;
        for (const auto& cell : mesh_->cells) {
            if (cell.size() != 4) {
                continue;
            }
            std::array<vtkIdType, 4> ids{
                static_cast<vtkIdType>(cell[0]), static_cast<vtkIdType>(cell[1]),
                static_cast<vtkIdType>(cell[2]), static_cast<vtkIdType>(cell[3])};
            cells->InsertNextCell(4, ids.data());
        }
        grid_->SetCells(VTK_TETRA, cells);

        lookup_ = vtkSmartPointer<vtkLookupTable>::New();
        configureLookupTable(lookup_, materialOnly_);

        surfaceMapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
        surfaceMapper_->SetInputData(grid_);
        surfaceMapper_->SetLookupTable(lookup_);
        surfaceMapper_->SetScalarModeToUseCellData();
        surfaceActor_ = vtkSmartPointer<vtkActor>::New();
        surfaceActor_->SetMapper(surfaceMapper_);
        surfaceActor_->GetProperty()->SetOpacity(0.22);
        surfaceActor_->GetProperty()->EdgeVisibilityOn();
        surfaceActor_->GetProperty()->SetEdgeColor(0.25, 0.25, 0.3);
        renderer_->AddActor(surfaceActor_);

        plane_ = vtkSmartPointer<vtkPlane>::New();
        cutter_ = vtkSmartPointer<vtkCutter>::New();
        cutter_->SetCutFunction(plane_);
        cutter_->SetInputData(grid_);
        cutterMapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
        cutterMapper_->SetInputConnection(cutter_->GetOutputPort());
        cutterMapper_->SetLookupTable(lookup_);
        cutterMapper_->SetScalarModeToUseCellData();
        cutterActor_ = vtkSmartPointer<vtkActor>::New();
        cutterActor_->SetMapper(cutterMapper_);
        cutterActor_->GetProperty()->SetOpacity(1.0);
        cutterActor_->GetProperty()->EdgeVisibilityOff();
        renderer_->AddActor(cutterActor_);

        glyphSource_ = vtkSmartPointer<vtkArrowSource>::New();
        glyphSource_->SetTipResolution(10);
        glyphSource_->SetShaftResolution(8);
        glyphFilter_ = vtkSmartPointer<vtkGlyph3D>::New();
        glyphFilter_->SetSourceConnection(glyphSource_->GetOutputPort());
        glyphFilter_->SetVectorModeToUseVector();
        glyphFilter_->SetScaleModeToScaleByVector();
        glyphFilter_->OrientOn();
        glyphMapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
        glyphMapper_->SetInputConnection(glyphFilter_->GetOutputPort());
        glyphMapper_->ScalarVisibilityOff();
        glyphActor_ = vtkSmartPointer<vtkActor>::New();
        glyphActor_->SetMapper(glyphMapper_);
        glyphActor_->GetProperty()->SetColor(1.0, 0.72, 0.16);
        renderer_->AddActor(glyphActor_);

        grid_->GetBounds(bounds_.data());
        rebuildScalars();
        updatePlane();
        renderer_->ResetCamera();
        renderWindow_->Render();
#else
        if (unavailable_) {
            unavailable_->setText(QStringLiteral(
                "VTK support is disabled; 3D mesh and field metadata remain readable."));
        }
#endif
    }

    void rebuildScalars() {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        if (!grid_ || !mesh_ || !fields_) {
            return;
        }
        std::vector<double> values(mesh_->cells.size(), 0.0);
        std::vector<std::size_t> counts(mesh_->cells.size(), 0);
        const std::vector<std::array<Complex, 3>>* vectorSource = nullptr;
        if (materialOnly_) {
            if (!material_ || material_->epsilonR.size() != mesh_->cells.size()
                || material_->muR.size() != mesh_->cells.size()) {
                return;
            }
            for (std::size_t cell = 0; cell < mesh_->cells.size(); ++cell) {
                const auto& epsilon = material_->epsilonR[cell];
                const auto& mu = material_->muR[cell];
                values[cell] = std::sqrt(std::max(
                    {std::abs(epsilon[0] * mu[0]), std::abs(epsilon[1] * mu[1]),
                     std::abs(epsilon[2] * mu[2])}));
                counts[cell] = 1;
            }
        } else {
            const auto& source = family_ == FieldFamily::Electric
                ? fields_->electric : fields_->magnetic;
            if (source.size() != mesh_->sampleOwnerCells.size()) {
                return;
            }
            vectorSource = &source;
            for (std::size_t sample = 0; sample < source.size(); ++sample) {
                const auto owner = static_cast<std::size_t>(mesh_->sampleOwnerCells[sample]);
                values[owner] += scalarValue(
                    source[sample][static_cast<std::size_t>(component_)], quantity_);
                ++counts[owner];
            }
        }
        auto minimum = std::numeric_limits<double>::infinity();
        auto maximum = -minimum;
        vtkNew<vtkDoubleArray> scalars;
        scalars->SetName("selected_field");
        scalars->SetNumberOfComponents(1);
        scalars->SetNumberOfTuples(static_cast<vtkIdType>(values.size()));
        for (std::size_t cell = 0; cell < values.size(); ++cell) {
            if (counts[cell] > 0) {
                values[cell] /= static_cast<double>(counts[cell]);
            }
            minimum = std::min(minimum, values[cell]);
            maximum = std::max(maximum, values[cell]);
            scalars->SetValue(static_cast<vtkIdType>(cell), values[cell]);
        }
        if (!(maximum > minimum)) {
            maximum = minimum + 1.0;
        }
        grid_->GetCellData()->SetScalars(scalars);
        surfaceMapper_->SetScalarRange(minimum, maximum);
        cutterMapper_->SetScalarRange(minimum, maximum);
        lookup_->SetTableRange(minimum, maximum);
        configureLookupTable(lookup_, materialOnly_);
        if (materialOnly_) {
            scalarBar_->SetTitle("material |n_eff|");
        } else {
            const auto selected = std::format(
                "{}{}", family_ == FieldFamily::Electric ? "E" : "H",
                "xyz"[static_cast<std::size_t>(component_)]);
            std::string title;
            switch (quantity_) {
            case ScalarQuantity::Magnitude:
                title = std::format("|{}|", selected);
                break;
            case ScalarQuantity::Real:
                title = std::format("Re({})", selected);
                break;
            case ScalarQuantity::Imaginary:
                title = std::format("Im({})", selected);
                break;
            case ScalarQuantity::Phase:
                title = std::format("phase({})", selected);
                break;
            }
            scalarBar_->SetTitle(title.c_str());
        }
        scalarBar_->SetLookupTable(lookup_);
        scalarBar_->SetVisibility(true);
        surfaceActor_->GetProperty()->SetOpacity(materialOnly_ ? 0.72 : 0.22);
        surfaceActor_->SetVisibility(!sliceEnabled_);
        cutterActor_->SetVisibility(sliceEnabled_);
        grid_->Modified();
        cutter_->Update();
        if (vectorSource != nullptr && !sliceEnabled_) {
            rebuildGlyphs(*vectorSource);
        } else {
            glyphActor_->SetVisibility(false);
            glyphData_ = nullptr;
        }
        renderWindow_->Render();
#endif
    }

#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
    void rebuildGlyphs(const std::vector<std::array<Complex, 3>>& source) {
        constexpr std::size_t maximumGlyphs = 1'200;
        glyphData_ = vtkSmartPointer<vtkPolyData>::New();
        vtkNew<vtkPoints> points;
        vtkNew<vtkDoubleArray> vectors;
        vectors->SetName("field_vector");
        vectors->SetNumberOfComponents(3);

        const auto count = std::min(source.size(), mesh_->samplePoints.size());
        const auto stride = std::max<std::size_t>(
            1, (count + maximumGlyphs - 1) / maximumGlyphs);
        double maximumMagnitude{};
        for (std::size_t sample = 0; sample < count; sample += stride) {
            const auto& location = mesh_->samplePoints[sample];
            std::array<double, 3> vector{};
            for (std::size_t component = 0; component < vector.size(); ++component) {
                // A complex phasor has no component-wise magnitude or phase
                // direction.  Use the physical t=0 vector for scalar
                // magnitude/phase views, and the quadrature vector only when
                // the user explicitly selects the imaginary quantity.
                vector[component] = quantity_ == ScalarQuantity::Imaginary
                    ? source[sample][component].imag()
                    : source[sample][component].real();
            }
            const auto magnitude = std::sqrt(
                vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
            maximumMagnitude = std::max(maximumMagnitude, magnitude);
            points->InsertNextPoint(location.data());
            vectors->InsertNextTuple(vector.data());
        }

        glyphData_->SetPoints(points);
        glyphData_->GetPointData()->SetVectors(vectors);
        glyphFilter_->SetInputData(glyphData_);
        const auto dx = bounds_[1] - bounds_[0];
        const auto dy = bounds_[3] - bounds_[2];
        const auto dz = bounds_[5] - bounds_[4];
        const auto diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
        const auto visible = maximumMagnitude > std::numeric_limits<double>::epsilon()
            && diagonal > 0.0;
        glyphFilter_->SetScaleFactor(
            visible ? 0.075 * diagonal / maximumMagnitude : 0.0);
        glyphActor_->SetVisibility(visible);
        glyphFilter_->Update();
    }
#endif

    void updatePlane() {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
        if (!plane_ || !grid_) {
            return;
        }
        std::array<double, 3> origin{
            0.5 * (bounds_[0] + bounds_[1]),
            0.5 * (bounds_[2] + bounds_[3]),
            0.5 * (bounds_[4] + bounds_[5])};
        origin[static_cast<std::size_t>(sliceAxis_)] =
            bounds_[static_cast<std::size_t>(2 * sliceAxis_)]
            + sliceFraction_ * (bounds_[static_cast<std::size_t>(2 * sliceAxis_ + 1)]
                                - bounds_[static_cast<std::size_t>(2 * sliceAxis_)]);
        std::array<double, 3> normal{0.0, 0.0, 0.0};
        normal[static_cast<std::size_t>(sliceAxis_)] = 1.0;
        plane_->SetOrigin(origin.data());
        plane_->SetNormal(normal.data());
        surfaceActor_->SetVisibility(!sliceEnabled_);
        if (sliceEnabled_) {
            glyphActor_->SetVisibility(false);
        }
        cutterActor_->SetVisibility(sliceEnabled_);
        cutter_->Update();
        renderWindow_->Render();
#endif
    }

    VtkFieldView* owner_{};
    MeshPtr mesh_;
    MaterialStatePtr material_;
    ModeFieldsPtr fields_;
    FieldFamily family_{FieldFamily::Electric};
    int component_{0};
    ScalarQuantity quantity_{ScalarQuantity::Magnitude};
    int sliceAxis_{2};
    double sliceFraction_{0.5};
    bool sliceEnabled_{false};
    bool materialOnly_{false};
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
    QVTKOpenGLNativeWidget* widget_{};
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkAxesActor> axes_;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationMarker_;
    vtkSmartPointer<vtkScalarBarActor> scalarBar_;
    vtkSmartPointer<vtkUnstructuredGrid> grid_;
    vtkSmartPointer<vtkLookupTable> lookup_;
    vtkSmartPointer<vtkDataSetMapper> surfaceMapper_;
    vtkSmartPointer<vtkActor> surfaceActor_;
    vtkSmartPointer<vtkPlane> plane_;
    vtkSmartPointer<vtkCutter> cutter_;
    vtkSmartPointer<vtkDataSetMapper> cutterMapper_;
    vtkSmartPointer<vtkActor> cutterActor_;
    vtkSmartPointer<vtkArrowSource> glyphSource_;
    vtkSmartPointer<vtkGlyph3D> glyphFilter_;
    vtkSmartPointer<vtkPolyData> glyphData_;
    vtkSmartPointer<vtkDataSetMapper> glyphMapper_;
    vtkSmartPointer<vtkActor> glyphActor_;
    std::array<double, 6> bounds_{};
#else
    QLabel* unavailable_{};
#endif
};

VtkFieldView::VtkFieldView(QWidget* parent)
    : QWidget(parent), impl_(std::make_unique<Impl>(this)) {}

VtkFieldView::~VtkFieldView() = default;

void VtkFieldView::setData(MeshPtr mesh, MaterialStatePtr material, ModeFieldsPtr fields) {
    impl_->setData(std::move(mesh), std::move(material), std::move(fields));
}

void VtkFieldView::setSelection(
    FieldFamily family, int component, ScalarQuantity quantity) {
    impl_->setSelection(family, component, quantity);
}

void VtkFieldView::setSlice(int axis, double fraction) {
    impl_->setSlice(axis, fraction);
}

void VtkFieldView::setSliceEnabled(bool enabled) {
    impl_->setSliceEnabled(enabled);
}

void VtkFieldView::setMaterialOnly(bool enabled) {
    impl_->setMaterialOnly(enabled);
}

void VtkFieldView::clearData(const QString& message) {
    impl_->clearData(message);
}

std::size_t VtkFieldView::glyphCount() const {
    return impl_->glyphCount();
}

bool VtkFieldView::sliceIsHeatMapOnly() const {
    return impl_->sliceIsHeatMapOnly();
}

bool VtkFieldView::annotationsVisible() const {
    return impl_->annotationsVisible();
}

bool VtkFieldView::available() {
#if defined(FEM_PERIODIC_MODE_VIEWER_WITH_VTK)
    return true;
#else
    return false;
#endif
}

} // namespace femperiodic
