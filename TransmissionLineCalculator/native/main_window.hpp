#pragma once

#include "model.hpp"

#include <QMainWindow>
#include <QString>

#include <cstdint>
#include <memory>
#include <vector>

class QCheckBox;
class QComboBox;
class QFormLayout;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPushButton;

namespace tl {

class FieldPlot;

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

    // These hooks make the asynchronous application path available to the
    // offscreen smoke runner without exposing individual widgets.
    void calculateForSmokeTest();
    [[nodiscard]] bool defaultsMatchForSmokeTest();
    [[nodiscard]] bool solveInProgress() const noexcept;
    [[nodiscard]] bool hasResult() const noexcept;

private:
    enum class InputKey {
        Frequency,
        InnerRadius,
        OuterRadius,
        OuterConductorThickness,
        TraceWidth,
        SubstrateHeight,
        ConductorThickness,
        GroundSpacing,
        CenterWidth,
        Gap,
        GroundWidth,
        EpsilonR,
        LossTangent,
        DomainPaddingFactor,
        MetalConductivity,
        MaxElementSize,
    };

    struct EntryDefinition {
        InputKey key;
        const char* label;
        const char* defaultText;
        double siScale;
        bool strictlyPositive{true};
        bool optional{};
    };

    struct EntryControl {
        EntryDefinition definition;
        QLineEdit* editor{};
    };

    struct SolveOutcome {
        std::shared_ptr<Result> result;
        QString error;
        std::uint64_t generation{};
        bool refinement{};
    };

    void buildUi();
    void configureEntries(LineType type);
    [[nodiscard]] static std::vector<EntryDefinition> definitionsFor(LineType type);
    [[nodiscard]] LineType selectedLineType() const;
    [[nodiscard]] Parameters readParameters() const;
    void calculate();
    void refine();
    void startSolve(Parameters parameters, bool refinement);
    void applyOutcome(const SolveOutcome& outcome);
    void invalidateSolution(const QString& status);
    void showResult();
    void setBusy(bool busy);
    void setStatus(const QString& message, bool error = false);
    void updateHeading();

    QComboBox* lineTypeCombo_{};
    QGroupBox* parametersGroup_{};
    QFormLayout* parameterForm_{};
    QPushButton* calculateButton_{};
    QPushButton* refineButton_{};
    QCheckBox* meshCheck_{};
    QLabel* statusLabel_{};
    QLabel* headingLabel_{};
    QLabel* resultsLabel_{};
    FieldPlot* electricPlot_{};
    FieldPlot* magneticPlot_{};
    std::vector<EntryControl> entries_;
    std::shared_ptr<const Result> result_;
    std::uint64_t generation_{};
    bool busy_{};
};

} // namespace tl
