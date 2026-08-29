#include "main_window.hpp"

#include "field_plot.hpp"
#include "solver.hpp"

#include <QtConcurrent/QtConcurrentRun>

#include <QCheckBox>
#include <QComboBox>
#include <QFontDatabase>
#include <QFormLayout>
#include <QFrame>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLayoutItem>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <stdexcept>
#include <utility>

namespace tl {
namespace {

template <typename Value>
[[nodiscard]] QString formatNumber(const Value& value, int digits = 7) {
    const std::complex<double> number{value};
    const auto real = number.real();
    const auto imaginary = number.imag();
    const auto threshold = std::pow(10.0, -(digits - 1)) * std::max(1.0, std::abs(real));
    if (std::abs(imaginary) <= threshold) {
        return QString::number(real, 'g', digits);
    }
    return QStringLiteral("%1%2%3j")
        .arg(QString::number(real, 'g', digits),
             imaginary >= 0.0 ? QStringLiteral("+") : QString{},
             QString::number(imaginary, 'g', digits));
}

template <typename Value>
[[nodiscard]] QString withUnit(const Value& value, const QString& unit) {
    return QStringLiteral("%1 %2").arg(formatNumber(value), unit);
}

[[nodiscard]] QString lineName(LineType type) {
    switch (type) {
    case LineType::Coaxial:
        return QStringLiteral("Coaxial");
    case LineType::Microstrip:
        return QStringLiteral("Microstrip");
    case LineType::Stripline:
        return QStringLiteral("Stripline");
    case LineType::CoplanarWaveguide:
        return QStringLiteral("CPW");
    }
    return QStringLiteral("Transmission line");
}

[[noreturn]] void inputError(const QString& message) {
    const auto utf8 = message.toUtf8();
    throw std::invalid_argument(utf8.constData());
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    buildUi();
    configureEntries(LineType::Coaxial);
    updateHeading();
    setBusy(false);
    setStatus(QStringLiteral("Ready"));
    resize(1460, 860);
}

void MainWindow::calculateForSmokeTest() {
    calculate();
}

bool MainWindow::solveInProgress() const noexcept {
    return busy_;
}

bool MainWindow::hasResult() const noexcept {
    return static_cast<bool>(result_);
}

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    auto* outer = new QHBoxLayout(central);
    outer->setContentsMargins(10, 10, 10, 10);
    outer->setSpacing(12);

    auto* controls = new QWidget;
    auto* controlsLayout = new QVBoxLayout(controls);
    controlsLayout->setContentsMargins(8, 8, 8, 8);

    auto* typeGroup = new QGroupBox(QStringLiteral("Transmission line"), controls);
    auto* typeLayout = new QVBoxLayout(typeGroup);
    lineTypeCombo_ = new QComboBox(typeGroup);
    lineTypeCombo_->addItems({QStringLiteral("Coaxial"), QStringLiteral("Microstrip"),
                              QStringLiteral("Stripline"), QStringLiteral("CPW")});
    typeLayout->addWidget(lineTypeCombo_);
    controlsLayout->addWidget(typeGroup);

    parametersGroup_ = new QGroupBox(QStringLiteral("Parameters"), controls);
    parameterForm_ = new QFormLayout(parametersGroup_);
    parameterForm_->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    parameterForm_->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
    parameterForm_->setVerticalSpacing(7);
    controlsLayout->addWidget(parametersGroup_);

    auto* actionLayout = new QHBoxLayout;
    calculateButton_ = new QPushButton(QStringLiteral("Calculate FEM"), controls);
    refineButton_ = new QPushButton(QStringLiteral("Refine x2"), controls);
    actionLayout->addWidget(calculateButton_);
    actionLayout->addWidget(refineButton_);
    controlsLayout->addLayout(actionLayout);

    meshCheck_ = new QCheckBox(QStringLiteral("Display mesh"), controls);
    meshCheck_->setChecked(false);
    controlsLayout->addWidget(meshCheck_);

    statusLabel_ = new QLabel(QStringLiteral("Ready"), controls);
    statusLabel_->setWordWrap(true);
    statusLabel_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    statusLabel_->setMinimumHeight(58);
    controlsLayout->addWidget(statusLabel_);
    controlsLayout->addStretch(1);

    auto* scroll = new QScrollArea(central);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setMinimumWidth(350);
    scroll->setMaximumWidth(430);
    scroll->setWidget(controls);
    outer->addWidget(scroll);

    auto* content = new QWidget(central);
    auto* contentLayout = new QVBoxLayout(content);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    headingLabel_ = new QLabel(content);
    QFont headingFont = headingLabel_->font();
    headingFont.setPointSize(std::max(headingFont.pointSize() + 4, 14));
    headingFont.setBold(true);
    headingLabel_->setFont(headingFont);
    headingLabel_->setAlignment(Qt::AlignCenter);
    contentLayout->addWidget(headingLabel_);

    auto* plots = new QSplitter(Qt::Horizontal, content);
    auto* electricGroup = new QGroupBox(QStringLiteral("Electric field"), plots);
    auto* electricLayout = new QVBoxLayout(electricGroup);
    electricLayout->setContentsMargins(2, 2, 2, 2);
    electricPlot_ = new FieldPlot(FieldFamily::Electric, electricGroup);
    electricLayout->addWidget(electricPlot_);
    auto* magneticGroup = new QGroupBox(QStringLiteral("Magnetic field"), plots);
    auto* magneticLayout = new QVBoxLayout(magneticGroup);
    magneticLayout->setContentsMargins(2, 2, 2, 2);
    magneticPlot_ = new FieldPlot(FieldFamily::Magnetic, magneticGroup);
    magneticLayout->addWidget(magneticPlot_);
    plots->addWidget(electricGroup);
    plots->addWidget(magneticGroup);
    plots->setStretchFactor(0, 1);
    plots->setStretchFactor(1, 1);
    contentLayout->addWidget(plots, 1);

    auto* resultsGroup = new QGroupBox(QStringLiteral("Extracted line parameters"), content);
    auto* resultsLayout = new QVBoxLayout(resultsGroup);
    resultsLabel_ = new QLabel(
        QStringLiteral("Enter the line dimensions, then calculate."), resultsGroup);
    resultsLabel_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    resultsLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    resultsLabel_->setMinimumHeight(64);
    resultsLabel_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    resultsLayout->addWidget(resultsLabel_);
    contentLayout->addWidget(resultsGroup);
    outer->addWidget(content, 1);
    setCentralWidget(central);

    connect(lineTypeCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        configureEntries(selectedLineType());
        updateHeading();
        invalidateSolution(QStringLiteral("Line changed; calculate the new geometry."));
    });
    connect(calculateButton_, &QPushButton::clicked, this, [this] { calculate(); });
    connect(refineButton_, &QPushButton::clicked, this, [this] { refine(); });
    connect(meshCheck_, &QCheckBox::toggled, this, [this](bool visible) {
        electricPlot_->setMeshVisible(visible);
        magneticPlot_->setMeshVisible(visible);
    });
}

std::vector<MainWindow::EntryDefinition> MainWindow::definitionsFor(LineType type) {
    constexpr EntryDefinition frequency{
        InputKey::Frequency, "Frequency (GHz)", "10", 1.0e9};
    constexpr EntryDefinition epsilon{
        InputKey::EpsilonR, "Relative permittivity", "3.55", 1.0};
    constexpr EntryDefinition loss{
        InputKey::LossTangent, "Loss tangent", "0.0027", 1.0, false};
    constexpr EntryDefinition conductivity{
        InputKey::MetalConductivity, "Metal σ (MS/m; blank=PEC)", "", 1.0e6,
        true, true};
    constexpr EntryDefinition mesh{
        InputKey::MaxElementSize, "Mesh size (mm)", "1.00", 1.0e-3};

    switch (type) {
    case LineType::Coaxial:
        return {
            frequency,
            {InputKey::InnerRadius, "Inner radius (mm)", "0.5", 1.0e-3},
            {InputKey::OuterRadius, "Outer radius (mm)", "1.67", 1.0e-3},
            {InputKey::OuterConductorThickness, "Outer metal (um)", "150", 1.0e-6},
            {InputKey::EpsilonR, "Relative permittivity", "2.1", 1.0},
            {InputKey::LossTangent, "Loss tangent", "0.0002", 1.0, false},
            conductivity,
            mesh,
        };
    case LineType::Microstrip:
        return {
            frequency,
            {InputKey::TraceWidth, "Trace width (mm)", "3", 1.0e-3},
            {InputKey::SubstrateHeight, "Substrate h (mm)", "1.524", 1.0e-3},
            {InputKey::ConductorThickness, "Metal thick. (um)", "35", 1.0e-6},
            {InputKey::DomainPaddingFactor, "Domain padding (x)", "1", 1.0},
            epsilon,
            loss,
            conductivity,
            mesh,
        };
    case LineType::Stripline:
        return {
            frequency,
            {InputKey::TraceWidth, "Trace width (mm)", "0.8", 1.0e-3},
            {InputKey::GroundSpacing, "Ground gap (mm)", "1.524", 1.0e-3},
            {InputKey::ConductorThickness, "Metal thick. (um)", "35", 1.0e-6},
            {InputKey::DomainPaddingFactor, "Domain padding (x)", "1", 1.0},
            epsilon,
            loss,
            conductivity,
            mesh,
        };
    case LineType::CoplanarWaveguide:
        return {
            frequency,
            {InputKey::CenterWidth, "Signal width (mm)", "0.6", 1.0e-3},
            {InputKey::Gap, "Slot gap (mm)", "0.25", 1.0e-3},
            {InputKey::GroundWidth, "Ground width (mm)", "1.5", 1.0e-3},
            {InputKey::SubstrateHeight, "Substrate h (mm)", "0.8", 1.0e-3},
            {InputKey::ConductorThickness, "Metal thick. (um)", "35", 1.0e-6},
            {InputKey::DomainPaddingFactor, "Domain padding (x)", "1", 1.0},
            epsilon,
            loss,
            conductivity,
            mesh,
        };
    }
    return {};
}

void MainWindow::configureEntries(LineType type) {
    entries_.clear();
    while (auto* item = parameterForm_->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    for (const auto& definition : definitionsFor(type)) {
        auto* editor = new QLineEdit(QString::fromUtf8(definition.defaultText), parametersGroup_);
        editor->setClearButtonEnabled(true);
        if (definition.optional) {
            editor->setPlaceholderText(QStringLiteral("PEC"));
        }
        parameterForm_->addRow(QString::fromUtf8(definition.label), editor);
        entries_.push_back({definition, editor});
        connect(editor, &QLineEdit::textEdited, this, [this] {
            invalidateSolution(QStringLiteral("Parameters changed; calculate again."));
        });
    }
    parametersGroup_->setTitle(QStringLiteral("%1 parameters").arg(lineName(type)));
}

LineType MainWindow::selectedLineType() const {
    switch (lineTypeCombo_->currentIndex()) {
    case 1:
        return LineType::Microstrip;
    case 2:
        return LineType::Stripline;
    case 3:
        return LineType::CoplanarWaveguide;
    default:
        return LineType::Coaxial;
    }
}

Parameters MainWindow::readParameters() const {
    auto parameters = defaultParameters(selectedLineType());
    parameters.type = selectedLineType();
    parameters.refinementFactor = 1.0;
    for (const auto& entry : entries_) {
        const auto label = QString::fromUtf8(entry.definition.label);
        const auto raw = entry.editor->text().trimmed();
        if (raw.isEmpty() && entry.definition.optional) {
            parameters.metalConductivity.reset();
            continue;
        }
        bool parsed = false;
        const auto displayed = raw.toDouble(&parsed);
        if (!parsed) {
            inputError(QStringLiteral("%1 must be a number; received %2.")
                           .arg(label, raw.isEmpty() ? QStringLiteral("blank") : raw));
        }
        if (!std::isfinite(displayed)) {
            inputError(QStringLiteral("%1 must be finite.").arg(label));
        }
        if (entry.definition.strictlyPositive && displayed <= 0.0) {
            inputError(QStringLiteral("%1 must be greater than zero.").arg(label));
        }
        if (!entry.definition.strictlyPositive && displayed < 0.0) {
            inputError(QStringLiteral("%1 must not be negative.").arg(label));
        }
        const auto value = displayed * entry.definition.siScale;
        switch (entry.definition.key) {
        case InputKey::Frequency:
            parameters.frequencyHz = value;
            break;
        case InputKey::InnerRadius:
            parameters.innerRadius = value;
            break;
        case InputKey::OuterRadius:
            parameters.outerRadius = value;
            break;
        case InputKey::OuterConductorThickness:
            parameters.outerConductorThickness = value;
            break;
        case InputKey::TraceWidth:
            parameters.traceWidth = value;
            break;
        case InputKey::SubstrateHeight:
            parameters.substrateHeight = value;
            break;
        case InputKey::ConductorThickness:
            parameters.conductorThickness = value;
            break;
        case InputKey::GroundSpacing:
            parameters.groundSpacing = value;
            break;
        case InputKey::CenterWidth:
            parameters.centerWidth = value;
            break;
        case InputKey::Gap:
            parameters.gap = value;
            break;
        case InputKey::GroundWidth:
            parameters.groundWidth = value;
            break;
        case InputKey::EpsilonR:
            parameters.epsilonR = value;
            break;
        case InputKey::LossTangent:
            parameters.lossTangent = value;
            break;
        case InputKey::DomainPaddingFactor:
            parameters.domainPaddingFactor = value;
            break;
        case InputKey::MetalConductivity:
            parameters.metalConductivity = value;
            break;
        case InputKey::MaxElementSize:
            parameters.maxElementSize = value;
            break;
        }
    }
    if (parameters.type == LineType::Coaxial
        && parameters.outerRadius <= parameters.innerRadius) {
        inputError(QStringLiteral("Outer radius (mm) must be greater than Inner radius (mm)."));
    }
    if (parameters.type == LineType::Stripline
        && parameters.conductorThickness >= parameters.groundSpacing) {
        inputError(QStringLiteral("Metal thick. (um) must be smaller than Ground gap (mm)."));
    }
    return parameters;
}

void MainWindow::calculate() {
    try {
        startSolve(readParameters(), false);
    } catch (const std::exception& error) {
        setStatus(QStringLiteral("Error: %1").arg(QString::fromUtf8(error.what())), true);
    }
}

void MainWindow::refine() {
    if (!result_) {
        setStatus(QStringLiteral("Calculate a line before requesting refinement."), true);
        return;
    }
    auto parameters = result_->parameters;
    parameters.refinementFactor *= 2.0;
    startSolve(parameters, true);
}

void MainWindow::startSolve(Parameters parameters, bool refinement) {
    if (busy_) {
        return;
    }
    const auto generation = ++generation_;
    setBusy(true);
    if (!refinement) {
        result_.reset();
        resultsLabel_->setText(QStringLiteral("Meshing and solving …"));
        electricPlot_->setEmptyMessage(QStringLiteral("Calculating E-field …"));
        magneticPlot_->setEmptyMessage(QStringLiteral("Calculating H-field …"));
    }
    setStatus(refinement ? QStringLiteral("Refining mesh x2 and solving …")
                         : QStringLiteral("Meshing and solving …"));

    auto* watcher = new QFutureWatcher<SolveOutcome>(this);
    connect(watcher, &QFutureWatcher<SolveOutcome>::finished, this, [this, watcher] {
        const auto outcome = watcher->result();
        watcher->deleteLater();
        setBusy(false);
        if (outcome.generation != generation_) {
            return;
        }
        applyOutcome(outcome);
    });
    watcher->setFuture(QtConcurrent::run([parameters, generation, refinement] {
        SolveOutcome outcome;
        outcome.generation = generation;
        outcome.refinement = refinement;
        try {
            outcome.result = std::make_shared<Result>(solve(parameters));
        } catch (const std::exception& error) {
            outcome.error = QString::fromUtf8(error.what());
        } catch (...) {
            outcome.error = QStringLiteral("The native FEM solver failed with an unknown error.");
        }
        return outcome;
    }));
}

void MainWindow::applyOutcome(const SolveOutcome& outcome) {
    if (!outcome.error.isEmpty() || !outcome.result) {
        if (!outcome.refinement) {
            resultsLabel_->setText(QStringLiteral(
                "No solution was produced. Correct the inputs and calculate again."));
            electricPlot_->setEmptyMessage(
                QStringLiteral("E-field unavailable because calculation failed"));
            magneticPlot_->setEmptyMessage(
                QStringLiteral("H-field unavailable because calculation failed"));
        }
        setStatus(QStringLiteral("Error: %1").arg(
                      outcome.error.isEmpty()
                          ? QStringLiteral("The FEM calculator completed without a solution.")
                          : outcome.error),
                  true);
        return;
    }
    result_ = outcome.result;
    electricPlot_->setResult(result_);
    magneticPlot_->setResult(result_);
    showResult();
    setBusy(false);
    const auto timing = QStringLiteral(" Mesh %1 ms; solve %2 ms.")
        .arg(QString::number(result_->meshMilliseconds, 'f', 1),
             QString::number(result_->solveMilliseconds, 'f', 1));
    setStatus((outcome.refinement
                   ? QStringLiteral("Refined FEM solution complete.")
                   : QStringLiteral("Solved. Refine x2 halves the current FEM element size."))
              + timing);
}

void MainWindow::invalidateSolution(const QString& status) {
    ++generation_;
    result_.reset();
    resultsLabel_->setText(QStringLiteral("Enter the line dimensions, then calculate."));
    electricPlot_->setEmptyMessage(QStringLiteral("E-field appears after calculation"));
    magneticPlot_->setEmptyMessage(QStringLiteral("H-field appears after calculation"));
    setBusy(busy_);
    setStatus(status);
}

void MainWindow::showResult() {
    if (!result_) {
        return;
    }
    const auto alpha = -result_->beta.imag();
    resultsLabel_->setText(
        QStringLiteral("n_eff = %1  Zc = %2 ohm  Zwave = %3 ohm\n"
                       "R' = %4  L' = %5  G' = %6  C' = %7\n"
                       "alpha = %8  P = %9")
            .arg(formatNumber(result_->neff),
                 formatNumber(result_->characteristicImpedance),
                 formatNumber(result_->waveImpedance),
                 withUnit(result_->resistancePerLength, QStringLiteral("ohm/m")),
                 withUnit(result_->inductancePerLength, QStringLiteral("H/m")),
                 withUnit(result_->conductancePerLength, QStringLiteral("S/m")),
                 withUnit(result_->capacitancePerLength, QStringLiteral("F/m")),
                 withUnit(alpha, QStringLiteral("1/m")),
                 withUnit(result_->power, QStringLiteral("W"))));
}

void MainWindow::setBusy(bool busy) {
    busy_ = busy;
    calculateButton_->setEnabled(!busy_);
    refineButton_->setEnabled(!busy_ && static_cast<bool>(result_));
}

void MainWindow::setStatus(const QString& message, bool error) {
    statusLabel_->setText(message);
    statusLabel_->setStyleSheet(error
        ? QStringLiteral("QLabel { color: crimson; }") : QString{});
}

void MainWindow::updateHeading() {
    const auto name = lineName(selectedLineType());
    headingLabel_->setText(QStringLiteral("FEM transmission-line calculator — %1").arg(name));
    setWindowTitle(QStringLiteral("FEM Transmission-Line Calculator — %1").arg(name));
}

} // namespace tl
