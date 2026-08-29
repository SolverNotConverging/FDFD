#include "main_window.hpp"

#include "h5_reader.hpp"
#include "plot_widget.hpp"

#include <QtConcurrent/QtConcurrentRun>

#include <QComboBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSplitter>
#include <QStatusBar>
#include <QTableWidget>
#include <QTabWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <tuple>

namespace wavefem {
namespace {

QString formatFrequency(double value) {
    if (!std::isfinite(value)) {
        return QStringLiteral("unknown frequency");
    }
    struct Unit {
        double threshold;
        double divisor;
        const char* suffix;
    };
    constexpr std::array<Unit, 4> units{{
        {1e12, 1e12, "THz"}, {1e9, 1e9, "GHz"},
        {1e6, 1e6, "MHz"}, {1e3, 1e3, "kHz"}}};
    for (const auto& unit : units) {
        if (std::abs(value) >= unit.threshold) {
            return QStringLiteral("%1 %2")
                .arg(value / unit.divisor, 0, 'g', 9)
                .arg(QString::fromLatin1(unit.suffix));
        }
    }
    return QStringLiteral("%1 Hz").arg(value, 0, 'g', 9);
}

QString sLabel(const SParameter& value) {
    return QStringLiteral("S(%1,%2 ← %3)")
        .arg(QString::fromStdString(value.side))
        .arg(value.outMode)
        .arg(value.inMode);
}

double sValue(Complex value, int quantity) {
    switch (quantity) {
    case 0:
        return 20.0 * std::log10(std::max(std::abs(value), 1e-15));
    case 1:
        return std::abs(value);
    case 2:
        return std::arg(value) * 180.0 / std::numbers::pi;
    case 3:
        return value.real();
    default:
        return value.imag();
    }
}

QString sAxisLabel(int quantity) {
    constexpr std::array<const char*, 5> labels{
        "Magnitude (dB)", "Magnitude", "Phase (deg)", "Real", "Imaginary"};
    return QString::fromLatin1(labels.at(static_cast<std::size_t>(quantity)));
}

ScalarQuantity modalQuantity(int index) {
    return index == 0 ? ScalarQuantity::Absolute
        : (index == 1 ? ScalarQuantity::Real : ScalarQuantity::Imaginary);
}

ScalarQuantity vectorQuantity(int index) {
    return index == 0 ? ScalarQuantity::Real : ScalarQuantity::Imaginary;
}

FieldPart fieldPart(int index) {
    return index == 0 ? FieldPart::Total
        : (index == 1 ? FieldPart::Incident : FieldPart::Scattered);
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("WaveFEM Viewer (native Qt)"));
    resize(1360, 860);
    buildUi();
}

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(7, 7, 7, 7);
    layout->setSpacing(6);

    auto* controls = new QHBoxLayout;
    auto* openButton = new QPushButton(QStringLiteral("Open HDF5…"), central);
    auto* directoryButton = new QPushButton(QStringLiteral("Open directory…"), central);
    pathLabel_ = new QLabel(QStringLiteral("No file loaded"), central);
    pathLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    fileCombo_ = new QComboBox(central);
    fileCombo_->setMinimumContentsLength(22);
    fileCombo_->setEnabled(false);
    frequencyCombo_ = new QComboBox(central);
    frequencyCombo_->setMinimumContentsLength(24);
    frequencyCombo_->setEnabled(false);
    controls->addWidget(openButton);
    controls->addWidget(directoryButton);
    controls->addWidget(pathLabel_, 1);
    controls->addWidget(new QLabel(QStringLiteral("File:"), central));
    controls->addWidget(fileCombo_);
    controls->addWidget(new QLabel(QStringLiteral("Frequency:"), central));
    controls->addWidget(frequencyCombo_);
    layout->addLayout(controls);

    tabs_ = new QTabWidget(central);
    tabs_->addTab(buildSParameterTab(), QStringLiteral("S-parameters"));
    tabs_->addTab(buildModalTab(FieldName::Electric, modal_[0]), QStringLiteral("Modal E"));
    tabs_->addTab(buildModalTab(FieldName::Magnetic, modal_[1]), QStringLiteral("Modal H"));
    tabs_->addTab(buildVectorTab(FieldName::Electric, vector_[0]), QStringLiteral("2D Vector E"));
    tabs_->addTab(buildVectorTab(FieldName::Magnetic, vector_[1]), QStringLiteral("2D Vector H"));
    layout->addWidget(tabs_, 1);
    setCentralWidget(central);

    connect(openButton, &QPushButton::clicked, this, [this] { chooseFile(); });
    connect(directoryButton, &QPushButton::clicked, this, [this] { chooseDirectory(); });
    connect(fileCombo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index >= 0) {
            const auto path = fileCombo_->itemData(index).toString();
            if (!path.isEmpty()) {
                loadPath(path);
            }
        }
    });
    connect(frequencyCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        if (fileIndex_) {
            refreshSParameters();
            loadSelectedResult();
        }
    });
    connect(tabs_, &QTabWidget::currentChanged, this,
            [this](int) { refreshCurrentTab(); });
    statusBar()->showMessage(QStringLiteral("Ready"));
}

QWidget* MainWindow::buildSParameterTab() {
    auto* tab = new QWidget;
    auto* layout = new QVBoxLayout(tab);
    auto* controls = new QHBoxLayout;
    controls->addWidget(new QLabel(QStringLiteral("Plot:"), tab));
    sQuantity_ = new QComboBox(tab);
    sQuantity_->addItems({QStringLiteral("magnitude_db"), QStringLiteral("magnitude"),
                          QStringLiteral("phase_deg"), QStringLiteral("real"),
                          QStringLiteral("imag")});
    controls->addWidget(sQuantity_);
    controls->addStretch(1);
    layout->addLayout(controls);

    auto* splitter = new QSplitter(Qt::Horizontal, tab);
    sTable_ = new QTableWidget(0, 6, splitter);
    sTable_->setHorizontalHeaderLabels({QStringLiteral("Side"), QStringLiteral("Out"),
                                       QStringLiteral("In"), QStringLiteral("Complex"),
                                       QStringLiteral("|S|"), QStringLiteral("Phase (deg)")});
    sTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    sTable_->horizontalHeader()->setStretchLastSection(true);
    sTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    sTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    sPlot_ = new PlotWidget(splitter);
    splitter->addWidget(sTable_);
    splitter->addWidget(sPlot_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 3);
    layout->addWidget(splitter, 1);
    connect(sQuantity_, &QComboBox::currentIndexChanged, this,
            [this](int) { refreshSParameters(); });
    return tab;
}

QWidget* MainWindow::buildModalTab(FieldName field, ModalControls& controls) {
    auto* tab = new QWidget;
    auto* layout = new QVBoxLayout(tab);
    auto* row = new QHBoxLayout;
    row->addWidget(new QLabel(QStringLiteral("Mode:"), tab));
    controls.mode = new QComboBox(tab);
    controls.mode->setMinimumContentsLength(28);
    row->addWidget(controls.mode);
    row->addWidget(new QLabel(QStringLiteral("Component:"), tab));
    controls.component = new QComboBox(tab);
    controls.component->addItems({QStringLiteral("norm"), QStringLiteral("x"),
                                  QStringLiteral("y"), QStringLiteral("z")});
    row->addWidget(controls.component);
    row->addWidget(new QLabel(QStringLiteral("Quantity:"), tab));
    controls.quantity = new QComboBox(tab);
    controls.quantity->addItems({QStringLiteral("abs"), QStringLiteral("real"),
                                 QStringLiteral("imag")});
    row->addWidget(controls.quantity);
    row->addStretch(1);
    layout->addLayout(row);
    controls.plot = new PlotWidget(tab);
    layout->addWidget(controls.plot, 1);
    const auto refresh = [this, field](int) { refreshModal(field); };
    connect(controls.mode, &QComboBox::currentIndexChanged, this, refresh);
    connect(controls.component, &QComboBox::currentIndexChanged, this, refresh);
    connect(controls.quantity, &QComboBox::currentIndexChanged, this, refresh);
    return tab;
}

QWidget* MainWindow::buildVectorTab(FieldName field, VectorControls& controls) {
    auto* tab = new QWidget;
    auto* layout = new QVBoxLayout(tab);
    auto* row = new QHBoxLayout;
    row->addWidget(new QLabel(QStringLiteral("Field part:"), tab));
    controls.part = new QComboBox(tab);
    controls.part->addItems({QStringLiteral("total"), QStringLiteral("incident"),
                             QStringLiteral("scattered")});
    row->addWidget(controls.part);
    row->addWidget(new QLabel(QStringLiteral("Quantity:"), tab));
    controls.quantity = new QComboBox(tab);
    controls.quantity->addItems({QStringLiteral("real"), QStringLiteral("imag")});
    row->addWidget(controls.quantity);
    row->addStretch(1);
    layout->addLayout(row);
    controls.plot = new PlotWidget(tab);
    layout->addWidget(controls.plot, 1);
    const auto refresh = [this, field](int) { refreshVector(field); };
    connect(controls.part, &QComboBox::currentIndexChanged, this, refresh);
    connect(controls.quantity, &QComboBox::currentIndexChanged, this, refresh);
    return tab;
}

void MainWindow::chooseFile() {
    const auto path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open WaveFEM HDF5 result"), QString(),
        QStringLiteral("HDF5 files (*.h5 *.hdf5);;All files (*)"));
    if (!path.isEmpty()) {
        loadPath(path);
    }
}

void MainWindow::chooseDirectory() {
    const auto directory = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Choose directory containing WaveFEM HDF5 results"));
    if (directory.isEmpty()) {
        return;
    }

    loadDirectory(directory);
}

void MainWindow::loadDirectory(const QString& directoryPath) {
    QDir selectedDirectory(directoryPath);
    const auto entries = selectedDirectory.entryInfoList(
        {QStringLiteral("*.h5"), QStringLiteral("*.hdf5")},
        QDir::Files | QDir::Readable, QDir::Name | QDir::IgnoreCase);
    if (entries.isEmpty()) {
        fileCombo_->blockSignals(true);
        fileCombo_->clear();
        fileCombo_->setEnabled(false);
        fileCombo_->blockSignals(false);
        QMessageBox::information(
            this, QStringLiteral("No HDF5 results"),
            QStringLiteral("The selected directory contains no .h5 or .hdf5 files."));
        return;
    }
    refreshFileChoices(entries.front().absoluteFilePath());
    loadPath(entries.front().absoluteFilePath());
}

void MainWindow::refreshFileChoices(const QString& selectedPath) {
    const QFileInfo selectedFile(selectedPath);
    QDir directory(selectedFile.absolutePath());
    const auto entries = directory.entryInfoList(
        {QStringLiteral("*.h5"), QStringLiteral("*.hdf5")},
        QDir::Files | QDir::Readable, QDir::Name | QDir::IgnoreCase);
    fileCombo_->blockSignals(true);
    fileCombo_->clear();
    int selectedIndex = -1;
    for (const auto& entry : entries) {
        const auto absolutePath = entry.absoluteFilePath();
        fileCombo_->addItem(entry.fileName(), absolutePath);
        if (absolutePath.compare(selectedFile.absoluteFilePath(), Qt::CaseInsensitive) == 0) {
            selectedIndex = fileCombo_->count() - 1;
        }
    }
    if (selectedIndex >= 0) {
        fileCombo_->setCurrentIndex(selectedIndex);
    }
    fileCombo_->setEnabled(fileCombo_->count() > 0);
    fileCombo_->setToolTip(directory.absolutePath());
    fileCombo_->blockSignals(false);
}

void MainWindow::loadPath(const QString& path) {
    if (QFileInfo(path).isDir()) {
        loadDirectory(path);
        return;
    }

    QElapsedTimer timer;
    timer.start();
    try {
        const std::filesystem::path native(path.toStdWString());
        fileIndex_ = H5Reader::loadIndex(native);
    } catch (const std::exception& error) {
        QMessageBox::critical(this, QStringLiteral("Could not open HDF5 file"),
                              QString::fromUtf8(error.what()));
        return;
    }
    ++loadGeneration_;
    result_.reset();
    refreshFileChoices(path);
    pathLabel_->setText(path + QStringLiteral("  (")
                        + QString::fromStdString(fileIndex_->kind) + QLatin1Char(')'));
    frequencyCombo_->blockSignals(true);
    frequencyCombo_->clear();
    for (std::size_t index = 0; index < fileIndex_->frequenciesHz.size(); ++index) {
        frequencyCombo_->addItem(QStringLiteral("%1: %2")
            .arg(index).arg(formatFrequency(fileIndex_->frequenciesHz[index])));
    }
    frequencyCombo_->setCurrentIndex(0);
    frequencyCombo_->setEnabled(true);
    frequencyCombo_->blockSignals(false);
    refreshSParameters();
    statusBar()->showMessage(QStringLiteral("Indexed %1 result(s) in %2 ms; loading fields…")
        .arg(fileIndex_->frequenciesHz.size()).arg(timer.elapsed()));
    loadSelectedResult();
}

int MainWindow::selectedResultIndex() const {
    return std::max(0, frequencyCombo_->currentIndex());
}

void MainWindow::loadSelectedResult() {
    if (!fileIndex_) {
        return;
    }
    const auto selected = selectedResultIndex();
    const auto generation = ++loadGeneration_;
    const auto index = fileIndex_;
    result_.reset();
    setResultControlsEnabled(false);
    for (std::size_t field = 0; field < modal_.size(); ++field) {
        modal_[field].plot->setEmpty(
            field == 0 ? QStringLiteral("Modal E") : QStringLiteral("Modal H"),
            QStringLiteral("Loading selected result…"));
        vector_[field].plot->setEmpty(
            field == 0 ? QStringLiteral("2D Vector E") : QStringLiteral("2D Vector H"),
            QStringLiteral("Loading selected result…"));
    }
    statusBar()->showMessage(QStringLiteral("Loading frequency %1…").arg(selected));

    auto* watcher = new QFutureWatcher<LoadOutcome>(this);
    connect(watcher, &QFutureWatcher<LoadOutcome>::finished, this, [this, watcher] {
        const auto outcome = watcher->result();
        watcher->deleteLater();
        applyLoadedResult(outcome);
    });
    watcher->setFuture(QtConcurrent::run([index, selected, generation] {
        QElapsedTimer timer;
        timer.start();
        LoadOutcome outcome;
        outcome.index = selected;
        outcome.generation = generation;
        try {
            outcome.result = H5Reader::loadResult(*index, static_cast<std::size_t>(selected));
        } catch (const std::exception& error) {
            outcome.error = QString::fromUtf8(error.what());
        }
        outcome.milliseconds = timer.elapsed();
        return outcome;
    }));
}

void MainWindow::applyLoadedResult(const LoadOutcome& outcome) {
    if (outcome.generation != loadGeneration_ || outcome.index != selectedResultIndex()) {
        return;
    }
    if (!outcome.error.isEmpty()) {
        QMessageBox::critical(this, QStringLiteral("Could not load HDF5 result"), outcome.error);
        statusBar()->showMessage(QStringLiteral("Loading failed"));
        return;
    }
    result_ = outcome.result;
    setResultControlsEnabled(true);
    for (std::size_t field = 0; field < modal_.size(); ++field) {
        auto& controls = modal_[field];
        const auto previous = std::max(0, controls.mode->currentIndex());
        controls.mode->blockSignals(true);
        controls.mode->clear();
        for (const auto& mode : result_->modes) {
            controls.mode->addItem(QString::fromStdString(mode.label));
        }
        if (!result_->modes.empty()) {
            controls.mode->setCurrentIndex(std::min<int>(previous,
                static_cast<int>(result_->modes.size()) - 1));
        }
        controls.mode->blockSignals(false);
    }
    for (std::size_t field = 0; field < modal_.size(); ++field) {
        modal_[field].plot->setEmpty(
            field == 0 ? QStringLiteral("Modal E") : QStringLiteral("Modal H"),
            QStringLiteral("Select this tab to render"));
        vector_[field].plot->setEmpty(
            field == 0 ? QStringLiteral("2D Vector E") : QStringLiteral("2D Vector H"),
            QStringLiteral("Select this tab to render"));
    }
    refreshSParameters();
    refreshCurrentTab();
    statusBar()->showMessage(
        QStringLiteral("Loaded %1 field samples, %2 mode(s), and %3 scene triangle(s) in %4 ms")
            .arg(result_->coordinates.columns)
            .arg(result_->modes.size())
            .arg(result_->scene ? result_->scene->triangles.columns : 0)
            .arg(outcome.milliseconds));
}

void MainWindow::refreshSParameters() {
    if (!fileIndex_ || fileIndex_->frequenciesHz.empty()) {
        sTable_->setRowCount(0);
        sPlot_->setEmpty(QStringLiteral("Modal S-parameters"),
                         QStringLiteral("Open an HDF5 result file"));
        return;
    }
    const auto selected = static_cast<std::size_t>(selectedResultIndex());
    const auto& current = fileIndex_->sParameters.at(selected);
    sTable_->setRowCount(static_cast<int>(current.size()));
    for (std::size_t row = 0; row < current.size(); ++row) {
        const auto& item = current[row];
        const auto magnitude = std::abs(item.value);
        const auto phase = std::arg(item.value) * 180.0 / std::numbers::pi;
        const std::array<QString, 6> values{
            QString::fromStdString(item.side), QString::number(item.outMode),
            QString::number(item.inMode),
            QStringLiteral("%1%2%3j")
                .arg(item.value.real(), 0, 'e', 6)
                .arg(item.value.imag() >= 0 ? QLatin1Char('+') : QLatin1Char('-'))
                .arg(std::abs(item.value.imag()), 0, 'e', 6),
            QString::number(magnitude, 'e', 6), QString::number(phase, 'f', 4)};
        for (int column = 0; column < static_cast<int>(values.size()); ++column) {
            sTable_->setItem(static_cast<int>(row), column,
                             new QTableWidgetItem(values[static_cast<std::size_t>(column)]));
        }
    }

    using Key = std::tuple<std::string, std::int64_t, std::int64_t>;
    std::map<Key, std::vector<double>> valuesByKey;
    for (std::size_t resultIndex = 0; resultIndex < fileIndex_->sParameters.size();
         ++resultIndex) {
        for (const auto& parameter : fileIndex_->sParameters[resultIndex]) {
            auto& values = valuesByKey[{parameter.side, parameter.outMode, parameter.inMode}];
            if (values.empty()) {
                values.assign(fileIndex_->frequenciesHz.size(),
                              std::numeric_limits<double>::quiet_NaN());
            }
            values[resultIndex] = sValue(parameter.value, sQuantity_->currentIndex());
        }
    }
    const bool knownFrequencies = std::ranges::all_of(
        fileIndex_->frequenciesHz, [](double value) { return std::isfinite(value); });
    std::vector<double> horizontal = fileIndex_->frequenciesHz;
    if (!knownFrequencies) {
        for (std::size_t index = 0; index < horizontal.size(); ++index) {
            horizontal[index] = static_cast<double>(index);
        }
    }
    std::vector<PlotSeries> series;
    for (auto& [key, values] : valuesByKey) {
        const auto& [side, outMode, inMode] = key;
        series.push_back({sLabel({side, outMode, inMode, {}}), horizontal, std::move(values)});
    }
    sPlot_->setLines(std::move(series), QStringLiteral("Modal S-parameters"),
                     knownFrequencies ? QStringLiteral("Frequency (Hz)")
                                      : QStringLiteral("Saved result index"),
                     sAxisLabel(sQuantity_->currentIndex()), horizontal[selected]);
}

void MainWindow::refreshModal(FieldName field) {
    auto& controls = modal_[field == FieldName::Electric ? 0 : 1];
    if (!result_ || result_->modes.empty() || controls.mode->currentIndex() < 0) {
        controls.plot->setEmpty(
            field == FieldName::Electric ? QStringLiteral("Modal E") : QStringLiteral("Modal H"),
            QStringLiteral("No saved modes"));
        return;
    }
    const auto modeIndex = static_cast<std::size_t>(controls.mode->currentIndex());
    if (modeIndex >= result_->modes.size()) {
        return;
    }
    controls.plot->setModal(result_->modes[modeIndex], field,
                            controls.component->currentIndex() - 1,
                            modalQuantity(controls.quantity->currentIndex()));
}

void MainWindow::refreshVector(FieldName field) {
    auto& controls = vector_[field == FieldName::Electric ? 0 : 1];
    if (!result_) {
        controls.plot->setEmpty(
            field == FieldName::Electric ? QStringLiteral("2D Vector E")
                                         : QStringLiteral("2D Vector H"),
            QStringLiteral("Loading field samples…"));
        return;
    }
    controls.plot->setVector(result_, field, fieldPart(controls.part->currentIndex()),
                             vectorQuantity(controls.quantity->currentIndex()));
}

void MainWindow::setResultControlsEnabled(bool enabled) {
    for (auto& controls : modal_) {
        controls.mode->setEnabled(enabled);
        controls.component->setEnabled(enabled);
        controls.quantity->setEnabled(enabled);
    }
    for (auto& controls : vector_) {
        controls.part->setEnabled(enabled);
        controls.quantity->setEnabled(enabled);
    }
}

void MainWindow::refreshCurrentTab() {
    switch (tabs_->currentIndex()) {
    case 0:
        refreshSParameters();
        break;
    case 1:
        refreshModal(FieldName::Electric);
        break;
    case 2:
        refreshModal(FieldName::Magnetic);
        break;
    case 3:
        refreshVector(FieldName::Electric);
        break;
    case 4:
        refreshVector(FieldName::Magnetic);
        break;
    default:
        break;
    }
}

} // namespace wavefem
