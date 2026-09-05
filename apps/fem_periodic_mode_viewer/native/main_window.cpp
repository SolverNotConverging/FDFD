#include "main_window.hpp"

#include "field_plot_2d.hpp"
#include "h5_reader.hpp"
#include "path_qt.hpp"
#include "vtk_field_view.hpp"

#include <QtConcurrent/QtConcurrentRun>

#include <QComboBox>
#include <QCheckBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QSplitter>
#include <QStackedWidget>
#include <QTabWidget>
#include <QStatusBar>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <limits>
#include <utility>

namespace femperiodic {

namespace {

QString qString(const std::string& value) {
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

QString formatComplex(Complex value) {
    return QStringLiteral("%1 %2 %3j")
        .arg(value.real(), 0, 'g', 9)
        .arg(value.imag() < 0.0 ? QStringLiteral("-") : QStringLiteral("+"))
        .arg(std::abs(value.imag()), 0, 'g', 9);
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("FEM Periodic Mode Viewer"));
    resize(1240, 820);
    setMinimumSize(900, 620);

    auto* central = new QWidget(this);
    auto* outer = new QVBoxLayout(central);
    auto* fileRow = new QHBoxLayout;
    auto* open = new QPushButton(QStringLiteral("Open HDF5…"), central);
    auto* openDirectory = new QPushButton(QStringLiteral("Open directory…"), central);
    pathLabel_ = new QLabel(QStringLiteral("No result loaded"), central);
    pathLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    fileCombo_ = new QComboBox(central);
    fileCombo_->setMinimumContentsLength(22);
    fileCombo_->setEnabled(false);
    fileRow->addWidget(open);
    fileRow->addWidget(openDirectory);
    fileRow->addWidget(pathLabel_, 1);
    fileRow->addWidget(new QLabel(QStringLiteral("File:"), central));
    fileRow->addWidget(fileCombo_);
    outer->addLayout(fileRow);

    auto* selectionRow = new QHBoxLayout;
    const auto addSelectionCombo =
        [central, selectionRow](const QString& label, QComboBox*& combo) {
            selectionRow->addWidget(new QLabel(label, central));
            combo = new QComboBox(central);
            selectionRow->addWidget(combo);
        };
    addSelectionCombo(QStringLiteral("Case:"), caseCombo_);
    addSelectionCombo(QStringLiteral("Mode:"), modeCombo_);
    selectionRow->addStretch(1);
    outer->addLayout(selectionRow);

    auto* splitter = new QSplitter(Qt::Horizontal, central);
    dimensionStack_ = new QStackedWidget(splitter);

    auto* page2D = new QWidget(dimensionStack_);
    auto* page2DLayout = new QVBoxLayout(page2D);
    page2DLayout->setContentsMargins(0, 0, 0, 0);
    auto* controls2D = new QHBoxLayout;
    controls2D->addWidget(new QLabel(QStringLiteral("2D field:"), page2D));
    familyCombo2D_ = new QComboBox(page2D);
    familyCombo2D_->addItems({QStringLiteral("E"), QStringLiteral("H")});
    controls2D->addWidget(familyCombo2D_);
    controls2D->addWidget(new QLabel(QStringLiteral("Component:"), page2D));
    componentCombo2D_ = new QComboBox(page2D);
    componentCombo2D_->addItems(
        {QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z")});
    controls2D->addWidget(componentCombo2D_);
    controls2D->addWidget(new QLabel(QStringLiteral("Quantity:"), page2D));
    quantityCombo2D_ = new QComboBox(page2D);
    quantityCombo2D_->addItems({QStringLiteral("Magnitude"), QStringLiteral("Real"),
                                QStringLiteral("Imaginary"), QStringLiteral("Phase")});
    controls2D->addWidget(quantityCombo2D_);
    controls2D->addStretch(1);
    page2DLayout->addLayout(controls2D);
    tabs2D_ = new QTabWidget(page2D);
    materialPlot2D_ = new FieldPlot2D(tabs2D_);
    materialPlot2D_->setMaterialOnly(true);
    plot2D_ = new FieldPlot2D(tabs2D_);
    tabs2D_->addTab(materialPlot2D_, QStringLiteral("Material"));
    tabs2D_->addTab(plot2D_, QStringLiteral("Field"));
    page2DLayout->addWidget(tabs2D_, 1);
    dimensionStack_->addWidget(page2D);

    auto* page3D = new QWidget(dimensionStack_);
    auto* page3DLayout = new QVBoxLayout(page3D);
    page3DLayout->setContentsMargins(0, 0, 0, 0);
    auto* controls3D = new QHBoxLayout;
    controls3D->addWidget(new QLabel(QStringLiteral("3D field:"), page3D));
    familyCombo3D_ = new QComboBox(page3D);
    familyCombo3D_->addItems({QStringLiteral("E"), QStringLiteral("H")});
    controls3D->addWidget(familyCombo3D_);
    controls3D->addWidget(new QLabel(QStringLiteral("Component:"), page3D));
    componentCombo3D_ = new QComboBox(page3D);
    componentCombo3D_->addItems(
        {QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z")});
    controls3D->addWidget(componentCombo3D_);
    controls3D->addWidget(new QLabel(QStringLiteral("Quantity:"), page3D));
    quantityCombo3D_ = new QComboBox(page3D);
    quantityCombo3D_->addItems({QStringLiteral("Magnitude"), QStringLiteral("Real"),
                                QStringLiteral("Imaginary"), QStringLiteral("Phase")});
    controls3D->addWidget(quantityCombo3D_);
    controls3D->addStretch(1);
    page3DLayout->addLayout(controls3D);

    auto* sliceRow = new QHBoxLayout;
    sliceEnabled_ = new QCheckBox(QStringLiteral("Heat-map slice"), page3D);
    sliceEnabled_->setChecked(false);
    sliceRow->addWidget(sliceEnabled_);
    sliceRow->addWidget(new QLabel(QStringLiteral("Axis:"), page3D));
    sliceAxisCombo_ = new QComboBox(page3D);
    sliceAxisCombo_->addItems({QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z")});
    sliceAxisCombo_->setCurrentIndex(2);
    sliceRow->addWidget(sliceAxisCombo_);
    sliceRow->addWidget(new QLabel(QStringLiteral("Position:"), page3D));
    sliceSlider_ = new QSlider(Qt::Horizontal, page3D);
    sliceSlider_->setRange(0, 1000);
    sliceSlider_->setValue(500);
    sliceRow->addWidget(sliceSlider_, 1);
    sliceAxisCombo_->setEnabled(false);
    sliceSlider_->setEnabled(false);
    page3DLayout->addLayout(sliceRow);

    tabs3D_ = new QTabWidget(page3D);
    materialView3D_ = new VtkFieldView(tabs3D_);
    materialView3D_->setMaterialOnly(true);
    view3D_ = new VtkFieldView(tabs3D_);
    tabs3D_->addTab(materialView3D_, QStringLiteral("Material"));
    tabs3D_->addTab(view3D_, QStringLiteral("Field"));
    page3DLayout->addWidget(tabs3D_, 1);
    dimensionStack_->addWidget(page3D);

    metadata_ = new QTextEdit(splitter);
    metadata_->setReadOnly(true);
    metadata_->setMinimumWidth(285);
    metadata_->setMaximumWidth(430);
    metadata_->setPlaceholderText(QStringLiteral("Mode metadata"));
    splitter->addWidget(dimensionStack_);
    splitter->addWidget(metadata_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);
    outer->addWidget(splitter, 1);
    setCentralWidget(central);

    connect(open, &QPushButton::clicked, this, [this] { chooseFile(); });
    connect(openDirectory, &QPushButton::clicked, this, [this] { chooseDirectory(); });
    connect(fileCombo_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index >= 0) {
            const auto path = fileCombo_->itemData(index).toString();
            if (!path.isEmpty()) {
                loadPath(pathFromQString(path));
            }
        }
    });
    connect(caseCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        populateModes();
        requestSelectedMode();
    });
    connect(modeCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        requestSelectedMode();
    });
    for (auto* combo : {familyCombo2D_, componentCombo2D_, quantityCombo2D_,
                        familyCombo3D_, componentCombo3D_, quantityCombo3D_}) {
        connect(combo, &QComboBox::currentIndexChanged, this, [this](int) {
            refreshFieldSelection();
        });
    }
    connect(sliceAxisCombo_, &QComboBox::currentIndexChanged, this, [this](int axis) {
        materialView3D_->setSlice(axis, static_cast<double>(sliceSlider_->value()) / 1000.0);
        view3D_->setSlice(axis, static_cast<double>(sliceSlider_->value()) / 1000.0);
    });
    connect(sliceSlider_, &QSlider::valueChanged, this, [this](int value) {
        materialView3D_->setSlice(sliceAxisCombo_->currentIndex(), static_cast<double>(value) / 1000.0);
        view3D_->setSlice(sliceAxisCombo_->currentIndex(), static_cast<double>(value) / 1000.0);
    });
    connect(sliceEnabled_, &QCheckBox::toggled, this, [this](bool enabled) {
        sliceAxisCombo_->setEnabled(enabled);
        sliceSlider_->setEnabled(enabled);
        materialView3D_->setSliceEnabled(enabled);
        view3D_->setSliceEnabled(enabled);
    });
    statusBar()->showMessage(QStringLiteral("Open a fem-periodic-modes HDF5 file"));
}

bool MainWindow::verifySliceRenderingForTest() {
    if (!mesh_ || mesh_->dimension != 3 || !VtkFieldView::available()) {
        return false;
    }
    sliceEnabled_->setChecked(true);
    return materialView3D_->sliceIsHeatMapOnly()
        && view3D_->sliceIsHeatMapOnly()
        && materialView3D_->annotationsVisible()
        && view3D_->annotationsVisible();
}

void MainWindow::chooseFile() {
    const auto path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open FEM periodic HDF5 result"), QString(),
        QStringLiteral("HDF5 files (*.h5 *.hdf5);;All files (*)"));
    if (!path.isEmpty()) {
        loadPath(pathFromQString(path));
    }
}

void MainWindow::chooseDirectory() {
    const auto directory = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Choose directory containing FEM periodic HDF5 results"));
    if (!directory.isEmpty()) {
        loadDirectory(pathFromQString(directory));
    }
}

void MainWindow::loadDirectory(const std::filesystem::path& directoryPath) {
    ++loadGeneration_;
    directoryScanIndex_.reset();
    QDir directory(qStringFromPath(std::filesystem::absolute(directoryPath)));
    const auto entries = directory.entryInfoList(
        {QStringLiteral("*.h5"), QStringLiteral("*.hdf5")},
        QDir::Files | QDir::Readable, QDir::Name | QDir::IgnoreCase);
    if (entries.isEmpty()) {
        fileCombo_->blockSignals(true);
        fileCombo_->clear();
        fileCombo_->setEnabled(false);
        fileCombo_->blockSignals(false);
        const auto error = QStringLiteral(
            "The selected directory contains no readable .h5 or .hdf5 files.");
        if (loadCompletionHandler_) {
            loadCompletionHandler_(false, error);
        } else {
            QMessageBox::information(this, QStringLiteral("No HDF5 results"), error);
        }
        return;
    }
    const auto firstPath = pathFromQString(entries.front().absoluteFilePath());
    refreshFileChoices(firstPath);
    directoryScanIndex_ = 0;
    loadFile(firstPath);
}

void MainWindow::refreshFileChoices(const std::filesystem::path& selectedPath) {
    const QFileInfo selectedFile(qStringFromPath(std::filesystem::absolute(selectedPath)));
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

void MainWindow::setLoadCompletionHandler(
    std::function<void(bool, const QString&)> handler) {
    loadCompletionHandler_ = std::move(handler);
}

void MainWindow::loadPath(const std::filesystem::path& path) {
    if (QFileInfo(qStringFromPath(path)).isDir()) {
        loadDirectory(path);
        return;
    }
    directoryScanIndex_.reset();
    loadFile(path);
}

void MainWindow::loadFile(const std::filesystem::path& filePath) {
    const auto absolutePath = std::filesystem::absolute(filePath);
    const auto generation = ++loadGeneration_;
    index_.reset();
    mesh_.reset();
    material_.reset();
    fields_.reset();
    cachedMeshIndex_.reset();
    cachedMesh_.reset();
    cachedMaterialIndex_.reset();
    cachedMaterial_.reset();
    caseCombo_->blockSignals(true);
    modeCombo_->blockSignals(true);
    caseCombo_->clear();
    modeCombo_->clear();
    caseCombo_->blockSignals(false);
    modeCombo_->blockSignals(false);
    plot2D_->clearData(QStringLiteral("Indexing the selected result…"));
    materialPlot2D_->clearData(QStringLiteral("Indexing the selected result…"));
    view3D_->clearData(QStringLiteral("Indexing the selected result…"));
    materialView3D_->clearData(QStringLiteral("Indexing the selected result…"));
    metadata_->clear();
    pathLabel_->setText(qStringFromPath(absolutePath));
    statusBar()->showMessage(QStringLiteral("Indexing result…"));
    caseCombo_->setEnabled(false);
    modeCombo_->setEnabled(false);
    auto* watcher = new QFutureWatcher<IndexOutcome>(this);
    connect(watcher, &QFutureWatcher<IndexOutcome>::finished, this, [this, watcher] {
        const auto outcome = watcher->result();
        watcher->deleteLater();
        applyIndex(outcome);
    });
    watcher->setFuture(QtConcurrent::run([absolutePath, generation] {
        IndexOutcome outcome;
        outcome.generation = generation;
        try {
            outcome.index = H5Reader::loadIndex(absolutePath);
        } catch (const std::exception& error) {
            outcome.error = qString(error.what());
        }
        return outcome;
    }));
}

bool MainWindow::tryNextDirectoryCandidate() {
    if (!directoryScanIndex_) {
        return false;
    }
    const auto nextIndex = *directoryScanIndex_ + 1;
    if (nextIndex >= fileCombo_->count()) {
        directoryScanIndex_.reset();
        return false;
    }
    directoryScanIndex_ = nextIndex;
    fileCombo_->blockSignals(true);
    fileCombo_->setCurrentIndex(nextIndex);
    const auto nextPath = fileCombo_->itemData(nextIndex).toString();
    fileCombo_->blockSignals(false);
    loadFile(pathFromQString(nextPath));
    return true;
}

void MainWindow::applyIndex(const IndexOutcome& outcome) {
    if (outcome.generation != loadGeneration_) {
        return;
    }
    if (!outcome.error.isEmpty()) {
        if (tryNextDirectoryCandidate()) {
            return;
        }
        plot2D_->clearData(outcome.error);
        materialPlot2D_->clearData(outcome.error);
        view3D_->clearData(outcome.error);
        materialView3D_->clearData(outcome.error);
        metadata_->setPlainText(outcome.error);
        statusBar()->showMessage(QStringLiteral("Indexing failed"));
        if (loadCompletionHandler_) {
            loadCompletionHandler_(false, outcome.error);
        } else {
            QMessageBox::critical(this, QStringLiteral("Could not open result"), outcome.error);
        }
        return;
    }
    index_ = outcome.index;
    refreshFileChoices(index_->path);
    mesh_.reset();
    material_.reset();
    fields_.reset();
    cachedMeshIndex_.reset();
    cachedMesh_.reset();
    cachedMaterialIndex_.reset();
    cachedMaterial_.reset();
    caseCombo_->blockSignals(true);
    caseCombo_->clear();
    for (std::size_t index = 0; index < index_->cases.size(); ++index) {
        caseCombo_->addItem(QStringLiteral("%1 · %2 GHz")
                                .arg(index)
                                .arg(index_->cases[index].frequencyHz / 1.0e9, 0, 'g', 8));
    }
    caseCombo_->setCurrentIndex(0);
    caseCombo_->blockSignals(false);
    caseCombo_->setEnabled(true);
    populateModes();
    if (modeCombo_->count() == 0) {
        const auto error = QStringLiteral("The archive contains no modes in the selected case.");
        if (tryNextDirectoryCandidate()) {
            return;
        }
        statusBar()->showMessage(error);
        if (loadCompletionHandler_) {
            loadCompletionHandler_(false, error);
        }
        return;
    }
    statusBar()->showMessage(QStringLiteral("Indexed %1 case(s), %2 mode(s)")
                                 .arg(index_->cases.size()).arg(index_->modes.size()));
    requestSelectedMode();
}

void MainWindow::populateModes() {
    modeCombo_->blockSignals(true);
    modeCombo_->clear();
    if (index_ && selectedCase() < index_->cases.size()) {
        const auto& selected = index_->cases[selectedCase()];
        for (std::size_t local = 0; local < selected.modeCount; ++local) {
            const auto& mode = index_->modes[selected.modeBegin + local];
            modeCombo_->addItem(QStringLiteral("%1 · %2 · neff %3")
                                    .arg(local + 1)
                                    .arg(qString(mode.polarization))
                                    .arg(formatComplex(mode.neff)));
        }
    }
    if (modeCombo_->count() > 0) {
        modeCombo_->setCurrentIndex(0);
    }
    modeCombo_->setEnabled(modeCombo_->count() > 0);
    modeCombo_->blockSignals(false);
}

void MainWindow::requestSelectedMode() {
    if (!index_ || selectedCase() >= index_->cases.size()
        || selectedMode() >= index_->cases[selectedCase()].modeCount) {
        return;
    }
    const auto generation = ++loadGeneration_;
    const auto caseIndex = selectedCase();
    const auto modeIndex = selectedMode();
    const auto index = index_;
    const auto& selectedIndex = index_->cases[caseIndex];
    const auto cachedMesh = cachedMeshIndex_ == selectedIndex.meshIndex
        ? cachedMesh_ : MeshPtr{};
    const auto cachedMaterial = cachedMaterialIndex_ == selectedIndex.materialStateIndex
        ? cachedMaterial_ : MaterialStatePtr{};
    mesh_.reset();
    material_.reset();
    fields_.reset();
    plot2D_->clearData(QStringLiteral("Loading the selected mode…"));
    materialPlot2D_->clearData(QStringLiteral("Loading the selected mode…"));
    view3D_->clearData(QStringLiteral("Loading the selected mode…"));
    materialView3D_->clearData(QStringLiteral("Loading the selected mode…"));
    metadata_->clear();
    statusBar()->showMessage(QStringLiteral("Loading case %1, mode %2…")
                                 .arg(caseIndex).arg(modeIndex + 1));
    auto* watcher = new QFutureWatcher<ModeOutcome>(this);
    connect(watcher, &QFutureWatcher<ModeOutcome>::finished, this, [this, watcher] {
        const auto outcome = watcher->result();
        watcher->deleteLater();
        applyMode(outcome);
    });
    watcher->setFuture(QtConcurrent::run(
        [generation, caseIndex, modeIndex, index, cachedMesh, cachedMaterial] {
            ModeOutcome outcome;
            outcome.generation = generation;
            outcome.caseIndex = caseIndex;
            outcome.modeIndex = modeIndex;
            try {
                const auto& selected = index->cases[caseIndex];
                outcome.mesh = cachedMesh
                    ? cachedMesh : H5Reader::loadMesh(*index, selected.meshIndex);
                outcome.material = cachedMaterial
                    ? cachedMaterial
                    : H5Reader::loadMaterialState(*index, selected.materialStateIndex);
                if (outcome.material->meshIndex != outcome.mesh->index
                    || outcome.material->epsilonR.size() != outcome.mesh->cells.size()) {
                    throw std::runtime_error("Mesh/material-state references are inconsistent.");
                }
                outcome.fields = H5Reader::loadModeFields(*index, caseIndex, modeIndex);
                if (outcome.fields->electric.size() != outcome.mesh->samplePoints.size()) {
                    throw std::runtime_error("Mesh/visualization sample counts are inconsistent.");
                }
            } catch (const std::exception& error) {
                outcome.error = qString(error.what());
            }
            return outcome;
        }));
}

void MainWindow::applyMode(const ModeOutcome& outcome) {
    if (outcome.generation != loadGeneration_ || outcome.caseIndex != selectedCase()
        || outcome.modeIndex != selectedMode()) {
        return;
    }
    if (!outcome.error.isEmpty()) {
        if (tryNextDirectoryCandidate()) {
            return;
        }
        plot2D_->clearData(outcome.error);
        materialPlot2D_->clearData(outcome.error);
        view3D_->clearData(outcome.error);
        materialView3D_->clearData(outcome.error);
        metadata_->setPlainText(outcome.error);
        statusBar()->showMessage(QStringLiteral("Mode loading failed"));
        if (loadCompletionHandler_) {
            loadCompletionHandler_(false, outcome.error);
        } else {
            QMessageBox::critical(this, QStringLiteral("Could not load mode"), outcome.error);
        }
        return;
    }
    mesh_ = outcome.mesh;
    material_ = outcome.material;
    fields_ = outcome.fields;
    const auto& selectedIndex = index_->cases[outcome.caseIndex];
    cachedMeshIndex_ = selectedIndex.meshIndex;
    cachedMesh_ = mesh_;
    cachedMaterialIndex_ = selectedIndex.materialStateIndex;
    cachedMaterial_ = material_;
    if (mesh_->dimension == 2) {
        dimensionStack_->setCurrentIndex(0);
        tabs2D_->setCurrentIndex(0);
        materialPlot2D_->setData(mesh_, material_, fields_);
        plot2D_->setData(mesh_, material_, fields_);
    } else {
        dimensionStack_->setCurrentIndex(1);
        tabs3D_->setCurrentIndex(0);
        materialView3D_->setData(mesh_, material_, fields_);
        view3D_->setData(mesh_, material_, fields_);
        materialView3D_->setSliceEnabled(sliceEnabled_->isChecked());
        view3D_->setSliceEnabled(sliceEnabled_->isChecked());
        materialView3D_->setSlice(sliceAxisCombo_->currentIndex(),
                                  static_cast<double>(sliceSlider_->value()) / 1000.0);
        view3D_->setSlice(sliceAxisCombo_->currentIndex(),
                          static_cast<double>(sliceSlider_->value()) / 1000.0);
    }
    refreshFieldSelection();
    refreshMetadata();
    directoryScanIndex_.reset();
    statusBar()->showMessage(QStringLiteral("Loaded %1 cells and %2 field samples")
                                 .arg(mesh_->cells.size()).arg(fields_->electric.size()));
    if (loadCompletionHandler_) {
        loadCompletionHandler_(true, {});
    }
}

void MainWindow::refreshFieldSelection() {
    const auto family2D = familyCombo2D_->currentIndex() == 0
        ? FieldFamily::Electric : FieldFamily::Magnetic;
    const auto quantity2D = static_cast<ScalarQuantity>(
        std::clamp(quantityCombo2D_->currentIndex(), 0, 3));
    plot2D_->setSelection(family2D, componentCombo2D_->currentIndex(), quantity2D);

    const auto family3D = familyCombo3D_->currentIndex() == 0
        ? FieldFamily::Electric : FieldFamily::Magnetic;
    const auto quantity3D = static_cast<ScalarQuantity>(
        std::clamp(quantityCombo3D_->currentIndex(), 0, 3));
    view3D_->setSelection(family3D, componentCombo3D_->currentIndex(), quantity3D);
}

void MainWindow::refreshMetadata() {
    if (!index_ || selectedCase() >= index_->cases.size()) {
        metadata_->clear();
        return;
    }
    const auto& selectedCaseData = index_->cases[selectedCase()];
    if (selectedMode() >= selectedCaseData.modeCount) {
        metadata_->clear();
        return;
    }
    const auto& mode = index_->modes[selectedCaseData.modeBegin + selectedMode()];
    QString text;
    text += QStringLiteral("Archive\n");
    text += QStringLiteral("  schema: %1.%2\n").arg(index_->schemaMajor).arg(index_->schemaMinor);
    text += QStringLiteral("  producer: %1 %2\n")
                .arg(qString(index_->producer), qString(index_->producerVersion));
    text += QStringLiteral("  convention: %1\n\n").arg(qString(index_->timeConvention));
    text += QStringLiteral("Case\n");
    text += QStringLiteral("  frequency: %1 GHz\n")
                .arg(selectedCaseData.frequencyHz / 1.0e9, 0, 'g', 10);
    if (mesh_) {
        text += QStringLiteral("  dimension: %1D\n").arg(mesh_->dimension);
        text += QStringLiteral("  topology: %1\n").arg(qString(mesh_->topology));
        text += QStringLiteral("  period: %1 m\n").arg(mesh_->periodM, 0, 'g', 10);
        text += QStringLiteral("  points/cells: %1 / %2\n\n")
                    .arg(mesh_->points.size()).arg(mesh_->cells.size());
    }
    text += QStringLiteral("Mode %1\n").arg(selectedMode() + 1);
    text += QStringLiteral("  polarization: %1\n").arg(qString(mode.polarization));
    text += QStringLiteral("  direction: %1\n").arg(qString(mode.direction));
    text += QStringLiteral("  gamma [1/m]: %1\n").arg(formatComplex(mode.gammaPerM));
    text += QStringLiteral("  neff: %1\n").arg(formatComplex(mode.neff));
    text += QStringLiteral("  folded neff: %1\n").arg(formatComplex(mode.neffFolded));
    text += QStringLiteral("  Bloch multiplier: %1\n").arg(formatComplex(mode.blochMultiplier));
    text += QStringLiteral("  residual: %1\n").arg(mode.residual, 0, 'g', 7);
    if (mode.gaussResidual) {
        text += QStringLiteral("  Gauss residual: %1\n").arg(*mode.gaussResidual, 0, 'g', 7);
    }
    text += QStringLiteral("  PML fraction: %1\n").arg(mode.pmlFraction, 0, 'g', 7);
    text += QStringLiteral("  normalization: %1\n").arg(qString(mode.normalization));
    metadata_->setPlainText(text);
}

std::size_t MainWindow::selectedCase() const {
    return caseCombo_->currentIndex() < 0
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(caseCombo_->currentIndex());
}

std::size_t MainWindow::selectedMode() const {
    return modeCombo_->currentIndex() < 0
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(modeCombo_->currentIndex());
}

} // namespace femperiodic
