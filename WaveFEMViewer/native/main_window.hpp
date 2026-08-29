#pragma once

#include "model.hpp"

#include <QMainWindow>

#include <array>
#include <memory>

class QComboBox;
class QLabel;
class QTableWidget;
class QTabWidget;

namespace wavefem {

class PlotWidget;

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);
    void loadPath(const QString& path);

private:
    struct ModalControls {
        QComboBox* mode{};
        QComboBox* component{};
        QComboBox* quantity{};
        PlotWidget* plot{};
    };

    struct VectorControls {
        QComboBox* part{};
        QComboBox* quantity{};
        PlotWidget* plot{};
    };

    struct LoadOutcome {
        ResultPtr result;
        QString error;
        int index{};
        qint64 milliseconds{};
        quint64 generation{};
    };

    void buildUi();
    QWidget* buildSParameterTab();
    QWidget* buildModalTab(FieldName field, ModalControls& controls);
    QWidget* buildVectorTab(FieldName field, VectorControls& controls);
    void chooseFile();
    void chooseDirectory();
    void loadDirectory(const QString& directoryPath);
    void refreshFileChoices(const QString& selectedPath);
    void loadSelectedResult();
    void applyLoadedResult(const LoadOutcome& outcome);
    void refreshSParameters();
    void refreshModal(FieldName field);
    void refreshVector(FieldName field);
    void refreshCurrentTab();
    void setResultControlsEnabled(bool enabled);
    [[nodiscard]] int selectedResultIndex() const;

    QLabel* pathLabel_{};
    QComboBox* fileCombo_{};
    QComboBox* frequencyCombo_{};
    QTabWidget* tabs_{};
    QComboBox* sQuantity_{};
    QTableWidget* sTable_{};
    PlotWidget* sPlot_{};
    std::array<ModalControls, 2> modal_{};
    std::array<VectorControls, 2> vector_{};
    std::shared_ptr<const FileIndex> fileIndex_;
    ResultPtr result_;
    quint64 loadGeneration_{};
};

} // namespace wavefem
