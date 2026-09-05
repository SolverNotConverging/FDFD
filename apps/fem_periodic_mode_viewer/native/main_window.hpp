#pragma once

#include "model.hpp"

#include <QMainWindow>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

class QComboBox;
class QCheckBox;
class QLabel;
class QPushButton;
class QSlider;
class QStackedWidget;
class QTabWidget;
class QTextEdit;

namespace femperiodic {

class FieldPlot2D;
class VtkFieldView;

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

    void loadPath(const std::filesystem::path& path);
    void setLoadCompletionHandler(std::function<void(bool, const QString&)> handler);
    [[nodiscard]] bool verifySliceRenderingForTest();

private:
    struct IndexOutcome {
        std::size_t generation{};
        FileIndexPtr index;
        QString error;
    };

    struct ModeOutcome {
        std::size_t generation{};
        std::size_t caseIndex{};
        std::size_t modeIndex{};
        MeshPtr mesh;
        MaterialStatePtr material;
        ModeFieldsPtr fields;
        QString error;
    };

    void chooseFile();
    void chooseDirectory();
    void loadDirectory(const std::filesystem::path& directoryPath);
    void loadFile(const std::filesystem::path& filePath);
    bool tryNextDirectoryCandidate();
    void refreshFileChoices(const std::filesystem::path& selectedPath);
    void applyIndex(const IndexOutcome& outcome);
    void populateModes();
    void requestSelectedMode();
    void applyMode(const ModeOutcome& outcome);
    void refreshFieldSelection();
    void refreshMetadata();
    [[nodiscard]] std::size_t selectedCase() const;
    [[nodiscard]] std::size_t selectedMode() const;

    FileIndexPtr index_;
    MeshPtr mesh_;
    MaterialStatePtr material_;
    ModeFieldsPtr fields_;
    std::size_t loadGeneration_{};
    std::optional<int> directoryScanIndex_;
    std::optional<std::size_t> cachedMeshIndex_;
    MeshPtr cachedMesh_;
    std::optional<std::size_t> cachedMaterialIndex_;
    MaterialStatePtr cachedMaterial_;
    std::function<void(bool, const QString&)> loadCompletionHandler_;

    QLabel* pathLabel_{};
    QComboBox* fileCombo_{};
    QComboBox* caseCombo_{};
    QComboBox* modeCombo_{};
    QComboBox* familyCombo2D_{};
    QComboBox* componentCombo2D_{};
    QComboBox* quantityCombo2D_{};
    QComboBox* familyCombo3D_{};
    QComboBox* componentCombo3D_{};
    QComboBox* quantityCombo3D_{};
    QComboBox* sliceAxisCombo_{};
    QCheckBox* sliceEnabled_{};
    QSlider* sliceSlider_{};
    QTextEdit* metadata_{};
    QStackedWidget* dimensionStack_{};
    QTabWidget* tabs2D_{};
    QTabWidget* tabs3D_{};
    FieldPlot2D* materialPlot2D_{};
    FieldPlot2D* plot2D_{};
    VtkFieldView* materialView3D_{};
    VtkFieldView* view3D_{};
};

} // namespace femperiodic
