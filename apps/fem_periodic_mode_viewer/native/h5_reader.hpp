#pragma once

#include "model.hpp"

#include <filesystem>
#include <memory>

namespace femperiodic {

class H5Reader final {
public:
    [[nodiscard]] static FileIndexPtr loadIndex(const std::filesystem::path& path);
    [[nodiscard]] static MeshPtr loadMesh(const FileIndex& index, std::size_t meshIndex);
    [[nodiscard]] static MaterialStatePtr loadMaterialState(
        const FileIndex& index, std::size_t materialStateIndex);
    [[nodiscard]] static ModeFieldsPtr loadModeFields(
        const FileIndex& index, std::size_t caseIndex, std::size_t localModeIndex);
    [[nodiscard]] static ModeCoefficients loadModeCoefficients(
        const FileIndex& index, std::size_t caseIndex, std::size_t localModeIndex);
};

} // namespace femperiodic
