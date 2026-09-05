#pragma once

#include "model.hpp"

#include <filesystem>
#include <memory>

namespace fem_waveguide_scattering {

class H5Reader final {
public:
    [[nodiscard]] static std::shared_ptr<FileIndex> loadIndex(
        const std::filesystem::path& path);
    [[nodiscard]] static std::shared_ptr<ResultData> loadResult(
        const FileIndex& index, std::size_t resultIndex);
};

} // namespace fem_waveguide_scattering
