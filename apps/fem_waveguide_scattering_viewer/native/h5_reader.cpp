#include "h5_reader.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace fem_waveguide_scattering {
namespace {

std::mutex hdf5Mutex;

class H5Handle final {
public:
    using Closer = herr_t (*)(hid_t);

    H5Handle() = default;
    H5Handle(hid_t id, Closer closer) : id_(id), closer_(closer) {}
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept
        : id_(std::exchange(other.id_, -1)), closer_(other.closer_) {}
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = std::exchange(other.id_, -1);
            closer_ = other.closer_;
        }
        return *this;
    }
    ~H5Handle() { reset(); }

    [[nodiscard]] hid_t get() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }

private:
    void reset() {
        if (id_ >= 0 && closer_ != nullptr) {
            closer_(id_);
        }
        id_ = -1;
    }

    hid_t id_{-1};
    Closer closer_{nullptr};
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string nativePath(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

H5Handle openFile(const std::filesystem::path& path) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    const auto name = nativePath(path);
    H5Handle file(H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (!file) {
        fail(std::format("Could not open HDF5 file: {}", name));
    }
    return file;
}

H5Handle openGroup(hid_t parent, const std::string& name) {
    H5Handle group(H5Gopen2(parent, name.c_str(), H5P_DEFAULT), H5Gclose);
    if (!group) {
        fail(std::format("Missing HDF5 group: {}", name));
    }
    return group;
}

H5Handle openDataset(hid_t parent, const std::string& name) {
    H5Handle dataset(H5Dopen2(parent, name.c_str(), H5P_DEFAULT), H5Dclose);
    if (!dataset) {
        fail(std::format("Missing HDF5 dataset: {}", name));
    }
    return dataset;
}

bool hasLink(hid_t parent, const std::string& name) {
    return H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0;
}

bool hasAttribute(hid_t parent, const std::string& name) {
    return H5Aexists(parent, name.c_str()) > 0;
}

std::vector<hsize_t> dimensions(hid_t dataset) {
    H5Handle space(H5Dget_space(dataset), H5Sclose);
    if (!space) {
        fail("Could not inspect HDF5 dataset dimensions.");
    }
    const int rank = H5Sget_simple_extent_ndims(space.get());
    if (rank < 0) {
        fail("Could not inspect HDF5 dataset rank.");
    }
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    if (rank > 0 && H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) < 0) {
        fail("Could not inspect HDF5 dataset dimensions.");
    }
    return dims;
}

std::size_t elementCount(const std::vector<hsize_t>& dims) {
    std::size_t count = 1;
    for (const auto value : dims) {
        if (value > std::numeric_limits<std::size_t>::max() / count) {
            fail("HDF5 dataset is too large for this process.");
        }
        count *= static_cast<std::size_t>(value);
    }
    return dims.empty() ? 1U : count;
}

std::string readStringAttribute(hid_t parent, const std::string& name) {
    H5Handle attribute(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (!attribute) {
        fail(std::format("Missing HDF5 attribute: {}", name));
    }
    H5Handle type(H5Aget_type(attribute.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_STRING) {
        fail(std::format("HDF5 attribute {} is not text.", name));
    }
    if (H5Tis_variable_str(type.get()) > 0) {
        char* value = nullptr;
        if (H5Aread(attribute.get(), type.get(), &value) < 0 || value == nullptr) {
            fail(std::format("Could not read HDF5 attribute {}.", name));
        }
        std::string result(value);
        H5free_memory(value);
        return result;
    }
    const auto width = H5Tget_size(type.get());
    std::string result(width, '\0');
    if (H5Aread(attribute.get(), type.get(), result.data()) < 0) {
        fail(std::format("Could not read HDF5 attribute {}.", name));
    }
    result.resize(std::strlen(result.c_str()));
    return result;
}

std::int64_t readIntegerAttribute(hid_t parent, const std::string& name) {
    H5Handle attribute(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (!attribute) {
        fail(std::format("Missing HDF5 integer attribute: {}", name));
    }
    std::int64_t value{};
    if (H5Aread(attribute.get(), H5T_NATIVE_LLONG, &value) < 0) {
        fail(std::format("Could not read HDF5 integer attribute {}.", name));
    }
    return value;
}

double readDoubleAttribute(hid_t parent, const std::string& name) {
    H5Handle attribute(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (!attribute) {
        fail(std::format("Missing HDF5 real attribute: {}", name));
    }
    double value{};
    if (H5Aread(attribute.get(), H5T_NATIVE_DOUBLE, &value) < 0) {
        fail(std::format("Could not read HDF5 real attribute {}.", name));
    }
    return value;
}

std::vector<double> readDoubles(hid_t parent, const std::string& name,
                                std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    auto dims = dimensions(dataset.get());
    std::vector<double> values(elementCount(dims));
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data()) < 0) {
        fail(std::format("Could not read real dataset {}.", name));
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return values;
}

std::vector<std::int64_t> readIndices(hid_t parent, const std::string& name,
                                      std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    auto dims = dimensions(dataset.get());
    std::vector<std::int64_t> values(elementCount(dims));
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data()) < 0) {
        fail(std::format("Could not read index dataset {}.", name));
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return values;
}

struct NativeComplex {
    double real;
    double imaginary;
};

std::vector<Complex> readComplex(hid_t parent, const std::string& name,
                                 std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    auto dims = dimensions(dataset.get());
    const auto count = elementCount(dims);
    std::vector<NativeComplex> raw(count);
    H5Handle fileType(H5Dget_type(dataset.get()), H5Tclose);
    if (!fileType) {
        fail(std::format("Could not inspect complex dataset {}.", name));
    }
    H5Handle memoryType;
    const auto typeClass = H5Tget_class(fileType.get());
#if H5_VERSION_GE(2, 0, 0)
    if (typeClass == H5T_COMPLEX) {
        if (H5Dread(dataset.get(), H5T_NATIVE_DOUBLE_COMPLEX, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, raw.data()) < 0) {
            fail(std::format("Could not read native complex dataset {}.", name));
        }
    } else
#endif
    if (typeClass == H5T_COMPOUND) {
        memoryType = H5Handle(H5Tcreate(H5T_COMPOUND, sizeof(NativeComplex)), H5Tclose);
        if (!memoryType
            || H5Tinsert(memoryType.get(), "r", HOFFSET(NativeComplex, real),
                         H5T_NATIVE_DOUBLE) < 0
            || H5Tinsert(memoryType.get(), "i", HOFFSET(NativeComplex, imaginary),
                         H5T_NATIVE_DOUBLE) < 0
            || H5Dread(dataset.get(), memoryType.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       raw.data()) < 0) {
            fail(std::format("Could not read compound complex dataset {}.", name));
        }
    } else if (typeClass == H5T_FLOAT) {
        std::vector<double> real(count);
        if (H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    real.data()) < 0) {
            fail(std::format("Could not read real-valued complex dataset {}.", name));
        }
        std::transform(real.begin(), real.end(), raw.begin(), [](double value) {
            return NativeComplex{value, 0.0};
        });
    } else {
        fail(std::format("Dataset {} is not numeric complex data.", name));
    }
    std::vector<Complex> values;
    values.reserve(count);
    for (const auto& value : raw) {
        values.emplace_back(value.real, value.imaginary);
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return values;
}

std::vector<std::string> readStrings(hid_t parent, const std::string& name) {
    auto dataset = openDataset(parent, name);
    const auto dims = dimensions(dataset.get());
    if (dims.size() != 1) {
        fail(std::format("Text dataset {} must be one-dimensional.", name));
    }
    const auto count = elementCount(dims);
    H5Handle type(H5Dget_type(dataset.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_STRING) {
        fail(std::format("Dataset {} is not text.", name));
    }
    std::vector<std::string> result;
    result.reserve(count);
    if (H5Tis_variable_str(type.get()) > 0) {
        std::vector<char*> values(count, nullptr);
        if (count > 0 && H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, values.data()) < 0) {
            fail(std::format("Could not read text dataset {}.", name));
        }
        for (const auto* value : values) {
            result.emplace_back(value == nullptr ? "" : value);
        }
        H5Handle space(H5Dget_space(dataset.get()), H5Sclose);
        H5Dvlen_reclaim(type.get(), space.get(), H5P_DEFAULT, values.data());
        return result;
    }
    const auto width = H5Tget_size(type.get());
    std::vector<char> values(count * width, '\0');
    if (!values.empty() && H5Dread(dataset.get(), type.get(), H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data()) < 0) {
        fail(std::format("Could not read text dataset {}.", name));
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto* start = values.data() + index * width;
        const auto length = std::find(start, start + width, '\0') - start;
        result.emplace_back(start, static_cast<std::size_t>(length));
    }
    return result;
}

RealMatrix readRealMatrix(hid_t parent, const std::string& name,
                          std::size_t expectedRows = 0) {
    std::vector<hsize_t> dims;
    auto values = readDoubles(parent, name, &dims);
    if (dims.size() != 2 || (expectedRows != 0 && dims[0] != expectedRows)) {
        fail(std::format("Dataset {} has an unexpected matrix shape.", name));
    }
    return {static_cast<std::size_t>(dims[0]), static_cast<std::size_t>(dims[1]),
            std::move(values)};
}

ComplexMatrix readComplexMatrix(hid_t parent, const std::string& name,
                                std::size_t expectedRows = 0,
                                std::size_t expectedColumns = 0) {
    std::vector<hsize_t> dims;
    auto values = readComplex(parent, name, &dims);
    if (dims.size() != 2 || (expectedRows != 0 && dims[0] != expectedRows)
        || (expectedColumns != 0 && dims[1] != expectedColumns)) {
        fail(std::format("Dataset {} has an unexpected complex matrix shape.", name));
    }
    return {static_cast<std::size_t>(dims[0]), static_cast<std::size_t>(dims[1]),
            std::move(values)};
}

std::vector<SParameter> readSParameters(hid_t resultGroup) {
    auto group = openGroup(resultGroup, "s_parameters");
    const auto sides = readStrings(group.get(), "side");
    const auto outModes = readIndices(group.get(), "out_mode");
    const auto inModes = readIndices(group.get(), "in_mode");
    const auto values = readComplex(group.get(), "value");
    if (sides.size() != values.size() || outModes.size() != values.size()
        || inModes.size() != values.size()) {
        fail("Inconsistent HDF5 S-parameter record lengths.");
    }
    std::vector<SParameter> result;
    result.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result.push_back({sides[index], outModes[index], inModes[index], values[index]});
    }
    return result;
}

ModeData readMode(hid_t modesGroup, std::size_t index) {
    auto group = openGroup(modesGroup, std::format("{:06d}", index));
    std::vector<hsize_t> xDims;
    auto x = readDoubles(group.get(), "x", &xDims);
    if (xDims.size() != 1 || x.empty()) {
        fail("Modal x dataset must be a nonempty vector.");
    }
    auto electric = readComplexMatrix(group.get(), "E", 3, x.size());
    auto magnetic = readComplexMatrix(group.get(), "H", 3, x.size());
    return {std::move(x), std::move(electric), std::move(magnetic),
            std::format("mode {}", index)};
}

SceneData readScene(hid_t resultGroup) {
    auto group = openGroup(resultGroup, "scene");
    auto points = readRealMatrix(group.get(), "points", 2);
    std::vector<hsize_t> triangleDims;
    auto triangleValues = readIndices(group.get(), "triangles", &triangleDims);
    if (triangleDims.size() != 2 || triangleDims[0] != 3) {
        fail("Scene triangles must have shape (3, M).");
    }
    IndexMatrix triangles{3, static_cast<std::size_t>(triangleDims[1]),
                          std::move(triangleValues)};
    auto eps = readComplex(group.get(), "eps_r");
    if (eps.size() != triangles.columns) {
        fail("Scene permittivity count does not match its triangles.");
    }
    const auto xSpanValues = readDoubles(group.get(), "x_span");
    const auto zSpanValues = readDoubles(group.get(), "z_span");
    if (xSpanValues.size() != 2 || zSpanValues.size() != 2) {
        fail("Scene spans must contain two values.");
    }
    auto lineGroup = openGroup(group.get(), "lines");
    const auto kinds = readStrings(lineGroup.get(), "kind");
    const auto labels = readStrings(lineGroup.get(), "label");
    std::vector<hsize_t> endpointDims;
    const auto endpoints = readDoubles(lineGroup.get(), "endpoints", &endpointDims);
    if (endpointDims.size() != 3 || endpointDims[0] != kinds.size()
        || endpointDims[1] != 2 || endpointDims[2] != 2 || labels.size() != kinds.size()) {
        fail("Scene line datasets have inconsistent shapes.");
    }
    std::vector<SceneLine> lines;
    lines.reserve(kinds.size());
    for (std::size_t index = 0; index < kinds.size(); ++index) {
        const auto offset = index * 4;
        lines.push_back({kinds[index], labels[index],
                         {endpoints[offset], endpoints[offset + 1],
                          endpoints[offset + 2], endpoints[offset + 3]}});
    }
    return {std::move(points), std::move(triangles), std::move(eps),
            {xSpanValues[0], xSpanValues[1]}, {zSpanValues[0], zSpanValues[1]},
            std::move(lines)};
}

std::string resultName(std::size_t index) {
    return std::format("{:06d}", index);
}

} // namespace

std::shared_ptr<FileIndex> H5Reader::loadIndex(const std::filesystem::path& path) {
    const std::scoped_lock lock(hdf5Mutex);
    if (!std::filesystem::is_regular_file(path)) {
        fail(std::format("HDF5 file does not exist: {}", nativePath(path)));
    }
    auto file = openFile(path);
    if (readStringAttribute(file.get(), "format") != "cem-fem-results") {
        fail("The selected file is not a FEM Waveguide Scattering HDF5 result.");
    }
    if (readStringAttribute(file.get(), "schema") != "1.0"
        || readStringAttribute(file.get(), "solver_family") != "waveguide_scattering"
        || readStringAttribute(file.get(), "time_convention") != "exp(+i*omega*t)"
        || readStringAttribute(file.get(), "units") != "SI"
        || readStringAttribute(file.get(), "field_representation") != "sampled-fields; exp(-i*ky*y)"
        || readIntegerAttribute(file.get(), "dimension") != 2) {
        fail("Incompatible computational electromagnetics scattering archive.");
    }
    if (readIntegerAttribute(file.get(), "schema_version") != 1) {
        fail("This viewer supports FEM Waveguide Scattering HDF5 schema version 1 only.");
    }
    const auto kind = readStringAttribute(file.get(), "kind");
    const auto resultCount = readIntegerAttribute(file.get(), "result_count");
    if ((kind != "single" && kind != "sweep") || resultCount <= 0) {
        fail("FEM Waveguide Scattering result kind/count metadata is invalid.");
    }
    auto frequencies = readDoubles(file.get(), "frequencies_hz");
    if (frequencies.size() != static_cast<std::size_t>(resultCount)) {
        fail("FEM Waveguide Scattering frequency and result counts do not match.");
    }
    auto resultsGroup = openGroup(file.get(), "results");
    std::vector<std::vector<SParameter>> summaries;
    summaries.reserve(frequencies.size());
    for (std::size_t index = 0; index < frequencies.size(); ++index) {
        auto resultGroup = openGroup(resultsGroup.get(), resultName(index));
        summaries.push_back(readSParameters(resultGroup.get()));
    }
    return std::make_shared<FileIndex>(FileIndex{
        std::filesystem::absolute(path), kind, std::move(frequencies), std::move(summaries)});
}

std::shared_ptr<ResultData> H5Reader::loadResult(const FileIndex& index,
                                                 std::size_t resultIndex) {
    const std::scoped_lock lock(hdf5Mutex);
    if (resultIndex >= index.frequenciesHz.size()) {
        fail("Requested result index is outside the saved frequency sweep.");
    }
    auto file = openFile(index.path);
    auto resultsGroup = openGroup(file.get(), "results");
    auto group = openGroup(resultsGroup.get(), resultName(resultIndex));
    auto coordinates = readRealMatrix(group.get(), "coordinates", 2);
    const auto pointCount = coordinates.columns;
    auto fieldsGroup = openGroup(group.get(), "fields");
    constexpr std::array<std::string_view, 6> fieldNames{
        "E_incident", "E_scattered", "E_total",
        "H_incident", "H_scattered", "H_total"};
    std::array<ComplexMatrix, 6> fields;
    for (std::size_t fieldIndex = 0; fieldIndex < fieldNames.size(); ++fieldIndex) {
        fields[fieldIndex] = readComplexMatrix(
            fieldsGroup.get(), std::string(fieldNames[fieldIndex]), 3, pointCount);
    }
    auto modesGroup = openGroup(group.get(), "modes");
    const auto modeCount = readIntegerAttribute(modesGroup.get(), "count");
    if (modeCount < 0) {
        fail("HDF5 mode count cannot be negative.");
    }
    std::vector<ModeData> modes;
    modes.reserve(static_cast<std::size_t>(modeCount));
    for (std::size_t modeIndex = 0; modeIndex < static_cast<std::size_t>(modeCount);
         ++modeIndex) {
        modes.push_back(readMode(modesGroup.get(), modeIndex));
    }
    std::optional<SceneData> scene;
    if (hasLink(group.get(), "scene")) {
        scene = readScene(group.get());
    }
    std::optional<double> ky;
    if (hasAttribute(group.get(), "ky")) {
        ky = readDoubleAttribute(group.get(), "ky");
    }
    auto result = std::make_shared<ResultData>();
    result->frequencyHz = index.frequenciesHz[resultIndex];
    result->ky = ky;
    result->coordinates = std::move(coordinates);
    result->fields = std::move(fields);
    result->sParameters = index.sParameters[resultIndex];
    result->modes = std::move(modes);
    result->scene = std::move(scene);
    return result;
}

} // namespace fem_waveguide_scattering
