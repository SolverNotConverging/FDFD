#include "h5_reader.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace femperiodic {
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
    const auto rank = H5Sget_simple_extent_ndims(space.get());
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
    if (std::find(dims.begin(), dims.end(), hsize_t{0}) != dims.end()) {
        return 0;
    }
    std::size_t count = 1;
    for (const auto value : dims) {
        if (value > std::numeric_limits<std::size_t>::max() / count) {
            fail("HDF5 dataset is too large for this process.");
        }
        count *= static_cast<std::size_t>(value);
    }
    return dims.empty() ? 1U : count;
}

void requireScalarAttribute(hid_t attribute, const std::string& name) {
    H5Handle space(H5Aget_space(attribute), H5Sclose);
    if (!space || H5Sget_simple_extent_type(space.get()) != H5S_SCALAR) {
        fail(std::format("HDF5 attribute {} must be scalar.", name));
    }
}

std::string readStringAttribute(hid_t parent, const std::string& name) {
    H5Handle attribute(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (!attribute) {
        fail(std::format("Missing HDF5 text attribute: {}", name));
    }
    requireScalarAttribute(attribute.get(), name);
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
    result.resize(std::find(result.begin(), result.end(), '\0') - result.begin());
    return result;
}

std::string optionalStringAttribute(
    hid_t parent, const std::string& name, const std::string& fallback = {}) {
    return hasAttribute(parent, name) ? readStringAttribute(parent, name) : fallback;
}

std::int64_t readIntegerAttribute(hid_t parent, const std::string& name) {
    H5Handle attribute(H5Aopen(parent, name.c_str(), H5P_DEFAULT), H5Aclose);
    if (!attribute) {
        fail(std::format("Missing HDF5 integer attribute: {}", name));
    }
    requireScalarAttribute(attribute.get(), name);
    H5Handle type(H5Aget_type(attribute.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_INTEGER) {
        fail(std::format("HDF5 attribute {} must contain an integer.", name));
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
        fail(std::format("Missing HDF5 floating-point attribute: {}", name));
    }
    requireScalarAttribute(attribute.get(), name);
    H5Handle type(H5Aget_type(attribute.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_FLOAT) {
        fail(std::format("HDF5 attribute {} must contain a floating-point value.", name));
    }
    double value{};
    if (H5Aread(attribute.get(), H5T_NATIVE_DOUBLE, &value) < 0 || !std::isfinite(value)) {
        fail(std::format("Could not read finite HDF5 attribute {}.", name));
    }
    return value;
}

std::vector<double> readDoubles(hid_t parent, const std::string& name,
                                std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    H5Handle type(H5Dget_type(dataset.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_FLOAT) {
        fail(std::format("Real dataset {} must use a floating-point datatype.", name));
    }
    auto dims = dimensions(dataset.get());
    std::vector<double> values(elementCount(dims));
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data()) < 0) {
        fail(std::format("Could not read real dataset {}.", name));
    }
    if (!std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        fail(std::format("Real dataset {} contains a non-finite value.", name));
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return values;
}

std::vector<std::int64_t> readIndices(hid_t parent, const std::string& name,
                                      std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    H5Handle type(H5Dget_type(dataset.get()), H5Tclose);
    if (!type || H5Tget_class(type.get()) != H5T_INTEGER) {
        fail(std::format("Integer dataset {} must use an integer datatype.", name));
    }
    auto dims = dimensions(dataset.get());
    std::vector<std::int64_t> values(elementCount(dims));
    if (!values.empty() && H5Dread(dataset.get(), H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, values.data()) < 0) {
        fail(std::format("Could not read integer dataset {}.", name));
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return values;
}

struct ComplexPair {
    double real;
    double imaginary;
};

H5Handle compoundComplexMemoryType() {
    H5Handle type(H5Tcreate(H5T_COMPOUND, sizeof(ComplexPair)), H5Tclose);
    if (!type
        || H5Tinsert(type.get(), "r", HOFFSET(ComplexPair, real), H5T_NATIVE_DOUBLE) < 0
        || H5Tinsert(type.get(), "i", HOFFSET(ComplexPair, imaginary), H5T_NATIVE_DOUBLE) < 0) {
        fail("Could not create the HDF5 compound-complex memory type.");
    }
    return type;
}

void readComplexRaw(hid_t dataset, hid_t memorySpace, hid_t fileSpace,
                    std::vector<ComplexPair>& raw, const std::string& name) {
    H5Handle fileType(H5Dget_type(dataset), H5Tclose);
    if (!fileType) {
        fail(std::format("Could not inspect complex dataset {}.", name));
    }
    const auto typeClass = H5Tget_class(fileType.get());
#if H5_VERSION_GE(2, 0, 0)
    if (typeClass == H5T_COMPLEX) {
        if (H5Tget_size(fileType.get()) != sizeof(ComplexPair)) {
            fail(std::format("Native complex dataset {} must use complex128 values.", name));
        }
        if (H5Dread(dataset, H5T_NATIVE_DOUBLE_COMPLEX, memorySpace, fileSpace,
                    H5P_DEFAULT, raw.data()) < 0) {
            fail(std::format("Could not read native complex dataset {}.", name));
        }
        return;
    }
#endif
    if (typeClass != H5T_COMPOUND) {
        fail(std::format("Dataset {} is not complex data.", name));
    }
    if (H5Tget_nmembers(fileType.get()) != 2) {
        fail(std::format("Compound complex dataset {} must have exactly r/i members.", name));
    }
    std::set<std::string> memberNames;
    for (unsigned member = 0; member < 2; ++member) {
        char* rawName = H5Tget_member_name(fileType.get(), member);
        if (rawName == nullptr) {
            fail(std::format("Could not inspect compound complex dataset {}.", name));
        }
        memberNames.emplace(rawName);
        H5free_memory(rawName);
        H5Handle memberType(H5Tget_member_type(fileType.get(), member), H5Tclose);
        if (!memberType || H5Tget_class(memberType.get()) != H5T_FLOAT
            || H5Tget_size(memberType.get()) != sizeof(double)) {
            fail(std::format("Compound complex dataset {} must use float64 r/i members.", name));
        }
    }
    if (memberNames != std::set<std::string>{"i", "r"}) {
        fail(std::format("Compound complex dataset {} must have exact r/i members.", name));
    }
    auto memoryType = compoundComplexMemoryType();
    if (H5Dread(dataset, memoryType.get(), memorySpace, fileSpace,
                H5P_DEFAULT, raw.data()) < 0) {
        fail(std::format("Could not read compound complex dataset {}.", name));
    }
}

std::vector<Complex> toComplex(const std::vector<ComplexPair>& raw,
                               const std::string& name) {
    std::vector<Complex> result;
    result.reserve(raw.size());
    for (const auto& value : raw) {
        if (!std::isfinite(value.real) || !std::isfinite(value.imaginary)) {
            fail(std::format("Complex dataset {} contains a non-finite value.", name));
        }
        result.emplace_back(value.real, value.imaginary);
    }
    return result;
}

std::vector<Complex> readComplex(hid_t parent, const std::string& name,
                                 std::vector<hsize_t>* outputDims = nullptr) {
    auto dataset = openDataset(parent, name);
    auto dims = dimensions(dataset.get());
    std::vector<ComplexPair> raw(elementCount(dims));
    if (!raw.empty()) {
        readComplexRaw(dataset.get(), H5S_ALL, H5S_ALL, raw, name);
    }
    if (outputDims != nullptr) {
        *outputDims = std::move(dims);
    }
    return toComplex(raw, name);
}

std::vector<double> readDoubleVector(hid_t parent, const std::string& name) {
    std::vector<hsize_t> dims;
    auto values = readDoubles(parent, name, &dims);
    if (dims.size() != 1) {
        fail(std::format("Dataset {} must be one-dimensional.", name));
    }
    return values;
}

std::vector<std::int64_t> readIndexVector(hid_t parent, const std::string& name) {
    std::vector<hsize_t> dims;
    auto values = readIndices(parent, name, &dims);
    if (dims.size() != 1) {
        fail(std::format("Dataset {} must be one-dimensional.", name));
    }
    return values;
}

std::vector<Complex> readComplexVector(hid_t parent, const std::string& name) {
    std::vector<hsize_t> dims;
    auto values = readComplex(parent, name, &dims);
    if (dims.size() != 1) {
        fail(std::format("Dataset {} must be one-dimensional.", name));
    }
    return values;
}

std::vector<Complex> readComplexMode(hid_t parent, const std::string& name,
                                     std::size_t modeIndex,
                                     std::vector<hsize_t>* fullDims = nullptr) {
    auto dataset = openDataset(parent, name);
    auto dims = dimensions(dataset.get());
    if (dims.size() < 2 || modeIndex >= dims[0]) {
        fail(std::format("Dataset {} has no requested mode {}.", name, modeIndex));
    }
    std::vector<hsize_t> start(dims.size(), 0);
    std::vector<hsize_t> count = dims;
    start[0] = static_cast<hsize_t>(modeIndex);
    count[0] = 1;

    H5Handle fileSpace(H5Dget_space(dataset.get()), H5Sclose);
    if (!fileSpace
        || H5Sselect_hyperslab(fileSpace.get(), H5S_SELECT_SET, start.data(), nullptr,
                               count.data(), nullptr) < 0) {
        fail(std::format("Could not select mode {} from dataset {}.", modeIndex, name));
    }
    H5Handle memorySpace(
        H5Screate_simple(static_cast<int>(count.size()), count.data(), nullptr), H5Sclose);
    if (!memorySpace) {
        fail("Could not create an HDF5 mode hyperslab memory space.");
    }
    std::vector<ComplexPair> raw(elementCount(count));
    if (!raw.empty()) {
        readComplexRaw(dataset.get(), memorySpace.get(), fileSpace.get(), raw, name);
    }
    if (fullDims != nullptr) {
        *fullDims = std::move(dims);
    }
    return toComplex(raw, name);
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

std::string objectName(std::size_t index) {
    return std::format("{:06d}", index);
}

std::size_t checkedIndex(std::int64_t value, const std::string& name) {
    if (value < 0) {
        fail(std::format("{} cannot be negative.", name));
    }
    return static_cast<std::size_t>(value);
}

void requireLength(std::size_t actual, std::size_t expected, const std::string& name) {
    if (actual != expected) {
        fail(std::format("Dataset {} has length {}; expected {}.", name, actual, expected));
    }
}

void validateComplexFinite(const Complex& value, const std::string& name) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
        fail(std::format("{} is not finite.", name));
    }
}

void validateOptionalModeMetadata(hid_t caseGroup, std::size_t expectedModes) {
    if (!hasLink(caseGroup, "mode_metadata")) {
        return;
    }
    auto metadata = openGroup(caseGroup, "mode_metadata");
    const auto hasMask = hasLink(metadata.get(), "has_power");
    const auto hasPower = hasLink(metadata.get(), "power");
    if (hasMask != hasPower) {
        fail("mode_metadata must provide has_power and power together.");
    }
    if (!hasMask) {
        return;
    }
    const auto mask = readIndexVector(metadata.get(), "has_power");
    requireLength(mask.size(), expectedModes, "mode_metadata/has_power");
    if (!std::all_of(mask.begin(), mask.end(), [](const auto value) {
            return value == 0 || value == 1;
        })) {
        fail("mode_metadata/has_power must contain only zero or one.");
    }
    const auto power = readComplexVector(metadata.get(), "power");
    requireLength(power.size(), expectedModes, "mode_metadata/power");
}

} // namespace

FileIndexPtr H5Reader::loadIndex(const std::filesystem::path& path) {
    const std::scoped_lock lock(hdf5Mutex);
    if (!std::filesystem::is_regular_file(path)) {
        fail(std::format("HDF5 file does not exist: {}", nativePath(path)));
    }
    auto file = openFile(path);
    if (readStringAttribute(file.get(), "format") != "cem-fem-results") {
        fail("The selected file is not a FEM periodic-mode result.");
    }
    if (readStringAttribute(file.get(), "schema") != "1.0"
        || readStringAttribute(file.get(), "solver_family") != "periodic_modes"
        || readStringAttribute(file.get(), "units") != "SI") {
        fail("Incompatible computational electromagnetics periodic archive.");
    }
    const auto schemaMajor = readIntegerAttribute(file.get(), "schema_major");
    const auto schemaMinor = readIntegerAttribute(file.get(), "schema_minor");
    if (schemaMajor != 1) {
        fail(std::format("Unsupported FEM periodic HDF5 schema major version {}.", schemaMajor));
    }
    if (schemaMinor < 0) {
        fail("FEM periodic HDF5 schema minor version cannot be negative.");
    }
    const auto kind = readStringAttribute(file.get(), "kind");
    const auto caseCountRaw = readIntegerAttribute(file.get(), "case_count");
    if ((kind != "single" && kind != "sweep") || caseCountRaw <= 0
        || (kind == "single" && caseCountRaw != 1)) {
        fail("FEM periodic HDF5 kind/case_count metadata is inconsistent.");
    }
    const auto caseCount = static_cast<std::size_t>(caseCountRaw);
    const auto timeConvention = readStringAttribute(file.get(), "time_convention");
    const auto fieldRepresentation = readStringAttribute(file.get(), "field_representation");
    if (timeConvention != "exp(+i*omega*t)" || fieldRepresentation != "periodic-envelope") {
        fail("Unsupported phasor or field representation in FEM periodic HDF5 file.");
    }

    auto indexGroup = openGroup(file.get(), "/index");
    const auto frequencies = readDoubleVector(indexGroup.get(), "frequency_hz");
    const auto offsets = readIndexVector(indexGroup.get(), "mode_offsets");
    const auto meshIndices = readIndexVector(indexGroup.get(), "mesh_index");
    const auto materialIndices = readIndexVector(indexGroup.get(), "material_state_index");
    requireLength(frequencies.size(), caseCount, "frequency_hz");
    requireLength(offsets.size(), caseCount + 1, "mode_offsets");
    requireLength(meshIndices.size(), caseCount, "mesh_index");
    requireLength(materialIndices.size(), caseCount, "material_state_index");
    if (offsets.front() != 0
        || !std::is_sorted(offsets.begin(), offsets.end())
        || offsets.back() < 0) {
        fail("FEM periodic mode_offsets is invalid.");
    }
    const auto totalModes = static_cast<std::size_t>(offsets.back());

    const auto gamma = readComplexVector(indexGroup.get(), "gamma_per_m");
    const auto neff = readComplexVector(indexGroup.get(), "neff");
    const auto neffFolded = readComplexVector(indexGroup.get(), "neff_folded");
    const auto bloch = readComplexVector(indexGroup.get(), "bloch_multiplier");
    const auto alpha = readDoubleVector(indexGroup.get(), "alpha_per_m");
    const auto beta = readDoubleVector(indexGroup.get(), "beta_per_m");
    const auto betaFolded = readDoubleVector(indexGroup.get(), "beta_folded_per_m");
    const auto residual = readDoubleVector(indexGroup.get(), "residual");
    const auto pmlFraction = readDoubleVector(indexGroup.get(), "pml_fraction");
    const auto polarizations = readStrings(indexGroup.get(), "polarization");
    const auto directions = readStrings(indexGroup.get(), "direction");
    const auto normalizations = readStrings(indexGroup.get(), "normalization");
    const auto hasGauss = hasLink(indexGroup.get(), "gauss_residual");
    const auto hasGaussAvailable = hasLink(indexGroup.get(), "gauss_available");
    const auto gauss = hasGauss
        ? readDoubleVector(indexGroup.get(), "gauss_residual") : std::vector<double>{};
    const auto gaussAvailable = hasGaussAvailable
        ? readIndexVector(indexGroup.get(), "gauss_available")
        : std::vector<std::int64_t>{};

    for (const auto& [name, size] : std::array<std::pair<std::string_view, std::size_t>, 12>{
             {{"gamma_per_m", gamma.size()}, {"neff", neff.size()},
              {"neff_folded", neffFolded.size()}, {"bloch_multiplier", bloch.size()},
              {"alpha_per_m", alpha.size()}, {"beta_per_m", beta.size()},
              {"beta_folded_per_m", betaFolded.size()}, {"residual", residual.size()},
              {"pml_fraction", pmlFraction.size()}, {"polarization", polarizations.size()},
              {"direction", directions.size()}, {"normalization", normalizations.size()}}}) {
        requireLength(size, totalModes, std::string(name));
    }
    if (hasGauss) {
        requireLength(gauss.size(), totalModes, "gauss_residual");
    }
    if (hasGaussAvailable) {
        if (!hasGauss) {
            fail("gauss_available requires a matching gauss_residual dataset.");
        }
        requireLength(gaussAvailable.size(), totalModes, "gauss_available");
        if (!std::all_of(gaussAvailable.begin(), gaussAvailable.end(),
                         [](const auto value) { return value == 0 || value == 1; })) {
            fail("gauss_available must contain only zero or one.");
        }
    }

    auto result = std::make_shared<FileIndex>();
    result->path = std::filesystem::absolute(path);
    result->kind = kind;
    result->producer = optionalStringAttribute(file.get(), "producer", "unknown");
    result->producerVersion = optionalStringAttribute(file.get(), "producer_version", "unknown");
    result->timeConvention = timeConvention;
    result->fieldRepresentation = fieldRepresentation;
    result->schemaMajor = schemaMajor;
    result->schemaMinor = schemaMinor;
    result->cases.reserve(caseCount);
    for (std::size_t index = 0; index < caseCount; ++index) {
        if (!(frequencies[index] > 0.0)) {
            fail("Every FEM periodic case frequency must be positive.");
        }
        const auto begin = checkedIndex(offsets[index], "mode offset");
        const auto end = checkedIndex(offsets[index + 1], "mode offset");
        if (end <= begin) {
            fail("Every FEM periodic case must contain at least one mode.");
        }
        result->cases.push_back({frequencies[index], begin, end - begin,
                                 checkedIndex(meshIndices[index], "mesh index"),
                                 checkedIndex(materialIndices[index], "material state index")});
    }
    result->modes.reserve(totalModes);
    for (std::size_t index = 0; index < totalModes; ++index) {
        validateComplexFinite(gamma[index], "gamma");
        validateComplexFinite(neff[index], "neff");
        validateComplexFinite(neffFolded[index], "folded neff");
        validateComplexFinite(bloch[index], "Bloch multiplier");
        if (residual[index] < 0.0 || pmlFraction[index] < 0.0 || pmlFraction[index] > 1.0) {
            fail("Modal residual/PML-fraction data is invalid.");
        }
        result->modes.push_back({
            gamma[index], neff[index], neffFolded[index], bloch[index],
            alpha[index], beta[index], betaFolded[index], residual[index],
            !hasGauss || (hasGaussAvailable && gaussAvailable[index] == 0)
                ? std::nullopt : std::optional<double>(gauss[index]),
            pmlFraction[index], polarizations[index], directions[index], normalizations[index]});
    }
    return result;
}

MeshPtr H5Reader::loadMesh(const FileIndex& index, std::size_t meshIndex) {
    const std::scoped_lock lock(hdf5Mutex);
    auto file = openFile(index.path);
    auto meshes = openGroup(file.get(), "/meshes");
    auto group = openGroup(meshes.get(), objectName(meshIndex));
    const auto dimension = static_cast<int>(readIntegerAttribute(group.get(), "dimension"));
    const auto topology = readStringAttribute(group.get(), "topology");
    if ((dimension != 2 && dimension != 3)
        || (dimension == 2 && topology != "triangle3")
        || (dimension == 3 && topology != "tetra4")) {
        fail("FEM periodic mesh dimension/topology is unsupported.");
    }
    std::vector<hsize_t> pointDims;
    const auto pointValues = readDoubles(group.get(), "points", &pointDims);
    if (pointDims.size() != 2 || pointDims[1] != 3 || pointDims[0] == 0) {
        fail("FEM periodic mesh points must have shape (N, 3).");
    }
    std::vector<hsize_t> cellDims;
    const auto cellValues = readIndices(group.get(), "cells", &cellDims);
    const auto verticesPerCell = dimension == 2 ? 3U : 4U;
    if (cellDims.size() != 2 || cellDims[1] != verticesPerCell || cellDims[0] == 0) {
        fail("FEM periodic mesh cells have an invalid shape.");
    }

    auto result = std::make_shared<MeshData>();
    result->index = meshIndex;
    result->dimension = dimension;
    result->topology = topology;
    result->periodicAxis = readStringAttribute(group.get(), "periodic_axis");
    result->periodM = readDoubleAttribute(group.get(), "period_m");
    result->referenceZM = readDoubleAttribute(group.get(), "reference_z_m");
    if (result->periodicAxis != "z" || !(result->periodM > 0.0)) {
        fail("FEM periodic mesh has invalid periodic-axis metadata.");
    }
    result->points.reserve(static_cast<std::size_t>(pointDims[0]));
    for (std::size_t row = 0; row < pointDims[0]; ++row) {
        result->points.push_back({pointValues[row * 3], pointValues[row * 3 + 1],
                                  pointValues[row * 3 + 2]});
    }
    result->cells.resize(static_cast<std::size_t>(cellDims[0]));
    for (std::size_t row = 0; row < cellDims[0]; ++row) {
        auto& cell = result->cells[row];
        cell.reserve(verticesPerCell);
        for (std::size_t column = 0; column < verticesPerCell; ++column) {
            const auto vertex = cellValues[row * verticesPerCell + column];
            if (vertex < 0 || static_cast<std::size_t>(vertex) >= result->points.size()) {
                fail("FEM periodic mesh contains an out-of-range cell vertex.");
            }
            cell.push_back(vertex);
        }
    }
    if (hasLink(group.get(), "cell_region_id")) {
        result->cellRegionIds = readIndexVector(group.get(), "cell_region_id");
        requireLength(result->cellRegionIds.size(), result->cells.size(), "cell_region_id");
    } else {
        result->cellRegionIds.assign(result->cells.size(), 0);
    }

    auto samples = openGroup(group.get(), "samples");
    std::vector<hsize_t> sampleDims;
    const auto sampleValues = readDoubles(samples.get(), "points", &sampleDims);
    if (sampleDims.size() != 2 || sampleDims[1] != 3 || sampleDims[0] == 0) {
        fail("FEM periodic visualization sample points must have shape (N, 3).");
    }
    result->samplePoints.reserve(static_cast<std::size_t>(sampleDims[0]));
    for (std::size_t row = 0; row < sampleDims[0]; ++row) {
        result->samplePoints.push_back({sampleValues[row * 3], sampleValues[row * 3 + 1],
                                        sampleValues[row * 3 + 2]});
    }
    result->sampleOwnerCells = readIndexVector(samples.get(), "owner_cell");
    requireLength(result->sampleOwnerCells.size(), result->samplePoints.size(), "owner_cell");
    for (const auto owner : result->sampleOwnerCells) {
        if (owner < 0 || static_cast<std::size_t>(owner) >= result->cells.size()) {
            fail("Visualization sample owner_cell is outside the mesh.");
        }
    }

    if (dimension == 3 && !hasLink(group.get(), "edge_nodes")) {
        fail("A 3D FEM periodic mesh must provide canonical edge_nodes.");
    }
    if (dimension == 3) {
        std::vector<hsize_t> edgeDims;
        const auto edges = readIndices(group.get(), "edge_nodes", &edgeDims);
        if (edgeDims.size() != 2 || edgeDims[1] != 2) {
            fail("FEM periodic edge_nodes must have shape (N, 2).");
        }
        result->edgeNodes.reserve(static_cast<std::size_t>(edgeDims[0]));
        for (std::size_t row = 0; row < edgeDims[0]; ++row) {
            const auto first = edges[row * 2];
            const auto second = edges[row * 2 + 1];
            if (first < 0 || second < 0 || first >= second
                || static_cast<std::size_t>(second) >= result->points.size()) {
                fail("FEM periodic edge_nodes is not canonical or is out of range.");
            }
            result->edgeNodes.push_back({first, second});
        }
        const std::set<std::array<std::int64_t, 2>> uniqueEdges(
            result->edgeNodes.begin(), result->edgeNodes.end());
        if (uniqueEdges.size() != result->edgeNodes.size()) {
            fail("FEM periodic edge_nodes contains a duplicate canonical edge.");
        }
    }

    if (hasLink(group.get(), "boundary")) {
        auto boundary = openGroup(group.get(), "boundary");
        std::vector<hsize_t> facetDims;
        const auto facets = readIndices(boundary.get(), "facets", &facetDims);
        const auto verticesPerFacet = dimension == 2 ? 2U : 3U;
        if (facetDims.size() != 2 || facetDims[1] != verticesPerFacet) {
            fail("Boundary facets have an invalid shape.");
        }
        result->boundaryFacets.resize(static_cast<std::size_t>(facetDims[0]));
        for (std::size_t row = 0; row < facetDims[0]; ++row) {
            for (std::size_t column = 0; column < verticesPerFacet; ++column) {
                const auto vertex = facets[row * verticesPerFacet + column];
                if (vertex < 0 || static_cast<std::size_t>(vertex) >= result->points.size()) {
                    fail("Boundary facet contains an out-of-range vertex.");
                }
                result->boundaryFacets[row].push_back(vertex);
            }
        }
        result->boundaryTags = readIndexVector(boundary.get(), "tag");
        requireLength(result->boundaryTags.size(), result->boundaryFacets.size(), "boundary/tag");
    }

    if (hasLink(group.get(), "periodic")) {
        auto periodic = openGroup(group.get(), "periodic");
        std::vector<hsize_t> pairDims;
        const auto nodePairs = readIndices(periodic.get(), "node_pairs", &pairDims);
        if (pairDims.size() != 2 || pairDims[1] != 2) {
            fail("Periodic node_pairs must have shape (N, 2).");
        }
        result->periodicNodePairs.reserve(static_cast<std::size_t>(pairDims[0]));
        std::set<std::int64_t> periodicSlaves;
        std::set<std::int64_t> periodicMasters;
        for (std::size_t row = 0; row < pairDims[0]; ++row) {
            const auto slave = nodePairs[row * 2];
            const auto master = nodePairs[row * 2 + 1];
            if (slave < 0 || master < 0 || slave == master
                || static_cast<std::size_t>(slave) >= result->points.size()
                || static_cast<std::size_t>(master) >= result->points.size()) {
                fail("Periodic node pair is invalid.");
            }
            if (!periodicSlaves.insert(slave).second
                || !periodicMasters.insert(master).second) {
                fail("Periodic node_pairs must form a one-to-one map.");
            }
            result->periodicNodePairs.push_back({slave, master});
        }
        if (hasLink(periodic.get(), "affine")) {
            std::vector<hsize_t> affineDims;
            const auto affine = readDoubles(periodic.get(), "affine", &affineDims);
            if (affineDims != std::vector<hsize_t>{4, 4}) {
                fail("Periodic affine map must have shape (4, 4).");
            }
            std::copy(affine.begin(), affine.end(), result->periodicAffine.begin());
            result->hasPeriodicAffine = true;
        }
        if (dimension == 3 && hasLink(periodic.get(), "edge_pairs")) {
            std::vector<hsize_t> edgePairDims;
            const auto edgePairs = readIndices(periodic.get(), "edge_pairs", &edgePairDims);
            if (edgePairDims.size() != 2 || edgePairDims[1] != 2) {
                fail("Periodic edge_pairs must have shape (N, 2).");
            }
            result->periodicEdgePairs.reserve(static_cast<std::size_t>(edgePairDims[0]));
            for (std::size_t row = 0; row < edgePairDims[0]; ++row) {
                const auto slave = edgePairs[row * 2];
                const auto master = edgePairs[row * 2 + 1];
                if (slave < 0 || master < 0
                    || static_cast<std::size_t>(slave) >= result->edgeNodes.size()
                    || static_cast<std::size_t>(master) >= result->edgeNodes.size()) {
                    fail("Periodic edge pair is invalid.");
                }
                result->periodicEdgePairs.push_back({slave, master});
            }
            result->periodicEdgeSigns = readIndexVector(periodic.get(), "edge_sign");
            requireLength(result->periodicEdgeSigns.size(), result->periodicEdgePairs.size(),
                          "periodic/edge_sign");
            if (!std::all_of(result->periodicEdgeSigns.begin(), result->periodicEdgeSigns.end(),
                             [](auto sign) { return sign == -1 || sign == 1; })) {
                fail("Periodic edge orientation signs must be +/-1.");
            }
            std::map<std::int64_t, std::int64_t> nodeMap;
            for (const auto& pair : result->periodicNodePairs) {
                nodeMap.emplace(pair[0], pair[1]);
            }
            std::set<std::int64_t> slaveEdges;
            std::set<std::int64_t> masterEdges;
            for (std::size_t row = 0; row < result->periodicEdgePairs.size(); ++row) {
                const auto slaveEdge = result->periodicEdgePairs[row][0];
                const auto masterEdge = result->periodicEdgePairs[row][1];
                if (!slaveEdges.insert(slaveEdge).second
                    || !masterEdges.insert(masterEdge).second) {
                    fail("Periodic edge_pairs must form a one-to-one map.");
                }
                const auto& slaveEndpoints = result->edgeNodes[static_cast<std::size_t>(slaveEdge)];
                const auto first = nodeMap.find(slaveEndpoints[0]);
                const auto second = nodeMap.find(slaveEndpoints[1]);
                if (first == nodeMap.end() || second == nodeMap.end()) {
                    fail("A periodic slave edge endpoint is absent from periodic node_pairs.");
                }
                const std::array<std::int64_t, 2> expectedMaster{
                    std::min(first->second, second->second),
                    std::max(first->second, second->second)};
                const auto& actualMaster = result->edgeNodes[static_cast<std::size_t>(masterEdge)];
                const auto expectedSign = first->second < second->second ? 1 : -1;
                if (actualMaster != expectedMaster
                    || result->periodicEdgeSigns[row] != expectedSign) {
                    fail("Periodic edge pair/sign disagrees with the periodic node map.");
                }
            }
        }
    }

    if (dimension == 3 && (!hasLink(group.get(), "cell_edges")
                           || !hasLink(group.get(), "cell_edge_sign"))) {
        fail("A 3D FEM periodic mesh must provide cell_edges and cell_edge_sign.");
    }
    if (dimension == 3) {
        if (result->edgeNodes.empty()) {
            fail("A mesh with cell_edges must also provide edge_nodes.");
        }
        std::vector<hsize_t> cellEdgeDims;
        const auto cellEdges = readIndices(group.get(), "cell_edges", &cellEdgeDims);
        std::vector<hsize_t> signDims;
        const auto cellSigns = readIndices(group.get(), "cell_edge_sign", &signDims);
        if (cellEdgeDims != std::vector<hsize_t>{result->cells.size(), 6}
            || signDims != cellEdgeDims) {
            fail("cell_edges and cell_edge_sign must have shape (Ncells, 6).");
        }
        result->cellEdges.resize(result->cells.size());
        result->cellEdgeSigns.resize(result->cells.size());
        constexpr std::array<std::array<std::size_t, 2>, 6> localPairs{{
            {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}}};
        for (std::size_t row = 0; row < result->cells.size(); ++row) {
            for (std::size_t column = 0; column < 6; ++column) {
                const auto edge = cellEdges[row * 6 + column];
                const auto sign = cellSigns[row * 6 + column];
                if (edge < 0 || static_cast<std::size_t>(edge) >= result->edgeNodes.size()
                    || (sign != -1 && sign != 1)) {
                    fail("cell_edges contains an invalid edge or orientation sign.");
                }
                const auto& endpoints = result->edgeNodes[static_cast<std::size_t>(edge)];
                const auto firstNode = result->cells[row][localPairs[column][0]];
                const auto secondNode = result->cells[row][localPairs[column][1]];
                const std::array<std::int64_t, 2> expectedEndpoints{
                    std::min(firstNode, secondNode), std::max(firstNode, secondNode)};
                if (endpoints != expectedEndpoints) {
                    fail("cell_edges columns do not follow the declared local Nedelec edge order.");
                }
                const auto expectedSign = firstNode < secondNode ? 1 : -1;
                if (sign != expectedSign) {
                    fail("cell_edge_sign is inconsistent with canonical edge orientation.");
                }
                result->cellEdges[row][column] = edge;
                result->cellEdgeSigns[row][column] = sign;
            }
        }
    }
    return result;
}

MaterialStatePtr H5Reader::loadMaterialState(
    const FileIndex& index, std::size_t materialStateIndex) {
    const std::scoped_lock lock(hdf5Mutex);
    auto file = openFile(index.path);
    auto states = openGroup(file.get(), "/material_states");
    auto group = openGroup(states.get(), objectName(materialStateIndex));
    std::vector<hsize_t> epsilonDims;
    auto epsilon = readComplex(group.get(), "epsilon_r", &epsilonDims);
    std::vector<hsize_t> muDims;
    auto mu = readComplex(group.get(), "mu_r", &muDims);
    if (epsilonDims.size() != 2 || epsilonDims[1] != 3 || epsilonDims != muDims) {
        fail("FEM periodic epsilon_r/mu_r must share shape (Ncells, 3).");
    }
    auto pml = readDoubleVector(group.get(), "pml_fraction");
    requireLength(pml.size(), static_cast<std::size_t>(epsilonDims[0]), "pml_fraction");

    auto result = std::make_shared<MaterialState>();
    result->index = materialStateIndex;
    result->meshIndex = checkedIndex(readIntegerAttribute(group.get(), "mesh_index"), "mesh index");
    result->epsilonR.reserve(static_cast<std::size_t>(epsilonDims[0]));
    result->muR.reserve(static_cast<std::size_t>(muDims[0]));
    for (std::size_t row = 0; row < epsilonDims[0]; ++row) {
        result->epsilonR.push_back({epsilon[row * 3], epsilon[row * 3 + 1], epsilon[row * 3 + 2]});
        result->muR.push_back({mu[row * 3], mu[row * 3 + 1], mu[row * 3 + 2]});
        if (pml[row] < 0.0 || pml[row] > 1.0) {
            fail("Material-state pml_fraction must lie in [0, 1].");
        }
    }
    result->pmlFraction = std::move(pml);
    return result;
}

ModeFieldsPtr H5Reader::loadModeFields(
    const FileIndex& index, std::size_t caseIndex, std::size_t localModeIndex) {
    if (caseIndex >= index.cases.size() || localModeIndex >= index.cases[caseIndex].modeCount) {
        fail("Requested FEM periodic case/mode is outside the archive index.");
    }
    const std::scoped_lock lock(hdf5Mutex);
    auto file = openFile(index.path);
    auto cases = openGroup(file.get(), "/cases");
    auto caseGroup = openGroup(cases.get(), objectName(caseIndex));
    validateOptionalModeMetadata(caseGroup.get(), index.cases[caseIndex].modeCount);
    auto visualization = openGroup(caseGroup.get(), "visualization");
    std::vector<hsize_t> electricDims;
    const auto electric = readComplexMode(
        visualization.get(), "E", localModeIndex, &electricDims);
    std::vector<hsize_t> magneticDims;
    const auto magnetic = readComplexMode(
        visualization.get(), "H", localModeIndex, &magneticDims);
    if (electricDims.size() != 3 || electricDims[2] != 3 || electricDims != magneticDims
        || electricDims[0] != index.cases[caseIndex].modeCount) {
        fail("FEM periodic E/H visualization arrays must share shape (modes, samples, 3).");
    }
    const auto samples = static_cast<std::size_t>(electricDims[1]);
    auto result = std::make_shared<ModeFields>();
    result->caseIndex = caseIndex;
    result->localModeIndex = localModeIndex;
    result->electric.reserve(samples);
    result->magnetic.reserve(samples);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        result->electric.push_back({electric[sample * 3], electric[sample * 3 + 1],
                                    electric[sample * 3 + 2]});
        result->magnetic.push_back({magnetic[sample * 3], magnetic[sample * 3 + 1],
                                    magnetic[sample * 3 + 2]});
    }
    return result;
}

ModeCoefficients H5Reader::loadModeCoefficients(
    const FileIndex& index, std::size_t caseIndex, std::size_t localModeIndex) {
    if (caseIndex >= index.cases.size() || localModeIndex >= index.cases[caseIndex].modeCount) {
        fail("Requested FEM periodic case/mode is outside the archive index.");
    }
    const std::scoped_lock lock(hdf5Mutex);
    auto file = openFile(index.path);
    auto cases = openGroup(file.get(), "/cases");
    auto caseGroup = openGroup(cases.get(), objectName(caseIndex));
    auto coefficients = openGroup(caseGroup.get(), "coefficients");
    if (readIntegerAttribute(coefficients.get(), "full_expanded") != 1) {
        fail("FEM periodic coefficients must declare full_expanded=1.");
    }
    std::vector<hsize_t> dims;
    auto values = readComplexMode(coefficients.get(), "values", localModeIndex, &dims);
    if (dims.size() != 2 || dims[0] != index.cases[caseIndex].modeCount || dims[1] == 0) {
        fail("FEM periodic coefficients must have shape (modes, ndofs).");
    }
    auto space = readStringAttribute(coefficients.get(), "space");
    auto meshes = openGroup(file.get(), "/meshes");
    auto mesh = openGroup(meshes.get(), objectName(index.cases[caseIndex].meshIndex));
    const auto dimension = readIntegerAttribute(mesh.get(), "dimension");
    if ((dimension == 2 && space != "P1-scalar-nodal")
        || (dimension == 3 && space != "Nedelec-N1-canonical-edges")
        || (dimension != 2 && dimension != 3)) {
        fail(std::format("Coefficient space {} is incompatible with the {}D mesh.",
                         space, dimension));
    }
    std::size_t expectedDofs{};
    if (dimension == 2) {
        const auto pointDims = dimensions(openDataset(mesh.get(), "points").get());
        if (pointDims.size() != 2 || pointDims[1] != 3) {
            fail("Cannot validate P1 coefficient count against mesh points.");
        }
        expectedDofs = static_cast<std::size_t>(pointDims[0]);
    } else {
        const auto edgeDims = dimensions(openDataset(mesh.get(), "edge_nodes").get());
        if (edgeDims.size() != 2 || edgeDims[1] != 2) {
            fail("Cannot validate Nedelec coefficient count against canonical edges.");
        }
        expectedDofs = static_cast<std::size_t>(edgeDims[0]);
    }
    if (dims[1] != expectedDofs) {
        fail(std::format("Expanded coefficient count {} does not match mesh DOF count {}.",
                         dims[1], expectedDofs));
    }
    std::string primaryUnknown;
    if (hasLink(coefficients.get(), "primary_unknown")) {
        const auto unknowns = readStrings(coefficients.get(), "primary_unknown");
        requireLength(unknowns.size(), index.cases[caseIndex].modeCount, "primary_unknown");
        primaryUnknown = unknowns[localModeIndex];
    } else {
        if (!hasAttribute(coefficients.get(), "primary_unknown")) {
            fail("FEM periodic coefficients are missing primary_unknown metadata.");
        }
        primaryUnknown = readStringAttribute(coefficients.get(), "primary_unknown");
    }
    return {caseIndex, localModeIndex, std::move(space),
            std::move(primaryUnknown), std::move(values)};
}

} // namespace femperiodic
