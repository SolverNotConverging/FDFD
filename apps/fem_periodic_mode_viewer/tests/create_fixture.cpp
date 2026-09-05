#include <hdf5.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class Handle final {
public:
    using Closer = herr_t (*)(hid_t);
    Handle(hid_t id, Closer closer) : id_(id), closer_(closer) {
        if (id_ < 0) throw std::runtime_error("HDF5 object creation failed.");
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) {
        other.id_ = -1;
    }
    ~Handle() { if (id_ >= 0) closer_(id_); }
    [[nodiscard]] hid_t get() const { return id_; }
private:
    hid_t id_;
    Closer closer_;
};

struct ComplexPair {
    double r;
    double i;
};

template <typename T>
std::vector<T> duplicateIf(std::vector<T> values, bool duplicate) {
    if (duplicate) {
        const auto copy = values;
        values.insert(values.end(), copy.begin(), copy.end());
    }
    return values;
}

void check(herr_t status, const char* message) {
    if (status < 0) throw std::runtime_error(message);
}

Handle makeGroup(hid_t parent, const char* name) {
    return {H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose};
}

void writeStringAttribute(hid_t parent, const char* name, std::string_view value) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type.get(), value.size() + 1), "Could not size string type.");
    check(H5Tset_strpad(type.get(), H5T_STR_NULLTERM), "Could not set string padding.");
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle attribute(H5Acreate2(parent, name, type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    std::string storage(value);
    check(H5Awrite(attribute.get(), type.get(), storage.c_str()), "Could not write string attribute.");
}

void writeVlenStringAttribute(hid_t parent, const char* name, std::string_view value) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type.get(), H5T_VARIABLE), "Could not size variable string type.");
    check(H5Tset_cset(type.get(), H5T_CSET_UTF8), "Could not set UTF-8 string encoding.");
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle attribute(H5Acreate2(parent, name, type.get(), space.get(), H5P_DEFAULT,
                                H5P_DEFAULT), H5Aclose);
    const auto* storage = value.data();
    check(H5Awrite(attribute.get(), type.get(), &storage),
          "Could not write variable string attribute.");
}

void writeIntegerAttribute(hid_t parent, const char* name, std::int64_t value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle attribute(H5Acreate2(parent, name, H5T_STD_I64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(attribute.get(), H5T_NATIVE_LLONG, &value), "Could not write integer attribute.");
}

void writeIntegerArrayAttribute(hid_t parent, const char* name,
                                const std::vector<std::int64_t>& values) {
    const hsize_t size = values.size();
    Handle space(H5Screate_simple(1, &size, nullptr), H5Sclose);
    Handle attribute(H5Acreate2(parent, name, H5T_STD_I64LE, space.get(),
                                H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(attribute.get(), H5T_NATIVE_LLONG, values.data()),
          "Could not write integer array attribute.");
}

void writeDoubleAttribute(hid_t parent, const char* name, double value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle attribute(H5Acreate2(parent, name, H5T_IEEE_F64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(H5Awrite(attribute.get(), H5T_NATIVE_DOUBLE, &value), "Could not write real attribute.");
}

template <typename T>
void writeDataset(hid_t parent, const char* name, const std::vector<hsize_t>& dims,
                  hid_t fileType, hid_t memoryType, const std::vector<T>& values,
                  bool compressed = false) {
    Handle space(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
    Handle properties(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    if (compressed && !values.empty()) {
        auto chunks = dims;
        chunks[0] = 1;
        check(H5Pset_chunk(properties.get(), static_cast<int>(chunks.size()), chunks.data()),
              "Could not set HDF5 chunks.");
        check(H5Pset_shuffle(properties.get()), "Could not enable HDF5 shuffle.");
        check(H5Pset_deflate(properties.get(), 4), "Could not enable HDF5 gzip.");
        check(H5Pset_fletcher32(properties.get()), "Could not enable HDF5 checksum.");
    }
    Handle dataset(H5Dcreate2(parent, name, fileType, space.get(), H5P_DEFAULT,
                              properties.get(), H5P_DEFAULT), H5Dclose);
    if (!values.empty()) {
        check(H5Dwrite(dataset.get(), memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       values.data()), "Could not write HDF5 dataset.");
    }
}

void writeStrings(hid_t parent, const char* name, const std::vector<std::string>& values) {
    const auto width = std::max<std::size_t>(
        1, std::max_element(values.begin(), values.end(), [](const auto& a, const auto& b) {
            return a.size() < b.size();
        })->size() + 1);
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type.get(), width), "Could not size HDF5 text dataset.");
    check(H5Tset_strpad(type.get(), H5T_STR_NULLTERM), "Could not set text padding.");
    std::vector<char> raw(values.size() * width, '\0');
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::copy(values[index].begin(), values[index].end(), raw.begin() + index * width);
    }
    writeDataset(parent, name, {values.size()}, type.get(), type.get(), raw);
}

void writeVlenStrings(hid_t parent, const char* name,
                      const std::vector<std::string>& values) {
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    check(H5Tset_size(type.get(), H5T_VARIABLE), "Could not size variable text type.");
    check(H5Tset_cset(type.get(), H5T_CSET_UTF8), "Could not set UTF-8 text encoding.");
    std::vector<const char*> pointers;
    pointers.reserve(values.size());
    for (const auto& value : values) {
        pointers.push_back(value.c_str());
    }
    writeDataset(parent, name, {values.size()}, type.get(), type.get(), pointers);
}

Handle compoundComplexType() {
    Handle type(H5Tcreate(H5T_COMPOUND, sizeof(ComplexPair)), H5Tclose);
    check(H5Tinsert(type.get(), "r", HOFFSET(ComplexPair, r), H5T_IEEE_F64LE),
          "Could not create compound real member.");
    check(H5Tinsert(type.get(), "i", HOFFSET(ComplexPair, i), H5T_IEEE_F64LE),
          "Could not create compound imaginary member.");
    return type;
}

Handle compoundComplexMemoryType() {
    Handle type(H5Tcreate(H5T_COMPOUND, sizeof(ComplexPair)), H5Tclose);
    check(H5Tinsert(type.get(), "r", HOFFSET(ComplexPair, r), H5T_NATIVE_DOUBLE),
          "Could not create compound real memory member.");
    check(H5Tinsert(type.get(), "i", HOFFSET(ComplexPair, i), H5T_NATIVE_DOUBLE),
          "Could not create compound imaginary memory member.");
    return type;
}

void writeComplex(hid_t parent, const char* name, const std::vector<hsize_t>& dims,
                  const std::vector<ComplexPair>& values, bool native, bool compressed = false) {
#if H5_VERSION_GE(2, 0, 0)
    if (native) {
        writeDataset(parent, name, dims, H5T_COMPLEX_IEEE_F64LE, H5T_NATIVE_DOUBLE_COMPLEX,
                     values, compressed);
        return;
    }
#else
    if (native) throw std::runtime_error("Native complex needs HDF5 2.x.");
#endif
    auto fileType = compoundComplexType();
    auto memoryType = compoundComplexMemoryType();
    writeDataset(parent, name, dims, fileType.get(), memoryType.get(), values, compressed);
}

void writeFixture(const std::filesystem::path& path, bool native, bool badSchema,
                   bool badAttribute, bool badLinks, bool threeD, bool variableStrings,
                   bool sweep, bool badCoefficientSpace, bool badCellEdgeOrder,
                   bool badPeriodicEdgeSign, bool badVectorRank, bool badModeMetadata) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    const auto encoded = path.u8string();
    const std::string filename(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    const auto textAttribute = [variableStrings](hid_t parent, const char* name,
                                                  std::string_view value) {
        if (variableStrings) {
            writeVlenStringAttribute(parent, name, value);
        } else {
            writeStringAttribute(parent, name, value);
        }
    };
    const auto textDataset = [variableStrings](hid_t parent, const char* name,
                                                const std::vector<std::string>& values) {
        if (variableStrings) {
            writeVlenStrings(parent, name, values);
        } else {
            writeStrings(parent, name, values);
        }
    };
    textAttribute(file.get(), "format", "cem-fem-results");
    textAttribute(file.get(), "schema", "1.0");
    textAttribute(file.get(), "solver_family", "periodic_modes");
    textAttribute(file.get(), "units", "SI");
    textAttribute(file.get(), "result_kind", "modes");
    if (badAttribute) {
        writeIntegerArrayAttribute(file.get(), "schema_major", {1, 1});
    } else {
        writeIntegerAttribute(file.get(), "schema_major", badSchema ? 99 : 1);
    }
    writeIntegerAttribute(file.get(), "schema_minor", 0);
    textAttribute(file.get(), "kind", sweep ? "sweep" : "single");
    writeIntegerAttribute(file.get(), "case_count", sweep ? 2 : 1);
    textAttribute(file.get(), "time_convention", "exp(+i*omega*t)");
    textAttribute(file.get(), "field_representation", "periodic-envelope");
    textAttribute(file.get(), "length_unit", "m");
    textAttribute(file.get(), "frequency_unit", "Hz");
    textAttribute(file.get(), "producer", variableStrings
        ? "fem-periodic-fixture-vlen-\xC2\xB5" : "fem-periodic-fixture");
    textAttribute(file.get(), "producer_version", "1.0");
    textAttribute(file.get(), "complex_storage", native ? "hdf5-native" : "compound-r-i");

    auto index = makeGroup(file.get(), "index");
    const auto frequencies = sweep ? std::vector<double>{10.0e9, 11.0e9}
                                   : std::vector<double>{10.0e9};
    const auto offsets = sweep ? std::vector<std::int64_t>{0, 2, 4}
                               : std::vector<std::int64_t>{0, 2};
    const auto objectIndices = sweep ? std::vector<std::int64_t>{0, 0}
                                     : std::vector<std::int64_t>{0};
    writeDataset(index.get(), "frequency_hz", {frequencies.size()}, H5T_IEEE_F64LE,
                 H5T_NATIVE_DOUBLE, frequencies);
    writeDataset(index.get(), "mode_offsets", {offsets.size()}, H5T_STD_I64LE,
                 H5T_NATIVE_LLONG, offsets);
    writeDataset(index.get(), "mesh_index", {objectIndices.size()}, H5T_STD_I64LE,
                 H5T_NATIVE_LLONG, objectIndices);
    writeDataset(index.get(), "material_state_index", {objectIndices.size()}, H5T_STD_I64LE,
                 H5T_NATIVE_LLONG, objectIndices);
    const auto gamma = duplicateIf(std::vector<ComplexPair>{{0.1, 20.0}, {0.2, -20.0}}, sweep);
    const auto neff = duplicateIf(std::vector<ComplexPair>{{1.0, -0.005}, {-1.0, -0.01}}, sweep);
    const auto folded = duplicateIf(std::vector<ComplexPair>{{1.0, -0.005}, {-1.0, -0.01}}, sweep);
    const auto bloch = duplicateIf(std::vector<ComplexPair>{{0.39, -0.71}, {0.38, 0.70}}, sweep);
    const auto modalCount = gamma.size();
    writeComplex(index.get(), "gamma_per_m", {modalCount}, gamma, native);
    writeComplex(index.get(), "neff", {modalCount}, neff, native);
    writeComplex(index.get(), "neff_folded", {modalCount}, folded, native);
    writeComplex(index.get(), "bloch_multiplier", {modalCount}, bloch, native);
    writeDataset(index.get(), "alpha_per_m", {modalCount}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                 duplicateIf(std::vector<double>{0.1, 0.2}, sweep));
    writeDataset(index.get(), "beta_per_m", {modalCount}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                 duplicateIf(std::vector<double>{20.0, -20.0}, sweep));
    writeDataset(index.get(), "beta_folded_per_m", {modalCount}, H5T_IEEE_F64LE,
                 H5T_NATIVE_DOUBLE, duplicateIf(std::vector<double>{20.0, -20.0}, sweep));
    writeDataset(index.get(), "residual", {modalCount}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                 duplicateIf(std::vector<double>{1.0e-10, 2.0e-10}, sweep));
    writeDataset(index.get(), "gauss_residual", {modalCount}, H5T_IEEE_F64LE,
                 H5T_NATIVE_DOUBLE, duplicateIf(std::vector<double>{3.0e-8, 0.0}, sweep));
    writeDataset(index.get(), "gauss_available", {modalCount}, H5T_STD_U8LE,
                 H5T_NATIVE_UCHAR,
                 duplicateIf(std::vector<unsigned char>{1, 0}, sweep));
    writeDataset(index.get(), "pml_fraction", {modalCount}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                 duplicateIf(std::vector<double>{0.02, 0.03}, sweep));
    textDataset(index.get(), "polarization",
                duplicateIf(std::vector<std::string>{"TE", "TM"}, sweep));
    textDataset(index.get(), "direction",
                duplicateIf(std::vector<std::string>{"forward", "backward"}, sweep));
    textDataset(index.get(), "normalization",
                duplicateIf(std::vector<std::string>{"unit-power", "unit-power"}, sweep));

    auto meshes = makeGroup(file.get(), "meshes");
    auto mesh = makeGroup(meshes.get(), "000000");
    writeIntegerAttribute(mesh.get(), "dimension", threeD ? 3 : 2);
    textAttribute(mesh.get(), "topology", threeD ? "tetra4" : "triangle3");
    textAttribute(mesh.get(), "periodic_axis", "z");
    writeDoubleAttribute(mesh.get(), "period_m", 2.0);
    writeDoubleAttribute(mesh.get(), "reference_z_m", 0.0);
    if (threeD) {
        writeDataset(mesh.get(), "points", {4, 3}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                     std::vector<double>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2}, true);
        writeDataset(mesh.get(), "cells", {1, 4}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1, 2, 3}, true);
        writeDataset(mesh.get(), "cell_region_id", {1}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{1});
        writeDataset(mesh.get(), "edge_nodes", {6, 2}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3});
    } else {
        writeDataset(mesh.get(), "points", {4, 3}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                     std::vector<double>{0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 2}, true);
        writeDataset(mesh.get(), "cells", {2, 3}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1, 2, 0, 2, 3}, true);
        writeDataset(mesh.get(), "cell_region_id", {2}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{1, 2});
    }
    auto samples = makeGroup(mesh.get(), "samples");
    if (threeD) {
        writeDataset(samples.get(), "points", {1, 3}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                     std::vector<double>{0.25, 0.25, 0.5});
        writeDataset(samples.get(), "owner_cell", {1}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0});
    } else {
        writeDataset(samples.get(), "points", {2, 3}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                     std::vector<double>{2.0 / 3.0, 0, 2.0 / 3.0,
                                         1.0 / 3.0, 0, 4.0 / 3.0});
        writeDataset(samples.get(), "owner_cell",
                     badVectorRank ? std::vector<hsize_t>{1, 2}
                                   : std::vector<hsize_t>{2},
                     H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1});
    }
    auto boundary = makeGroup(mesh.get(), "boundary");
    if (threeD) {
        writeDataset(boundary.get(), "facets", {4, 3}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3});
        writeDataset(boundary.get(), "tag", {4}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{1, 2, 3, 4});
        writeDataset(mesh.get(), "cell_edges", {1, 6}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     badCellEdgeOrder
                         ? std::vector<std::int64_t>{3, 0, 1, 2, 4, 5}
                         : std::vector<std::int64_t>{0, 3, 1, 2, 4, 5});
        writeDataset(mesh.get(), "cell_edge_sign", {1, 6}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{1, 1, 1, 1, 1, 1});
    } else {
        writeDataset(boundary.get(), "facets", {4, 2}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{0, 1, 1, 2, 2, 3, 3, 0});
        writeDataset(boundary.get(), "tag", {4}, H5T_STD_I64LE, H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{1, 2, 3, 4});
    }
    auto periodic = makeGroup(mesh.get(), "periodic");
    writeDataset(periodic.get(), "node_pairs", {2, 2},
                 H5T_STD_I64LE, H5T_NATIVE_LLONG,
                 threeD ? std::vector<std::int64_t>{2, 1, 3, 0}
                        : std::vector<std::int64_t>{3, 0, 2, 1});
    writeDataset(periodic.get(), "affine", {4, 4}, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                 std::vector<double>{1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, -2,
                                     0, 0, 0, 1});
    if (threeD) {
        writeDataset(periodic.get(), "edge_pairs", {1, 2}, H5T_STD_I64LE,
                     H5T_NATIVE_LLONG, std::vector<std::int64_t>{5, 0});
        writeDataset(periodic.get(), "edge_sign", {1}, H5T_STD_I64LE,
                     H5T_NATIVE_LLONG,
                     std::vector<std::int64_t>{badPeriodicEdgeSign ? 1 : -1});
    }

    auto states = makeGroup(file.get(), "material_states");
    auto state = makeGroup(states.get(), "000000");
    writeIntegerAttribute(state.get(), "mesh_index", badLinks ? 1 : 0);
    const auto materialCells = threeD ? 1U : 2U;
    std::vector<ComplexPair> epsilon(materialCells * 3, {2.25, -0.01});
    std::vector<ComplexPair> mu(materialCells * 3, {1.0, 0.0});
    writeComplex(state.get(), "epsilon_r", {materialCells, 3}, epsilon, native, true);
    writeComplex(state.get(), "mu_r", {materialCells, 3}, mu, native, true);
    writeDataset(state.get(), "pml_fraction", {materialCells}, H5T_IEEE_F64LE,
                 H5T_NATIVE_DOUBLE, std::vector<double>(materialCells, threeD ? 0.0 : 0.25));

    auto cases = makeGroup(file.get(), "cases");
    const auto writeCase = [&](const char* name, double frequency) {
        auto group = makeGroup(cases.get(), name);
        writeDoubleAttribute(group.get(), "frequency_hz", frequency);
        writeDoubleAttribute(group.get(), "omega", 2.0 * 3.141592653589793 * frequency);
        writeDoubleAttribute(group.get(), "k0", 2.0 * 3.141592653589793 * frequency / 299792458.0);
        writeIntegerAttribute(group.get(), "mesh_index", 0);
        writeIntegerAttribute(group.get(), "material_state_index", 0);
        writeIntegerAttribute(group.get(), "mode_count", 2);
        textAttribute(group.get(), "backend", "cython-refined-arnoldi");
        auto coefficients = makeGroup(group.get(), "coefficients");
        writeIntegerAttribute(coefficients.get(), "full_expanded", 1);
        textAttribute(coefficients.get(), "space",
                      threeD && !badCoefficientSpace
                          ? "Nedelec-N1-canonical-edges" : "P1-scalar-nodal");
        textDataset(coefficients.get(), "primary_unknown", threeD
            ? std::vector<std::string>{"E", "E"}
            : std::vector<std::string>{"Ey", "Hy"});
        if (threeD && !badCoefficientSpace) {
            writeComplex(coefficients.get(), "values", {2, 6},
                         {{1, 0}, {0.5, 0.1}, {0.2, -0.1}, {0.1, 0}, {0.3, 0}, {0.4, 0},
                          {0, 0}, {0.2, 0.2}, {0.4, 0.1}, {1, 0}, {0.3, 0}, {0.1, 0}},
                         native, true);
        } else {
            writeComplex(coefficients.get(), "values", {2, 4},
                         {{1, 0}, {0.5, 0.1}, {0.2, -0.1}, {0, 0},
                          {0, 0}, {0.2, 0.2}, {0.4, 0.1}, {1, 0}}, native, true);
        }
        auto metadata = makeGroup(group.get(), "mode_metadata");
        writeDataset(metadata.get(), "has_power",
                     badModeMetadata ? std::vector<hsize_t>{1, 2}
                                     : std::vector<hsize_t>{2},
                     H5T_STD_U8LE, H5T_NATIVE_UCHAR,
                     std::vector<unsigned char>{1, 0});
        writeComplex(metadata.get(), "power", {2}, {{1.0, 0.0}, {0.0, 0.0}}, native);
        auto visualization = makeGroup(group.get(), "visualization");
        if (threeD) {
            writeComplex(visualization.get(), "E", {2, 1, 3},
                         {{0.2, 0.1}, {1, 0}, {0.4, -0.1},
                          {0.6, 0}, {0.2, 0.1}, {1.2, 0}}, native, true);
            writeComplex(visualization.get(), "H", {2, 1, 3},
                         {{0.1, 0}, {0.3, 0}, {0.7, 0.1},
                          {0.2, 0}, {0.5, 0}, {0.8, -0.1}}, native, true);
        } else {
            writeComplex(visualization.get(), "E", {2, 2, 3},
                         {{0, 0}, {1, 0}, {0, 0}, {0, 0}, {0.5, 0.1}, {0, 0},
                          {0.2, 0}, {0, 0}, {0.1, 0}, {0.4, 0}, {0, 0}, {0.3, 0}},
                         native, true);
            writeComplex(visualization.get(), "H", {2, 2, 3},
                         {{0.2, 0}, {0, 0}, {0.1, 0}, {0.3, 0}, {0, 0}, {0.2, 0},
                          {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0.6, 0.1}, {0, 0}},
                         native, true);
        }
    };
    writeCase("000000", 10.0e9);
    if (sweep) {
        writeCase("000001", 11.0e9);
    }
    check(H5Fflush(file.get(), H5F_SCOPE_GLOBAL), "Could not flush fixture.");
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: fem-periodic-fixture FILE "
                     "compound|native|vlen|sweep|bad-schema|bad-attribute|bad-links|"
                     "bad-coefficient-space|bad-cell-edge-order|bad-periodic-edge-sign|"
                     "bad-vector-rank|bad-mode-metadata|tetra\n";
        return EXIT_FAILURE;
    }
    try {
        const std::string_view mode(argv[2]);
        if (mode != "compound" && mode != "native" && mode != "bad-schema"
            && mode != "bad-attribute" && mode != "tetra" && mode != "vlen"
            && mode != "sweep" && mode != "bad-links"
            && mode != "bad-coefficient-space" && mode != "bad-cell-edge-order"
            && mode != "bad-periodic-edge-sign" && mode != "bad-vector-rank"
            && mode != "bad-mode-metadata") {
            throw std::runtime_error("Unknown fixture encoding.");
        }
        writeFixture(std::filesystem::path(argv[1]), mode == "native", mode == "bad-schema",
                     mode == "bad-attribute", mode == "bad-links",
                     mode == "tetra" || mode == "bad-coefficient-space"
                         || mode == "bad-cell-edge-order" || mode == "bad-periodic-edge-sign",
                      mode == "vlen", mode == "sweep", mode == "bad-coefficient-space",
                      mode == "bad-cell-edge-order", mode == "bad-periodic-edge-sign",
                      mode == "bad-vector-rank", mode == "bad-mode-metadata");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "fem-periodic-fixture: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
