#include "h5_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(double left, double right, double tolerance = 1.0e-12) {
    return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: fem-periodic-reader-test FILE\n";
        return EXIT_FAILURE;
    }
    try {
        const auto archive = femperiodic::H5Reader::loadIndex(std::filesystem::path(argv[1]));
        require(archive->schemaMajor == 1 && archive->schemaMinor == 0, "schema mismatch");
        require(archive->producer == "fem-periodic-fixture"
                    || archive->producer == "fem-periodic-fixture-vlen-\xC2\xB5",
                "producer UTF-8 mismatch");
        const auto sweep = archive->kind == "sweep";
        require(archive->kind == "single" || sweep, "kind mismatch");
        require(archive->cases.size() == (sweep ? 2U : 1U)
                    && archive->modes.size() == (sweep ? 4U : 2U),
                "index count mismatch");
        require(close(archive->cases[0].frequencyHz, 10.0e9), "frequency mismatch");
        if (sweep) {
            require(close(archive->cases[1].frequencyHz, 11.0e9)
                        && archive->cases[1].modeBegin == 2
                        && archive->cases[1].modeCount == 2,
                    "sweep offset mismatch");
        }
        require(archive->modes[0].polarization == "TE", "polarization mismatch");
        require(close(archive->modes[1].gammaPerM.real(), 0.2), "gamma mismatch");
        require(archive->modes[0].gaussResidual.has_value()
                    && close(*archive->modes[0].gaussResidual, 3.0e-8),
                "available Gauss residual mismatch");
        require(!archive->modes[1].gaussResidual.has_value(),
                "unavailable Gauss residual was not masked");

        const auto mesh = femperiodic::H5Reader::loadMesh(*archive, 0);
        require(mesh->dimension == 2 && mesh->topology == "triangle3", "mesh type mismatch");
        require(mesh->points.size() == 4 && mesh->cells.size() == 2, "mesh size mismatch");
        require(mesh->samplePoints.size() == 2, "sample size mismatch");
        require(mesh->boundaryFacets.size() == 4 && mesh->periodicNodePairs.size() == 2,
                "boundary/periodic topology mismatch");
        require(mesh->hasPeriodicAffine && close(mesh->periodicAffine[11], -2.0),
                "periodic affine mismatch");

        const auto material = femperiodic::H5Reader::loadMaterialState(*archive, 0);
        require(material->meshIndex == 0 && material->epsilonR.size() == 2,
                "material size mismatch");
        require(close(material->epsilonR[0][0].real(), 2.25), "epsilon mismatch");

        const auto selectedCase = sweep ? 1U : 0U;
        const auto fields = femperiodic::H5Reader::loadModeFields(*archive, selectedCase, 1);
        require(fields->electric.size() == 2 && fields->magnetic.size() == 2,
                "field hyperslab size mismatch");
        require(close(fields->electric[0][0].real(), 0.2), "selected field mode mismatch");

        const auto coefficients = femperiodic::H5Reader::loadModeCoefficients(
            *archive, selectedCase, 1);
        require(coefficients.values.size() == 4, "coefficient hyperslab size mismatch");
        require(coefficients.primaryUnknown == "Hy", "primary unknown mismatch");
        require(close(coefficients.values[3].real(), 1.0), "coefficient value mismatch");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "fem-periodic-reader-test: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
