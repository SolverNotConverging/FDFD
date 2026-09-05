#include "main_window.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QTimer>
#include <cstdlib>

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("FEM Waveguide Scattering Viewer"));
    QApplication::setApplicationVersion(QStringLiteral(CEM_SCATTERING_VIEWER_VERSION));
    QApplication::setOrganizationName(QStringLiteral("FEM Waveguide Scattering"));

    fem_waveguide_scattering::MainWindow window;
    window.show();
    const auto arguments = application.arguments();
    const bool smokeTest = arguments.size() > 2
        && arguments.at(1) == QStringLiteral("--smoke-test");
    const int pathIndex = smokeTest ? 2 : 1;
    if (arguments.size() > pathIndex) {
        window.loadPath(QFileInfo(arguments.at(pathIndex)).absoluteFilePath());
    }
    if (smokeTest) {
        QTimer::singleShot(1800, &application, [&application, &window] {
            application.exit(window.hasLoadedResult() ? EXIT_SUCCESS : EXIT_FAILURE);
        });
    }
    return application.exec();
}
