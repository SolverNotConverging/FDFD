#include "main_window.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QTimer>

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("WaveFEM Viewer"));
    QApplication::setApplicationVersion(QStringLiteral(WAVEFEM_VIEWER_VERSION));
    QApplication::setOrganizationName(QStringLiteral("WaveFEM"));

    wavefem::MainWindow window;
    window.show();
    const auto arguments = application.arguments();
    const bool smokeTest = arguments.size() > 2
        && arguments.at(1) == QStringLiteral("--smoke-test");
    const int pathIndex = smokeTest ? 2 : 1;
    if (arguments.size() > pathIndex) {
        window.loadPath(QFileInfo(arguments.at(pathIndex)).absoluteFilePath());
    }
    if (smokeTest) {
        QTimer::singleShot(1800, &application, &QApplication::quit);
    }
    return application.exec();
}
