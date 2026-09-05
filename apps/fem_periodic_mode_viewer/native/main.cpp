#include "main_window.hpp"
#include "path_qt.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QThreadPool>
#include <QTimer>
#include <QDebug>

#ifdef FEM_PERIODIC_MODE_VIEWER_WITH_VTK
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>
#endif

#include <filesystem>
#include <cstdlib>
#include <system_error>

int main(int argc, char* argv[]) {
#ifdef FEM_PERIODIC_MODE_VIEWER_WITH_VTK
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
#endif
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("FEM Periodic Mode Viewer"));
    QApplication::setApplicationVersion(QStringLiteral(FEM_PERIODIC_MODE_VIEWER_VERSION));
    const auto arguments = application.arguments();
    bool smoke = false;
    bool smokeSlice = false;
    bool removeSourceOnExit = false;
    bool testExitImmediately = false;
    QString pathArgument;
    for (qsizetype index = 1; index < arguments.size(); ++index) {
        const auto& argument = arguments.at(index);
        if (argument == QStringLiteral("--smoke-test")) {
            smoke = true;
        } else if (argument == QStringLiteral("--smoke-test-slice")) {
            smoke = true;
            smokeSlice = true;
        } else if (argument == QStringLiteral("--remove-source-on-exit")) {
            removeSourceOnExit = true;
        } else if (argument == QStringLiteral("--test-exit-immediately")) {
            testExitImmediately = true;
        } else if (argument.startsWith(QStringLiteral("--"))) {
            qCritical().noquote() << QStringLiteral("Unknown option: %1").arg(argument);
            return EXIT_FAILURE;
        } else if (pathArgument.isEmpty()) {
            pathArgument = argument;
        } else {
            qCritical() << "Only one HDF5 file or directory may be opened at a time.";
            return EXIT_FAILURE;
        }
    }
    if (smoke && pathArgument.isEmpty()) {
        qCritical() << "Smoke-test mode requires an HDF5 path.";
        return EXIT_FAILURE;
    }
    if (removeSourceOnExit &&
        (pathArgument.isEmpty() || QFileInfo(pathArgument).isDir())) {
        qCritical() << "Source cleanup requires a single HDF5 file.";
        return EXIT_FAILURE;
    }

    const auto sourcePath = pathArgument.isEmpty()
        ? std::filesystem::path{}
        : femperiodic::pathFromQString(pathArgument);
    int exitCode = EXIT_FAILURE;
    {
        femperiodic::MainWindow window;
        window.show();
        if (smoke) {
            window.setLoadCompletionHandler(
                [&application, &window, smokeSlice](bool success, const QString& error) {
                if (success && smokeSlice && !window.verifySliceRenderingForTest()) {
                    success = false;
                    qCritical() << "Slice smoke test did not produce a heat-map-only cut "
                                   "with visible axes and scalar bar.";
                }
                if (!success) {
                    qCritical().noquote()
                        << QStringLiteral("Smoke-test load failed: %1").arg(error);
                }
                QTimer::singleShot(250, &application, [&application, success] {
                    application.exit(success ? EXIT_SUCCESS : EXIT_FAILURE);
                });
            });
            QTimer::singleShot(10'000, &application, [&application] {
                qCritical() << "Smoke-test load timed out.";
                application.exit(EXIT_FAILURE);
            });
        }
        if (!sourcePath.empty()) {
            window.loadPath(sourcePath);
        }
        if (testExitImmediately) {
            QTimer::singleShot(0, &application, [&application] {
                application.exit(EXIT_SUCCESS);
            });
        }
        exitCode = application.exec();
    }
    if (removeSourceOnExit) {
        // QtConcurrent readers use the global pool.  A QFutureWatcher being
        // destroyed disconnects it but does not wait for its worker, so drain
        // the pool before unlinking a file that a closing window may still be
        // reading on Windows.
        QThreadPool::globalInstance()->waitForDone();
        std::error_code error;
        const auto removed = std::filesystem::remove(sourcePath, error);
        if (!removed || error) {
            qCritical().noquote()
                << QStringLiteral("Could not remove temporary source after viewer exit: %1")
                       .arg(femperiodic::qStringFromPath(sourcePath));
            if (exitCode == EXIT_SUCCESS) {
                exitCode = EXIT_FAILURE;
            }
        }
    }
    return exitCode;
}
