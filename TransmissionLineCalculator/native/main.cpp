#include "main_window.hpp"

#include <QApplication>
#include <QTimer>

int main(int argc, char* argv[]) {
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("Transmission Line Calculator"));
    QApplication::setOrganizationName(QStringLiteral("FDFD"));

    tl::MainWindow window;
    window.show();

    const auto arguments = application.arguments();
    if (arguments.contains(QStringLiteral("--calculate-smoke-test"))) {
        QTimer::singleShot(0, &window, [&window] { window.calculateForSmokeTest(); });
        auto* completionTimer = new QTimer(&application);
        QObject::connect(completionTimer, &QTimer::timeout, &application,
                         [&application, &window, completionTimer] {
            if (!window.solveInProgress()) {
                completionTimer->stop();
                application.exit(window.hasResult() ? 0 : 1);
            }
        });
        completionTimer->start(50);
        QTimer::singleShot(120000, &application, [&application, &window] {
            if (window.solveInProgress()) {
                application.exit(2);
            }
        });
    } else if (arguments.contains(QStringLiteral("--smoke-test"))) {
        const int status = window.defaultsMatchForSmokeTest() ? 0 : 1;
        QTimer::singleShot(1500, &application,
                           [&application, status] { application.exit(status); });
    }

    return application.exec();
}
