// Compare the four native FEM templates with adaptive refinement disabled.
#include "solver.hpp"
#include "field_plot.hpp"
#include <QApplication>
#include <QHBoxLayout>
#include <QPixmap>
#include <QTabWidget>
#include <QTimer>
#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTabWidget window;
    window.setWindowTitle(QStringLiteral("Fixed-mesh transmission line comparison"));
    const std::array types{tl::LineType::Coaxial, tl::LineType::Microstrip,
                          tl::LineType::Stripline, tl::LineType::CoplanarWaveguide};
    const std::array names{"coaxial", "microstrip", "stripline", "coplanar waveguide"};
    for (std::size_t i = 0; i < types.size(); ++i) {
        auto parameters = tl::defaultParameters(types[i]);
        parameters.maxRefinements = 0;
        const auto result = std::make_shared<const tl::Result>(tl::solve(parameters));
        if (result->adaptiveHistory.size() != 1) {
            throw std::runtime_error("example must perform exactly one mesh solve");
        }
        std::cout << names[i] << ": neff=" << result->neff
                  << ", Zc=" << result->characteristicImpedance << " ohm\n";
        auto* tab = new QWidget(&window);
        auto* layout = new QHBoxLayout(tab);
        for (auto family : {tl::FieldFamily::Electric, tl::FieldFamily::Magnetic}) {
            auto* plot = new tl::FieldPlot(family, tab);
            plot->setResult(result);
            plot->setMeshVisible(true);
            layout->addWidget(plot);
        }
        window.addTab(tab, QString::fromLatin1(names[i]));
    }
    window.resize(1200, 720);
    window.show();
    if (app.arguments().contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(0, &window, [&] {
            for (int i = 0; i < window.count(); ++i) {
                window.setCurrentIndex(i);
                if (window.grab().isNull()) {
                    app.exit(1);
                    return;
                }
            }
            app.quit();
        });
    }
    return app.exec();
}
