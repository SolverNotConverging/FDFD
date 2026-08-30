#include "tui.hpp"

#include "solver.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace tl::tui {
namespace {

using Clock = std::chrono::steady_clock;
using namespace ftxui;

enum class InputKey {
    Frequency,
    InnerRadius,
    OuterRadius,
    OuterConductorThickness,
    TraceWidth,
    SubstrateHeight,
    ConductorThickness,
    GroundSpacing,
    CenterWidth,
    Gap,
    GroundWidth,
    EpsilonR,
    LossTangent,
    DomainPaddingFactor,
    MetalConductivity,
    MaxElementSize,
};

struct FieldDefinition {
    InputKey key;
    std::string label;
    std::string defaultText;
    double siScale;
    bool strictlyPositive{true};
    bool optional{};
};

struct FieldState {
    FieldDefinition definition;
    std::string value;
    std::string error;
    Component input;
    Component row;
};

struct GeometryForm {
    LineType type{};
    std::vector<std::shared_ptr<FieldState>> fields;
    Component component;
};

[[nodiscard]] std::string_view lineTitle(const LineType type) {
    switch (type) {
        case LineType::Coaxial:
            return "Coaxial";
        case LineType::Microstrip:
            return "Microstrip";
        case LineType::Stripline:
            return "Stripline";
        case LineType::CoplanarWaveguide:
            return "CPW";
    }
    return "Unknown";
}

[[nodiscard]] std::vector<FieldDefinition> definitionsFor(const LineType type) {
    const FieldDefinition frequency{InputKey::Frequency, "Frequency (GHz)",
                                    "10", 1.0e9};
    const FieldDefinition epsilon{InputKey::EpsilonR, "Relative permittivity",
                                  "3.55", 1.0};
    const FieldDefinition loss{InputKey::LossTangent, "Loss tangent", "0.0027",
                               1.0, false};
    const FieldDefinition conductivity{InputKey::MetalConductivity,
                                       "Metal sigma (MS/m; blank=PEC)", "",
                                       1.0e6, true, true};
    const FieldDefinition mesh{InputKey::MaxElementSize, "Mesh size (mm)",
                               "1.00", 1.0e-3};

    switch (type) {
        case LineType::Coaxial:
            return {
                frequency,
                {InputKey::InnerRadius, "Inner radius (mm)", "0.5", 1.0e-3},
                {InputKey::OuterRadius, "Outer radius (mm)", "1.67", 1.0e-3},
                {InputKey::OuterConductorThickness, "Outer metal (um)", "150",
                 1.0e-6},
                {InputKey::EpsilonR, "Relative permittivity", "2.1", 1.0},
                {InputKey::LossTangent, "Loss tangent", "0.0002", 1.0, false},
                conductivity,
                mesh,
            };
        case LineType::Microstrip:
            return {
                frequency,
                {InputKey::TraceWidth, "Trace width (mm)", "3", 1.0e-3},
                {InputKey::SubstrateHeight, "Substrate h (mm)", "1.524",
                 1.0e-3},
                {InputKey::ConductorThickness, "Metal thick. (um)", "35",
                 1.0e-6},
                {InputKey::DomainPaddingFactor, "Domain padding factor", "3", 1.0},
                epsilon,
                loss,
                conductivity,
                mesh,
            };
        case LineType::Stripline:
            return {
                frequency,
                {InputKey::TraceWidth, "Trace width (mm)", "0.8", 1.0e-3},
                {InputKey::GroundSpacing, "Ground gap (mm)", "1.524", 1.0e-3},
                {InputKey::ConductorThickness, "Metal thick. (um)", "35",
                 1.0e-6},
                {InputKey::DomainPaddingFactor, "Domain padding factor", "3", 1.0},
                epsilon,
                loss,
                conductivity,
                mesh,
            };
        case LineType::CoplanarWaveguide:
            return {
                frequency,
                {InputKey::CenterWidth, "Signal width (mm)", "0.6", 1.0e-3},
                {InputKey::Gap, "Slot gap (mm)", "0.25", 1.0e-3},
                {InputKey::GroundWidth, "Ground width (mm)", "1.5", 1.0e-3},
                {InputKey::SubstrateHeight, "Substrate h (mm)", "0.8", 1.0e-3},
                {InputKey::ConductorThickness, "Metal thick. (um)", "35",
                 1.0e-6},
                {InputKey::DomainPaddingFactor, "Domain padding factor", "3", 1.0},
                epsilon,
                loss,
                conductivity,
                mesh,
            };
    }
    return {};
}

[[nodiscard]] std::string formatScientific(const double value,
                                           const std::string_view unit = {}) {
    std::ostringstream output;
    output << std::scientific << std::setprecision(9) << value;
    if (!unit.empty()) {
        output << ' ' << unit;
    }
    return output.str();
}

[[nodiscard]] std::string formatComplexValue(
    const std::complex<double> value, const std::string_view unit = {}) {
    std::ostringstream output;
    output << std::scientific << std::setprecision(9) << value.real()
           << (value.imag() < 0.0 ? " - j" : " + j")
           << std::abs(value.imag());
    if (!unit.empty()) {
        output << ' ' << unit;
    }
    return output.str();
}

[[nodiscard]] std::string compactNumber(const double value) {
    std::ostringstream output;
    output << std::setprecision(10) << value;
    return output.str();
}

[[nodiscard]] Element valueRow(const std::string_view label,
                               const std::string& value) {
    return hbox({text(std::string(label)) | size(WIDTH, EQUAL, 13), text(value)});
}

[[nodiscard]] bool nearlyEqual(const double left, const double right) {
    return std::abs(left - right) <=
           1.0e-12 * std::max({1.0, std::abs(left), std::abs(right)});
}

[[nodiscard]] bool sameParameters(const Parameters& left,
                                  const Parameters& right) {
    const bool conductivityMatches =
        left.metalConductivity.has_value() ==
            right.metalConductivity.has_value() &&
        (!left.metalConductivity.has_value() ||
         nearlyEqual(*left.metalConductivity, *right.metalConductivity));
    return left.type == right.type && conductivityMatches &&
           nearlyEqual(left.frequencyHz, right.frequencyHz) &&
           nearlyEqual(left.maxElementSize, right.maxElementSize) &&
           nearlyEqual(left.refinementFactor, right.refinementFactor) &&
           nearlyEqual(left.innerRadius, right.innerRadius) &&
           nearlyEqual(left.outerRadius, right.outerRadius) &&
           nearlyEqual(left.outerConductorThickness,
                       right.outerConductorThickness) &&
           nearlyEqual(left.traceWidth, right.traceWidth) &&
           nearlyEqual(left.substrateHeight, right.substrateHeight) &&
           nearlyEqual(left.conductorThickness, right.conductorThickness) &&
           nearlyEqual(left.groundSpacing, right.groundSpacing) &&
           nearlyEqual(left.centerWidth, right.centerWidth) &&
           nearlyEqual(left.gap, right.gap) &&
           nearlyEqual(left.groundWidth, right.groundWidth) &&
           nearlyEqual(left.epsilonR, right.epsilonR) &&
           nearlyEqual(left.lossTangent, right.lossTangent) &&
           nearlyEqual(left.domainPaddingFactor, right.domainPaddingFactor);
}

class TuiApplication final {
public:
    TuiApplication() : screen_(ScreenInteractive::Fullscreen()) {
        buildComponents();
    }

    ~TuiApplication() {
        stopRequested_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    int loop() {
        screen_.TrackMouse(true);
        screen_.Loop(root_);
        stopRequested_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
        return EXIT_SUCCESS;
    }

    [[nodiscard]] bool renderSmokeFrame() {
        const std::vector<std::string> expectedLineNames{
            "Microstrip", "CPW", "Stripline", "Coaxial"
        };
        if (lineIndex_ != 0 || lineNames_ != expectedLineNames
            || forms_.front().type != LineType::Microstrip) {
            return false;
        }
        for (auto& form : forms_) {
            Component firstInvalid;
            const auto parsed = readParameters(form, firstInvalid);
            if (!parsed.has_value() || firstInvalid ||
                !sameParameters(*parsed, defaultParameters(form.type))) {
                return false;
            }
        }

        const auto renderAt = [this](const int width, const int height) {
            Screen screen = Screen::Create(Dimension::Fixed(width),
                                           Dimension::Fixed(height));
            Render(screen, root_->Render());
            return screen.ToString();
        };
        const std::string wide = renderAt(120, 40);
        const std::string compact = renderAt(80, 24);
        if (wide.find("Transmission-Line Calculator") == std::string::npos ||
            wide.find("Calculate FEM") == std::string::npos ||
            wide.find("Coaxial") == std::string::npos ||
            compact.find("Workspace") == std::string::npos ||
            compact.find("Setup") == std::string::npos ||
            compact.find("Calculate FEM") == std::string::npos) {
            return false;
        }

        auto sampleResult = std::make_shared<Result>();
        sampleResult->parameters = defaultParameters(LineType::Coaxial);
        sampleResult->parameters.metalConductivity = 5.8e7;
        result_ = std::move(sampleResult);
        resultName_ = "Coaxial";
        viewIndex_ = 1;

        resultTabIndex_ = 0;
        const std::string overview = renderAt(80, 24);
        resultTabIndex_ = 1;
        const std::string rlgc = renderAt(80, 24);
        resultTabIndex_ = 2;
        resultBenchmark_ = true;
        resultCompletedRuns_ = 5;
        resultRequestedRuns_ = 5;
        const std::string performance = renderAt(80, 24);
        return overview.find("Current") != std::string::npos &&
               rlgc.find("Geometry") != std::string::npos &&
               performance.find("Runs") != std::string::npos;
    }

private:
    void buildComponents() {
        lineNames_ = {"Microstrip", "CPW", "Stripline", "Coaxial"};
        MenuOption lineOptions = MenuOption::HorizontalAnimated();
        lineOptions.on_change = [this] {
            viewIndex_ = 0;
            setupChanged("Geometry changed; edit its inputs or press F5.");
        };
        lineMenu_ = Menu(&lineNames_, &lineIndex_, lineOptions);
        lineMenu_ = CatchEvent(lineMenu_, [this](const Event event) {
            if (busy_) {
                return true;
            }
            if (event == Event::Tab) {
                viewMenu_->TakeFocus();
                return true;
            }
            return false;
        });
        visibleLineMenu_ =
            Maybe(lineMenu_, [this] { return viewIndex_ == 0; });

        constexpr std::array types{
            LineType::Microstrip,
            LineType::CoplanarWaveguide,
            LineType::Stripline,
            LineType::Coaxial,
        };
        Components formChildren;
        for (std::size_t index = 0; index < forms_.size(); ++index) {
            forms_[index] = makeForm(types[index]);
            formChildren.push_back(Maybe(
                forms_[index].component,
                [this, index] { return lineIndex_ == static_cast<int>(index); }));
        }
        formStack_ = Container::Vertical(std::move(formChildren));

        refinementInput_ = makeStandaloneInput(
            "Mesh density factor", refinementText_, refinementError_, "1",
            refinementRawInput_,
            [this] { setupChanged("Mesh density changed; press F5 to calculate."); });

        CheckboxOption benchmarkOptions;
        benchmarkOptions.on_change = [this] {
            repetitionsError_.clear();
            setupChanged(benchmark_ ? "Benchmark enabled; choose the run count."
                                    : "Benchmark disabled; the solver runs once.");
        };
        benchmarkCheckbox_ =
            Checkbox("Benchmark repeated full solves", &benchmark_, benchmarkOptions);
        repetitionsInput_ = makeStandaloneInput(
            "Benchmark runs", repetitionsText_, repetitionsError_, "5",
            repetitionsRawInput_,
            [this] { setupChanged("Benchmark count changed; press F5 to calculate."); });
        repetitionsInput_ =
            Maybe(repetitionsInput_, [this] { return benchmark_; });

        calculateButton_ = Button(
            "Calculate FEM", [this] { startRun(false); }, ButtonOption::Animated());
        refineButton_ = Button(
            "Refine x2 + solve", [this] { startRun(true); }, ButtonOption::Animated());
        resetButton_ = Button("Reset defaults", [this] { resetDefaults(); },
                              ButtonOption::Animated());
        stopButton_ = Button("Stop after current solve", [this] { requestStop(); },
                             ButtonOption::Animated(Color::Red));

        actionRow_ = Container::Horizontal({
            Maybe(calculateButton_, [this] { return !busy_; }),
            Maybe(refineButton_, [this] { return !busy_; }),
            Maybe(resetButton_, [this] { return !busy_; }),
            Maybe(stopButton_, [this] { return busy_; }),
        });

        setupControls_ = Container::Vertical(
            {formStack_, refinementInput_, benchmarkCheckbox_, repetitionsInput_});
        setupControls_ = CatchEvent(setupControls_, [this](Event) { return busy_; });

        resultTabs_ = {"Overview", "RLGC", "Performance"};
        resultTabMenu_ = Menu(&resultTabs_, &resultTabIndex_,
                              MenuOption::HorizontalAnimated());

        viewNames_ = {"Setup", "Results"};
        viewMenu_ = Menu(&viewNames_, &viewIndex_,
                         MenuOption::HorizontalAnimated());
        viewMenu_ = CatchEvent(viewMenu_, [this](const Event event) {
            if (event == Event::Tab) {
                if (viewIndex_ == 0) {
                    forms_[static_cast<std::size_t>(lineIndex_)]
                        .fields.front()
                        ->input->TakeFocus();
                } else {
                    resultTabMenu_->TakeFocus();
                }
                return true;
            }
            return false;
        });
        setupTab_ = Container::Vertical({setupControls_, actionRow_});
        contentTabs_ =
            Container::Tab({setupTab_, resultTabMenu_}, &viewIndex_);

        helpCloseButton_ = Button("Close", [this] { showHelp_ = false; },
                                  ButtonOption::Animated());
        helpComponent_ = Renderer(helpCloseButton_, [this] {
            return window(
                       text(" Help ") | bold,
                       vbox({
                           text("F5        Calculate the selected geometry"),
                           text("F6        Double mesh density and calculate"),
                           text("Ctrl+R    Restore all audited defaults"),
                           text("Tab       Move focus; Shift+Tab moves back"),
                           text("Arrows    Select geometry and result tabs"),
                           text("F1        Toggle this help"),
                           text("Ctrl+Q    Quit after the current Gmsh solve"),
                           separator(),
                           paragraph("Inputs use the engineering units shown in their "
                                     "labels. A blank metal conductivity means ideal PEC. "
                                     "The active Gmsh solve is opaque, so stop takes "
                                     "effect between benchmark repetitions."),
                           separator(),
                           helpCloseButton_->Render() | center,
                       })) |
                   size(WIDTH, LESS_THAN, 76) | center;
        });

        mainContainer_ = Container::Vertical(
            {visibleLineMenu_, viewMenu_, contentTabs_});
        auto mainRenderer = Renderer(mainContainer_, [this] { return renderMain(); });
        root_ = Modal(mainRenderer, helpComponent_, &showHelp_);
        root_ = CatchEvent(root_, [this](const Event event) {
            if (event == Event::F1) {
                showHelp_ = !showHelp_;
                return true;
            }
            if (showHelp_ && event == Event::Escape) {
                showHelp_ = false;
                return true;
            }
            if (!showHelp_ && event == Event::F5) {
                startRun(false);
                return true;
            }
            if (!showHelp_ && event == Event::F6) {
                startRun(true);
                return true;
            }
            if (!showHelp_ && event == Event::Special({'\x12'})) {
                resetDefaults();
                return true;
            }
            if (event == Event::Special({'\x11'})) {
                requestExit();
                return true;
            }
            return false;
        });
    }

    GeometryForm makeForm(const LineType type) {
        GeometryForm form;
        form.type = type;
        Components rows;
        for (auto definition : definitionsFor(type)) {
            auto field = std::make_shared<FieldState>();
            field->definition = std::move(definition);
            field->value = field->definition.defaultText;

            InputOption options = InputOption::Default();
            options.multiline = false;
            options.on_change = [this, field] {
                field->error.clear();
                setupChanged("Parameters changed; press F5 to calculate.");
            };
            const std::string placeholder = field->definition.optional
                                                ? "PEC"
                                                : field->definition.defaultText;
            field->input = Input(&field->value, placeholder, options);
            field->row = Renderer(field->input, [field] {
                Element editor = field->input->Render() | borderRounded | flex;
                if (!field->error.empty()) {
                    editor |= color(Color::Red);
                }
                Elements lines{
                    hbox({text(field->definition.label) | size(WIDTH, EQUAL, 29),
                          editor}),
                };
                if (!field->error.empty()) {
                    lines.push_back(text("  ! " + field->error) | color(Color::Red));
                }
                return vbox(std::move(lines));
            });
            rows.push_back(field->row);
            form.fields.push_back(std::move(field));
        }
        form.component = Container::Vertical(std::move(rows));
        return form;
    }

    Component makeStandaloneInput(const std::string& label, std::string& value,
                                  std::string& error,
                                  const std::string& placeholder,
                                  Component& rawInput,
                                  std::function<void()> onChange) {
        InputOption options = InputOption::Default();
        options.multiline = false;
        options.on_change = std::move(onChange);
        rawInput = Input(&value, placeholder, options);
        return Renderer(rawInput, [input = rawInput, label, &error] {
            Element editor = input->Render() | borderRounded | flex;
            if (!error.empty()) {
                editor |= color(Color::Red);
            }
            Elements lines{hbox({text(label) | size(WIDTH, EQUAL, 29), editor})};
            if (!error.empty()) {
                lines.push_back(text("  ! " + error) | color(Color::Red));
            }
            return vbox(std::move(lines));
        });
    }

    void setupChanged(std::string message) {
        if (result_) {
            resultStale_ = true;
        }
        status_ = std::move(message);
        statusError_ = false;
    }

    void resetDefaults() {
        if (busy_) {
            return;
        }
        for (auto& form : forms_) {
            for (auto& field : form.fields) {
                field->value = field->definition.defaultText;
                field->error.clear();
            }
        }
        refinementText_ = "1";
        repetitionsText_ = "5";
        refinementError_.clear();
        repetitionsError_.clear();
        benchmark_ = false;
        viewIndex_ = 0;
        resultStale_ = static_cast<bool>(result_);
        status_ = "Audited defaults restored; press F5 to calculate.";
        statusError_ = false;
    }

    [[nodiscard]] static std::optional<double> parseDisplayedField(
        const std::shared_ptr<FieldState>& field) {
        if (field->value.empty() && field->definition.optional) {
            return std::nullopt;
        }
        try {
            std::size_t consumed{};
            const double displayed = std::stod(field->value, &consumed);
            if (consumed != field->value.size()) {
                field->error = "Enter a number only.";
                return std::nullopt;
            }
            if (!std::isfinite(displayed)) {
                field->error = "Value must be finite.";
                return std::nullopt;
            }
            if (field->definition.strictlyPositive && displayed <= 0.0) {
                field->error = "Value must be greater than zero.";
                return std::nullopt;
            }
            if (!field->definition.strictlyPositive && displayed < 0.0) {
                field->error = "Value must not be negative.";
                return std::nullopt;
            }
            return displayed * field->definition.siScale;
        } catch (const std::exception&) {
            field->error = field->value.empty() ? "A value is required."
                                                : "Enter a valid number.";
            return std::nullopt;
        }
    }

    [[nodiscard]] std::shared_ptr<FieldState> findField(GeometryForm& form,
                                                        const InputKey key) {
        const auto found = std::find_if(
            form.fields.begin(), form.fields.end(),
            [key](const auto& field) { return field->definition.key == key; });
        if (found == form.fields.end()) {
            throw std::logic_error("TUI field definition is incomplete");
        }
        return *found;
    }

    [[nodiscard]] double currentRefinement(Component& firstInvalid) {
        refinementError_.clear();
        try {
            std::size_t consumed{};
            const double parsed = std::stod(refinementText_, &consumed);
            if (consumed != refinementText_.size() || !std::isfinite(parsed) ||
                parsed <= 0.0) {
                throw std::invalid_argument("invalid refinement");
            }
            return parsed;
        } catch (const std::exception&) {
            refinementError_ = "Enter a finite number greater than zero.";
            if (!firstInvalid) {
                firstInvalid = refinementRawInput_;
            }
            return 1.0;
        }
    }

    [[nodiscard]] int currentRepetitions(Component& firstInvalid) {
        repetitionsError_.clear();
        if (!benchmark_) {
            return 1;
        }
        try {
            std::size_t consumed{};
            const long parsed = std::stol(repetitionsText_, &consumed);
            if (consumed != repetitionsText_.size() || parsed < 1L ||
                parsed > 1000L) {
                throw std::invalid_argument("invalid repetition count");
            }
            return static_cast<int>(parsed);
        } catch (const std::exception&) {
            repetitionsError_ = "Enter an integer from 1 to 1000.";
            if (!firstInvalid) {
                firstInvalid = repetitionsRawInput_;
            }
            return 0;
        }
    }

    [[nodiscard]] std::optional<Parameters> readParameters(
        GeometryForm& form, Component& firstInvalid) {
        Parameters parameters = defaultParameters(form.type);
        parameters.refinementFactor = currentRefinement(firstInvalid);
        if (!refinementError_.empty()) {
            return std::nullopt;
        }

        bool valid = true;
        for (const auto& field : form.fields) {
            field->error.clear();
            const auto value = parseDisplayedField(field);
            if (field->definition.optional && field->value.empty()) {
                parameters.metalConductivity.reset();
                continue;
            }
            if (!value.has_value()) {
                if (!firstInvalid) {
                    firstInvalid = field->input;
                }
                valid = false;
                continue;
            }
            switch (field->definition.key) {
                case InputKey::Frequency:
                    parameters.frequencyHz = *value;
                    break;
                case InputKey::InnerRadius:
                    parameters.innerRadius = *value;
                    break;
                case InputKey::OuterRadius:
                    parameters.outerRadius = *value;
                    break;
                case InputKey::OuterConductorThickness:
                    parameters.outerConductorThickness = *value;
                    break;
                case InputKey::TraceWidth:
                    parameters.traceWidth = *value;
                    break;
                case InputKey::SubstrateHeight:
                    parameters.substrateHeight = *value;
                    break;
                case InputKey::ConductorThickness:
                    parameters.conductorThickness = *value;
                    break;
                case InputKey::GroundSpacing:
                    parameters.groundSpacing = *value;
                    break;
                case InputKey::CenterWidth:
                    parameters.centerWidth = *value;
                    break;
                case InputKey::Gap:
                    parameters.gap = *value;
                    break;
                case InputKey::GroundWidth:
                    parameters.groundWidth = *value;
                    break;
                case InputKey::EpsilonR:
                    parameters.epsilonR = *value;
                    break;
                case InputKey::LossTangent:
                    parameters.lossTangent = *value;
                    break;
                case InputKey::DomainPaddingFactor:
                    parameters.domainPaddingFactor = *value;
                    break;
                case InputKey::MetalConductivity:
                    parameters.metalConductivity = *value;
                    break;
                case InputKey::MaxElementSize:
                    parameters.maxElementSize = *value;
                    break;
            }
        }
        if (!valid) {
            return std::nullopt;
        }
        if (parameters.type == LineType::Coaxial &&
            parameters.outerRadius <= parameters.innerRadius) {
            auto field = findField(form, InputKey::OuterRadius);
            field->error = "Must be greater than the inner radius.";
            firstInvalid = field->input;
            return std::nullopt;
        }
        if (parameters.type == LineType::Stripline &&
            parameters.conductorThickness >= parameters.groundSpacing) {
            auto field = findField(form, InputKey::ConductorThickness);
            field->error = "Must be smaller than the ground gap.";
            firstInvalid = field->input;
            return std::nullopt;
        }
        return parameters;
    }

    void enqueueUiTask(Closure task) {
        screen_.Post([this, task = std::move(task)]() mutable {
            task();
            screen_.RequestAnimationFrame();
        });
    }

    void startRun(const bool refine) {
        if (busy_) {
            return;
        }
        if (worker_.joinable()) {
            worker_.join();
        }

        Component firstInvalid;
        std::optional<double> refinedDensity;
        if (refine) {
            const double current = currentRefinement(firstInvalid);
            if (!refinementError_.empty()) {
                viewIndex_ = 0;
                status_ = "Correct the highlighted mesh-density value first.";
                statusError_ = true;
                firstInvalid->TakeFocus();
                return;
            }
            refinedDensity = current * 2.0;
        }

        auto parameters =
            readParameters(forms_[static_cast<std::size_t>(lineIndex_)], firstInvalid);
        const int repetitions = currentRepetitions(firstInvalid);
        if (!parameters.has_value() || repetitions == 0) {
            viewIndex_ = 0;
            status_ = "Correct the highlighted input before calculating.";
            statusError_ = true;
            if (firstInvalid) {
                firstInvalid->TakeFocus();
            }
            return;
        }

        if (refinedDensity.has_value()) {
            refinementText_ = compactNumber(*refinedDensity);
            parameters->refinementFactor = *refinedDensity;
            resultStale_ = static_cast<bool>(result_);
        }

        viewIndex_ = 0;
        busy_ = true;
        stopButton_->TakeFocus();
        stopRequested_.store(false);
        statusError_ = false;
        lastSolveError_.clear();
        status_ = "Starting FEM calculation...";
        const std::string geometryName(lineTitle(parameters->type));
        const bool benchmarkRequested = benchmark_;

        worker_ = std::thread(
            [this, parameters = *parameters, repetitions, geometryName,
             benchmarkRequested] {
                std::vector<double> timings;
                timings.reserve(static_cast<std::size_t>(repetitions));
                std::shared_ptr<Result> lastResult;
                std::string error;
                bool stopped = false;
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    if (stopRequested_.load()) {
                        stopped = true;
                        break;
                    }
                    enqueueUiTask([this, repetition, repetitions, geometryName] {
                        status_ = "Solving " + geometryName + " (run " +
                                  std::to_string(repetition + 1) + "/" +
                                  std::to_string(repetitions) + ")...";
                    });
                    try {
                        const auto begin = Clock::now();
                        lastResult = std::make_shared<Result>(solve(parameters));
                        const auto end = Clock::now();
                        timings.push_back(std::chrono::duration<double, std::milli>(
                                              end - begin)
                                              .count());
                    } catch (const std::exception& failure) {
                        error = failure.what();
                        break;
                    } catch (...) {
                        error = "The native FEM solver failed with an unknown error.";
                        break;
                    }
                }

                double median = 0.0;
                if (!timings.empty()) {
                    std::sort(timings.begin(), timings.end());
                    const auto middle = timings.size() / 2U;
                    median = timings.size() % 2U == 0U
                                 ? 0.5 * (timings[middle - 1U] + timings[middle])
                                 : timings[middle];
                }
                const int completed = static_cast<int>(timings.size());
                enqueueUiTask([this, lastResult = std::move(lastResult),
                               error = std::move(error), median, completed,
                               repetitions, geometryName, stopped,
                               benchmarkRequested] {
                    busy_ = false;
                    lastSolveError_ = error;
                    if (lastResult) {
                        result_ = std::move(lastResult);
                        resultName_ = geometryName;
                        resultMedianMilliseconds_ = median;
                        resultCompletedRuns_ = completed;
                        resultRequestedRuns_ = repetitions;
                        resultBenchmark_ = benchmarkRequested;
                        resultStale_ = false;
                    }
                    viewIndex_ = 1;
                    resultTabMenu_->TakeFocus();
                    if (!error.empty()) {
                        status_ = "Solver error: " + error;
                        statusError_ = true;
                    } else if (stopped || stopRequested_.load()) {
                        status_ = "Stopped between solves; completed results were retained.";
                        statusError_ = false;
                    } else {
                        status_ = "Calculation complete. Choose a result tab.";
                        statusError_ = false;
                    }
                    if (exitAfterRun_) {
                        screen_.Exit();
                    }
                });
            });
    }

    void requestStop() {
        if (!busy_) {
            return;
        }
        stopRequested_.store(true);
        status_ = "Stop requested; waiting for the current Gmsh solve to finish...";
        statusError_ = false;
    }

    void requestExit() {
        if (busy_) {
            exitAfterRun_ = true;
            requestStop();
            return;
        }
        screen_.Exit();
    }

    [[nodiscard]] Element renderMain() {
        const auto title = hbox({
            text(" FEM Transmission-Line Calculator ") | bold | color(Color::Cyan),
            filler(),
            text(std::string("v") + TL_CALCULATOR_VERSION + " / FTXUI") | dim,
        });
        const auto target = hbox({text(" Geometry  ") | bold,
                                  lineMenu_->Render() | xflex});

        const auto viewPicker = hbox({text(" Workspace ") | bold,
                                      viewMenu_->Render() | xflex});

        Element body;
        if (viewIndex_ == 0) {
            const auto setupBody =
                vbox({setupControls_->Render() | yframe | flex, separator(),
                      actionRow_->Render()}) |
                flex;
            body = window(text(" Inputs ") | bold, setupBody) | flex;
        } else {
            std::string resultTitle = " Results";
            if (!resultName_.empty()) {
                resultTitle += " - " + resultName_;
            }
            if (resultStale_) {
                resultTitle += " (stale)";
            }
            resultTitle += " ";
            Elements resultElements{resultTabMenu_->Render(), separator()};
            if (!lastSolveError_.empty()) {
                resultElements.push_back(
                    paragraph("Last solve failed: " + lastSolveError_) |
                    color(Color::Red));
                resultElements.push_back(separator());
            }
            resultElements.push_back(renderResult() | yframe | flex);
            body = window(text(resultTitle) | bold,
                          vbox(std::move(resultElements))) |
                   flex;
        }

        Element status = text(" " + status_ + " ");
        if (statusError_) {
            status |= color(Color::Red) | bold;
        } else if (busy_) {
            status |= color(Color::Yellow) | bold;
        } else {
            status |= color(Color::Green);
        }

        Elements layout{title};
        if (viewIndex_ == 0) {
            layout.push_back(target);
        }
        layout.push_back(viewPicker);
        layout.push_back(body);
        layout.push_back(separator());
        layout.push_back(status);
        layout.push_back(
            text(" F5 Calculate  F6 Refine x2  F1 Help  Tab Focus  Ctrl+Q Quit ") |
            dim);
        return vbox(std::move(layout)) | borderRounded;
    }

    [[nodiscard]] Element renderResult() const {
        if (!result_) {
            return vbox({
                text("No FEM result yet.") | bold,
                paragraph("Choose a geometry, edit its engineering-unit inputs, "
                          "then press F5 or activate Calculate FEM."),
            });
        }

        const auto& result = *result_;
        if (resultTabIndex_ == 1) {
            Elements rows{
                valueRow("R'", formatScientific(result.resistancePerLength, "ohm/m")),
                valueRow("L'", formatScientific(result.inductancePerLength, "H/m")),
                valueRow("G'", formatScientific(result.conductancePerLength, "S/m")),
                valueRow("C'", formatComplexValue(result.capacitancePerLength, "F/m")),
                valueRow("C0'", formatScientific(result.vacuumCapacitancePerLength, "F/m")),
                valueRow("Power", formatComplexValue(result.power, "W")),
            };
            if (result.parameters.metalConductivity.has_value()) {
                rows.push_back(valueRow(
                    "Rs", formatScientific(result.surfaceResistance, "ohm")));
                rows.push_back(valueRow(
                    "Geometry",
                    formatScientific(result.conductorGeometryFactorPerLength, "1/m")));
            }
            return vbox(std::move(rows));
        }
        if (resultTabIndex_ == 2) {
            Elements rows{
                valueRow("Nodes", std::to_string(result.mesh.nodes.size())),
                valueRow("Triangles", std::to_string(result.mesh.triangles.size())),
                valueRow("Mesh", formatScientific(result.meshMilliseconds, "ms")),
                valueRow("Assembly", formatScientific(result.assemblyMilliseconds, "ms")),
                valueRow("Factorize", formatScientific(result.factorizationMilliseconds, "ms")),
                valueRow("Solve", formatScientific(result.solveMilliseconds, "ms")),
            };
            if (resultBenchmark_) {
                rows.push_back(valueRow(
                    "Median", formatScientific(resultMedianMilliseconds_, "ms")));
                rows.push_back(valueRow(
                    "Runs", std::to_string(resultCompletedRuns_) + "/" +
                                std::to_string(resultRequestedRuns_)));
            }
            return vbox(std::move(rows));
        }
        return vbox({
            valueRow("n_eff", formatComplexValue(result.neff)),
            valueRow("beta", formatComplexValue(result.beta, "1/m")),
            valueRow("Zc", formatComplexValue(result.characteristicImpedance, "ohm")),
            valueRow("Zwave", formatComplexValue(result.waveImpedance, "ohm")),
            valueRow("Voltage", formatComplexValue(result.voltage, "V")),
            valueRow("Current", formatComplexValue(result.current, "A")),
        });
    }

    ScreenInteractive screen_;
    Component root_;
    Component mainContainer_;
    Component lineMenu_;
    Component visibleLineMenu_;
    Component viewMenu_;
    Component contentTabs_;
    Component setupTab_;
    Component formStack_;
    Component setupControls_;
    Component refinementInput_;
    Component refinementRawInput_;
    Component benchmarkCheckbox_;
    Component repetitionsInput_;
    Component repetitionsRawInput_;
    Component actionRow_;
    Component calculateButton_;
    Component refineButton_;
    Component resetButton_;
    Component stopButton_;
    Component resultTabMenu_;
    Component helpComponent_;
    Component helpCloseButton_;

    std::array<GeometryForm, 4> forms_;
    std::vector<std::string> lineNames_;
    std::vector<std::string> viewNames_;
    std::vector<std::string> resultTabs_;
    int lineIndex_{};
    int viewIndex_{};
    int resultTabIndex_{};
    std::string refinementText_{"1"};
    std::string repetitionsText_{"5"};
    std::string refinementError_;
    std::string repetitionsError_;
    std::string status_{"Ready. Edit inputs or press F5 for the microstrip default."};
    std::string lastSolveError_;
    std::string resultName_;
    std::shared_ptr<const Result> result_;
    double resultMedianMilliseconds_{};
    int resultCompletedRuns_{};
    int resultRequestedRuns_{};
    bool benchmark_{};
    bool busy_{};
    bool showHelp_{};
    bool statusError_{};
    bool resultStale_{};
    bool resultBenchmark_{};
    bool exitAfterRun_{};
    std::atomic<bool> stopRequested_{};
    std::thread worker_;
};

}  // namespace

int run() {
    TuiApplication application;
    return application.loop();
}

bool smokeTest() {
    TuiApplication application;
    return application.renderSmokeFrame();
}

}  // namespace tl::tui
