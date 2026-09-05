#pragma once

#include <QByteArray>
#include <QString>

#include <filesystem>
#include <string>

namespace femperiodic {

inline std::filesystem::path pathFromQString(const QString& value) {
#if defined(_WIN32)
    return std::filesystem::path(value.toStdWString());
#else
    const auto encoded = value.toUtf8();
    return std::filesystem::path(
        std::string(encoded.constData(), static_cast<std::size_t>(encoded.size())));
#endif
}

inline QString qStringFromPath(const std::filesystem::path& value) {
#if defined(_WIN32)
    return QString::fromStdWString(value.wstring());
#else
    const auto encoded = value.u8string();
    return QString::fromUtf8(reinterpret_cast<const char*>(encoded.data()),
                             static_cast<qsizetype>(encoded.size()));
#endif
}

} // namespace femperiodic
