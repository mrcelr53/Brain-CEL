/*
* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum class LogLevel : int {
    Off   = 0,
    Error = 1,
    Warn  = 2,
    Info  = 3,
    Debug = 4,
    Trace = 5
};

#ifndef BRAINCEL_LOG_MAX_LEVEL
#define BRAINCEL_LOG_MAX_LEVEL 5
#endif

class LiveView {
public:
    void setTitle(const std::string& title);

    // Progress bar
    void setProgress(double current, double total);
    void clearProgress();

    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, double value, int precision = 2);
    void set(const std::string& key, long long value);

    void setUnit(const std::string& key, double value, const std::string& unit,
                 int precision = 2);

    void remove(const std::string& key);
    void clearFields();

    void show();
    void hide();

    bool active() const { return m_active; }

    void refresh(bool force = false);

    void setMinRedrawInterval(std::chrono::milliseconds interval) { m_minInterval = interval; }

private:
    friend class Log;

    std::string composeLine(int width) const;

    void eraseUnlocked();
    void drawUnlocked();
    int  drawnLines() const { return m_drawnLines; }

    std::string m_title;
    std::vector<std::pair<std::string, std::string>> m_fields;

    bool   m_hasProgress = false;
    double m_progressCur = 0.0;
    double m_progressTot = 0.0;

    bool m_active     = false;
    int  m_drawnLines = 0;      // box mode
    bool m_useBox     = true;   // line mode
    bool m_lineDrawn  = false;  // line mode
    int  m_spin       = 0;      // line mode

    std::chrono::steady_clock::time_point m_lastDraw{};
    std::chrono::milliseconds             m_minInterval{66};
};

// The logger
class Log {
public:
    static void     setLevel(LogLevel level);
    static LogLevel level();

    static bool setLevelFromString(const std::string& name);

    static void configureFromEnv();

    static void  setConsoleStream(std::FILE* stream);
    static std::FILE* consoleStream();

    static bool openFile(const std::string& path, bool append = false);
    static void closeFile();
    static void flush();

    static void setColorEnabled(bool enabled);
    static bool colorEnabled();
    static void setLiveEnabled(bool enabled);
    static bool liveEnabled();
    static bool lineModeEnabled();
    static bool cursorCapable();
    static bool progressEnabled();

    static bool enabled(const LogLevel l) { return static_cast<int>(l) <= s_level; }

    // Emission
    static void emit(LogLevel level, std::string_view tag, std::string_view message);
    static void result(std::string_view line);
    static void result(std::string_view plain, std::string_view colored);

    static void banner(std::string_view title, int width = 34);
    static void endBanner(int width = 34);
    static void kv(std::string_view key, std::string_view value, int keyWidth = 12);

    static void rule(int width = 60);
    static void blank();

    // Rendering
    static void heading(std::string_view title, int width = 72);

    static void section(std::string_view title, int width = 72);

    static void fields(const std::vector<std::pair<std::string, std::string>>& rows,
                       int columns = 2, int width = 72);

    static std::string count(long long n);              // 36760   -> "36 760"
    static std::string duration(double ms);             // 2654.1  -> "2.65 s"
    static std::string energy(long long microjoules);   // 1158418 -> "1.16 J"  (-1 -> "n/a")
    static std::string ratio(double v, int precision = 3);

    // Live area
    static LiveView& live();

private:
    static void writeHistory(const std::string& plain, const std::string& colored);

    static int s_level;
};

// Call-site macros
#define BC_LOG_AT_(lvl, tag, ...)                                                            \
    do {                                                                                     \
        if (Log::enabled(lvl)) {                                                             \
            Log::emit((lvl), (tag), std::format(__VA_ARGS__));                                \
        }                                                                                    \
    } while (0)

#if BRAINCEL_LOG_MAX_LEVEL >= 1
#define BC_ERROR(tag, ...) BC_LOG_AT_(LogLevel::Error, tag, __VA_ARGS__)
#else
#define BC_ERROR(tag, ...) ((void)0)
#endif

#if BRAINCEL_LOG_MAX_LEVEL >= 2
#define BC_WARN(tag, ...) BC_LOG_AT_(LogLevel::Warn, tag, __VA_ARGS__)
#else
#define BC_WARN(tag, ...) ((void)0)
#endif

#if BRAINCEL_LOG_MAX_LEVEL >= 3
#define BC_INFO(tag, ...) BC_LOG_AT_(LogLevel::Info, tag, __VA_ARGS__)
#else
#define BC_INFO(tag, ...) ((void)0)
#endif

#if BRAINCEL_LOG_MAX_LEVEL >= 4
#define BC_DEBUG(tag, ...) BC_LOG_AT_(LogLevel::Debug, tag, __VA_ARGS__)
#else
#define BC_DEBUG(tag, ...) ((void)0)
#endif

#if BRAINCEL_LOG_MAX_LEVEL >= 5
#define BC_TRACE(tag, ...) BC_LOG_AT_(LogLevel::Trace, tag, __VA_ARGS__)
#else
#define BC_TRACE(tag, ...) ((void)0)
#endif

#define BC_RESULT(...) Log::result(std::format(__VA_ARGS__))

// Latched variants: emit max once per call site
#define BC_LOG_ONCE_AT_(lvl, tag, ...)                                                      \
    do {                                                                                    \
        if (Log::enabled(lvl)) {                                                            \
            static std::atomic<bool> bcOnceFired_{false};                                   \
            if (!bcOnceFired_.load(std::memory_order_relaxed) &&                            \
                !bcOnceFired_.exchange(true, std::memory_order_relaxed)) {                  \
                Log::emit((lvl), (tag),                                                     \
                          std::format(__VA_ARGS__) +                                        \
                          " [further identical reports suppressed]");                       \
            }                                                                               \
        }                                                                                   \
    } while (0)

#define BC_WARN_ONCE(tag, ...)  BC_LOG_ONCE_AT_(LogLevel::Warn,  tag, __VA_ARGS__)
#define BC_ERROR_ONCE(tag, ...) BC_LOG_ONCE_AT_(LogLevel::Error, tag, __VA_ARGS__)
