/*
* SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#include <braincel/Log.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#include <unistd.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <termios.h>

namespace {

// State


std::mutex    g_mutex;                    // mutex guards the console cursor

std::FILE*    g_console      = nullptr;   // resolved to stdout on first use
bool          g_colorSet     = false;     // color override
bool          g_color        = false;
bool          g_liveSet      = false;     // live area override
bool          g_live         = false;
bool          g_ttyResolved  = false;
bool          g_isTty        = false;
bool          g_cursorCapable = false;    // append-only fallback for non-cursor terminals (CSI)
bool          g_probeRan      = false;

std::ofstream g_file;
int           g_sinceFlush   = 0;
constexpr int kFlushEvery    = 32;

LiveView      g_liveView;

// Terminal probing
std::FILE* console() {
    if (!g_console) g_console = stdout;
    return g_console;
}

// Check terminals CSI capability
bool probeCursorSupport(const int outFd, const int timeoutMs = 200) {
    const int inFd = STDIN_FILENO;
    if (!::isatty(outFd) || !::isatty(inFd)) return false;

    termios saved{};
    if (::tcgetattr(inFd, &saved) != 0) return false;

    termios raw = saved;
    raw.c_lflag &= ~(ICANON | ECHO);   // unbuffered, and do not echo the reply
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    if (::tcsetattr(inFd, TCSANOW, &raw) != 0) return false;

    bool answered = false;
    if (::write(outFd, "\033[6n", 4) == 4) {
        char   buf[64];
        size_t used = 0;
        pollfd pfd{inFd, POLLIN, 0};
        while (used + 1 < sizeof(buf) && ::poll(&pfd, 1, timeoutMs) > 0) {
            const ssize_t got = ::read(inFd, buf + used, sizeof(buf) - 1 - used);
            if (got <= 0) break;
            used += static_cast<size_t>(got);
            // The reply is terminated by 'R'. Any well-formed answer proves CSI is honoured.
            if (std::memchr(buf, 'R', used) != nullptr) { answered = true; break; }
        }
    }

    ::tcsetattr(inFd, TCSANOW, &saved);
    return answered;
}

void resolveTty() {
    if (g_ttyResolved) return;
    g_isTty       = ::isatty(::fileno(console())) != 0;
    g_ttyResolved = true;

    const char* term = std::getenv("TERM");   // JetBrains terminal tool window (JediTerm)
    const bool usableTerm =
        term && *term && std::strcmp(term, "dumb") != 0 && std::strcmp(term, "unknown") != 0;

    if (!g_isTty || !usableTerm) {
        g_cursorCapable = false;
    } else if (const char* probe = std::getenv("BRAINCEL_LOG_PROBE");
               probe && (std::strcmp(probe, "0") == 0 || std::strcmp(probe, "off") == 0)) {
        g_cursorCapable = true;
    } else {
        g_cursorCapable = probeCursorSupport(::fileno(console()));
    }

    g_probeRan = true;

    if (!g_colorSet) g_color = g_isTty;
    if (!g_liveSet)  g_live  = g_cursorCapable;
}

int terminalWidth() {
    resolveTty();
    if (!g_isTty) return 100;
    winsize ws{};
    if (::ioctl(::fileno(console()), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20) {
        return static_cast<int>(ws.ws_col);
    }
    return 100;
}

// Color
constexpr const char* kReset = "\033[0m";
constexpr const char* kDim   = "\033[2m";
constexpr const char* kBold  = "\033[1m";

const char* levelColor(const LogLevel l) {
    switch (l) {
        case LogLevel::Error: return "\033[1;31m";  // bold red
        case LogLevel::Warn:  return "\033[1;33m";  // bold yellow
        case LogLevel::Info:  return "\033[1;36m";  // bold cyan
        case LogLevel::Debug: return "\033[2;37m";  // dim grey
        case LogLevel::Trace: return "\033[2;35m";  // dim magenta
        default:              return "";
    }
}

const char* levelName(const LogLevel l) {
    switch (l) {
        case LogLevel::Error: return "ERROR";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Trace: return "TRACE";
        default:              return "     ";
    }
}

std::string timestamp() {
    using namespace std::chrono;
    const auto now  = system_clock::now();
    const auto secs = time_point_cast<seconds>(now);
    const auto ms   = duration_cast<milliseconds>(now - secs).count();

    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    ::localtime_r(&t, &tm);

    return std::format("{:02}:{:02}:{:02}.{:03}", tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
}

std::string lower(std::string s) {
    std::ranges::transform(s, s.begin(),
                           [](const unsigned char c) { return std::tolower(c); });
    return s;
}


// Creates a new string by repeating the given glyph n times
std::string repeat(const std::string_view glyph, const int n) {
    std::string out;
    out.reserve(glyph.size() * std::max(0, n));
    for (int i = 0; i < n; ++i) out += glyph;
    return out;
}


// Calculates the visible width of a string
size_t visibleWidth(const std::string_view s) {
    size_t w = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\033') {
            while (i < s.size() && s[i] != 'm') ++i;
            continue;
        }
        // Continuation bytes (0b10xxxxxx) belong to the glyph already counted.
        if ((static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) continue;
        ++w;
    }
    return w;
}

} // namespace


// Log configuration
int Log::s_level = static_cast<int>(LogLevel::Info);

void Log::setLevel(const LogLevel level) { s_level = static_cast<int>(level); }

LogLevel Log::level() { return static_cast<LogLevel>(s_level); }

bool Log::setLevelFromString(const std::string& name) {
    const std::string n = lower(name);
    if (n == "off")                     { setLevel(LogLevel::Off);   return true; }
    if (n == "error" || n == "err")     { setLevel(LogLevel::Error); return true; }
    if (n == "warn"  || n == "warning") { setLevel(LogLevel::Warn);  return true; }
    if (n == "info")                    { setLevel(LogLevel::Info);  return true; }
    if (n == "debug")                   { setLevel(LogLevel::Debug); return true; }
    if (n == "trace")                   { setLevel(LogLevel::Trace); return true; }
    return false;
}

void Log::configureFromEnv() {
    if (const char* lvl = std::getenv("BRAINCEL_LOG_LEVEL")) {
        setLevelFromString(lvl);
    }
    if (const char* live = std::getenv("BRAINCEL_LOG_LIVE")) {
        setLiveEnabled(!(std::strcmp(live, "0") == 0 || lower(live) == "off"));
    }
    if (const char* col = std::getenv("BRAINCEL_LOG_COLOR")) {
        setColorEnabled(!(std::strcmp(col, "0") == 0 || lower(col) == "off"));
    }
    if (const char* path = std::getenv("BRAINCEL_LOG_FILE")) {
        if (*path) openFile(path);
    }

    // Console-capability diagnosis
    if (enabled(LogLevel::Debug)) {
        const char* term = std::getenv("TERM");
        const char* emu  = std::getenv("TERMINAL_EMULATOR");
        BC_DEBUG("Console",
                 "isatty={} stdin_tty={} TERM='{}' TERMINAL_EMULATOR='{}' probe={} "
                 "-> cursorCapable={} boxMode={} lineMode={}",
                 g_isTty, ::isatty(STDIN_FILENO) != 0,
                 term ? term : "<unset>", emu ? emu : "<unset>",
                 g_probeRan ? "ran" : "skipped",
                 g_cursorCapable, liveEnabled(), lineModeEnabled());
    }
}

void Log::setConsoleStream(std::FILE* stream) {
    std::lock_guard lock(g_mutex);
    g_console     = stream ? stream : stdout;
    g_ttyResolved = false;
}

std::FILE* Log::consoleStream() { return console(); }

bool Log::openFile(const std::string& path, const bool append) {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) g_file.close();
    g_file.open(path, append ? std::ios::app : std::ios::trunc);
    return g_file.is_open();
}

void Log::closeFile() {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) { g_file.flush(); g_file.close(); }
}

void Log::flush() {
    std::lock_guard lock(g_mutex);
    if (g_file.is_open()) g_file.flush();
    std::fflush(console());
    g_sinceFlush = 0;
}

void Log::setColorEnabled(const bool enabled) {
    resolveTty();
    g_color    = enabled;
    g_colorSet = true;
}

bool Log::colorEnabled() { resolveTty(); return g_color; }

void Log::setLiveEnabled(const bool enabled) {
    resolveTty();
    // The live area is a terminal-only feature
    const bool wanted = enabled && g_isTty;
    if (!wanted && g_liveView.active()) g_liveView.hide();
    g_live    = wanted;
    g_liveSet = true;
}

bool Log::liveEnabled() { resolveTty(); return g_live; }

bool Log::cursorCapable() { resolveTty(); return g_cursorCapable; }

bool Log::progressEnabled() { return liveEnabled() || lineModeEnabled(); }

bool Log::lineModeEnabled() {
    resolveTty();
    // Interactive but the cursor cannot be moved
    return g_isTty && !g_live;
}

LiveView& Log::live() { return g_liveView; }

// Log emission

void Log::writeHistory(const std::string& plain, const std::string& colored) {
    std::lock_guard lock(g_mutex);
    resolveTty();

    const bool hadLive = g_liveView.active() && g_liveView.drawnLines() > 0;
    if (hadLive) g_liveView.eraseUnlocked();

    std::fputs((g_color ? colored : plain).c_str(), console());
    std::fputc('\n', console());

    if (hadLive) {
        g_liveView.drawUnlocked();
        std::fflush(console());
    }

    // The file sink records the history area only, always uncoloured.
    if (g_file.is_open()) {
        g_file << plain << '\n';
        if (++g_sinceFlush >= kFlushEvery) { g_file.flush(); g_sinceFlush = 0; }
    }
}

void Log::emit(const LogLevel level, const std::string_view tag, const std::string_view message) {
    const std::string ts = timestamp();

    std::string plain;
    if (tag.empty()) {
        plain = std::format("{} {} {}", ts, levelName(level), message);
    } else {
        plain = std::format("{} {} [{}] {}", ts, levelName(level), tag, message);
    }

    std::string colored;
    if (tag.empty()) {
        colored = std::format("{}{}{} {}{}{} {}",
                              kDim, ts, kReset,
                              levelColor(level), levelName(level), kReset,
                              message);
    } else {
        colored = std::format("{}{}{} {}{}{} {}[{}]{} {}",
                              kDim, ts, kReset,
                              levelColor(level), levelName(level), kReset,
                              kDim, tag, kReset,
                              message);
    }

    writeHistory(plain, colored);
}

void Log::result(const std::string_view line) {
    const std::string s(line);
    writeHistory(s, s);
}

void Log::result(const std::string_view plain, const std::string_view colored) {
    writeHistory(std::string(plain), std::string(colored));
}

void Log::banner(const std::string_view title, const int width) {
    const int titleLen = static_cast<int>(title.size());
    const int dashes   = std::max(0, width - titleLen - 2);
    const int left     = dashes / 2;
    const int right    = dashes - left;

    const std::string plain =
        std::format("{} {} {}", std::string(left, '-'), title, std::string(right, '-'));
    const std::string colored =
        std::format("{}{} {}{}{}{} {}{}",
                    kDim, std::string(left, '-'), kReset,
                    kBold, title, kReset,
                    std::string(right, '-'), kReset);

    writeHistory(plain, g_color ? colored : plain);
}

void Log::endBanner(const int width) {
    const std::string line(width, '-');
    writeHistory(line, std::format("{}{}{}", kDim, line, kReset));
}

void Log::kv(const std::string_view key, const std::string_view value, const int keyWidth) {
    // routed through result()
    std::string k(key);
    k += ":";
    if (static_cast<int>(k.size()) < keyWidth) k.resize(keyWidth, ' ');
    result(std::format("  {}{}", k, value));
}

void Log::rule(const int width) {
    const std::string line(width, '-');
    writeHistory(line, std::format("{}{}{}", kDim, line, kReset));
}

void Log::blank() { writeHistory("", ""); }

// Rendering

void Log::heading(const std::string_view title, const int width) {
    blank();
    const std::string t(title);
    const std::string ruleLine   = repeat("━", std::max(0, width));
    const std::string titlePlain = std::format("  {}", t);
    const std::string rulePlain  = std::format("  {}", ruleLine);
    writeHistory(titlePlain, g_color ? std::format("  {}{}{}", kBold, t, kReset) : titlePlain);
    writeHistory(rulePlain,  g_color ? std::format("  {}{}{}", kDim, ruleLine, kReset) : rulePlain);
}

void Log::section(const std::string_view title, const int width) {
    const int used  = static_cast<int>(visibleWidth(title)) + 3;
    const int fill  = std::max(0, width - used);
    const std::string plain   = std::format("  {} {}", title, repeat("─", fill));
    const std::string colored = std::format("  {}{}{} {}{}{}",
                                            kBold, title, kReset,
                                            kDim, repeat("─", fill), kReset);
    writeHistory(plain, colored);
}

void Log::fields(const std::vector<std::pair<std::string, std::string>>& rows,
                 const int columns, const int width) {
    if (rows.empty()) return;
    const int cols = std::max(1, columns);

    size_t keyW = 0, valW = 0;
    for (const auto& [k, v] : rows) {
        keyW = std::max(keyW, visibleWidth(k));
        valW = std::max(valW, visibleWidth(v));
    }

    // Column pitch
    const int cellW  = static_cast<int>(keyW + valW) + 3;
    const int pitch  = std::min(cellW + 4, std::max(1, (width - 4) / cols));
    const bool color = g_color;

    for (size_t i = 0; i < rows.size(); i += cols) {
        std::string plain = "    ";
        std::string col   = "    ";
        for (int c = 0; c < cols && i + c < rows.size(); ++c) {
            const auto& [k, v] = rows[i + c];
            const int kPad = static_cast<int>(keyW - visibleWidth(k));
            const int vPad = static_cast<int>(valW - visibleWidth(v));   // right-align values

            const std::string cellPlain =
                std::format("{}{}  {}{}", k, std::string(kPad, ' '), std::string(vPad, ' '), v);
            const std::string cellCol =
                std::format("{}{}{}{}  {}{}{}{}",
                            kDim, k, kReset, std::string(kPad, ' '),
                            std::string(vPad, ' '), kBold, v, kReset);

            plain += cellPlain;
            col   += cellCol;
            const bool last = (c + 1 == cols) || (i + c + 1 >= rows.size());
            if (!last) {
                const int gap = std::max(2, pitch - static_cast<int>(visibleWidth(cellPlain)));
                plain += std::string(gap, ' ');
                col   += std::string(gap, ' ');
            }
        }
        writeHistory(plain, color ? col : plain);
    }
}

std::string Log::count(const long long n) {
    std::string digits = std::to_string(n < 0 ? -n : n);       // Thousands separated by a space
    std::string out;
    const int lead = static_cast<int>(digits.size()) % 3;
    for (int i = 0; i < static_cast<int>(digits.size()); ++i) {
        if (i > 0 && (i - lead) % 3 == 0) out += ' ';
        out += digits[i];
    }
    return (n < 0 ? "-" : "") + out;
}

std::string Log::duration(const double ms) {
    if (ms < 1.0)      return std::format("{:.2f} ms", ms);
    if (ms < 1000.0)   return std::format("{:.0f} ms", ms);
    if (ms < 60000.0)  return std::format("{:.2f} s", ms / 1000.0);
    const long long totalSec = static_cast<long long>(ms / 1000.0);
    return std::format("{}m {:02}s", totalSec / 60, totalSec % 60);
}

std::string Log::energy(const long long microjoules) {
    if (microjoules < 0)          return "n/a";
    if (microjoules < 1000)       return std::format("{} µJ", microjoules);
    if (microjoules < 1000000)    return std::format("{:.2f} mJ", microjoules / 1e3);
    return std::format("{:.2f} J", microjoules / 1e6);
}

std::string Log::ratio(const double v, const int precision) {
    return std::format("{:.{}f}", v, precision);
}

// LiveView
void LiveView::setTitle(const std::string& title) { m_title = title; }

void LiveView::setProgress(const double current, const double total) {
    m_hasProgress = true;
    m_progressCur = current;
    m_progressTot = total;
}

void LiveView::clearProgress() { m_hasProgress = false; }

void LiveView::set(const std::string& key, const std::string& value) {
    for (auto& [k, v] : m_fields) {
        if (k == key) { v = value; return; }
    }
    m_fields.emplace_back(key, value);
}

void LiveView::set(const std::string& key, const double value, const int precision) {
    set(key, std::format("{:.{}f}", value, precision));
}

void LiveView::set(const std::string& key, const long long value) {
    set(key, std::format("{}", value));
}

void LiveView::setUnit(const std::string& key, const double value, const std::string& unit,
                       const int precision) {
    set(key, std::format("{:.{}f} {}", value, precision, unit));
}

void LiveView::remove(const std::string& key) {
    std::erase_if(m_fields, [&](const auto& p) { return p.first == key; });
}

void LiveView::clearFields() { m_fields.clear(); }

void LiveView::show() {
    const bool box  = Log::liveEnabled();
    const bool line = !box && Log::lineModeEnabled();
    if (!box && !line) return;

    std::lock_guard lock(g_mutex);
    if (m_active) return;
    m_useBox = box;
    m_minInterval = box ? std::chrono::milliseconds{66} : std::chrono::milliseconds{200};  // Line mode writes to a shared scrollback line
    m_active = true;
    drawUnlocked();
    std::fflush(console());
}

void LiveView::hide() {
    std::lock_guard lock(g_mutex);
    if (!m_active) return;
    eraseUnlocked();
    m_active = false;
    std::fflush(console());
}

void LiveView::refresh(const bool force) {
    if (!m_active) return;

    // Wall-clock rate limit
    if (!force) {
        const auto now = std::chrono::steady_clock::now();
        if (now - m_lastDraw < m_minInterval) return;
        m_lastDraw = now;
    } else {
        m_lastDraw = std::chrono::steady_clock::now();
    }

    std::lock_guard lock(g_mutex);
    eraseUnlocked();
    drawUnlocked();
    std::fflush(console());
}

// Single-line rendering
std::string LiveView::composeLine(const int width) const {
    const bool  color = Log::colorEnabled();
    const char* dim   = color ? kDim  : "";
    const char* bold  = color ? kBold : "";
    const char* accent= color ? "\033[1;36m" : "";
    const char* reset = color ? kReset : "";

    static constexpr const char* kSpin[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    static constexpr const char* kPart[] = {"", "▏","▎","▍","▌","▋","▊","▉"};

    std::string out = "\r ";
    out += accent; out += kSpin[m_spin % 10]; out += reset;

    if (!m_title.empty()) {
        std::string_view shortTitle{m_title};
        if (const auto dash = shortTitle.find(" — "); dash != std::string_view::npos)
            shortTitle = shortTitle.substr(0, dash);
        out += "  ";
        out += bold; out += shortTitle; out += reset;
    }

    if (m_hasProgress && m_progressTot > 0.0) {
        const double frac = std::clamp(m_progressCur / m_progressTot, 0.0, 1.0);
        constexpr int kCells = 18;
        const double  exact  = frac * kCells;
        const int     full   = static_cast<int>(exact);
        const int     part   = static_cast<int>((exact - full) * 8.0);

        std::string bar;
        for (int i = 0; i < full && i < kCells; ++i) bar += "█";
        if (full < kCells && part > 0) bar += kPart[part];
        const int drawn = full + ((full < kCells && part > 0) ? 1 : 0);

        out += "  ";
        out += dim; out += "▕"; out += reset;
        out += accent; out += bar; out += reset;
        out += dim;
        for (int i = drawn; i < kCells; ++i) out += "░";
        out += "▏"; out += reset;
        out += bold; out += std::format(" {:5.1f}%", frac * 100.0); out += reset;
    }

    // Drop fileds if they dont fit width
    for (const auto& [key, value] : m_fields) {
        std::string cell;
        cell += dim;   cell += "  ·  "; cell += key; cell += " "; cell += reset;
        cell += bold;  cell += value;   cell += reset;
        if (static_cast<int>(visibleWidth(out) + visibleWidth(cell)) >= width - 1) break;
        out += cell;
    }

    // Pad to the full width
    const int pad = width - static_cast<int>(visibleWidth(out));
    if (pad > 0) out += std::string(pad, ' ');
    return out;
}

void LiveView::eraseUnlocked() {
    if (!m_useBox) {
        if (!m_lineDrawn) return;
        const int width = std::min(terminalWidth(), 120);
        std::fprintf(console(), "\r%*s\r", width, "");
        m_lineDrawn = false;
        return;
    }
    if (m_drawnLines <= 0) return;

    // Move the cursor back to the top of the block
    std::fprintf(console(), "\033[%dA\r\033[J", m_drawnLines);
    m_drawnLines = 0;
}

void LiveView::drawUnlocked() {
    if (!m_useBox) {
        // Single line
        const int width = std::min(terminalWidth(), 120);
        ++m_spin;
        std::fputs(composeLine(width).c_str(), console());
        m_lineDrawn  = true;
        m_drawnLines = 0;
        return;
    }

    const int width   = std::min(terminalWidth(), 100);
    const int inner   = width - 2;
    const bool color  = Log::colorEnabled();

    const char* dim   = color ? kDim  : "";
    const char* bold  = color ? kBold : "";
    const char* reset = color ? kReset : "";

    int lines = 0;

    // Top border
    if (!m_title.empty()) {
        const int titleRoom =
            std::max(0, inner - static_cast<int>(visibleWidth(m_title)) - 3);
        std::fprintf(console(), "%s┌─%s %s%s%s %s%s┐%s\n",
                     dim, reset,
                     bold, m_title.c_str(), reset,
                     dim, repeat("─", titleRoom).c_str(), reset);
    } else {
        std::fprintf(console(), "%s┌%s┐%s\n", dim, repeat("─", inner).c_str(), reset);
    }
    ++lines;

    // Progress bar
    if (m_hasProgress) {
        const double frac = (m_progressTot > 0.0)
                                ? std::clamp(m_progressCur / m_progressTot, 0.0, 1.0)
                                : 0.0;
        const std::string pct = std::format("{:5.1f}%", frac * 100.0);
        // Row is: "│" space bar space pct space "│" -> borders(2) + 3 spaces + pct.
        const int barRoom = inner - static_cast<int>(pct.size()) - 3;
        const int filled  = (barRoom > 0) ? static_cast<int>(frac * barRoom) : 0;

        std::string bar;
        for (int i = 0; i < barRoom; ++i) bar += (i < filled) ? "█" : "░";

        std::fprintf(console(), "%s│%s %s%s%s %s %s│%s\n",
                     dim, reset, bold, bar.c_str(), reset, pct.c_str(), dim, reset);
        ++lines;
    }

    // Field rows
    const int colWidth = inner / 2 - 1;
    for (size_t i = 0; i < m_fields.size(); i += 2) {
        std::string cell1 = std::format("{}: {}{}{}",
                                        m_fields[i].first, bold, m_fields[i].second, reset);
        std::string row = " " + cell1;
        int used = 1 + static_cast<int>(visibleWidth(cell1));

        if (i + 1 < m_fields.size()) {
            const int pad = std::max(1, colWidth - used + 1);
            row += std::string(pad, ' ');
            used += pad;
            std::string cell2 = std::format("{}: {}{}{}", m_fields[i + 1].first,
                                            bold, m_fields[i + 1].second, reset);
            row  += cell2;
            used += static_cast<int>(visibleWidth(cell2));
        }

        const int trail = std::max(0, inner - used);
        std::fprintf(console(), "%s│%s%s%s%s│%s\n",
                     dim, reset, row.c_str(), std::string(trail, ' ').c_str(), dim, reset);
        ++lines;
    }

    // Bottom border
    std::fprintf(console(), "%s└%s┘%s\n", dim, repeat("─", inner).c_str(), reset);
    ++lines;

    m_drawnLines = lines;
}
