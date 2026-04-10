#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string_view>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace kakuhen::util {

/*!
 * @brief A terminal progress bar with percentage and ETA display.
 *
 * Renders a fixed-width progress bar that updates in place on TTY terminals.
 * Format: `[===>    ] 45% ETA: 2m 15s`
 *
 * For non-TTY output, prints milestone updates on separate lines.
 */
class ProgressBar {
 public:
  /*!
   * @brief Construct a progress bar.
   *
   * @param bar_width Width of the progress bar in characters (default 30).
   * @param non_tty_step_pct Percentage step for non-TTY output (default 10%).
   */
  explicit ProgressBar(int bar_width = 30, int non_tty_step_pct = 10)
      : bar_width_(std::max(10, bar_width)),
        non_tty_step_pct_(std::clamp(non_tty_step_pct, 1, 100)),
        tty_(stderr_is_tty()),
        start_time_(std::chrono::steady_clock::now()) {}

  ~ProgressBar() {
    finish();
  }

  // Non-copyable, non-movable (manages terminal state)
  ProgressBar(const ProgressBar&) = delete;
  ProgressBar& operator=(const ProgressBar&) = delete;
  ProgressBar(ProgressBar&&) = delete;
  ProgressBar& operator=(ProgressBar&&) = delete;

  /*!
   * @brief Update the progress bar.
   *
   * @param fraction Progress fraction in [0, 1].
   * @param label Optional label to display (default: "Progress").
   */
  void update(double fraction, std::string_view label = "Progress") {
    if (finished_) return;

    fraction = std::clamp(fraction, 0.0, 1.0);
    const int pct = static_cast<int>(fraction * 100.0 + 0.5);

    if (!tty_) {
      update_non_tty(pct, label);
      return;
    }

    // Rate-limit updates to avoid flooding (except at 0% and 100%)
    auto now = std::chrono::steady_clock::now();
    auto since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_);
    if (pct != 0 && pct != 100 && pct == last_pct_ && since_last.count() < 100) {
      return;
    }

    last_pct_ = pct;
    last_update_ = now;
    visible_ = true;

    render_tty(fraction, pct, label);

    if (pct >= 100) {
      std::cerr << '\n' << std::flush;
      visible_ = false;
      finished_ = true;
    }
  }

  /*!
   * @brief Finish the progress bar display.
   *
   * Clears the progress bar line if visible. Called automatically by destructor.
   */
  void finish() {
    if (tty_ && visible_ && !finished_) {
      std::cerr << "\r\x1b[2K" << std::flush;
      visible_ = false;
      finished_ = true;
    }
  }

  /*!
   * @brief Reset the progress bar for reuse.
   */
  void reset() {
    start_time_ = std::chrono::steady_clock::now();
    last_update_ = start_time_;
    last_pct_ = -1;
    visible_ = false;
    finished_ = false;
  }

 private:
  static bool stderr_is_tty() noexcept {
#if defined(_WIN32)
    return _isatty(_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
  }

  void update_non_tty(int pct, std::string_view label) {
    if (pct >= 100) {
      if (!finished_) {
        std::cerr << label << " 100%\n" << std::flush;
        finished_ = true;
      }
    } else if (pct >= last_pct_ + non_tty_step_pct_) {
      last_pct_ = pct;
      std::cerr << label << " " << pct << "%\n" << std::flush;
    }
  }

  void render_tty(double fraction, int pct, std::string_view label) {
    // Build the bar into a stack-allocated buffer: [====>     ]
    // bar_width_ is clamped to >= 10 in the constructor; add 2 for '[' and ']'.
    char bar_buf[256];
    const int filled = static_cast<int>(fraction * bar_width_);
    const int empty = bar_width_ - filled;
    int pos = 0;
    bar_buf[pos++] = '[';
    for (int k = 0; k < filled - 1; ++k) bar_buf[pos++] = '=';
    if (filled > 0) bar_buf[pos++] = '>';
    for (int k = 0; k < empty; ++k) bar_buf[pos++] = ' ';
    bar_buf[pos++] = ']';
    bar_buf[pos] = '\0';

    // Calculate ETA into a stack buffer
    char eta_buf[32];
    format_eta(fraction, eta_buf, sizeof(eta_buf));

    // Render: \r[====>     ] 45% ETA: 2m 15s  label
    std::cerr << "\r\x1b[2K" << bar_buf << ' ' << std::setw(3) << pct << "% " << eta_buf;
    if (!label.empty()) {
      std::cerr << "  " << label;
    }
    std::cerr << std::flush;
  }

  void format_eta(double fraction, char* buf, std::size_t buf_size) const {
    if (fraction < 0.01) {
      std::snprintf(buf, buf_size, "ETA: --:--");
      return;
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed < 0.1) {
      std::snprintf(buf, buf_size, "ETA: --:--");
      return;
    }
    // remaining = elapsed * (1 - fraction) / fraction
    double remaining = elapsed * (1.0 - fraction) / fraction;
    char dur[24];
    format_duration(remaining, dur, sizeof(dur));
    std::snprintf(buf, buf_size, "ETA: %s", dur);
  }

  static void format_duration(double seconds, char* buf, std::size_t buf_size) {
    if (seconds < 0 || seconds > 86400.0 * 7) {  // Cap at 7 days
      std::snprintf(buf, buf_size, "--:--");
      return;
    }
    const int total_secs = static_cast<int>(seconds + 0.5);
    const int hours = total_secs / 3600;
    const int mins = (total_secs % 3600) / 60;
    const int secs = total_secs % 60;
    if (hours > 0) {
      std::snprintf(buf, buf_size, "%dh %02dm", hours, mins);
    } else if (mins > 0) {
      std::snprintf(buf, buf_size, "%dm %02ds", mins, secs);
    } else {
      std::snprintf(buf, buf_size, "%ds", secs);
    }
  }

  int bar_width_;
  int non_tty_step_pct_;
  bool tty_;
  int last_pct_{-1};
  bool visible_{false};
  bool finished_{false};
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_update_{start_time_};
};

}  // namespace kakuhen::util
