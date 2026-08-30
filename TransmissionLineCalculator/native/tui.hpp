#pragma once

namespace tl::tui {

// Run the full-screen interactive calculator. The caller must provide a real
// terminal; the application owns the alternate screen until it exits.
int run();

// Render the initial component tree into an in-memory screen for CI.
bool smokeTest();

}  // namespace tl::tui
