#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>

/**
 * Split a string into tokens using a single-character delimiter.
 * @param str Input string to split.
 * @param delimiter Character used to separate tokens.
 * @returns A vector containing the tokens in order; empty vector if input is empty.
 */

/**
 * Remove leading and trailing whitespace characters (space, tab, newline, carriage return).
 * @param str Input string to trim.
 * @returns The trimmed string; empty string if no non-whitespace characters are present.
 */

/**
 * Convert all characters in a string to lowercase using the C locale's tolower.
 * @param str Input string to convert.
 * @returns A lowercase copy of the input string.
 */

/**
 * Check whether a string begins with a given prefix.
 * @param str String to inspect.
 * @param prefix Prefix to check for at the start of `str`.
 * @returns `true` if `str` starts with `prefix`, `false` otherwise.
 */

/**
 * Check whether a string ends with a given suffix.
 * @param str String to inspect.
 * @param suffix Suffix to check for at the end of `str`.
 * @returns `true` if `str` ends with `suffix`, `false` otherwise.
 */

/**
 * Obtain the current system time as milliseconds since the Unix epoch.
 * @returns Current time in milliseconds since 1970-01-01T00:00:00Z.
 */

/**
 * Obtain the current system time formatted as an ISO 8601 UTC string.
 * @returns Current time in ISO 8601 UTC format (e.g., "2024-01-02T15:04:05Z").
 */

/**
 * Generate a pseudo-random version-4-like UUID string in the form 8-4-4-4-12 hex digits.
 * @returns A UUID-like hexadecimal string (lowercase, hyphen-separated).
 */

/**
 * Check whether a file exists and is readable.
 * @param path Filesystem path to the file.
 * @returns `true` if the file can be opened for reading, `false` otherwise.
 */

/**
 * Read the entire contents of a file into a string.
 * @param path Filesystem path to the file.
 * @returns The file's contents as a string.
 * @throws std::runtime_error If the file cannot be opened for reading.
 */

/**
 * Write a string to a file, replacing its contents.
 * @param path Filesystem path to the file to write.
 * @param content Content to write into the file.
 * @throws std::runtime_error If the file cannot be opened for writing.
 */
namespace atomic {
namespace utils {

// String utilities
inline std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

inline std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

inline bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Time utilities
inline int64_t timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

inline std::string timestamp_iso() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// UUID generation (simple version)
inline std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);
    
    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
        ss << dis(gen);
    };
    return ss.str();
}

// File utilities
inline bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

inline std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline void write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << content;
}

} // namespace utils
} // namespace atomic