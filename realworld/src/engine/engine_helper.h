#pragma once
#include <vector>

namespace engine {
namespace helper {

std::vector<uint64_t> readFile(const std::string& file_name, uint64_t& file_size);

} // namespace helper
} // namespace engine
