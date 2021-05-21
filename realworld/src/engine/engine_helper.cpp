#include <fstream>
#include <vector>
#include <string>

#include "engine_helper.h"

namespace engine {
namespace helper {

std::vector<uint64_t> readFile(const std::string& file_name, uint64_t& file_size) {
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::string error_message = std::string("failed to open file! :") + file_name;
        throw std::runtime_error(error_message);
    }

    file_size = (uint64_t)file.tellg();
    std::vector<uint64_t> buffer((file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    file.close();
    return buffer;
}

} // namespace helper
} // namespace engine
