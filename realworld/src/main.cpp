#include <vulkan/vulkan.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <memory>
#include <thread>

#include "application.h"

extern glm::vec4 getMainImage(const glm::vec2& frag_coord, const glm::ivec2& screen_size);

struct FillBlockParams {
    glm::ivec2 blk_ofs;
    glm::ivec2 blk_size;
    glm::ivec2 size;
    uint8_t* image;
    glm::vec2* height_map;
};

void fillBlock(FillBlockParams* params) {
    for (int y = 0; y < params->blk_size.y; y++) {
        for (int x = 0; x < params->blk_size.x; x++) {
            auto ofs = params->blk_ofs + glm::ivec2(x, y);
            auto index = ofs.y * params->size.x + ofs.x;

            //glm::vec4 c = getMainImage(ofs, params->size);
            auto height = params->height_map[index];
            float normal_height = (height.x + 255.0f) / 150.0f;
            glm::vec4 c = glm::vec4(normal_height, normal_height, normal_height, 1.0f);

            params->image[index * 3] = static_cast<uint8_t>(glm::clamp(c.z, 0.0f, 1.0f) * 255.0f);
            params->image[index * 3 + 1] = static_cast<uint8_t>(glm::clamp(c.y, 0.0f, 1.0f) * 255.0f);
            params->image[index * 3 + 2] = static_cast<uint8_t>(glm::clamp(c.x, 0.0f, 1.0f) * 255.0f);
        }
    }
}


int main() {
    if (0) {
        std::ofstream myfile;
        myfile.open("out.bmp", std::ios::binary);

        glm::vec3 corners[4] = { {-100.0f, 0.0f, -100.0f},
                                {-100.0f, 0.0f, 100.0f},
                                {100.0f, 0.0f, 100.0f},
                                {100.0f, 0.0f, -100.0f} };

        const uint32_t w = 1920, h = 1080;
        auto height_map = generateTileMeshVertex(corners, glm::uvec2(w - 1, h - 1));

        uint32_t file_size = 54 + 3 * w * h;

        uint8_t bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
        uint8_t bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };

        bmpfileheader[2] = (uint8_t)(file_size & 0xff);
        bmpfileheader[3] = (uint8_t)((file_size >> 8) & 0xff);
        bmpfileheader[4] = (uint8_t)((file_size >> 16) & 0xff);
        bmpfileheader[5] = (uint8_t)(file_size >> 24);

        bmpinfoheader[4] = (uint8_t)(w & 0xff);
        bmpinfoheader[5] = (uint8_t)((w >> 8) & 0xff);
        bmpinfoheader[6] = (uint8_t)((w >> 16) & 0xff);
        bmpinfoheader[7] = (uint8_t)(w >> 24);
        bmpinfoheader[8] = (uint8_t)(h & 0xff);
        bmpinfoheader[9] = (uint8_t)((h >> 8) & 0xff);
        bmpinfoheader[10] = (uint8_t)((h >> 16) & 0xff);
        bmpinfoheader[11] = (uint8_t)(h >> 24);

        myfile.write((char*)bmpfileheader, 14);
        myfile.write((char*)bmpinfoheader, 40);

        std::vector<uint8_t> image;
        image.resize(w * h * 3);

        const uint32_t kBlockSize = 64;

        const int bw = (w + kBlockSize - 1) / kBlockSize;
        const int bh = (h + kBlockSize - 1) / kBlockSize;
        std::vector<FillBlockParams> params_list;
        params_list.reserve(bw * bh);

        for (int by = 0; by < bh; by++) {
            for (int bx = 0; bx < bw; bx++) {
                auto ox = bx * kBlockSize;
                auto oy = by * kBlockSize;
                FillBlockParams params;
                params.blk_ofs = glm::ivec2(ox, oy);
                params.blk_size = glm::ivec2(glm::min(w - ox, kBlockSize), glm::min(h - oy, kBlockSize));
                params.size = glm::ivec2(w, h);
                params.image = image.data();
                params.height_map = height_map.data();
                params_list.push_back(params);
            }
        }

        std::vector<std::thread> threads;
        for (auto& params : params_list) {
            threads.push_back(std::thread(fillBlock, &params));
        }

        for (auto& item : threads) {
            item.join();
        }

        myfile.write((char*)image.data(), w * h * 3);
        myfile.close();
    }

    auto app = std::make_shared<work::app::RealWorldApplication>();

    try {
        app->run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}