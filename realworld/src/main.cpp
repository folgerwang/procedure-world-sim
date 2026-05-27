#include <vulkan/vulkan.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <filesystem>
#include <ctime>
#include <mutex>
#include <streambuf>
#include <string>

#include "application.h"
#include "helper/cluster_mesh.h"  // engine::helper::clusterRenderingEnabled()

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


// ─── Command-line flag parsing ──────────────────────────────────────────────
// Supported flags (parsed in main(), before the app is constructed):
//
//   --cluster-debug      Turn on the "Nanite-lite" cluster debug visualisation.
//                        Builds a ClusterMesh parallel to every static mesh at
//                        load time and renders each cluster in a unique flat
//                        color so you can eyeball cluster boundaries/sizes.
//                        (Also accepted: -cluster-debug, --cluster, -cluster.)
//
// Keeping this as a simple argv-sweep rather than pulling in a CLI lib — we
// only have one flag and the engine's startup is already self-contained.
static bool parseClusterDebugFlag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (a == nullptr) continue;
        if (std::strcmp(a, "--cluster-debug") == 0 ||
            std::strcmp(a,  "-cluster-debug") == 0 ||
            std::strcmp(a, "--cluster")        == 0 ||
            std::strcmp(a,  "-cluster")        == 0) {
            return true;
        }
    }
    return false;
}

// ── Line-routing stream buffer: splits physics/IK/collision log lines into
// a separate file ──────────────────────────────────────────────────────
// std::cout carries everything; this buffer inspects each completed LINE and,
// if it begins with a physics/IK/collision tag (or is an indented
// continuation of one), writes it to the physics log instead of the main
// engine log.  Centralising the split here means no call site has to change.
// A mutex guards the shared line buffer because a few lines come from
// background threads (async BVH build / classifier).
namespace {
class PhysicsRouteBuf : public std::streambuf {
public:
    PhysicsRouteBuf(std::streambuf* main_buf, std::streambuf* phys_buf)
        : main_(main_buf), phys_(phys_buf) {}
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lk(mtx_);
        for (std::streamsize i = 0; i < n; ++i) putLocked(s[i]);
        return n;
    }
    int overflow(int ch) override {
        if (ch == traits_type::eof()) return ch;
        std::lock_guard<std::mutex> lk(mtx_);
        putLocked(static_cast<char>(ch));
        return ch;
    }
    int sync() override {
        // Flush the underlying targets; keep any partial (newline-less) line
        // buffered so we can still route it once its tag is complete.
        if (main_) main_->pubsync();
        if (phys_) phys_->pubsync();
        return 0;
    }
private:
    void putLocked(char c) {
        line_ += c;
        if (c == '\n') { writeLine(); line_.clear(); }
    }
    void writeLine() {
        std::streambuf* dst = routeIsPhysics(line_) ? phys_ : main_;
        if (!dst) dst = main_ ? main_ : phys_;
        if (dst) dst->sputn(line_.data(),
                            static_cast<std::streamsize>(line_.size()));
    }
    bool routeIsPhysics(const std::string& s) {
        std::size_t i = 0;
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (i > 0) return last_phys_;   // indented continuation of prev line
        static const char* const kPhys[] = {
            "[collision", "[foot_", "[follow]", "[cw.", "[player.", "[QEM" };
        last_phys_ = false;
        for (const char* p : kPhys) {
            const std::size_t lp = std::strlen(p);
            if (s.size() >= lp && s.compare(0, lp, p) == 0) {
                last_phys_ = true;
                break;
            }
        }
        return last_phys_;
    }
    std::streambuf* main_ = nullptr;
    std::streambuf* phys_ = nullptr;
    std::string     line_;
    bool            last_phys_ = false;
    std::mutex      mtx_;
};
} // namespace

int main(int argc, char** argv) {
    // ── Route stdout to a log file ───────────────────────────────────
    // The engine prints a lot of per-frame debug ([player.*], [cw.*],
    // [foot_ray], [follow], …) plus one-shot load diagnostics
    // ([collision.*], [mesh.fbx], [QEM.holes], …).  That floods the
    // console, so send std::cout to a timestamped file under logs/.
    // std::cerr is left on the console for genuine errors.  The
    // ofstream is static so its buffer outlives every std::cout for the
    // whole run.  Run with ENGINE_LOG_CONSOLE=1 to keep stdout on the
    // terminal instead (e.g. for quick interactive debugging).
    static std::ofstream s_stdout_log;
    static std::ofstream s_physics_log;
    {
        char*  keep_console = nullptr;
        size_t keep_len     = 0;
        _dupenv_s(&keep_console, &keep_len, "ENGINE_LOG_CONSOLE");
        const bool to_console =
            keep_console && keep_len && keep_console[0] == '1';
        if (keep_console) free(keep_console);
        if (!to_console) {
            std::error_code ec;
            std::filesystem::create_directories("logs", ec);
            std::time_t now_t = std::time(nullptr);
            std::tm     tm_local{};
            localtime_s(&tm_local, &now_t);
            char fname[256];
            std::strftime(
                fname, sizeof(fname),
                "logs/engine_stdout_%Y-%m-%d_%H-%M-%S.log", &tm_local);
            // Physics / IK / collision lines are split into their own file.
            char pname[256];
            std::strftime(
                pname, sizeof(pname),
                "logs/physics_%Y-%m-%d_%H-%M-%S.log", &tm_local);
            s_stdout_log.open(fname);
            s_physics_log.open(pname);
            if (s_stdout_log.is_open() && s_physics_log.is_open()) {
                // Route physics/IK/collision lines to the physics log,
                // everything else to the main engine log.
                static PhysicsRouteBuf s_route(s_stdout_log.rdbuf(),
                                               s_physics_log.rdbuf());
                std::cout.rdbuf(&s_route);
                std::cerr << "[log] stdout -> " << fname
                          << "\n[log] physics/IK/collision -> " << pname
                          << "  (set ENGINE_LOG_CONSOLE=1 to keep both on "
                             "the console)" << std::endl;
            } else if (s_stdout_log.is_open()) {
                // Physics file failed to open -> single combined log.
                std::cout.rdbuf(s_stdout_log.rdbuf());
                std::cerr << "[log] stdout -> " << fname
                          << "  (physics split disabled; set "
                             "ENGINE_LOG_CONSOLE=1 to keep it on the console)"
                          << std::endl;
            }
        }
    }

    // Flip the global toggle before any mesh loads so the build path can
    // pick it up and produce ClusterMesh sidecars.
    if (parseClusterDebugFlag(argc, argv)) {
        engine::helper::clusterRenderingEnabled() = true;
        std::cout << "[CLUSTER] debug draw enabled via CLI flag\n";
    }
#if 0
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
#endif

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