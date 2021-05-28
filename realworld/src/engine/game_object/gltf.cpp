#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>

#include "engine/engine_helper.h"
#include "engine/game_object/gltf.h"
#include "engine/renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

#include "tiny_gltf.h"
#include "engine/tiny_mtx2.h"

namespace ego = engine::game_object;
namespace engine {

namespace {
static std::string getFilePathExtension(const std::string& file_name) {
    if (file_name.find_last_of(".") != std::string::npos)
        return file_name.substr(file_name.find_last_of(".") + 1);
    return "";
}

static void transformBbox(
    const glm::mat4& mat,
    const glm::vec3& bbox_min,
    const glm::vec3& bbox_max,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {

    glm::vec3 extent = bbox_max - bbox_min;
    glm::vec3 base = glm::vec3(mat * glm::vec4(bbox_min, 1.0f));
    output_bbox_min = base;
    output_bbox_max = base;
    auto mat_1 = glm::mat3(mat);
    glm::vec3 vec_x = mat_1 * glm::vec3(extent.x, 0, 0);
    glm::vec3 vec_y = mat_1 * glm::vec3(0, extent.y, 0);
    glm::vec3 vec_z = mat_1 * glm::vec3(0, 0, extent.z);

    glm::vec3 points[7];
    points[0] = base + vec_x;
    points[1] = base + vec_y;
    points[2] = base + vec_z;
    points[3] = points[0] + vec_y;
    points[4] = points[0] + vec_z;
    points[5] = points[1] + vec_z;
    points[6] = points[3] + vec_z;

    for (int i = 0; i < 7; i++) {
        output_bbox_min = min(output_bbox_min, points[i]);
        output_bbox_max = max(output_bbox_max, points[i]);
    }
}

static void calculateBbox(
    std::shared_ptr<ego::ObjectData>& gltf_object,
    int32_t node_idx,
    const glm::mat4& parent_matrix,
    glm::vec3& output_bbox_min,
    glm::vec3& output_bbox_max) {
    if (node_idx >= 0) {
        const auto& node = gltf_object->nodes_[node_idx];
        auto cur_matrix = parent_matrix;
        if (node.matrix) {
            cur_matrix *= *node.matrix;
        }
        if (node.mesh_idx >= 0) {
            glm::vec3 bbox_min, bbox_max;
            transformBbox(
                cur_matrix,
                gltf_object->meshes_[node.mesh_idx].bbox_min_,
                gltf_object->meshes_[node.mesh_idx].bbox_max_,
                bbox_min,
                bbox_max);
            output_bbox_min = min(output_bbox_min, bbox_min);
            output_bbox_max = max(output_bbox_max, bbox_max);
        }

        for (auto& child_idx : node.child_idx) {
            calculateBbox(gltf_object, child_idx, cur_matrix, output_bbox_min, output_bbox_max);
        }
    }
}

static void setupMeshState(
    const renderer::DeviceInfo& device_info,
    const tinygltf::Model& model,
    std::shared_ptr<ego::ObjectData>& gltf_object) {

    const auto& device = device_info.device;

    // Buffer
    {
        gltf_object->buffers_.resize(model.buffers.size());
        for (size_t i = 0; i < model.buffers.size(); i++) {
            auto buffer = model.buffers[i];
            renderer::Helper::createBufferWithSrcData(
                device_info,
                SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT) |
                SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
                buffer.data.size(),
                buffer.data.data(),
                gltf_object->buffers_[i].buffer,
                gltf_object->buffers_[i].memory);
        }
    }

    // Buffer views.
    {
        auto& buffer_views = gltf_object->buffer_views_;
        buffer_views.resize(model.bufferViews.size());

        for (size_t i = 0; i < model.bufferViews.size(); i++) {
            const tinygltf::BufferView& bufferView = model.bufferViews[i];
            buffer_views[i].buffer_idx = bufferView.buffer;
            buffer_views[i].offset = bufferView.byteOffset;
            buffer_views[i].range = bufferView.byteLength;
            buffer_views[i].stride = bufferView.byteStride;
        }
    }

    // allocate texture memory at first.
    gltf_object->textures_.resize(model.textures.size());

    // Material
    {
        gltf_object->materials_.resize(model.materials.size());
        for (size_t i_mat = 0; i_mat < model.materials.size(); i_mat++) {
            auto& dst_material = gltf_object->materials_[i_mat];
            const auto& src_material = model.materials[i_mat];

            dst_material.base_color_idx_ = src_material.pbrMetallicRoughness.baseColorTexture.index;
            dst_material.normal_idx_ = src_material.normalTexture.index;
            dst_material.metallic_roughness_idx_ = src_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            dst_material.emissive_idx_ = src_material.emissiveTexture.index;
            dst_material.occlusion_idx_ = src_material.occlusionTexture.index;

            if (dst_material.base_color_idx_ >= 0) {
                gltf_object->textures_[dst_material.base_color_idx_].linear = false;
            }

            if (dst_material.emissive_idx_ >= 0) {
                gltf_object->textures_[dst_material.emissive_idx_].linear = false;
            }

            device->createBuffer(
                sizeof(PbrMaterialParams),
                SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
                SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
                SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
                dst_material.uniform_buffer_.buffer,
                dst_material.uniform_buffer_.memory);

            PbrMaterialParams ubo{};
            ubo.base_color_factor = glm::vec4(
                src_material.pbrMetallicRoughness.baseColorFactor[0],
                src_material.pbrMetallicRoughness.baseColorFactor[1],
                src_material.pbrMetallicRoughness.baseColorFactor[2],
                src_material.pbrMetallicRoughness.baseColorFactor[3]);

            ubo.glossiness_factor = 1.0f;
            ubo.metallic_roughness_specular_factor = 1.0f;
            ubo.metallic_factor = static_cast<float>(src_material.pbrMetallicRoughness.metallicFactor);
            ubo.roughness_factor = static_cast<float>(src_material.pbrMetallicRoughness.roughnessFactor);
            ubo.alpha_cutoff = static_cast<float>(src_material.alphaCutoff);
            ubo.mip_count = 11;
            ubo.normal_scale = static_cast<float>(src_material.normalTexture.scale);
            ubo.occlusion_strength = static_cast<float>(src_material.occlusionTexture.strength);

            ubo.emissive_factor = glm::vec3(
                src_material.emissiveFactor[0],
                src_material.emissiveFactor[1],
                src_material.emissiveFactor[2]);

            ubo.uv_set_flags = glm::vec4(0, 0, 0, 0);
            ubo.exposure = 1.0f;
            ubo.material_features = (src_material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0 ? FEATURE_HAS_METALLIC_ROUGHNESS_MAP : 0) | FEATURE_MATERIAL_METALLICROUGHNESS;
            ubo.material_features |= (src_material.pbrMetallicRoughness.baseColorTexture.index >= 0 ? FEATURE_HAS_BASE_COLOR_MAP : 0);
            ubo.material_features |= (src_material.emissiveTexture.index >= 0 ? FEATURE_HAS_EMISSIVE_MAP : 0);
            ubo.material_features |= (src_material.occlusionTexture.index >= 0 ? FEATURE_HAS_OCCLUSION_MAP : 0);
            ubo.material_features |= (src_material.normalTexture.index >= 0 ? FEATURE_HAS_NORMAL_MAP : 0);
            ubo.tonemap_type = TONEMAP_DEFAULT;
            ubo.specular_factor = vec3(1.0f, 1.0f, 1.0f);
            ubo.lights[0].type = LightType_Directional;
            ubo.lights[0].color = glm::vec3(1, 0, 0);
            ubo.lights[0].direction = glm::vec3(0, 0, -1);
            ubo.lights[0].intensity = 1.0f;
            ubo.lights[0].position = glm::vec3(0, 0, 0);

            device->updateBufferMemory(dst_material.uniform_buffer_.memory, sizeof(ubo), &ubo);
        }
    }

    // Texture
    {
        for (size_t i_tex = 0; i_tex < model.textures.size(); i_tex++) {
            auto& dst_tex = gltf_object->textures_[i_tex];
            const auto& src_tex = model.textures[i_tex];
            const auto& src_img = model.images[i_tex];
            auto format = renderer::Format::R8G8B8A8_UNORM;
            renderer::Helper::create2DTextureImage(
                device_info,
                format,
                src_img.width,
                src_img.height,
                src_img.component,
                src_img.image.data(),
                dst_tex.image,
                dst_tex.memory);

            dst_tex.view = device->createImageView(
                dst_tex.image,
                renderer::ImageViewType::VIEW_2D,
                format,
                SET_FLAG_BIT(ImageAspect, COLOR_BIT));
        }
    }
}

static void setupMesh(
    const tinygltf::Model& model,
    const tinygltf::Mesh& mesh,
    ego::MeshInfo& mesh_info) {

    for (size_t i = 0; i < mesh.primitives.size(); i++) {
        const tinygltf::Primitive& primitive = mesh.primitives[i];

        ego::PrimitiveInfo primitive_info;
        primitive_info.tag_.restart_enable = false;
        primitive_info.material_idx_ = primitive.material;

        auto mode = renderer::PrimitiveTopology::MAX_ENUM;
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
            mode = renderer::PrimitiveTopology::TRIANGLE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            mode = renderer::PrimitiveTopology::TRIANGLE_STRIP;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            mode = renderer::PrimitiveTopology::TRIANGLE_FAN;
        }
        else if (primitive.mode == TINYGLTF_MODE_POINTS) {
            mode = renderer::PrimitiveTopology::POINT_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE) {
            mode = renderer::PrimitiveTopology::LINE_LIST;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
            mode = renderer::PrimitiveTopology::LINE_STRIP;
        }
        else {
            assert(0);
        }

        primitive_info.tag_.topology = static_cast<uint32_t>(mode);

        if (primitive.indices < 0) return;

        std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
        std::map<std::string, int>::const_iterator itEnd(primitive.attributes.end());

        uint32_t dst_binding = 0;
        for (; it != itEnd; it++) {
            assert(it->second >= 0);
            const tinygltf::Accessor& accessor = model.accessors[it->second];

            dst_binding = static_cast<uint32_t>(primitive_info.binding_list_.size());
            assert(dst_binding < VINPUT_INSTANCE_BINDING_START);
            primitive_info.binding_list_.push_back(accessor.bufferView);

            engine::renderer::VertexInputBindingDescription binding = {};
            binding.binding = dst_binding;
            binding.stride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
            binding.input_rate = renderer::VertexInputRate::VERTEX;
            primitive_info.binding_descs_.push_back(binding);

            engine::renderer::VertexInputAttributeDescription attribute = {};
            attribute.buffer_view = accessor.bufferView;
            attribute.binding = dst_binding;
            attribute.offset = 0;
            attribute.buffer_offset = accessor.byteOffset + model.bufferViews[accessor.bufferView].byteOffset;
            if (it->first.compare("POSITION") == 0) {
                attribute.location = VINPUT_POSITION;
                primitive_info.bbox_min_ = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
                primitive_info.bbox_max_ = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
                mesh_info.bbox_min_ = min(mesh_info.bbox_min_, primitive_info.bbox_min_);
                mesh_info.bbox_max_ = max(mesh_info.bbox_max_, primitive_info.bbox_max_);
            }
            else if (it->first.compare("TEXCOORD_0") == 0) {
                attribute.location = VINPUT_TEXCOORD0;
                primitive_info.tag_.has_texcoord_0 = true;
            }
            else if (it->first.compare("NORMAL") == 0) {
                attribute.location = VINPUT_NORMAL;
                primitive_info.tag_.has_normal = true;
            }
            else if (it->first.compare("TANGENT") == 0) {
                attribute.location = VINPUT_TANGENT;
                primitive_info.tag_.has_tangent = true;
            }
            else if (it->first.compare("TEXCOORD_1") == 0) {
                attribute.location = VINPUT_TEXCOORD1;
            }
            else if (it->first.compare("COLOR") == 0) {
                attribute.location = VINPUT_COLOR;
            }
            else if (it->first.compare("JOINTS_0") == 0) {
                attribute.location = VINPUT_JOINTS_0;
                primitive_info.tag_.has_skin_set_0 = true;
            }
            else if (it->first.compare("WEIGHTS_0") == 0) {
                attribute.location = VINPUT_WEIGHTS_0;
                primitive_info.tag_.has_skin_set_0 = true;
            }
            else {
                // add support here.
                assert(0);
            }

            if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_FLOAT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = engine::renderer::Format::R32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = engine::renderer::Format::R32G32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = engine::renderer::Format::R32G32B32_SFLOAT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = engine::renderer::Format::R32G32B32A32_SFLOAT;
                }
                else {
                    assert(0);
                }
            }
            else if (accessor.componentType == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    attribute.format = engine::renderer::Format::R16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    attribute.format = engine::renderer::Format::R16G16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    attribute.format = engine::renderer::Format::R16G16B16_UINT;
                }
                else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    attribute.format = engine::renderer::Format::R16G16B16A16_UINT;
                }
                else {
                    assert(0);
                }

            }
            else {
                // add support here.
                assert(0);
            }
            primitive_info.attribute_descs_.push_back(attribute);
            dst_binding++;
        }

        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        primitive_info.index_desc_.binding = indexAccessor.bufferView;
        primitive_info.index_desc_.offset = indexAccessor.byteOffset;
        primitive_info.index_desc_.index_type = 
            indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 
            renderer::IndexType::UINT16 : 
            renderer::IndexType::UINT32;
        primitive_info.index_desc_.index_count = indexAccessor.count;

        primitive_info.generateHash();
        mesh_info.primitives_.push_back(primitive_info);
    }
}

static void setupMeshes(
    const tinygltf::Model& model,
    std::shared_ptr<ego::ObjectData>& gltf_object) {
    gltf_object->meshes_.resize(model.meshes.size());
    for (int i_mesh = 0; i_mesh < model.meshes.size(); i_mesh++) {
        setupMesh(model, model.meshes[i_mesh], gltf_object->meshes_[i_mesh]);
    }
}

static void setupNode(
    const tinygltf::Model& model,
    const tinygltf::Node& node,
    ego::NodeInfo& node_info) {

    bool has_matrix = false;
    glm::mat4 mesh_matrix(1);
    if (node.matrix.size() == 16) {
        // Use 'matrix' attribute
        const auto& m = node.matrix.data();
        mesh_matrix =
            glm::mat4(m[0], m[1], m[2], m[3],
                m[4], m[5], m[6], m[7],
                m[8], m[9], m[10], m[11],
                m[12], m[13], m[14], m[15]);
        has_matrix = true;
    }
    else {
        // Assume Trans x Rotate x Scale order
        if (node.scale.size() == 3) {
            mesh_matrix = glm::scale(mesh_matrix, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
            has_matrix = true;
        }

        if (node.rotation.size() == 4) {
            mesh_matrix = glm::rotate(mesh_matrix, glm::radians(static_cast<float>(node.rotation[0])), glm::vec3(node.rotation[1], node.rotation[2], node.rotation[3]));
            has_matrix = true;
        }

        if (node.translation.size() == 3) {
            mesh_matrix = glm::translate(mesh_matrix, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
            has_matrix = true;
        }
    }

    if (has_matrix) {
        node_info.matrix = std::make_shared<glm::mat4>(mesh_matrix);
    }

    node_info.mesh_idx = node.mesh;

    // Draw child nodes.
    node_info.child_idx.resize(node.children.size());
    for (size_t i = 0; i < node.children.size(); i++) {
        assert(node.children[i] < model.nodes.size());
        node_info.child_idx[i] = node.children[i];
    }
}

static void setupNodes(
    const tinygltf::Model& model, 
    std::shared_ptr<ego::ObjectData>& gltf_object) {
    gltf_object->nodes_.resize(model.nodes.size());
    for (int i_node = 0; i_node < model.nodes.size(); i_node++) {
        setupNode(model, model.nodes[i_node], gltf_object->nodes_[i_node]);
    }
}

static void setupModel(
    const tinygltf::Model& model,
    std::shared_ptr<ego::ObjectData>& gltf_object) {
    assert(model.scenes.size() > 0);
    gltf_object->default_scene_ = model.defaultScene;
    gltf_object->scenes_.resize(model.scenes.size());
    for (uint32_t i_scene = 0; i_scene < model.scenes.size(); i_scene++) {
        const auto& src_scene = model.scenes[i_scene];
        auto& dst_scene = gltf_object->scenes_[i_scene];

        dst_scene.nodes_.resize(src_scene.nodes.size());
        for (size_t i_node = 0; i_node < src_scene.nodes.size(); i_node++) {
            dst_scene.nodes_[i_node] = src_scene.nodes[i_node];
        }
    }
}

static std::shared_ptr<ego::ObjectData> loadGltfModel(
    const renderer::DeviceInfo& device_info,
    const std::string& input_filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::string ext = getFilePathExtension(input_filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret =
            loader.LoadBinaryFromFile(&model, &err, &warn, input_filename.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
    }

    if (!warn.empty()) {
        std::cout << "Warn: " << warn.c_str() << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err.c_str() << std::endl;
    }
    if (!ret) {
        std::cout << "Failed to load .glTF : " << input_filename << std::endl;
        return nullptr;
    }

    auto gltf_object = std::make_shared<ego::ObjectData>(device_info.device);
    gltf_object->meshes_.reserve(model.meshes.size());

    setupMeshState(device_info, model, gltf_object);
    setupMeshes(model, gltf_object);
    setupNodes(model, gltf_object);
    setupModel(model, gltf_object);
    for (auto& root : gltf_object->scenes_) {
        for (auto& node : root.nodes_) {
            calculateBbox(gltf_object, root.nodes_[0], glm::mat4(1.0f), root.bbox_min_, root.bbox_max_);
        }
    }

    // init indirect draw buffer.
    uint32_t num_prims = 0;
    for (auto& mesh : gltf_object->meshes_) {
        for (auto& prim : mesh.primitives_) {
            prim.indirect_draw_cmd_ofs_ =
                num_prims * sizeof(renderer::DrawIndexedIndirectCommand);
            num_prims++;
        }
    }

    std::vector<renderer::DrawIndexedIndirectCommand> indirect_draw_cmd_buffer(num_prims);
    uint32_t prim_idx = 0;
    for (const auto& mesh : gltf_object->meshes_) {
        for (const auto& prim : mesh.primitives_) {
            indirect_draw_cmd_buffer[prim_idx].first_index = 0;
            indirect_draw_cmd_buffer[prim_idx].first_instance = 0;
            indirect_draw_cmd_buffer[prim_idx].index_count =
                static_cast<uint32_t>(prim.index_desc_.index_count);
            indirect_draw_cmd_buffer[prim_idx].instance_count = 2;
            indirect_draw_cmd_buffer[prim_idx].vertex_offset = 0;
            prim_idx++;
        }
    }

    renderer::Helper::createBufferWithSrcData(
        device_info,
        SET_FLAG_BIT(BufferUsage, INDIRECT_BUFFER_BIT),
        num_prims * sizeof(renderer::DrawIndexedIndirectCommand),
        indirect_draw_cmd_buffer.data(),
        gltf_object->indirect_draw_cmd_.buffer,
        gltf_object->indirect_draw_cmd_.memory);

    return gltf_object;
}

static std::vector<renderer::TextureDescriptor> addGltfTextures(
    const std::shared_ptr<ego::ObjectData>& gltf_object,
    const ego::MaterialInfo& material,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex) {

    const auto& description_set = material.desc_set_;

    std::vector<renderer::TextureDescriptor> descriptor_writes;
    descriptor_writes.reserve(10);
    auto& textures = gltf_object->textures_;

    auto& black_tex = renderer::Helper::getBlackTexture();
    auto& white_tex = renderer::Helper::getWhiteTexture();

    // base color.
    auto& base_color_tex_view = material.base_color_idx_ < 0 ? black_tex : textures[material.base_color_idx_];
    renderer::Helper::addOneTexture(descriptor_writes, BASE_COLOR_TEX_INDEX, texture_sampler, base_color_tex_view.view, description_set);

    // normal.
    auto& normal_tex_view = material.normal_idx_ < 0 ? black_tex : textures[material.normal_idx_];
    renderer::Helper::addOneTexture(descriptor_writes, NORMAL_TEX_INDEX, texture_sampler, normal_tex_view.view, description_set);

    // metallic roughness.
    auto& metallic_roughness_tex = material.metallic_roughness_idx_ < 0 ? black_tex : textures[material.metallic_roughness_idx_];
    renderer::Helper::addOneTexture(descriptor_writes, METAL_ROUGHNESS_TEX_INDEX, texture_sampler, metallic_roughness_tex.view, description_set);

    // emisive.
    auto& emissive_tex = material.emissive_idx_ < 0 ? black_tex : textures[material.emissive_idx_];
    renderer::Helper::addOneTexture(descriptor_writes, EMISSIVE_TEX_INDEX, texture_sampler, emissive_tex.view, description_set);

    // occlusion.
    auto& occlusion_tex = material.occlusion_idx_ < 0 ? white_tex : textures[material.occlusion_idx_];
    renderer::Helper::addOneTexture(descriptor_writes, OCCLUSION_TEX_INDEX, texture_sampler, occlusion_tex.view, description_set);

    // thin_film_lut.
    renderer::Helper::addOneTexture(descriptor_writes, THIN_FILM_LUT_INDEX, texture_sampler, thin_film_lut_tex.view, description_set);

    return descriptor_writes;
}

static void updateDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& desc_set_layout,
    const std::shared_ptr<ego::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex)
{
    for (uint32_t i_mat = 0; i_mat < gltf_object->materials_.size(); i_mat++) {
        auto& material = gltf_object->materials_[i_mat];
        material.desc_set_ = device->createDescriptorSets(
            descriptor_pool, desc_set_layout, 1)[0];

        std::vector<renderer::BufferDescriptor> material_buffer_descs;
        renderer::Helper::addOneBuffer(
            material_buffer_descs,
            PBR_CONSTANT_INDEX,
            material.uniform_buffer_.buffer,
            material.desc_set_,
            renderer::DescriptorType::UNIFORM_BUFFER,
            sizeof(PbrMaterialParams));

        // create a global ibl texture descriptor set.
        auto material_tex_descs = addGltfTextures(gltf_object, material, texture_sampler, thin_film_lut_tex);

        device->updateDescriptorSets(material_tex_descs, material_buffer_descs);
    }
}

static void drawMesh(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<ego::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::PipelineLayout>& gltf_pipeline_layout,
    const renderer::DescriptorSetList& desc_set_list,
    const ego::MeshInfo& mesh_info,
    const ModelParams& model_params) {
    for (const auto& prim : mesh_info.primitives_) {
        const auto& attrib_list = prim.attribute_descs_;

        std::vector<std::shared_ptr<renderer::Buffer>> buffers(attrib_list.size());
        std::vector<uint64_t> offsets(attrib_list.size());
        for (int i_attrib = 0; i_attrib < attrib_list.size(); i_attrib++) {
            const auto& buffer_view = gltf_object->buffer_views_[attrib_list[i_attrib].buffer_view];
            buffers[i_attrib] = gltf_object->buffers_[buffer_view.buffer_idx].buffer;
            offsets[i_attrib] = attrib_list[i_attrib].buffer_offset;
        }
        cmd_buf->bindVertexBuffers(0, buffers, offsets);
        const auto& index_buffer_view = gltf_object->buffer_views_[prim.index_desc_.binding];
        cmd_buf->bindIndexBuffer(gltf_object->buffers_[index_buffer_view.buffer_idx].buffer,
            prim.index_desc_.offset + index_buffer_view.offset,
            prim.index_desc_.index_type);

        renderer::DescriptorSetList desc_sets = desc_set_list;
        if (prim.material_idx_ >= 0) {
            const auto& material = gltf_object->materials_[prim.material_idx_];
            desc_sets.push_back(material.desc_set_);
        }
        cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_layout, desc_sets);

        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, VERTEX_BIT), gltf_pipeline_layout, &model_params, sizeof(model_params));

        //cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
        cmd_buf->drawIndexedIndirect(
            gltf_object->indirect_draw_cmd_,
            prim.indirect_draw_cmd_ofs_);
    }
}

static void drawNodes(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<ego::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::PipelineLayout>& gltf_pipeline_layout,
    const renderer::DescriptorSetList& desc_set_list,
    int32_t node_idx,
    const glm::mat4& parent_matrix) {
    if (node_idx >= 0) {
        const auto& node = gltf_object->nodes_[node_idx];
        auto cur_matrix = parent_matrix;
        if (node.matrix) {
            cur_matrix *= *node.matrix;
        }
        if (node.mesh_idx >= 0) {
            ModelParams model_params{};
            model_params.model_mat = cur_matrix;
            auto invert_mat = inverse(model_params.model_mat);
            model_params.normal_mat = transpose(invert_mat);
            drawMesh(cmd_buf,
                gltf_object,
                gltf_pipeline_layout,
                desc_set_list,
                gltf_object->meshes_[node.mesh_idx],
                model_params);
        }

        for (auto& child_idx : node.child_idx) {
            drawNodes(cmd_buf,
                gltf_object,
                gltf_pipeline_layout,
                desc_set_list,
                child_idx,
                cur_matrix);
        }
    }
}

// material texture descriptor set layout.
static std::shared_ptr<renderer::DescriptorSetLayout> createDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(7);

    renderer::DescriptorSetLayoutBinding ubo_pbr_layout_binding{};
    ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
    ubo_pbr_layout_binding.descriptor_count = 1;
    ubo_pbr_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
    ubo_pbr_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    ubo_pbr_layout_binding.immutable_samplers = nullptr; // Optional
    bindings.push_back(ubo_pbr_layout_binding);

    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(BASE_COLOR_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createGltfPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& material_desc_set_layout) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, VERTEX_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(ModelParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(material_desc_set_layout);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static renderer::ShaderModuleList getGltfShaderModules(
    std::shared_ptr<renderer::Device> device,
    bool has_normals,
    bool has_tangent,
    bool has_texcoord_0,
    bool has_skin_set_0) {
    renderer::ShaderModuleList shader_modules(2);
    std::string feature_str = std::string(has_texcoord_0 ? "_TEX" : "") +
        (has_tangent ? "_TN" : (has_normals ? "_N" : "")) +
        (has_skin_set_0 ? "_SKIN" : "");
    uint64_t vert_code_size, frag_code_size;
    auto vert_shader_code = engine::helper::readFile("lib/shaders/base_vert" + feature_str + ".spv", vert_code_size);
    auto frag_shader_code = engine::helper::readFile("lib/shaders/base_frag" + feature_str + ".spv", frag_code_size);

    shader_modules[0] = device->createShaderModule(vert_code_size, vert_shader_code.data());
    shader_modules[1] = device->createShaderModule(frag_code_size, frag_shader_code.data());

    return shader_modules;
}

static std::shared_ptr<renderer::Pipeline> createGltfPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size,
    const ego::PrimitiveInfo& primitive) {
    auto shader_modules = getGltfShaderModules(
        device,
        primitive.tag_.has_normal,
        primitive.tag_.has_tangent,
        primitive.tag_.has_texcoord_0,
        primitive.tag_.has_skin_set_0);

    renderer::PipelineInputAssemblyStateCreateInfo topology_info;
    topology_info.restart_enable = primitive.tag_.restart_enable;
    topology_info.topology = static_cast<renderer::PrimitiveTopology>(primitive.tag_.topology);

    auto binding_descs = primitive.binding_descs_;
    auto attribute_descs = primitive.attribute_descs_;

    renderer::VertexInputBindingDescription desc;
    desc.binding = VINPUT_INSTANCE_BINDING_START;
    desc.input_rate = renderer::VertexInputRate::INSTANCE;
    desc.stride = sizeof(game_object::InstanceDataInfo);
    binding_descs.push_back(desc);

    renderer::VertexInputAttributeDescription attr;
    attr.binding = VINPUT_INSTANCE_BINDING_START;
    attr.buffer_offset = 0;
    attr.format = renderer::Format::R32G32B32_SFLOAT;
    attr.buffer_view = 0;
    attr.location = IINPUT_MAT_ROT_0;
    attr.offset = offsetof(game_object::InstanceDataInfo, mat_rot_0);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_1;
    attr.offset = offsetof(game_object::InstanceDataInfo, mat_rot_1);
    attribute_descs.push_back(attr);
    attr.location = IINPUT_MAT_ROT_2;
    attr.offset = offsetof(game_object::InstanceDataInfo, mat_rot_2);
    attribute_descs.push_back(attr);
    attr.format = renderer::Format::R32G32B32A32_SFLOAT;
    attr.location = IINPUT_MAT_POS_SCALE;
    attr.offset = offsetof(game_object::InstanceDataInfo, mat_pos_scale);
    attribute_descs.push_back(attr);

    auto gltf_pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        binding_descs,
        attribute_descs,
        topology_info,
        graphic_pipeline_info,
        shader_modules,
        display_size);

    for (auto& shader_module : shader_modules) {
        device->destroyShaderModule(shader_module);
    }

    return gltf_pipeline;
}

} // namespace

namespace game_object {

// static member definition.
std::shared_ptr<renderer::DescriptorSetLayout> GltfObject::material_desc_set_layout_;
std::shared_ptr<renderer::PipelineLayout> GltfObject::gltf_pipeline_layout_;
std::unordered_map<size_t, std::shared_ptr<renderer::Pipeline>> GltfObject::gltf_pipeline_list_;
std::unordered_map<std::string, std::shared_ptr<ObjectData>> GltfObject::object_list_;

void PrimitiveInfo::generateHash() {
    hash_ = std::hash<uint32_t>{}(tag_.data);
    for (auto& item : binding_descs_) {
        hash_combine(hash_, item.binding);
        hash_combine(hash_, item.input_rate);
        hash_combine(hash_, item.stride);
    }
    for (auto& item : attribute_descs_) {
        hash_combine(hash_, item.binding);
        hash_combine(hash_, item.buffer_offset);
        hash_combine(hash_, item.buffer_view);
        hash_combine(hash_, item.format);
        hash_combine(hash_, item.location);
        hash_combine(hash_, item.offset);
    }
}

void ObjectData::destroy() {
    for (auto& texture : textures_) {
        texture.destroy(device_);
    }

    for (auto& material : materials_) {
        material.uniform_buffer_.destroy(device_);
    }

    for (auto& buffer : buffers_) {
        buffer.destroy(device_);
    }
}

GltfObject::GltfObject(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex,
    const std::string& file_name,
    const glm::uvec2& display_size,
    glm::mat4 location/* = glm::mat4(1.0f)*/) {

    location_ = location;

    auto result = object_list_.find(file_name);
    if (result == object_list_.end()) {
        object_ = loadGltfModel(device_info, file_name);

        updateDescriptorSets(
            device_info.device,
            descriptor_pool,
            material_desc_set_layout_,
            object_,
            texture_sampler,
            thin_film_lut_tex);

        const auto& primitive = object_->meshes_[0].primitives_[0];
        auto hash_value = primitive.getHash();
        auto result = gltf_pipeline_list_.find(hash_value);
        if (result == gltf_pipeline_list_.end()) {
            gltf_pipeline_list_[hash_value] =
                createGltfPipeline(
                    device_info.device,
                    render_pass,
                    gltf_pipeline_layout_,
                    graphic_pipeline_info,
                    display_size,
                    primitive);
        }

        object_list_[file_name] = object_;
    }
    else {
        object_ = result->second;
    }

    std::vector<InstanceDataInfo> instance_data(2);
    instance_data[0].mat_rot_0 = location[0];
    instance_data[0].mat_rot_1 = location[1];
    instance_data[0].mat_rot_2 = location[2];
    instance_data[0].mat_pos_scale = location[3];

    instance_data[1].mat_rot_0 = location[0];
    instance_data[1].mat_rot_1 = location[1];
    instance_data[1].mat_rot_2 = location[2];
    instance_data[1].mat_pos_scale = location[3] + glm::vec4(1, 0, 0, 0);

    renderer::Helper::createBufferWithSrcData(
        device_info,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        instance_data.size() * sizeof(InstanceDataInfo),
        instance_data.data(),
        instance_buffer_.buffer,
        instance_buffer_.memory);
}

void GltfObject::initStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts) {
    if (material_desc_set_layout_ == nullptr) {
        material_desc_set_layout_ =
            createDescriptorSetLayout(device);
    }

    if (gltf_pipeline_layout_ == nullptr) {
        assert(material_desc_set_layout_);
        gltf_pipeline_layout_ =
            createGltfPipelineLayout(
                device,
                global_desc_set_layouts,
                material_desc_set_layout_);
    }
}

void GltfObject::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {

    if (gltf_pipeline_layout_) {
        device->destroyPipelineLayout(gltf_pipeline_layout_);
        assert(material_desc_set_layout_);
        gltf_pipeline_layout_ =
            createGltfPipelineLayout(device, global_desc_set_layouts, material_desc_set_layout_);
    }

    gltf_pipeline_list_.clear();

    for (auto& object : object_list_) {
        const auto& primitive = object.second->meshes_[0].primitives_[0];
        auto hash_value = primitive.getHash();
        auto result = gltf_pipeline_list_.find(hash_value);
        if (result == gltf_pipeline_list_.end()) {
            gltf_pipeline_list_[hash_value] = 
                createGltfPipeline(
                    device,
                    render_pass,
                    gltf_pipeline_layout_,
                    graphic_pipeline_info,
                    display_size,
                    primitive);
        }
    }
}

void GltfObject::generateDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& thin_film_lut_tex) {

    for (auto& object : object_list_) {
        updateDescriptorSets(
            device,
            descriptor_pool,
            material_desc_set_layout_,
            object.second,
            texture_sampler,
            thin_film_lut_tex);
    }
}

void GltfObject::destoryStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(material_desc_set_layout_);
    device->destroyPipelineLayout(gltf_pipeline_layout_);
    gltf_pipeline_list_.clear();
    object_list_.clear();
}

#if 0
void GltfObject::generateInstanceBuffers(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    uint32_t vx_count = segment_count_.x + 1;
    uint32_t vy_count = segment_count_.y + 1;

    cmd_buf->addBufferBarrier(
        vertex_buffer_.buffer,
        { SET_FLAG_BIT(Access, VERTEX_ATTRIBUTE_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        vertex_buffer_size_);

    cmd_buf->addBufferBarrier(
        index_buffer_.buffer,
        { SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        index_buffer_size_);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, tile_creator_pipeline_);
    TileParams tile_params = {};
    tile_params.min = min_;
    tile_params.max = max_;
    tile_params.segment_count = segment_count_;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        tile_creator_pipeline_layout_,
        &tile_params,
        sizeof(tile_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        tile_creator_pipeline_layout_,
        { buffer_desc_set_ });

    cmd_buf->dispatch((vx_count + 7) / 8, (vy_count + 7) / 8, 1);

    cmd_buf->addBufferBarrier(
        vertex_buffer_.buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, VERTEX_ATTRIBUTE_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        vertex_buffer_size_);

    cmd_buf->addBufferBarrier(
        index_buffer_.buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, INDIRECT_COMMAND_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_INPUT_BIT) },
        index_buffer_size_);
}
#endif

void GltfObject::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list) {

    const auto& primitive = object_->meshes_[0].primitives_[0];
    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        gltf_pipeline_list_[primitive.getHash()]);

    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = instance_buffer_.buffer;
    offsets[0] = 0;
    cmd_buf->bindVertexBuffers(VINPUT_INSTANCE_BINDING_START, buffers, offsets);

    int32_t root_node = object_->default_scene_ >= 0 ? object_->default_scene_ : 0;
    for (auto node_idx : object_->scenes_[root_node].nodes_) {
        drawNodes(
            cmd_buf,
            object_,
            gltf_pipeline_layout_,
            desc_set_list,
            node_idx,
            glm::mat4(1));
    }
}

} // game_object
} // engine