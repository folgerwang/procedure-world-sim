#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>

#include "gltf.h"
#include "../shaders/global_definition.glsl.h"

#include "tiny_gltf.h"
#include "tiny_mtx2.h"

namespace engine {
namespace {
static std::string getFilePathExtension(const std::string& file_name) {
    if (file_name.find_last_of(".") != std::string::npos)
        return file_name.substr(file_name.find_last_of(".") + 1);
    return "";
}

static void setupMeshState(
    const renderer::DeviceInfo& device_info,
    const tinygltf::Model& model,
    std::shared_ptr<renderer::ObjectData>& gltf_object) {

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
    renderer::MeshInfo& mesh_info) {

    for (size_t i = 0; i < mesh.primitives.size(); i++) {
        const tinygltf::Primitive& primitive = mesh.primitives[i];

        renderer::PrimitiveInfo primitive_info;
        primitive_info.topology_info_.restart_enable = false;
        primitive_info.material_idx_ = primitive.material;

        auto& mode = primitive_info.topology_info_.topology;
        mode = renderer::PrimitiveTopology::MAX_ENUM;
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

        if (primitive.indices < 0) return;

        std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
        std::map<std::string, int>::const_iterator itEnd(primitive.attributes.end());

        uint32_t dst_binding = 0;
        for (; it != itEnd; it++) {
            assert(it->second >= 0);
            const tinygltf::Accessor& accessor = model.accessors[it->second];

            dst_binding = static_cast<uint32_t>(primitive_info.binding_list_.size());
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
                primitive_info.has_texcoord_0_ = true;
            }
            else if (it->first.compare("NORMAL") == 0) {
                attribute.location = VINPUT_NORMAL;
                primitive_info.has_normal_ = true;
            }
            else if (it->first.compare("TANGENT") == 0) {
                attribute.location = VINPUT_TANGENT;
                primitive_info.has_tangent_ = true;
            }
            else if (it->first.compare("TEXCOORD_1") == 0) {
                attribute.location = VINPUT_TEXCOORD1;
            }
            else if (it->first.compare("COLOR") == 0) {
                attribute.location = VINPUT_COLOR;
            }
            else if (it->first.compare("JOINTS_0") == 0) {
                attribute.location = VINPUT_JOINTS_0;
                primitive_info.has_skin_set_0_ = true;
            }
            else if (it->first.compare("WEIGHTS_0") == 0) {
                attribute.location = VINPUT_WEIGHTS_0;
                primitive_info.has_skin_set_0_ = true;
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
        primitive_info.index_desc_.index_type = indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? renderer::IndexType::UINT16 : renderer::IndexType::UINT32;
        primitive_info.index_desc_.index_count = indexAccessor.count;

        mesh_info.primitives_.push_back(primitive_info);
    }
}

static void setupMeshes(const tinygltf::Model& model, std::shared_ptr<renderer::ObjectData>& gltf_object) {
    gltf_object->meshes_.resize(model.meshes.size());
    for (int i_mesh = 0; i_mesh < model.meshes.size(); i_mesh++) {
        setupMesh(model, model.meshes[i_mesh], gltf_object->meshes_[i_mesh]);
    }
}

static void setupNode(
    const tinygltf::Model& model,
    const tinygltf::Node& node,
    renderer::NodeInfo& node_info) {

    bool has_matrix = false;
    glm::mat4 mesh_matrix(1);
    if (node.matrix.size() == 16) {
        // Use 'matrix' attribute
        const auto& m = node.matrix.data();
        mesh_matrix = glm::mat4(m[0], m[1], m[2], m[3],
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

static void setupNodes(const tinygltf::Model& model, std::shared_ptr<renderer::ObjectData>& gltf_object) {
    gltf_object->nodes_.resize(model.nodes.size());
    for (int i_node = 0; i_node < model.nodes.size(); i_node++) {
        setupNode(model, model.nodes[i_node], gltf_object->nodes_[i_node]);
    }
}

static void setupModel(
    const tinygltf::Model& model,
    std::shared_ptr<renderer::ObjectData>& gltf_object) {
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
    std::shared_ptr<renderer::ObjectData>& gltf_object,
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

void drawMesh(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<renderer::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::PipelineLayout>& gltf_pipeline_layout,
    const std::shared_ptr<renderer::DescriptorSet>& global_tex_desc_set,
    const std::shared_ptr<renderer::DescriptorSet>& src_desc_set,
    const renderer::MeshInfo& mesh_info,
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

        renderer::DescriptorSetList desc_sets{ global_tex_desc_set, src_desc_set };
        if (prim.material_idx_ >= 0) {
            const auto& material = gltf_object->materials_[prim.material_idx_];
            desc_sets.push_back(material.desc_set_);
        }
        cmd_buf->bindDescriptorSets(renderer::PipelineBindPoint::GRAPHICS, gltf_pipeline_layout, desc_sets);

        cmd_buf->pushConstants(SET_FLAG_BIT(ShaderStage, VERTEX_BIT), gltf_pipeline_layout, &model_params, sizeof(model_params));

        cmd_buf->drawIndexed(static_cast<uint32_t>(prim.index_desc_.index_count));
    }
}

}

namespace renderer {

std::shared_ptr<renderer::ObjectData> loadGltfModel(
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

    auto gltf_object = std::make_shared<renderer::ObjectData>();
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

    return gltf_object;
}

std::vector<renderer::TextureDescriptor> addGltfTextures(
    const std::shared_ptr<renderer::ObjectData>& gltf_object,
    const renderer::MaterialInfo& material,
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

void drawNodes(
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const std::shared_ptr<renderer::ObjectData>& gltf_object,
    const std::shared_ptr<renderer::PipelineLayout>& gltf_pipeline_layout,
    const std::shared_ptr<renderer::DescriptorSet>& global_tex_desc_set,
    const std::shared_ptr<renderer::DescriptorSet>& src_desc_set,
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
                global_tex_desc_set,
                src_desc_set,
                gltf_object->meshes_[node.mesh_idx],
                model_params);
        }

        for (auto& child_idx : node.child_idx) {
            drawNodes(cmd_buf,
                gltf_object,
                gltf_pipeline_layout,
                global_tex_desc_set,
                src_desc_set,
                child_idx,
                cur_matrix);
        }
    }
}

void ObjectData::destroy(const std::shared_ptr<Device>& device) {
    for (auto& texture : textures_) {
        texture.destroy(device);
    }

    for (auto& material : materials_) {
        material.uniform_buffer_.destroy(device);
    }

    for (auto& buffer : buffers_) {
        buffer.destroy(device);
    }
}

} // renderer
} // work