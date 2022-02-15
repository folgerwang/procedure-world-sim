const uint32_t g_object_count = 3;
struct BottomlevelDataInfo {
    er::BufferInfo   vertex_buffer{};
    er::BufferInfo   index_buffer{};
    er::BufferInfo   transform_buffer{};
    er::BufferInfo   as_buffer{};
    er::BufferInfo   scratch_buffer{};

    er::AccelerationStructure   as_handle{};
    er::DeviceAddress   as_device_address = 0;
};

void initBottomLevelDataInfo(
    const er::DeviceInfo& device_info,
    BottomlevelDataInfo& bl_data_info) {
    // Setup vertices for a single triangle
    struct Vertex {
        float pos[3];
    };

    std::vector<Vertex> vertices = {
    { {  1.0f,  1.0f, 0.0f } },
    { { -1.0f,  1.0f, 0.0f } },
    { {  0.0f, -1.0f, 0.0f } }
    };

    // Setup indices
    std::vector<uint32_t> indices = { 0, 1, 2 };
    auto index_count = static_cast<uint32_t>(indices.size());

    std::vector<glm::mat3x4> transform_matrices(g_object_count);
    for (uint32_t i = 0; i < g_object_count; i++) {
        transform_matrices[i] = {
            1.0f, 0.0f, 0.0f, (float)i * 3.0f - 3.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
    }

    // Create buffers
    // For the sake of simplicity we won't stage the vertex data to the GPU memory
    // Vertex buffer
    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info.vertex_buffer.buffer,
        bl_data_info.vertex_buffer.memory,
        vertices.size() * sizeof(Vertex),
        vertices.data());

    // Index buffer
    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info.index_buffer.buffer,
        bl_data_info.index_buffer.memory,
        indices.size() * sizeof(uint32_t),
        indices.data());

    // Transform buffer
    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info.transform_buffer.buffer,
        bl_data_info.transform_buffer.memory,
        sizeof(glm::mat3x4) * g_object_count,
        &transform_matrices[0]);

    // Get size info
    er::AccelerationStructureBuildGeometryInfo as_build_geometry_info{};
    as_build_geometry_info.type = er::AccelerationStructureType::BOTTOM_LEVEL_KHR;
    as_build_geometry_info.flags = SET_FLAG_BIT(BuildAccelerationStructure, PREFER_FAST_TRACE_BIT_KHR);

    // Build
    for (uint32_t i = 0; i < g_object_count; i++) {
        auto as_geometry = std::make_shared<er::AccelerationStructureGeometry>();
        as_geometry->flags = SET_FLAG_BIT(Geometry, OPAQUE_BIT_KHR);
        as_geometry->geometry_type = er::GeometryType::TRIANGLES_KHR;
        as_geometry->geometry.triangles.struct_type =
            er::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        as_geometry->geometry.triangles.vertex_format = er::Format::R32G32B32_SFLOAT;
        as_geometry->geometry.triangles.vertex_data.device_address =
            bl_data_info.vertex_buffer.buffer->getDeviceAddress();
        as_geometry->geometry.triangles.max_vertex = 3;
        as_geometry->geometry.triangles.vertex_stride = sizeof(Vertex);
        as_geometry->geometry.triangles.index_type = er::IndexType::UINT32;
        as_geometry->geometry.triangles.index_data.device_address =
            bl_data_info.index_buffer.buffer->getDeviceAddress();
        as_geometry->geometry.triangles.transform_data.device_address =
            bl_data_info.transform_buffer.buffer->getDeviceAddress();
        as_geometry->max_primitive_count = 1;

        as_build_geometry_info.geometries.push_back(as_geometry);
    }

    er::AccelerationStructureBuildSizesInfo as_build_size_info{};
    as_build_size_info.struct_type = er::StructureType::ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    device_info.device->getAccelerationStructureBuildSizes(
        er::AccelerationStructureBuildType::DEVICE_KHR,
        as_build_geometry_info,
        as_build_size_info);

    device_info.device->createBuffer(
        as_build_size_info.as_size,
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info.as_buffer.buffer,
        bl_data_info.as_buffer.memory);

    bl_data_info.as_handle = device_info.device->createAccelerationStructure(
        bl_data_info.as_buffer.buffer,
        er::AccelerationStructureType::BOTTOM_LEVEL_KHR);

    device_info.device->createBuffer(
        as_build_size_info.build_scratch_size,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info.scratch_buffer.buffer,
        bl_data_info.scratch_buffer.memory);

    as_build_geometry_info.mode = er::BuildAccelerationStructureMode::BUILD_KHR;
    as_build_geometry_info.dst_as = bl_data_info.as_handle;
    as_build_geometry_info.scratch_data.device_address =
        bl_data_info.scratch_buffer.buffer->getDeviceAddress();

    uint32_t num_triangles = 1;
    std::vector<er::AccelerationStructureBuildRangeInfo> as_build_range_infos(g_object_count);
    for (auto i = 0; i < g_object_count; i++) {
        as_build_range_infos[i].primitive_count = num_triangles;
        as_build_range_infos[i].primitive_offset = 0;
        as_build_range_infos[i].first_vertex = 0;
        as_build_range_infos[i].transform_offset = i * sizeof(glm::mat3x4);
    }

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    auto command_buffers = device_info.device->allocateCommandBuffers(device_info.cmd_pool, 1);
    if (command_buffers.size() > 0) {
        auto& cmd_buf = command_buffers[0];
        if (cmd_buf) {
            cmd_buf->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));
            cmd_buf->buildAccelerationStructures({ as_build_geometry_info }, as_build_range_infos);
            cmd_buf->endCommandBuffer();
        }

        device_info.cmd_queue->submit(command_buffers);
        device_info.cmd_queue->waitIdle();
        device_info.device->freeCommandBuffers(device_info.cmd_pool, command_buffers);
    }

    bl_data_info.as_device_address =
        device_info.device->getAccelerationStructureDeviceAddress(bl_data_info.as_handle);
}

struct ToplevelDataInfo {
    er::BufferInfo   instance_buffer{};
    er::BufferInfo   as_buffer{};
    er::BufferInfo   scratch_buffer{};

    er::AccelerationStructure   as_handle{};
    er::DeviceAddress   as_device_address = 0;
};

void initTopLevelDataInfo(
    const er::DeviceInfo& device_info,
    const BottomlevelDataInfo& bl_data_info,
    ToplevelDataInfo& tl_data_info) {

    er::TransformMatrix transform_matrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f };

    uint32_t primitive_count = 1;

    er::AccelerationStructureInstance instance{};
    instance.transform = transform_matrix;
    instance.instance_custom_index = 0;
    instance.mask = 0xFF;
    instance.instance_shader_binding_table_record_offset = 0;
    instance.flags = SET_FLAG_BIT(GeometryInstance, TRIANGLE_FACING_CULL_DISABLE_BIT_KHR);
    instance.acceleration_structure_reference = bl_data_info.as_device_address;

    // Instance buffer
    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info.instance_buffer.buffer,
        tl_data_info.instance_buffer.memory,
        sizeof(instance),
        &instance);

    auto as_geometry = std::make_shared<er::AccelerationStructureGeometry>();
    as_geometry->flags = SET_FLAG_BIT(Geometry, OPAQUE_BIT_KHR);
    as_geometry->geometry_type = er::GeometryType::INSTANCES_KHR;
    as_geometry->geometry.instances.struct_type =
        er::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    as_geometry->geometry.instances.array_of_pointers = 0x00;
    as_geometry->geometry.instances.data.device_address =
        tl_data_info.instance_buffer.buffer->getDeviceAddress();
    as_geometry->max_primitive_count = primitive_count;

    // Get size info
    /*
    The pSrcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored. Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command, except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData will be examined to check if it is NULL.*
    */
    // Get size info
    er::AccelerationStructureBuildGeometryInfo as_build_geometry_info{};
    as_build_geometry_info.type = er::AccelerationStructureType::TOP_LEVEL_KHR;
    as_build_geometry_info.flags = SET_FLAG_BIT(BuildAccelerationStructure, PREFER_FAST_TRACE_BIT_KHR);
    as_build_geometry_info.geometries.push_back(as_geometry);

    er::AccelerationStructureBuildSizesInfo as_build_size_info{};
    as_build_size_info.struct_type = er::StructureType::ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    device_info.device->getAccelerationStructureBuildSizes(
        er::AccelerationStructureBuildType::DEVICE_KHR,
        as_build_geometry_info,
        as_build_size_info);

    device_info.device->createBuffer(
        as_build_size_info.as_size,
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info.as_buffer.buffer,
        tl_data_info.as_buffer.memory);


    tl_data_info.as_handle = device_info.device->createAccelerationStructure(
        tl_data_info.as_buffer.buffer,
        er::AccelerationStructureType::TOP_LEVEL_KHR);

    device_info.device->createBuffer(
        as_build_size_info.build_scratch_size,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info.scratch_buffer.buffer,
        tl_data_info.scratch_buffer.memory);

    as_build_geometry_info.mode = er::BuildAccelerationStructureMode::BUILD_KHR;
    as_build_geometry_info.dst_as = tl_data_info.as_handle;
    as_build_geometry_info.scratch_data.device_address =
        tl_data_info.scratch_buffer.buffer->getDeviceAddress();

    er::AccelerationStructureBuildRangeInfo as_build_range_info{};
    as_build_range_info.primitive_count = primitive_count;
    as_build_range_info.primitive_offset = 0;
    as_build_range_info.first_vertex = 0;
    as_build_range_info.transform_offset = 0;

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    auto command_buffers = device_info.device->allocateCommandBuffers(device_info.cmd_pool, 1);
    if (command_buffers.size() > 0) {
        auto& cmd_buf = command_buffers[0];
        if (cmd_buf) {
            cmd_buf->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));
            cmd_buf->buildAccelerationStructures({ as_build_geometry_info }, { as_build_range_info });
            cmd_buf->endCommandBuffer();
        }

        device_info.cmd_queue->submit(command_buffers);
        device_info.cmd_queue->waitIdle();
        device_info.device->freeCommandBuffers(device_info.cmd_pool, command_buffers);
    }

    tl_data_info.as_device_address =
        device_info.device->getAccelerationStructureDeviceAddress(tl_data_info.as_handle);
}

struct RayTracingRenderingInfo {
    er::ShaderModuleList shader_modules;
    er::RtShaderGroupCreateInfoList shader_groups;
    std::shared_ptr<er::DescriptorSetLayout> rt_desc_set_layout;
    std::shared_ptr<er::PipelineLayout> rt_pipeline_layout;
    std::shared_ptr<er::Pipeline> rt_pipeline;
    std::shared_ptr<er::DescriptorSet> rt_desc_set;
    er::StridedDeviceAddressRegion raygen_shader_sbt_entry;
    er::StridedDeviceAddressRegion miss_shader_sbt_entry;
    er::StridedDeviceAddressRegion hit_shader_sbt_entry;
    er::StridedDeviceAddressRegion callable_shader_sbt_entry;
    er::BufferInfo raygen_shader_binding_table;
    er::BufferInfo miss_shader_binding_table;
    er::BufferInfo hit_shader_binding_table;
    er::BufferInfo callable_shader_binding_table;
    er::BufferInfo ubo;
    er::TextureInfo result_image;
};

struct UniformData {
    glm::mat4 view_inverse;
    glm::mat4 proj_inverse;
};

uint32_t alignedSize(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

std::shared_ptr<er::DescriptorSetLayout> createRtDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(5);
    bindings[0] = er::helper::getBufferDescriptionSetLayoutBinding(
        0,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        er::DescriptorType::ACCELERATION_STRUCTURE_KHR);
    bindings[1] = er::helper::getBufferDescriptionSetLayoutBinding(
        1,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR),
        er::DescriptorType::STORAGE_IMAGE);
    bindings[2] = er::helper::getBufferDescriptionSetLayoutBinding(
        2,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, MISS_BIT_KHR),
        er::DescriptorType::UNIFORM_BUFFER);
    bindings[3] = er::helper::getBufferDescriptionSetLayoutBinding(
        3,
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        er::DescriptorType::STORAGE_BUFFER);
    bindings[4] = er::helper::getBufferDescriptionSetLayoutBinding(
        4,
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        er::DescriptorType::STORAGE_BUFFER);
    return device->createDescriptorSetLayout(bindings);
}

enum {
    kRayGenIndex,
    kRayMissIndex,
    kClosestHitIndex,
    kCallable1Index,
    kCallable2Index,
    kCallable3Index,
    kNumRtShaders
};
void createRayTracingPipeline(
    const er::DeviceInfo& device_info,
    RayTracingRenderingInfo& rt_render_info) {
    const auto& device = device_info.device;
    rt_render_info.shader_modules.resize(kNumRtShaders);
    rt_render_info.shader_modules[kRayGenIndex] =
        er::helper::loadShaderModule(
            device,
            "ray_tracing/callable_test/rt_raygen_rgen.spv",
            er::ShaderStageFlagBits::RAYGEN_BIT_KHR);
    rt_render_info.shader_modules[kRayMissIndex] =
        er::helper::loadShaderModule(
            device,
            "ray_tracing/callable_test/rt_miss_rmiss.spv",
            er::ShaderStageFlagBits::MISS_BIT_KHR);
    rt_render_info.shader_modules[kClosestHitIndex] =
        er::helper::loadShaderModule(
            device,
            "ray_tracing/callable_test/rt_closesthit_rchit.spv",
            er::ShaderStageFlagBits::CLOSEST_HIT_BIT_KHR);
    for (auto i = 0; i < g_object_count; i++) {
        auto index = kCallable1Index + static_cast<uint32_t>(i);
        auto callable_shader_name =
            std::string("ray_tracing/callable_test/rt_callable") +
            std::to_string(i+1) +
            "_rcall.spv";
        rt_render_info.shader_modules[index] =
            er::helper::loadShaderModule(
                device,
                callable_shader_name,
                er::ShaderStageFlagBits::CALLABLE_BIT_KHR);
    }

    rt_render_info.shader_groups.resize(kNumRtShaders);
    rt_render_info.shader_groups[kRayGenIndex].type = er::RayTracingShaderGroupType::GENERAL_KHR;
    rt_render_info.shader_groups[kRayGenIndex].general_shader = kRayGenIndex;
    rt_render_info.shader_groups[kRayMissIndex].type = er::RayTracingShaderGroupType::GENERAL_KHR;
    rt_render_info.shader_groups[kRayMissIndex].general_shader = kRayMissIndex;
    rt_render_info.shader_groups[kClosestHitIndex].type = er::RayTracingShaderGroupType::TRIANGLES_HIT_GROUP_KHR;
    rt_render_info.shader_groups[kClosestHitIndex].closest_hit_shader = kClosestHitIndex;
    for (auto i = 0; i < g_object_count; i++) {
        auto index = kCallable1Index + static_cast<uint32_t>(i);
        rt_render_info.shader_groups[index].type = er::RayTracingShaderGroupType::GENERAL_KHR;
        rt_render_info.shader_groups[index].general_shader = index;
    }

    rt_render_info.rt_desc_set_layout = createRtDescriptorSetLayout(device);
    rt_render_info.rt_pipeline_layout =
        device->createPipelineLayout(
            { rt_render_info.rt_desc_set_layout }, { });
    rt_render_info.rt_pipeline =
        device->createPipeline(
            rt_render_info.rt_pipeline_layout,
            rt_render_info.shader_modules,
            rt_render_info.shader_groups,
            2);
}

void createShaderBindingTable(
    const er::DeviceInfo& device_info,
    const er::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
    const er::PhysicalDeviceAccelerationStructureFeatures& as_features,
    RayTracingRenderingInfo& rt_render_info) {
    const auto& device = device_info.device;
    const uint32_t handle_size = rt_pipeline_properties.shader_group_handle_size;
    const uint32_t handle_size_aligned = 
        alignedSize(rt_pipeline_properties.shader_group_handle_size,
            rt_pipeline_properties.shader_group_handle_alignment);
    const uint32_t group_count =
        static_cast<uint32_t>(rt_render_info.shader_groups.size());
    const uint32_t sbt_size = group_count * handle_size_aligned;

    std::vector<uint8_t> shader_handle_storage(sbt_size);
    device->getRayTracingShaderGroupHandles(
        rt_render_info.rt_pipeline,
        group_count,
        sbt_size,
        shader_handle_storage.data());

    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info.raygen_shader_binding_table.buffer,
        rt_render_info.raygen_shader_binding_table.memory,
        handle_size,
        shader_handle_storage.data());

    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info.miss_shader_binding_table.buffer,
        rt_render_info.miss_shader_binding_table.memory,
        handle_size,
        shader_handle_storage.data() + handle_size_aligned);

    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info.hit_shader_binding_table.buffer,
        rt_render_info.hit_shader_binding_table.memory,
        handle_size,
        shader_handle_storage.data() + handle_size_aligned * 2);

    er::Helper::createBuffer(
        device_info,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info.callable_shader_binding_table.buffer,
        rt_render_info.callable_shader_binding_table.memory,
        handle_size * g_object_count,
        shader_handle_storage.data() + handle_size_aligned * 3);

    rt_render_info.raygen_shader_sbt_entry.device_address =
        rt_render_info.raygen_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info.raygen_shader_sbt_entry.size = handle_size_aligned;
    rt_render_info.raygen_shader_sbt_entry.stride = handle_size_aligned;

    rt_render_info.miss_shader_sbt_entry.device_address =
        rt_render_info.miss_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info.miss_shader_sbt_entry.size = handle_size_aligned;
    rt_render_info.miss_shader_sbt_entry.stride = handle_size_aligned;

    rt_render_info.hit_shader_sbt_entry.device_address =
        rt_render_info.hit_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info.hit_shader_sbt_entry.size = handle_size_aligned;
    rt_render_info.hit_shader_sbt_entry.stride = handle_size_aligned;
    
    rt_render_info.callable_shader_sbt_entry.device_address =
        rt_render_info.callable_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info.callable_shader_sbt_entry.size = handle_size_aligned * 3;
    rt_render_info.callable_shader_sbt_entry.stride = handle_size_aligned;
}

void createRtResources(
    const er::DeviceInfo& device_info,
    RayTracingRenderingInfo& rt_render_info) {
    device_info.device->createBuffer(
        sizeof(UniformData),
        SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        rt_render_info.ubo.buffer,
        rt_render_info.ubo.memory);

    er::Helper::create2DTextureImage(
        device_info,
        er::Format::R16G16B16A16_SFLOAT,
        glm::uvec2(1920, 1080),
        rt_render_info.result_image,
        SET_FLAG_BIT(ImageUsage, TRANSFER_SRC_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        er::ImageLayout::GENERAL);
}

void createDescriptorSets(
    const er::DeviceInfo& device_info,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const BottomlevelDataInfo& bl_data_info,
    const ToplevelDataInfo& tl_data_info,
    RayTracingRenderingInfo& rt_render_info) {
    const auto& device = device_info.device;

    rt_render_info.rt_desc_set =
        device->createDescriptorSets(
            descriptor_pool,
            rt_render_info.rt_desc_set_layout,
            1)[0];

    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(5);

    er::Helper::addOneAccelerationStructure(
        descriptor_writes,
        rt_render_info.rt_desc_set,
        er::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        0,
        { tl_data_info.as_handle });

    er::Helper::addOneTexture(
        descriptor_writes,
        rt_render_info.rt_desc_set,
        er::DescriptorType::STORAGE_IMAGE,
        1,
        nullptr,
        rt_render_info.result_image.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info.rt_desc_set,
        er::DescriptorType::UNIFORM_BUFFER,
        2,
        rt_render_info.ubo.buffer,
        rt_render_info.ubo.buffer->getSize());

    er::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info.rt_desc_set,
        er::DescriptorType::STORAGE_BUFFER,
        3,
        bl_data_info.vertex_buffer.buffer,
        bl_data_info.vertex_buffer.buffer->getSize());

    er::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info.rt_desc_set,
        er::DescriptorType::STORAGE_BUFFER,
        4,
        bl_data_info.index_buffer.buffer,
        bl_data_info.index_buffer.buffer->getSize());

    device->updateDescriptorSets(descriptor_writes);
}

er::TextureInfo testRayTracing(
    const er::DeviceInfo& device_info,
    const std::shared_ptr<er::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const glsl::ViewParams& view_params,
    er::PhysicalDeviceRayTracingPipelineProperties rt_pipeline_properties,
    er::PhysicalDeviceAccelerationStructureFeatures as_features) {

    static bool s_rt_init = false;
    static BottomlevelDataInfo s_bl_data_info;
    static ToplevelDataInfo s_tl_data_info;
    static RayTracingRenderingInfo s_rt_render_info;

    if (!s_rt_init) {
        initBottomLevelDataInfo(
            device_info,
            s_bl_data_info);
        initTopLevelDataInfo(
            device_info,
            s_bl_data_info,
            s_tl_data_info);
        createRayTracingPipeline(
            device_info,
            s_rt_render_info);
        createShaderBindingTable(
            device_info,
            rt_pipeline_properties,
            as_features,
            s_rt_render_info);
        createRtResources(
            device_info,
            s_rt_render_info);
        createDescriptorSets(
            device_info,
            descriptor_pool,
            s_bl_data_info,
            s_tl_data_info,
            s_rt_render_info);

        s_rt_init = true;
    }

    auto view = view_params.view;
    view[3] = glm::vec4(0, 0, -5, 1);
    UniformData uniform_data;
    uniform_data.proj_inverse = glm::inverse(view_params.proj);
    uniform_data.view_inverse = glm::inverse(view);

    device_info.device->updateBufferMemory(
        s_rt_render_info.ubo.memory,
        sizeof(uniform_data),
        &uniform_data);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::RAY_TRACING,
        s_rt_render_info.rt_pipeline);

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::RAY_TRACING,
        s_rt_render_info.rt_pipeline_layout,
        {s_rt_render_info.rt_desc_set });

    cmd_buf->traceRays(
        s_rt_render_info.raygen_shader_sbt_entry,
        s_rt_render_info.miss_shader_sbt_entry,
        s_rt_render_info.hit_shader_sbt_entry,
        s_rt_render_info.callable_shader_sbt_entry,
        glm::uvec3(1920, 1080, 1));

    return s_rt_render_info.result_image;
}
