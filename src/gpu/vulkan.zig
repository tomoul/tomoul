// src/gpu/vulkan.zig
// Vulkan Compute Backend for Tomoul
//
// Provides GPU-accelerated compute via Vulkan 1.1+ compute shaders.
// Used for SGEMM, attention, and full transformer forward passes.
//
// Architecture:
//   init() → enumerate devices, create logical device + compute queue
//   createBuffer() → allocate device-local or host-visible buffers
//   upload() → transfer data to device-local memory via staging buffer
//   createPipeline() → load SPIR-V, create compute pipeline + descriptor sets
//   dispatch() → record command buffer, submit to queue, wait
//   readback() → copy device-local buffer to host-visible, map and read
//   deinit() → destroy all Vulkan resources

const std = @import("std");
const vk = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const VulkanError = error{
    InstanceCreationFailed,
    NoPhysicalDevice,
    NoComputeQueue,
    DeviceCreationFailed,
    BufferCreationFailed,
    MemoryAllocationFailed,
    MemoryBindFailed,
    MemoryMapFailed,
    ShaderModuleCreationFailed,
    PipelineLayoutCreationFailed,
    PipelineCreationFailed,
    DescriptorPoolCreationFailed,
    DescriptorSetAllocationFailed,
    CommandPoolCreationFailed,
    CommandBufferAllocationFailed,
    CommandBufferBeginFailed,
    CommandBufferEndFailed,
    QueueSubmitFailed,
    FenceCreationFailed,
    FenceWaitFailed,
    OutOfMemory,
};

/// GPU buffer handle with metadata
pub const GpuBuffer = struct {
    buffer: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
    is_host_visible: bool,
};

/// Compute pipeline handle
pub const ComputePipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    descriptor_set_layout: vk.VkDescriptorSetLayout,
    shader_module: vk.VkShaderModule,
};

/// Vulkan compute context — single device, single compute queue
pub const VulkanContext = struct {
    allocator: std.mem.Allocator,

    // Core Vulkan objects
    instance: vk.VkInstance,
    physical_device: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    compute_queue: vk.VkQueue,
    compute_queue_family: u32,

    // Command submission
    command_pool: vk.VkCommandPool,
    command_buffer: vk.VkCommandBuffer,
    fence: vk.VkFence,

    // Descriptor pool (shared across pipelines)
    descriptor_pool: vk.VkDescriptorPool,

    // Device properties
    device_name: [256]u8,
    device_name_len: usize,
    max_compute_work_group_count: [3]u32,
    max_compute_work_group_size: [3]u32,
    max_compute_shared_memory: u32,
    subgroup_size: u32,

    const Self = @This();

    /// Initialize Vulkan: instance → physical device → logical device → queue → command pool
    /// Selects the best available compute device (discrete GPU preferred).
    pub fn init(allocator: std.mem.Allocator) VulkanError!Self {
        var self: Self = undefined;
        self.allocator = allocator;

        // 1. Create Vulkan instance
        const app_info = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "Tomoul",
            .applicationVersion = vk.VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "Tomoul GPU",
            .engineVersion = vk.VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = vk.VK_API_VERSION_1_1,
        };

        const instance_info = vk.VkInstanceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        };

        if (vk.vkCreateInstance(&instance_info, null, &self.instance) != vk.VK_SUCCESS) {
            return VulkanError.InstanceCreationFailed;
        }
        errdefer vk.vkDestroyInstance(self.instance, null);

        // 2. Select physical device (prefer discrete GPU)
        var device_count: u32 = 0;
        _ = vk.vkEnumeratePhysicalDevices(self.instance, &device_count, null);
        if (device_count == 0) {
            return VulkanError.NoPhysicalDevice;
        }

        const devices = allocator.alloc(vk.VkPhysicalDevice, device_count) catch return VulkanError.OutOfMemory;
        defer allocator.free(devices);
        _ = vk.vkEnumeratePhysicalDevices(self.instance, &device_count, devices.ptr);

        // Score devices: discrete > integrated > CPU
        var best_device: ?vk.VkPhysicalDevice = null;
        var best_score: i32 = -1;
        var best_queue_family: u32 = 0;

        for (devices[0..device_count]) |dev| {
            var props: vk.VkPhysicalDeviceProperties = undefined;
            vk.vkGetPhysicalDeviceProperties(dev, &props);

            const score: i32 = switch (props.deviceType) {
                vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 100,
                vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 50,
                vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 30,
                vk.VK_PHYSICAL_DEVICE_TYPE_CPU => 10,
                else => 1,
            };

            // Find compute queue family
            var queue_count: u32 = 0;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, null);
            const queue_families = allocator.alloc(vk.VkQueueFamilyProperties, queue_count) catch continue;
            defer allocator.free(queue_families);
            vk.vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, queue_families.ptr);

            var found_compute: bool = false;
            var compute_family: u32 = 0;
            for (queue_families[0..queue_count], 0..) |qf, idx| {
                if (qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                    found_compute = true;
                    compute_family = @intCast(idx);
                    break;
                }
            }

            if (found_compute and score > best_score) {
                best_device = dev;
                best_score = score;
                best_queue_family = compute_family;
            }
        }

        self.physical_device = best_device orelse return VulkanError.NoPhysicalDevice;
        self.compute_queue_family = best_queue_family;

        // Store device properties
        var props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(self.physical_device, &props);
        const name_slice = std.mem.sliceTo(&props.deviceName, 0);
        self.device_name_len = @min(name_slice.len, 256);
        @memcpy(self.device_name[0..self.device_name_len], name_slice[0..self.device_name_len]);

        self.max_compute_work_group_count = props.limits.maxComputeWorkGroupCount;
        self.max_compute_work_group_size = props.limits.maxComputeWorkGroupSize;
        self.max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

        // Get subgroup properties
        var subgroup_props = vk.VkPhysicalDeviceSubgroupProperties{
            .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
            .pNext = null,
            .subgroupSize = 0,
            .supportedStages = 0,
            .supportedOperations = 0,
            .quadOperationsInAllStages = 0,
        };
        var dev_props2 = vk.VkPhysicalDeviceProperties2{
            .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &subgroup_props,
            .properties = undefined,
        };
        vk.vkGetPhysicalDeviceProperties2(self.physical_device, &dev_props2);
        self.subgroup_size = subgroup_props.subgroupSize;

        // 3. Create logical device with compute queue
        const queue_priority: f32 = 1.0;
        const queue_create_info = vk.VkDeviceQueueCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = self.compute_queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        const device_create_info = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,
        };

        if (vk.vkCreateDevice(self.physical_device, &device_create_info, null, &self.device) != vk.VK_SUCCESS) {
            return VulkanError.DeviceCreationFailed;
        }
        errdefer vk.vkDestroyDevice(self.device, null);

        // Get compute queue
        vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0, &self.compute_queue);

        // 4. Create command pool
        const pool_info = vk.VkCommandPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.compute_queue_family,
        };

        if (vk.vkCreateCommandPool(self.device, &pool_info, null, &self.command_pool) != vk.VK_SUCCESS) {
            return VulkanError.CommandPoolCreationFailed;
        }
        errdefer vk.vkDestroyCommandPool(self.device, self.command_pool, null);

        // 5. Allocate command buffer
        const alloc_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.command_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        if (vk.vkAllocateCommandBuffers(self.device, &alloc_info, &self.command_buffer) != vk.VK_SUCCESS) {
            return VulkanError.CommandBufferAllocationFailed;
        }

        // 6. Create fence
        const fence_info = vk.VkFenceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };

        if (vk.vkCreateFence(self.device, &fence_info, null, &self.fence) != vk.VK_SUCCESS) {
            return VulkanError.FenceCreationFailed;
        }
        errdefer vk.vkDestroyFence(self.device, self.fence, null);

        // 7. Create descriptor pool (space for many storage buffer descriptors)
        const pool_sizes = [_]vk.VkDescriptorPoolSize{
            .{
                .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 512, // enough for full transformer forward pass
            },
        };

        const desc_pool_info = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 128,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        };

        if (vk.vkCreateDescriptorPool(self.device, &desc_pool_info, null, &self.descriptor_pool) != vk.VK_SUCCESS) {
            return VulkanError.DescriptorPoolCreationFailed;
        }

        return self;
    }

    /// Get device name as a slice
    pub fn getDeviceName(self: *const Self) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    /// Reset the descriptor pool, freeing all allocated descriptor sets.
    /// Call before re-allocating descriptor sets for a new forward pass.
    pub fn resetDescriptorPool(self: *Self) void {
        _ = vk.vkResetDescriptorPool(self.device, self.descriptor_pool, 0);
    }

    /// Create a storage buffer (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    pub fn createStorageBuffer(self: *Self, size: usize, host_visible: bool) VulkanError!GpuBuffer {
        return self.createBuffer(size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_visible);
    }

    // =========================================================================
    // Buffer Management
    // =========================================================================

    /// Create a GPU buffer with the specified size and usage.
    /// host_visible=true: CPU-accessible (staging, readback)
    /// host_visible=false: device-local (fast GPU memory)
    pub fn createBuffer(self: *Self, size: usize, usage: vk.VkBufferUsageFlags, host_visible: bool) VulkanError!GpuBuffer {
        const buffer_info = vk.VkBufferCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = @intCast(size),
            .usage = usage,
            .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var buffer: vk.VkBuffer = undefined;
        if (vk.vkCreateBuffer(self.device, &buffer_info, null, &buffer) != vk.VK_SUCCESS) {
            return VulkanError.BufferCreationFailed;
        }
        errdefer vk.vkDestroyBuffer(self.device, buffer, null);

        // Get memory requirements
        var mem_req: vk.VkMemoryRequirements = undefined;
        vk.vkGetBufferMemoryRequirements(self.device, buffer, &mem_req);

        // Find suitable memory type
        const required_props: vk.VkMemoryPropertyFlags = if (host_visible)
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        else
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        const mem_type_idx = self.findMemoryType(mem_req.memoryTypeBits, required_props) orelse {
            // Fallback: try host-visible if device-local not found
            if (!host_visible) {
                const fallback_props = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (self.findMemoryType(mem_req.memoryTypeBits, fallback_props)) |idx| {
                    return self.allocateAndBindBuffer(buffer, mem_req.size, idx, true);
                }
            }
            return VulkanError.MemoryAllocationFailed;
        };

        return self.allocateAndBindBuffer(buffer, mem_req.size, mem_type_idx, host_visible);
    }

    fn allocateAndBindBuffer(self: *Self, buffer: vk.VkBuffer, size: vk.VkDeviceSize, mem_type_idx: u32, host_visible: bool) VulkanError!GpuBuffer {
        const alloc_info = vk.VkMemoryAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = size,
            .memoryTypeIndex = mem_type_idx,
        };

        var memory: vk.VkDeviceMemory = undefined;
        if (vk.vkAllocateMemory(self.device, &alloc_info, null, &memory) != vk.VK_SUCCESS) {
            return VulkanError.MemoryAllocationFailed;
        }
        errdefer vk.vkFreeMemory(self.device, memory, null);

        if (vk.vkBindBufferMemory(self.device, buffer, memory, 0) != vk.VK_SUCCESS) {
            return VulkanError.MemoryBindFailed;
        }

        return GpuBuffer{
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .usage = 0,
            .is_host_visible = host_visible,
        };
    }

    /// Upload data from CPU to a host-visible buffer
    pub fn uploadToBuffer(self: *Self, buf: *const GpuBuffer, data: []const u8) VulkanError!void {
        var mapped: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, buf.memory, 0, @intCast(data.len), 0, &mapped) != vk.VK_SUCCESS) {
            return VulkanError.MemoryMapFailed;
        }
        const dst: [*]u8 = @ptrCast(mapped.?);
        @memcpy(dst[0..data.len], data);
        vk.vkUnmapMemory(self.device, buf.memory);
    }

    /// Read data from a host-visible buffer back to CPU
    pub fn readbackFromBuffer(self: *Self, buf: *const GpuBuffer, output: []u8) VulkanError!void {
        var mapped: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, buf.memory, 0, @intCast(output.len), 0, &mapped) != vk.VK_SUCCESS) {
            return VulkanError.MemoryMapFailed;
        }
        const src: [*]const u8 = @ptrCast(mapped.?);
        @memcpy(output, src[0..output.len]);
        vk.vkUnmapMemory(self.device, buf.memory);
    }

    /// Destroy a GPU buffer and free its memory
    pub fn destroyBuffer(self: *Self, buf: *const GpuBuffer) void {
        vk.vkDestroyBuffer(self.device, buf.buffer, null);
        vk.vkFreeMemory(self.device, buf.memory, null);
    }

    fn findMemoryType(self: *const Self, type_filter: u32, properties: vk.VkMemoryPropertyFlags) ?u32 {
        var mem_properties: vk.VkPhysicalDeviceMemoryProperties = undefined;
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mem_properties);

        for (0..mem_properties.memoryTypeCount) |i| {
            const idx: u5 = @intCast(i);
            if ((type_filter & (@as(u32, 1) << idx)) != 0 and
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return @intCast(i);
            }
        }
        return null;
    }

    // =========================================================================
    // Pipeline Management
    // =========================================================================

    /// Create a compute pipeline from SPIR-V bytecode.
    /// num_storage_buffers: number of storage buffer bindings (e.g., 3 for A, B, C)
    /// push_constant_size: size of push constant block in bytes (0 if none)
    pub fn createComputePipeline(
        self: *Self,
        spirv_code: []const u8,
        num_storage_buffers: u32,
        push_constant_size: u32,
    ) VulkanError!ComputePipeline {
        // Create shader module
        const shader_info = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spirv_code.len,
            .pCode = @ptrCast(@alignCast(spirv_code.ptr)),
        };

        var shader_module: vk.VkShaderModule = undefined;
        if (vk.vkCreateShaderModule(self.device, &shader_info, null, &shader_module) != vk.VK_SUCCESS) {
            return VulkanError.ShaderModuleCreationFailed;
        }
        errdefer vk.vkDestroyShaderModule(self.device, shader_module, null);

        // Create descriptor set layout
        var bindings = self.allocator.alloc(vk.VkDescriptorSetLayoutBinding, num_storage_buffers) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(bindings);

        for (0..num_storage_buffers) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            };
        }

        const layout_info = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .bindingCount = num_storage_buffers,
            .pBindings = bindings.ptr,
        };

        var descriptor_set_layout: vk.VkDescriptorSetLayout = undefined;
        if (vk.vkCreateDescriptorSetLayout(self.device, &layout_info, null, &descriptor_set_layout) != vk.VK_SUCCESS) {
            return VulkanError.DescriptorPoolCreationFailed;
        }
        errdefer vk.vkDestroyDescriptorSetLayout(self.device, descriptor_set_layout, null);

        // Create pipeline layout (with optional push constants)
        var push_constant_range: vk.VkPushConstantRange = undefined;
        const has_push_constants = push_constant_size > 0;
        if (has_push_constants) {
            push_constant_range = .{
                .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = push_constant_size,
            };
        }

        const pipeline_layout_info = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = if (has_push_constants) 1 else 0,
            .pPushConstantRanges = if (has_push_constants) &push_constant_range else null,
        };

        var pipeline_layout: vk.VkPipelineLayout = undefined;
        if (vk.vkCreatePipelineLayout(self.device, &pipeline_layout_info, null, &pipeline_layout) != vk.VK_SUCCESS) {
            return VulkanError.PipelineLayoutCreationFailed;
        }
        errdefer vk.vkDestroyPipelineLayout(self.device, pipeline_layout, null);

        // Create compute pipeline
        const stage_info = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
            .pSpecializationInfo = null,
        };

        const pipeline_info = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage_info,
            .layout = pipeline_layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };

        var pipeline: vk.VkPipeline = undefined;
        if (vk.vkCreateComputePipelines(self.device, null, 1, &pipeline_info, null, &pipeline) != vk.VK_SUCCESS) {
            return VulkanError.PipelineCreationFailed;
        }

        return ComputePipeline{
            .pipeline = pipeline,
            .layout = pipeline_layout,
            .descriptor_set_layout = descriptor_set_layout,
            .shader_module = shader_module,
        };
    }

    /// Destroy a compute pipeline and all associated resources
    pub fn destroyPipeline(self: *Self, pipe: *const ComputePipeline) void {
        vk.vkDestroyPipeline(self.device, pipe.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, pipe.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, pipe.descriptor_set_layout, null);
        vk.vkDestroyShaderModule(self.device, pipe.shader_module, null);
    }

    // =========================================================================
    // Dispatch
    // =========================================================================

    /// Allocate a descriptor set for a pipeline
    pub fn allocateDescriptorSet(self: *Self, pipe: *const ComputePipeline) VulkanError!vk.VkDescriptorSet {
        const alloc_info = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &pipe.descriptor_set_layout,
        };

        var descriptor_set: vk.VkDescriptorSet = undefined;
        if (vk.vkAllocateDescriptorSets(self.device, &alloc_info, &descriptor_set) != vk.VK_SUCCESS) {
            return VulkanError.DescriptorSetAllocationFailed;
        }
        return descriptor_set;
    }

    /// Bind storage buffers to a descriptor set
    pub fn bindBuffers(self: *Self, descriptor_set: vk.VkDescriptorSet, buffers: []const GpuBuffer) VulkanError!void {
        var writes = self.allocator.alloc(vk.VkWriteDescriptorSet, buffers.len) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(writes);

        var buffer_infos = self.allocator.alloc(vk.VkDescriptorBufferInfo, buffers.len) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(buffer_infos);

        for (buffers, 0..) |buf, i| {
            buffer_infos[i] = .{
                .buffer = buf.buffer,
                .offset = 0,
                .range = vk.VK_WHOLE_SIZE,
            };

            writes[i] = .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = descriptor_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }

        vk.vkUpdateDescriptorSets(self.device, @intCast(writes.len), writes.ptr, 0, null);
    }

    /// Dispatch a compute shader: record command buffer, submit, wait for completion.
    /// push_constants: raw bytes for push constant data (null if none)
    pub fn dispatch(
        self: *Self,
        pipe: *const ComputePipeline,
        descriptor_set: vk.VkDescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        push_constants: ?[]const u8,
    ) VulkanError!void {
        // Reset command buffer
        _ = vk.vkResetCommandBuffer(self.command_buffer, 0);

        // Begin recording
        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };

        if (vk.vkBeginCommandBuffer(self.command_buffer, &begin_info) != vk.VK_SUCCESS) {
            return VulkanError.CommandBufferBeginFailed;
        }

        // Bind pipeline and descriptor set
        vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipe.layout,
            0,
            1,
            &descriptor_set,
            0,
            null,
        );

        // Push constants
        if (push_constants) |pc| {
            vk.vkCmdPushConstants(
                self.command_buffer,
                pipe.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                @intCast(pc.len),
                pc.ptr,
            );
        }

        // Dispatch compute
        vk.vkCmdDispatch(self.command_buffer, group_count_x, group_count_y, group_count_z);

        // End recording
        if (vk.vkEndCommandBuffer(self.command_buffer) != vk.VK_SUCCESS) {
            return VulkanError.CommandBufferEndFailed;
        }

        // Submit
        _ = vk.vkResetFences(self.device, 1, &self.fence);

        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        if (vk.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.fence) != vk.VK_SUCCESS) {
            return VulkanError.QueueSubmitFailed;
        }

        // Wait for completion
        if (vk.vkWaitForFences(self.device, 1, &self.fence, vk.VK_TRUE, std.math.maxInt(u64)) != vk.VK_SUCCESS) {
            return VulkanError.FenceWaitFailed;
        }
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    pub fn deinit(self: *Self) void {
        _ = vk.vkDeviceWaitIdle(self.device);
        vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        vk.vkDestroyFence(self.device, self.fence, null);
        vk.vkDestroyCommandPool(self.device, self.command_pool, null);
        vk.vkDestroyDevice(self.device, null);
        vk.vkDestroyInstance(self.instance, null);
    }

    // =========================================================================
    // Batched Command Buffer Recording
    // =========================================================================

    /// Begin recording a command buffer for multiple dispatches.
    /// Use cmdDispatch() to record individual dispatches, then submitAndWait().
    pub fn beginCommandBuffer(self: *Self) VulkanError!void {
        _ = vk.vkResetCommandBuffer(self.command_buffer, 0);

        const begin_info = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };

        if (vk.vkBeginCommandBuffer(self.command_buffer, &begin_info) != vk.VK_SUCCESS) {
            return VulkanError.CommandBufferBeginFailed;
        }
    }

    /// Record a compute dispatch into the current command buffer (no submit).
    /// Automatically inserts a compute→compute memory barrier before the dispatch.
    pub fn cmdDispatch(
        self: *Self,
        pipe: *const ComputePipeline,
        descriptor_set: vk.VkDescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        push_constants: ?[]const u8,
    ) void {
        // Memory barrier: ensure previous compute writes are visible
        const barrier = vk.VkMemoryBarrier{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        };
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );

        vk.vkCmdBindPipeline(self.command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipe.layout,
            0,
            1,
            &descriptor_set,
            0,
            null,
        );

        if (push_constants) |pc| {
            vk.vkCmdPushConstants(
                self.command_buffer,
                pipe.layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                @intCast(pc.len),
                pc.ptr,
            );
        }

        vk.vkCmdDispatch(self.command_buffer, group_count_x, group_count_y, group_count_z);
    }

    /// End recording and submit the command buffer, then wait for completion.
    pub fn submitAndWait(self: *Self) VulkanError!void {
        if (vk.vkEndCommandBuffer(self.command_buffer) != vk.VK_SUCCESS) {
            return VulkanError.CommandBufferEndFailed;
        }

        _ = vk.vkResetFences(self.device, 1, &self.fence);

        const submit_info = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        if (vk.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.fence) != vk.VK_SUCCESS) {
            return VulkanError.QueueSubmitFailed;
        }

        if (vk.vkWaitForFences(self.device, 1, &self.fence, vk.VK_TRUE, std.math.maxInt(u64)) != vk.VK_SUCCESS) {
            return VulkanError.FenceWaitFailed;
        }
    }
};
