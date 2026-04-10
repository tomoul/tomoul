// src/gpu/vulkan.zig
// Vulkan Compute Backend for Tomoul
//
// Provides GPU-accelerated compute via Vulkan 1.1+ compute shaders.
// Used for SGEMM, attention, and full transformer forward passes.
//
// ZERO link-time dependencies: Vulkan is loaded at runtime via dlopen/LoadLibrary.
// If libvulkan is not found, init() returns VulkanNotAvailable and the HAL falls
// back to CPU silently.
//
// Architecture:
//   init() → dlopen libvulkan → enumerate devices → create logical device + compute queue
//   createBuffer() → allocate device-local or host-visible buffers
//   upload() → transfer data to device-local memory via staging buffer
//   createPipeline() → load SPIR-V, create compute pipeline + descriptor sets
//   dispatch() → record command buffer, submit to queue, wait
//   readback() → copy device-local buffer to host-visible, map and read
//   deinit() → destroy all Vulkan resources → close library handle

const std = @import("std");
const vkl = @import("vk_loader");

pub const VulkanError = error{
    VulkanNotAvailable,
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
    buffer: vkl.VkBuffer,
    memory: vkl.VkDeviceMemory,
    size: vkl.VkDeviceSize,
    usage: vkl.VkBufferUsageFlags,
    is_host_visible: bool,
};

/// Compute pipeline handle
pub const ComputePipeline = struct {
    pipeline: vkl.VkPipeline,
    layout: vkl.VkPipelineLayout,
    descriptor_set_layout: vkl.VkDescriptorSetLayout,
    shader_module: vkl.VkShaderModule,
};

/// Vulkan compute context — single device, single compute queue
pub const VulkanContext = struct {
    allocator: std.mem.Allocator,

    // Runtime-loaded Vulkan dispatch table
    vk: vkl.VkLoader,

    // Core Vulkan objects
    instance: vkl.VkInstance,
    physical_device: vkl.VkPhysicalDevice,
    device: vkl.VkDevice,
    compute_queue: vkl.VkQueue,
    compute_queue_family: u32,

    // Command submission
    command_pool: vkl.VkCommandPool,
    command_buffer: vkl.VkCommandBuffer,
    fence: vkl.VkFence,

    // Descriptor pool (shared across pipelines)
    descriptor_pool: vkl.VkDescriptorPool,

    // Device properties
    device_name: [256]u8,
    device_name_len: usize,
    max_compute_work_group_count: [3]u32,
    max_compute_work_group_size: [3]u32,
    max_compute_shared_memory: u32,
    subgroup_size: u32,

    const Self = @This();

    /// Initialize Vulkan: dlopen → instance → physical device → logical device → queue → command pool
    /// Selects the best available compute device (discrete GPU preferred).
    /// Returns VulkanNotAvailable if Vulkan runtime library is not found.
    pub fn init(allocator: std.mem.Allocator) VulkanError!Self {
        var self: Self = undefined;
        self.allocator = allocator;

        // 0. Load Vulkan at runtime (zero link-time dependency)
        self.vk = vkl.VkLoader.init() catch return VulkanError.VulkanNotAvailable;
        errdefer self.vk.deinit();

        // 1. Create Vulkan instance
        const app_info = vkl.VkApplicationInfo{
            .pApplicationName = "Tomoul",
            .applicationVersion = vkl.VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "Tomoul GPU",
            .engineVersion = vkl.VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = vkl.VK_API_VERSION_1_1,
        };

        const instance_info = vkl.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
        };

        if (self.vk.vkCreateInstance(&instance_info, null, &self.instance) != vkl.VK_SUCCESS) {
            return VulkanError.InstanceCreationFailed;
        }
        errdefer self.vk.vkDestroyInstance(self.instance, null);

        // 2. Select physical device (prefer discrete GPU)
        var device_count: u32 = 0;
        _ = self.vk.vkEnumeratePhysicalDevices(self.instance, &device_count, null);
        if (device_count == 0) {
            return VulkanError.NoPhysicalDevice;
        }

        const devices = allocator.alloc(vkl.VkPhysicalDevice, device_count) catch return VulkanError.OutOfMemory;
        defer allocator.free(devices);
        _ = self.vk.vkEnumeratePhysicalDevices(self.instance, &device_count, devices.ptr);

        // Score devices: discrete > integrated > CPU
        var best_device: ?vkl.VkPhysicalDevice = null;
        var best_score: i32 = -1;
        var best_queue_family: u32 = 0;

        for (devices[0..device_count]) |dev| {
            var props: vkl.VkPhysicalDeviceProperties = .{};
            self.vk.vkGetPhysicalDeviceProperties(dev, &props);

            const score: i32 = switch (props.deviceType) {
                vkl.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 100,
                vkl.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 50,
                vkl.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 30,
                vkl.VK_PHYSICAL_DEVICE_TYPE_CPU => 10,
                else => 1,
            };

            // Find compute queue family
            var queue_count: u32 = 0;
            self.vk.vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, null);
            const queue_families = allocator.alloc(vkl.VkQueueFamilyProperties, queue_count) catch continue;
            defer allocator.free(queue_families);
            self.vk.vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, queue_families.ptr);

            var found_compute: bool = false;
            var compute_family: u32 = 0;
            for (queue_families[0..queue_count], 0..) |qf, idx| {
                if (qf.queueFlags & vkl.VK_QUEUE_COMPUTE_BIT != 0) {
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
        var props: vkl.VkPhysicalDeviceProperties = .{};
        self.vk.vkGetPhysicalDeviceProperties(self.physical_device, &props);
        const name_slice = std.mem.sliceTo(&props.deviceName, 0);
        self.device_name_len = @min(name_slice.len, 256);
        @memcpy(self.device_name[0..self.device_name_len], name_slice[0..self.device_name_len]);

        self.max_compute_work_group_count = props.limits.maxComputeWorkGroupCount;
        self.max_compute_work_group_size = props.limits.maxComputeWorkGroupSize;
        self.max_compute_shared_memory = props.limits.maxComputeSharedMemorySize;

        // Get subgroup properties
        var subgroup_props = vkl.VkPhysicalDeviceSubgroupProperties{};
        var dev_props2 = vkl.VkPhysicalDeviceProperties2{
            .pNext = @ptrCast(&subgroup_props),
        };
        self.vk.vkGetPhysicalDeviceProperties2(self.physical_device, &dev_props2);
        self.subgroup_size = subgroup_props.subgroupSize;

        // 3. Create logical device with compute queue
        const queue_priority: f32 = 1.0;
        const queue_create_info = vkl.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = self.compute_queue_family,
            .queueCount = 1,
            .pQueuePriorities = @ptrCast(&queue_priority),
        };

        const device_create_info = vkl.VkDeviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = @ptrCast(&queue_create_info),
        };

        if (self.vk.vkCreateDevice(self.physical_device, &device_create_info, null, &self.device) != vkl.VK_SUCCESS) {
            return VulkanError.DeviceCreationFailed;
        }
        errdefer self.vk.vkDestroyDevice(self.device, null);

        // Get compute queue
        self.vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0, &self.compute_queue);

        // 4. Create command pool
        const pool_info = vkl.VkCommandPoolCreateInfo{
            .flags = vkl.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.compute_queue_family,
        };

        if (self.vk.vkCreateCommandPool(self.device, &pool_info, null, &self.command_pool) != vkl.VK_SUCCESS) {
            return VulkanError.CommandPoolCreationFailed;
        }
        errdefer self.vk.vkDestroyCommandPool(self.device, self.command_pool, null);

        // 5. Allocate command buffer
        const alloc_info = vkl.VkCommandBufferAllocateInfo{
            .commandPool = self.command_pool,
            .commandBufferCount = 1,
        };

        if (self.vk.vkAllocateCommandBuffers(self.device, &alloc_info, @ptrCast(&self.command_buffer)) != vkl.VK_SUCCESS) {
            return VulkanError.CommandBufferAllocationFailed;
        }

        // 6. Create fence
        const fence_info = vkl.VkFenceCreateInfo{};

        if (self.vk.vkCreateFence(self.device, &fence_info, null, &self.fence) != vkl.VK_SUCCESS) {
            return VulkanError.FenceCreationFailed;
        }
        errdefer self.vk.vkDestroyFence(self.device, self.fence, null);

        // 7. Create descriptor pool (space for many storage buffer descriptors)
        const pool_sizes = [_]vkl.VkDescriptorPoolSize{
            .{
                .type = vkl.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 512,
            },
        };

        const desc_pool_info = vkl.VkDescriptorPoolCreateInfo{
            .flags = vkl.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 128,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        };

        if (self.vk.vkCreateDescriptorPool(self.device, &desc_pool_info, null, &self.descriptor_pool) != vkl.VK_SUCCESS) {
            return VulkanError.DescriptorPoolCreationFailed;
        }

        return self;
    }

    /// Get device name as a slice
    pub fn getDeviceName(self: *const Self) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    /// Reset the descriptor pool, freeing all allocated descriptor sets.
    pub fn resetDescriptorPool(self: *Self) void {
        _ = self.vk.vkResetDescriptorPool(self.device, self.descriptor_pool, 0);
    }

    /// Create a storage buffer (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    pub fn createStorageBuffer(self: *Self, size: usize, host_visible: bool) VulkanError!GpuBuffer {
        return self.createBuffer(size, vkl.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_visible);
    }

    // =========================================================================
    // Buffer Management
    // =========================================================================

    pub fn createBuffer(self: *Self, size: usize, usage: vkl.VkBufferUsageFlags, host_visible: bool) VulkanError!GpuBuffer {
        const buffer_info = vkl.VkBufferCreateInfo{
            .size = @intCast(size),
            .usage = usage,
        };

        var buffer: vkl.VkBuffer = null;
        if (self.vk.vkCreateBuffer(self.device, &buffer_info, null, &buffer) != vkl.VK_SUCCESS) {
            return VulkanError.BufferCreationFailed;
        }
        errdefer self.vk.vkDestroyBuffer(self.device, buffer, null);

        // Get memory requirements
        var mem_req: vkl.VkMemoryRequirements = .{};
        self.vk.vkGetBufferMemoryRequirements(self.device, buffer, &mem_req);

        // Find suitable memory type
        const required_props: vkl.VkMemoryPropertyFlags = if (host_visible)
            vkl.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vkl.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        else
            vkl.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        const mem_type_idx = self.findMemoryType(mem_req.memoryTypeBits, required_props) orelse {
            if (!host_visible) {
                const fallback_props = vkl.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vkl.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (self.findMemoryType(mem_req.memoryTypeBits, fallback_props)) |idx| {
                    return self.allocateAndBindBuffer(buffer, mem_req.size, idx, true);
                }
            }
            return VulkanError.MemoryAllocationFailed;
        };

        return self.allocateAndBindBuffer(buffer, mem_req.size, mem_type_idx, host_visible);
    }

    fn allocateAndBindBuffer(self: *Self, buffer: vkl.VkBuffer, size: vkl.VkDeviceSize, mem_type_idx: u32, host_visible: bool) VulkanError!GpuBuffer {
        const alloc_info = vkl.VkMemoryAllocateInfo{
            .allocationSize = size,
            .memoryTypeIndex = mem_type_idx,
        };

        var memory: vkl.VkDeviceMemory = null;
        if (self.vk.vkAllocateMemory(self.device, &alloc_info, null, &memory) != vkl.VK_SUCCESS) {
            return VulkanError.MemoryAllocationFailed;
        }
        errdefer self.vk.vkFreeMemory(self.device, memory, null);

        if (self.vk.vkBindBufferMemory(self.device, buffer, memory, 0) != vkl.VK_SUCCESS) {
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
        if (self.vk.vkMapMemory(self.device, buf.memory, 0, @intCast(data.len), 0, &mapped) != vkl.VK_SUCCESS) {
            return VulkanError.MemoryMapFailed;
        }
        const dst: [*]u8 = @ptrCast(mapped.?);
        @memcpy(dst[0..data.len], data);
        self.vk.vkUnmapMemory(self.device, buf.memory);
    }

    /// Read data from a host-visible buffer back to CPU
    pub fn readbackFromBuffer(self: *Self, buf: *const GpuBuffer, output: []u8) VulkanError!void {
        var mapped: ?*anyopaque = null;
        if (self.vk.vkMapMemory(self.device, buf.memory, 0, @intCast(output.len), 0, &mapped) != vkl.VK_SUCCESS) {
            return VulkanError.MemoryMapFailed;
        }
        const src: [*]const u8 = @ptrCast(mapped.?);
        @memcpy(output, src[0..output.len]);
        self.vk.vkUnmapMemory(self.device, buf.memory);
    }

    /// Destroy a GPU buffer and free its memory
    pub fn destroyBuffer(self: *Self, buf: *const GpuBuffer) void {
        self.vk.vkDestroyBuffer(self.device, buf.buffer, null);
        self.vk.vkFreeMemory(self.device, buf.memory, null);
    }

    fn findMemoryType(self: *const Self, type_filter: u32, properties: vkl.VkMemoryPropertyFlags) ?u32 {
        var mem_properties: vkl.VkPhysicalDeviceMemoryProperties = .{};
        self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mem_properties);

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
    pub fn createComputePipeline(
        self: *Self,
        spirv_code: []const u8,
        num_storage_buffers: u32,
        push_constant_size: u32,
    ) VulkanError!ComputePipeline {
        // SPIR-V requires 4-byte aligned pCode. @embedFile data may not be aligned.
        const needs_copy = @intFromPtr(spirv_code.ptr) % 4 != 0;
        const aligned_ptr: [*]align(4) const u8 = if (needs_copy) blk: {
            const aligned = self.allocator.alignedAlloc(u8, .@"4", spirv_code.len) catch return VulkanError.OutOfMemory;
            @memcpy(aligned, spirv_code);
            break :blk aligned.ptr;
        } else @ptrCast(@alignCast(spirv_code.ptr));
        defer if (needs_copy) {
            const slice: []align(4) u8 = @as([*]align(4) u8, @ptrCast(@constCast(aligned_ptr)))[0..spirv_code.len];
            self.allocator.free(slice);
        };

        // Create shader module
        const shader_info = vkl.VkShaderModuleCreateInfo{
            .codeSize = spirv_code.len,
            .pCode = @ptrCast(@alignCast(aligned_ptr)),
        };

        var shader_module: vkl.VkShaderModule = null;
        if (self.vk.vkCreateShaderModule(self.device, &shader_info, null, &shader_module) != vkl.VK_SUCCESS) {
            return VulkanError.ShaderModuleCreationFailed;
        }
        errdefer self.vk.vkDestroyShaderModule(self.device, shader_module, null);

        // Create descriptor set layout
        var bindings = self.allocator.alloc(vkl.VkDescriptorSetLayoutBinding, num_storage_buffers) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(bindings);

        for (0..num_storage_buffers) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = vkl.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = vkl.VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }

        const layout_info = vkl.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = num_storage_buffers,
            .pBindings = bindings.ptr,
        };

        var descriptor_set_layout: vkl.VkDescriptorSetLayout = null;
        if (self.vk.vkCreateDescriptorSetLayout(self.device, &layout_info, null, &descriptor_set_layout) != vkl.VK_SUCCESS) {
            return VulkanError.DescriptorPoolCreationFailed;
        }
        errdefer self.vk.vkDestroyDescriptorSetLayout(self.device, descriptor_set_layout, null);

        // Create pipeline layout (with optional push constants)
        var push_constant_range: vkl.VkPushConstantRange = .{};
        const has_push_constants = push_constant_size > 0;
        if (has_push_constants) {
            push_constant_range = .{
                .stageFlags = vkl.VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = push_constant_size,
            };
        }

        const pipeline_layout_info = vkl.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = @ptrCast(&descriptor_set_layout),
            .pushConstantRangeCount = if (has_push_constants) 1 else 0,
            .pPushConstantRanges = if (has_push_constants) @as(?[*]const vkl.VkPushConstantRange, @ptrCast(&push_constant_range)) else null,
        };

        var pipeline_layout: vkl.VkPipelineLayout = null;
        if (self.vk.vkCreatePipelineLayout(self.device, &pipeline_layout_info, null, &pipeline_layout) != vkl.VK_SUCCESS) {
            return VulkanError.PipelineLayoutCreationFailed;
        }
        errdefer self.vk.vkDestroyPipelineLayout(self.device, pipeline_layout, null);

        // Create compute pipeline
        const stage_info = vkl.VkPipelineShaderStageCreateInfo{
            .stage = vkl.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
        };

        const pipeline_info = vkl.VkComputePipelineCreateInfo{
            .stage = stage_info,
            .layout = pipeline_layout,
        };

        var pipeline: vkl.VkPipeline = null;
        if (self.vk.vkCreateComputePipelines(self.device, null, 1, @ptrCast(&pipeline_info), null, @ptrCast(&pipeline)) != vkl.VK_SUCCESS) {
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
        self.vk.vkDestroyPipeline(self.device, pipe.pipeline, null);
        self.vk.vkDestroyPipelineLayout(self.device, pipe.layout, null);
        self.vk.vkDestroyDescriptorSetLayout(self.device, pipe.descriptor_set_layout, null);
        self.vk.vkDestroyShaderModule(self.device, pipe.shader_module, null);
    }

    // =========================================================================
    // Dispatch
    // =========================================================================

    /// Allocate a descriptor set for a pipeline
    pub fn allocateDescriptorSet(self: *Self, pipe: *const ComputePipeline) VulkanError!vkl.VkDescriptorSet {
        const alloc_info = vkl.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = @ptrCast(&pipe.descriptor_set_layout),
        };

        var descriptor_set: vkl.VkDescriptorSet = null;
        if (self.vk.vkAllocateDescriptorSets(self.device, &alloc_info, @ptrCast(&descriptor_set)) != vkl.VK_SUCCESS) {
            return VulkanError.DescriptorSetAllocationFailed;
        }
        return descriptor_set;
    }

    /// Bind storage buffers to a descriptor set
    pub fn bindBuffers(self: *Self, descriptor_set: vkl.VkDescriptorSet, buffers: []const GpuBuffer) VulkanError!void {
        var writes = self.allocator.alloc(vkl.VkWriteDescriptorSet, buffers.len) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(writes);

        var buffer_infos = self.allocator.alloc(vkl.VkDescriptorBufferInfo, buffers.len) catch return VulkanError.OutOfMemory;
        defer self.allocator.free(buffer_infos);

        for (buffers, 0..) |buf, i| {
            buffer_infos[i] = .{
                .buffer = buf.buffer,
                .offset = 0,
                .range = vkl.VK_WHOLE_SIZE,
            };

            writes[i] = .{
                .dstSet = descriptor_set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = vkl.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[i],
            };
        }

        self.vk.vkUpdateDescriptorSets(self.device, @intCast(writes.len), writes.ptr, 0, null);
    }

    /// Dispatch a compute shader: record command buffer, submit, wait for completion.
    pub fn dispatch(
        self: *Self,
        pipe: *const ComputePipeline,
        descriptor_set: vkl.VkDescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        push_constants: ?[]const u8,
    ) VulkanError!void {
        _ = self.vk.vkResetCommandBuffer(self.command_buffer, 0);

        const begin_info = vkl.VkCommandBufferBeginInfo{
            .flags = vkl.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        if (self.vk.vkBeginCommandBuffer(self.command_buffer, &begin_info) != vkl.VK_SUCCESS) {
            return VulkanError.CommandBufferBeginFailed;
        }

        self.vk.vkCmdBindPipeline(self.command_buffer, vkl.VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        self.vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vkl.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipe.layout,
            0,
            1,
            @ptrCast(&descriptor_set),
            0,
            null,
        );

        if (push_constants) |pc| {
            self.vk.vkCmdPushConstants(
                self.command_buffer,
                pipe.layout,
                vkl.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                @intCast(pc.len),
                pc.ptr,
            );
        }

        self.vk.vkCmdDispatch(self.command_buffer, group_count_x, group_count_y, group_count_z);

        if (self.vk.vkEndCommandBuffer(self.command_buffer) != vkl.VK_SUCCESS) {
            return VulkanError.CommandBufferEndFailed;
        }

        _ = self.vk.vkResetFences(self.device, 1, @ptrCast(&self.fence));

        const submit_info = vkl.VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&self.command_buffer),
        };

        if (self.vk.vkQueueSubmit(self.compute_queue, 1, &submit_info, self.fence) != vkl.VK_SUCCESS) {
            return VulkanError.QueueSubmitFailed;
        }

        if (self.vk.vkWaitForFences(self.device, 1, @ptrCast(&self.fence), vkl.VK_TRUE, std.math.maxInt(u64)) != vkl.VK_SUCCESS) {
            return VulkanError.FenceWaitFailed;
        }
    }

    // =========================================================================
    // Cleanup
    // =========================================================================

    pub fn deinit(self: *Self) void {
        _ = self.vk.vkDeviceWaitIdle(self.device);
        self.vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.vk.vkDestroyFence(self.device, self.fence, null);
        self.vk.vkDestroyCommandPool(self.device, self.command_pool, null);
        self.vk.vkDestroyDevice(self.device, null);
        self.vk.vkDestroyInstance(self.instance, null);
        self.vk.deinit();
    }

    // =========================================================================
    // Batched Command Buffer Recording
    // =========================================================================

    /// Begin recording a command buffer for multiple dispatches.
    pub fn beginCommandBuffer(self: *Self) VulkanError!void {
        _ = self.vk.vkResetCommandBuffer(self.command_buffer, 0);

        const begin_info = vkl.VkCommandBufferBeginInfo{
            .flags = vkl.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        if (self.vk.vkBeginCommandBuffer(self.command_buffer, &begin_info) != vkl.VK_SUCCESS) {
            return VulkanError.CommandBufferBeginFailed;
        }
    }

    /// Record a compute dispatch into the current command buffer (no submit).
    pub fn cmdDispatch(
        self: *Self,
        pipe: *const ComputePipeline,
        descriptor_set: vkl.VkDescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
        push_constants: ?[]const u8,
    ) void {
        const barrier = vkl.VkMemoryBarrier{
            .srcAccessMask = vkl.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vkl.VK_ACCESS_SHADER_READ_BIT | vkl.VK_ACCESS_SHADER_WRITE_BIT,
        };
        self.vk.vkCmdPipelineBarrier(
            self.command_buffer,
            vkl.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vkl.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            @ptrCast(&barrier),
            0,
            null,
            0,
            null,
        );

        self.vk.vkCmdBindPipeline(self.command_buffer, vkl.VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        self.vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vkl.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipe.layout,
            0,
            1,
            @ptrCast(&descriptor_set),
            0,
            null,
        );

        if (push_constants) |pc| {
            self.vk.vkCmdPushConstants(
                self.command_buffer,
                pipe.layout,
                vkl.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                @intCast(pc.len),
                pc.ptr,
            );
        }

        self.vk.vkCmdDispatch(self.command_buffer, group_count_x, group_count_y, group_count_z);
    }

    /// End recording and submit the command buffer, then wait for completion.
    pub fn submitAndWait(self: *Self) VulkanError!void {
        if (self.vk.vkEndCommandBuffer(self.command_buffer) != vkl.VK_SUCCESS) {
            return VulkanError.CommandBufferEndFailed;
        }

        _ = self.vk.vkResetFences(self.device, 1, @ptrCast(&self.fence));

        const submit_info = vkl.VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&self.command_buffer),
        };

        if (self.vk.vkQueueSubmit(self.compute_queue, 1, @ptrCast(&submit_info), self.fence) != vkl.VK_SUCCESS) {
            return VulkanError.QueueSubmitFailed;
        }

        if (self.vk.vkWaitForFences(self.device, 1, @ptrCast(&self.fence), vkl.VK_TRUE, std.math.maxInt(u64)) != vkl.VK_SUCCESS) {
            return VulkanError.FenceWaitFailed;
        }
    }
};
