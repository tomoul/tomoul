// src/gpu/vk_loader.zig
//
// Runtime Vulkan Function Loader
//
// Loads libvulkan at runtime via dlopen/LoadLibrary — ZERO link-time dependency.
// If the Vulkan library isn't found, init() returns error.VulkanNotAvailable
// and the HAL falls back to CPU.
//
// Usage:
//   const vk = try VkLoader.init();    // loads libvulkan + resolves all function pointers
//   defer vk.deinit();                 // closes the library handle
//   const instance = vk.createInstance(...);
//
// Platform dispatch:
//   Linux/Android/WSL2 → dlopen("libvulkan.so.1")
//   Windows            → LoadLibraryA("vulkan-1.dll")

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Vulkan Type Definitions (no @cImport needed)
// ============================================================================

// Opaque handles (pointers)
pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};
pub const VkCommandPool = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkFence = ?*opaque {};
pub const VkBuffer = ?*opaque {};
pub const VkDeviceMemory = ?*opaque {};
pub const VkShaderModule = ?*opaque {};
pub const VkPipeline = ?*opaque {};
pub const VkPipelineLayout = ?*opaque {};
pub const VkPipelineCache = ?*opaque {};
pub const VkDescriptorPool = ?*opaque {};
pub const VkDescriptorSet = ?*opaque {};
pub const VkDescriptorSetLayout = ?*opaque {};
pub const VkSemaphore = ?*opaque {};

// Numeric types
pub const VkResult = i32;
pub const VkBool32 = u32;
pub const VkDeviceSize = u64;
pub const VkFlags = u32;
pub const VkBufferUsageFlags = VkFlags;
pub const VkMemoryPropertyFlags = VkFlags;
pub const VkCommandBufferUsageFlags = VkFlags;
pub const VkCommandPoolCreateFlags = VkFlags;
pub const VkDescriptorPoolCreateFlags = VkFlags;
pub const VkPipelineStageFlags = VkFlags;
pub const VkAccessFlags = VkFlags;
pub const VkShaderStageFlags = VkFlags;
pub const VkDependencyFlags = VkFlags;
pub const VkFenceCreateFlags = VkFlags;
pub const VkCommandBufferResetFlags = VkFlags;
pub const VkDescriptorPoolResetFlags = VkFlags;

// Allocation callbacks (always null for us)
pub const VkAllocationCallbacks = opaque {};

// ============================================================================
// Vulkan Constants
// ============================================================================

pub const VK_SUCCESS: VkResult = 0;
pub const VK_TRUE: VkBool32 = 1;
pub const VK_FALSE: VkBool32 = 0;
pub const VK_WHOLE_SIZE: VkDeviceSize = ~@as(VkDeviceSize, 0);

// Structure types
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: i32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: i32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: i32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: i32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: i32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: i32 = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: i32 = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: i32 = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: i32 = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: i32 = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: i32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: i32 = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: i32 = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: i32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: i32 = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: i32 = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: i32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: i32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: i32 = 42;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER: i32 = 46;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: i32 = 1000094000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: i32 = 1000059001;

// API version
pub inline fn VK_MAKE_VERSION(major: u32, minor: u32, patch: u32) u32 {
    return (major << 22) | (minor << 12) | patch;
}
pub const VK_API_VERSION_1_1: u32 = VK_MAKE_VERSION(1, 1, 0);

// Physical device types
pub const VK_PHYSICAL_DEVICE_TYPE_OTHER: i32 = 0;
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: i32 = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: i32 = 2;
pub const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: i32 = 3;
pub const VK_PHYSICAL_DEVICE_TYPE_CPU: i32 = 4;

// Queue flags
pub const VK_QUEUE_COMPUTE_BIT: VkFlags = 0x00000002;

// Buffer usage
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;

// Memory property flags
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;

// Sharing mode
pub const VK_SHARING_MODE_EXCLUSIVE: i32 = 0;

// Descriptor type
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: i32 = 7;

// Shader stage
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlags = 0x00000020;

// Pipeline bind point
pub const VK_PIPELINE_BIND_POINT_COMPUTE: i32 = 1;

// Pipeline stage
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkPipelineStageFlags = 0x00000800;

// Access flags
pub const VK_ACCESS_SHADER_READ_BIT: VkAccessFlags = 0x00000020;
pub const VK_ACCESS_SHADER_WRITE_BIT: VkAccessFlags = 0x00000040;

// Command buffer level
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: i32 = 0;

// Command buffer usage
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkCommandBufferUsageFlags = 0x00000001;

// Command pool create flags
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkCommandPoolCreateFlags = 0x00000002;

// Descriptor pool create flags
pub const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: VkDescriptorPoolCreateFlags = 0x00000001;

// ============================================================================
// Vulkan Structures
// ============================================================================

pub const VkApplicationInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

pub const VkInstanceCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 0,
    pQueuePriorities: ?[*]const f32 = null,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

pub const VkBufferCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize = 0,
    usage: VkBufferUsageFlags = 0,
    sharingMode: i32 = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

pub const VkMemoryAllocateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize = 0,
    alignment: VkDeviceSize = 0,
    memoryTypeBits: u32 = 0,
};

pub const VkMemoryType = extern struct {
    propertyFlags: VkMemoryPropertyFlags = 0,
    heapIndex: u32 = 0,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize = 0,
    flags: VkFlags = 0,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32 = 0,
    memoryTypes: [32]VkMemoryType = [_]VkMemoryType{.{}} ** 32,
    memoryHeapCount: u32 = 0,
    memoryHeaps: [16]VkMemoryHeap = [_]VkMemoryHeap{.{}} ** 16,
};

pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32 = 0,
    maxImageDimension2D: u32 = 0,
    maxImageDimension3D: u32 = 0,
    maxImageDimensionCube: u32 = 0,
    maxImageArrayLayers: u32 = 0,
    maxTexelBufferElements: u32 = 0,
    maxUniformBufferRange: u32 = 0,
    maxStorageBufferRange: u32 = 0,
    maxPushConstantsSize: u32 = 0,
    maxMemoryAllocationCount: u32 = 0,
    maxSamplerAllocationCount: u32 = 0,
    bufferImageGranularity: VkDeviceSize = 0,
    sparseAddressSpaceSize: VkDeviceSize = 0,
    maxBoundDescriptorSets: u32 = 0,
    maxPerStageDescriptorSamplers: u32 = 0,
    maxPerStageDescriptorUniformBuffers: u32 = 0,
    maxPerStageDescriptorStorageBuffers: u32 = 0,
    maxPerStageDescriptorSampledImages: u32 = 0,
    maxPerStageDescriptorStorageImages: u32 = 0,
    maxPerStageDescriptorInputAttachments: u32 = 0,
    maxPerStageResources: u32 = 0,
    maxDescriptorSetSamplers: u32 = 0,
    maxDescriptorSetUniformBuffers: u32 = 0,
    maxDescriptorSetUniformBuffersDynamic: u32 = 0,
    maxDescriptorSetStorageBuffers: u32 = 0,
    maxDescriptorSetStorageBuffersDynamic: u32 = 0,
    maxDescriptorSetSampledImages: u32 = 0,
    maxDescriptorSetStorageImages: u32 = 0,
    maxDescriptorSetInputAttachments: u32 = 0,
    maxVertexInputAttributes: u32 = 0,
    maxVertexInputBindings: u32 = 0,
    maxVertexInputAttributeOffset: u32 = 0,
    maxVertexInputBindingStride: u32 = 0,
    maxVertexOutputComponents: u32 = 0,
    maxTessellationGenerationLevel: u32 = 0,
    maxTessellationPatchSize: u32 = 0,
    maxTessellationControlPerVertexInputComponents: u32 = 0,
    maxTessellationControlPerVertexOutputComponents: u32 = 0,
    maxTessellationControlPerPatchOutputComponents: u32 = 0,
    maxTessellationControlTotalOutputComponents: u32 = 0,
    maxTessellationEvaluationInputComponents: u32 = 0,
    maxTessellationEvaluationOutputComponents: u32 = 0,
    maxGeometryShaderInvocations: u32 = 0,
    maxGeometryInputComponents: u32 = 0,
    maxGeometryOutputComponents: u32 = 0,
    maxGeometryOutputVertices: u32 = 0,
    maxGeometryTotalOutputComponents: u32 = 0,
    maxFragmentInputComponents: u32 = 0,
    maxFragmentOutputAttachments: u32 = 0,
    maxFragmentDualSrcAttachments: u32 = 0,
    maxFragmentCombinedOutputResources: u32 = 0,
    maxComputeSharedMemorySize: u32 = 0,
    maxComputeWorkGroupCount: [3]u32 = .{ 0, 0, 0 },
    maxComputeWorkGroupInvocations: u32 = 0,
    maxComputeWorkGroupSize: [3]u32 = .{ 0, 0, 0 },
    subPixelPrecisionBits: u32 = 0,
    subTexelPrecisionBits: u32 = 0,
    mipmapPrecisionBits: u32 = 0,
    maxDrawIndexedIndexValue: u32 = 0,
    maxDrawIndirectCount: u32 = 0,
    maxSamplerLodBias: f32 = 0,
    maxSamplerAnisotropy: f32 = 0,
    maxViewports: u32 = 0,
    maxViewportDimensions: [2]u32 = .{ 0, 0 },
    viewportBoundsRange: [2]f32 = .{ 0, 0 },
    viewportSubPixelBits: u32 = 0,
    minMemoryMapAlignment: usize = 0,
    minTexelBufferOffsetAlignment: VkDeviceSize = 0,
    minUniformBufferOffsetAlignment: VkDeviceSize = 0,
    minStorageBufferOffsetAlignment: VkDeviceSize = 0,
    minTexelOffset: i32 = 0,
    maxTexelOffset: u32 = 0,
    minTexelGatherOffset: i32 = 0,
    maxTexelGatherOffset: u32 = 0,
    minInterpolationOffset: f32 = 0,
    maxInterpolationOffset: f32 = 0,
    subPixelInterpolationOffsetBits: u32 = 0,
    maxFramebufferWidth: u32 = 0,
    maxFramebufferHeight: u32 = 0,
    maxFramebufferLayers: u32 = 0,
    framebufferColorSampleCounts: VkFlags = 0,
    framebufferDepthSampleCounts: VkFlags = 0,
    framebufferStencilSampleCounts: VkFlags = 0,
    framebufferNoAttachmentsSampleCounts: VkFlags = 0,
    maxColorAttachments: u32 = 0,
    sampledImageColorSampleCounts: VkFlags = 0,
    sampledImageIntegerSampleCounts: VkFlags = 0,
    sampledImageDepthSampleCounts: VkFlags = 0,
    sampledImageStencilSampleCounts: VkFlags = 0,
    storageImageSampleCounts: VkFlags = 0,
    maxSampleMaskWords: u32 = 0,
    timestampComputeAndGraphics: VkBool32 = 0,
    timestampPeriod: f32 = 0,
    maxClipDistances: u32 = 0,
    maxCullDistances: u32 = 0,
    maxCombinedClipAndCullDistances: u32 = 0,
    discreteQueuePriorities: u32 = 0,
    pointSizeRange: [2]f32 = .{ 0, 0 },
    lineWidthRange: [2]f32 = .{ 0, 0 },
    pointSizeGranularity: f32 = 0,
    lineWidthGranularity: f32 = 0,
    strictLines: VkBool32 = 0,
    standardSampleLocations: VkBool32 = 0,
    optimalBufferCopyOffsetAlignment: VkDeviceSize = 0,
    optimalBufferCopyRowPitchAlignment: VkDeviceSize = 0,
    nonCoherentAtomSize: VkDeviceSize = 0,
};

pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: VkBool32 = 0,
    residencyStandard2DMultisampleBlockShape: VkBool32 = 0,
    residencyStandard3DBlockShape: VkBool32 = 0,
    residencyAlignedMipSize: VkBool32 = 0,
    residencyNonResidentStrict: VkBool32 = 0,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32 = 0,
    driverVersion: u32 = 0,
    vendorID: u32 = 0,
    deviceID: u32 = 0,
    deviceType: i32 = 0,
    deviceName: [256]u8 = [_]u8{0} ** 256,
    pipelineCacheUUID: [16]u8 = [_]u8{0} ** 16,
    limits: VkPhysicalDeviceLimits = .{},
    sparseProperties: VkPhysicalDeviceSparseProperties = .{},
};

pub const VkPhysicalDeviceProperties2 = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    properties: VkPhysicalDeviceProperties = .{},
};

pub const VkPhysicalDeviceSubgroupProperties = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    pNext: ?*anyopaque = null,
    subgroupSize: u32 = 0,
    supportedStages: VkFlags = 0,
    supportedOperations: VkFlags = 0,
    quadOperationsInAllStages: VkBool32 = 0,
};

pub const VkQueueFamilyProperties = extern struct {
    queueFlags: VkFlags = 0,
    queueCount: u32 = 0,
    timestampValidBits: u32 = 0,
    minImageTransferGranularity: extern struct {
        width: u32 = 0,
        height: u32 = 0,
        depth: u32 = 0,
    } = .{},
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize = 0,
    pCode: ?[*]const u32 = null,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: i32 = 0,
    descriptorCount: u32 = 0,
    stageFlags: VkShaderStageFlags = 0,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

pub const VkPushConstantRange = extern struct {
    stageFlags: VkShaderStageFlags = 0,
    offset: u32 = 0,
    size: u32 = 0,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?[*]const VkPushConstantRange = null,
};

pub const VkSpecializationInfo = opaque {};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkShaderStageFlags = 0,
    module: VkShaderModule = null,
    pName: ?[*:0]const u8 = null,
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo = .{},
    layout: VkPipelineLayout = null,
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

pub const VkDescriptorPoolSize = extern struct {
    type: i32 = 0,
    descriptorCount: u32 = 0,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorPoolCreateFlags = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool = null,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer = null,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = 0,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet = null,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 0,
    descriptorType: i32 = 0,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandPoolCreateFlags = 0,
    queueFamilyIndex: u32 = 0,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = null,
    level: i32 = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32 = 0,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandBufferUsageFlags = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const VkSubmitInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?[*]const VkSemaphore = null,
    pWaitDstStageMask: ?[*]const VkPipelineStageFlags = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?[*]const VkSemaphore = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFenceCreateFlags = 0,
};

pub const VkMemoryBarrier = extern struct {
    sType: i32 = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkAccessFlags = 0,
    dstAccessMask: VkAccessFlags = 0,
};

// ============================================================================
// Function Pointer Types
// ============================================================================

pub const PFN_vkCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const VkAllocationCallbacks, *VkInstance) callconv(.c) VkResult;
pub const PFN_vkDestroyInstance = *const fn (VkInstance, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
pub const PFN_vkGetPhysicalDeviceProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceProperties2 = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties2) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceQueueFamilyProperties = *const fn (VkPhysicalDevice, *u32, ?[*]VkQueueFamilyProperties) callconv(.c) void;
pub const PFN_vkCreateDevice = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*const VkAllocationCallbacks, *VkDevice) callconv(.c) VkResult;
pub const PFN_vkDestroyDevice = *const fn (VkDevice, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkGetDeviceQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
pub const PFN_vkCreateBuffer = *const fn (VkDevice, *const VkBufferCreateInfo, ?*const VkAllocationCallbacks, *VkBuffer) callconv(.c) VkResult;
pub const PFN_vkDestroyBuffer = *const fn (VkDevice, VkBuffer, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkGetBufferMemoryRequirements = *const fn (VkDevice, VkBuffer, *VkMemoryRequirements) callconv(.c) void;
pub const PFN_vkAllocateMemory = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*const VkAllocationCallbacks, *VkDeviceMemory) callconv(.c) VkResult;
pub const PFN_vkFreeMemory = *const fn (VkDevice, VkDeviceMemory, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkBindBufferMemory = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.c) VkResult;
pub const PFN_vkMapMemory = *const fn (VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, u32, *?*anyopaque) callconv(.c) VkResult;
pub const PFN_vkUnmapMemory = *const fn (VkDevice, VkDeviceMemory) callconv(.c) void;
pub const PFN_vkCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const VkAllocationCallbacks, *VkShaderModule) callconv(.c) VkResult;
pub const PFN_vkDestroyShaderModule = *const fn (VkDevice, VkShaderModule, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkCreateDescriptorSetLayout = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*const VkAllocationCallbacks, *VkDescriptorSetLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorSetLayout = *const fn (VkDevice, VkDescriptorSetLayout, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const VkAllocationCallbacks, *VkPipelineLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyPipelineLayout = *const fn (VkDevice, VkPipelineLayout, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkCreateComputePipelines = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*const VkAllocationCallbacks, [*]VkPipeline) callconv(.c) VkResult;
pub const PFN_vkDestroyPipeline = *const fn (VkDevice, VkPipeline, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkCreateDescriptorPool = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*const VkAllocationCallbacks, *VkDescriptorPool) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorPool = *const fn (VkDevice, VkDescriptorPool, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkResetDescriptorPool = *const fn (VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags) callconv(.c) VkResult;
pub const PFN_vkAllocateDescriptorSets = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(.c) VkResult;
pub const PFN_vkUpdateDescriptorSets = *const fn (VkDevice, u32, [*]const VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreateCommandPool = *const fn (VkDevice, *const VkCommandPoolCreateInfo, ?*const VkAllocationCallbacks, *VkCommandPool) callconv(.c) VkResult;
pub const PFN_vkDestroyCommandPool = *const fn (VkDevice, VkCommandPool, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkAllocateCommandBuffers = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.c) VkResult;
pub const PFN_vkBeginCommandBuffer = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub const PFN_vkEndCommandBuffer = *const fn (VkCommandBuffer) callconv(.c) VkResult;
pub const PFN_vkResetCommandBuffer = *const fn (VkCommandBuffer, VkCommandBufferResetFlags) callconv(.c) VkResult;
pub const PFN_vkCmdBindPipeline = *const fn (VkCommandBuffer, i32, VkPipeline) callconv(.c) void;
pub const PFN_vkCmdBindDescriptorSets = *const fn (VkCommandBuffer, i32, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?[*]const u32) callconv(.c) void;
pub const PFN_vkCmdPushConstants = *const fn (VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, u32, u32, *const anyopaque) callconv(.c) void;
pub const PFN_vkCmdDispatch = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;
pub const PFN_vkCmdPipelineBarrier = *const fn (VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, u32, ?[*]const VkMemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(.c) void;
pub const PFN_vkQueueSubmit = *const fn (VkQueue, u32, [*]const VkSubmitInfo, VkFence) callconv(.c) VkResult;
pub const PFN_vkCreateFence = *const fn (VkDevice, *const VkFenceCreateInfo, ?*const VkAllocationCallbacks, *VkFence) callconv(.c) VkResult;
pub const PFN_vkDestroyFence = *const fn (VkDevice, VkFence, ?*const VkAllocationCallbacks) callconv(.c) void;
pub const PFN_vkResetFences = *const fn (VkDevice, u32, [*]const VkFence) callconv(.c) VkResult;
pub const PFN_vkWaitForFences = *const fn (VkDevice, u32, [*]const VkFence, VkBool32, u64) callconv(.c) VkResult;
pub const PFN_vkDeviceWaitIdle = *const fn (VkDevice) callconv(.c) VkResult;

// ============================================================================
// vkGetInstanceProcAddr — the one function we load by symbol name
// ============================================================================
pub const PFN_vkGetInstanceProcAddr = *const fn (VkInstance, [*:0]const u8) callconv(.c) ?*const fn () callconv(.c) void;

// ============================================================================
// Vulkan Loader — runtime dispatch table
// ============================================================================

pub const VkLoader = struct {
    lib: std.DynLib,

    // Function pointers (populated from vkGetInstanceProcAddr)
    vkCreateInstance: PFN_vkCreateInstance,
    vkDestroyInstance: PFN_vkDestroyInstance,
    vkEnumeratePhysicalDevices: PFN_vkEnumeratePhysicalDevices,
    vkGetPhysicalDeviceProperties: PFN_vkGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceProperties2: PFN_vkGetPhysicalDeviceProperties2,
    vkGetPhysicalDeviceMemoryProperties: PFN_vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceQueueFamilyProperties: PFN_vkGetPhysicalDeviceQueueFamilyProperties,
    vkCreateDevice: PFN_vkCreateDevice,
    vkDestroyDevice: PFN_vkDestroyDevice,
    vkGetDeviceQueue: PFN_vkGetDeviceQueue,
    vkCreateBuffer: PFN_vkCreateBuffer,
    vkDestroyBuffer: PFN_vkDestroyBuffer,
    vkGetBufferMemoryRequirements: PFN_vkGetBufferMemoryRequirements,
    vkAllocateMemory: PFN_vkAllocateMemory,
    vkFreeMemory: PFN_vkFreeMemory,
    vkBindBufferMemory: PFN_vkBindBufferMemory,
    vkMapMemory: PFN_vkMapMemory,
    vkUnmapMemory: PFN_vkUnmapMemory,
    vkCreateShaderModule: PFN_vkCreateShaderModule,
    vkDestroyShaderModule: PFN_vkDestroyShaderModule,
    vkCreateDescriptorSetLayout: PFN_vkCreateDescriptorSetLayout,
    vkDestroyDescriptorSetLayout: PFN_vkDestroyDescriptorSetLayout,
    vkCreatePipelineLayout: PFN_vkCreatePipelineLayout,
    vkDestroyPipelineLayout: PFN_vkDestroyPipelineLayout,
    vkCreateComputePipelines: PFN_vkCreateComputePipelines,
    vkDestroyPipeline: PFN_vkDestroyPipeline,
    vkCreateDescriptorPool: PFN_vkCreateDescriptorPool,
    vkDestroyDescriptorPool: PFN_vkDestroyDescriptorPool,
    vkResetDescriptorPool: PFN_vkResetDescriptorPool,
    vkAllocateDescriptorSets: PFN_vkAllocateDescriptorSets,
    vkUpdateDescriptorSets: PFN_vkUpdateDescriptorSets,
    vkCreateCommandPool: PFN_vkCreateCommandPool,
    vkDestroyCommandPool: PFN_vkDestroyCommandPool,
    vkAllocateCommandBuffers: PFN_vkAllocateCommandBuffers,
    vkBeginCommandBuffer: PFN_vkBeginCommandBuffer,
    vkEndCommandBuffer: PFN_vkEndCommandBuffer,
    vkResetCommandBuffer: PFN_vkResetCommandBuffer,
    vkCmdBindPipeline: PFN_vkCmdBindPipeline,
    vkCmdBindDescriptorSets: PFN_vkCmdBindDescriptorSets,
    vkCmdPushConstants: PFN_vkCmdPushConstants,
    vkCmdDispatch: PFN_vkCmdDispatch,
    vkCmdPipelineBarrier: PFN_vkCmdPipelineBarrier,
    vkQueueSubmit: PFN_vkQueueSubmit,
    vkCreateFence: PFN_vkCreateFence,
    vkDestroyFence: PFN_vkDestroyFence,
    vkResetFences: PFN_vkResetFences,
    vkWaitForFences: PFN_vkWaitForFences,
    vkDeviceWaitIdle: PFN_vkDeviceWaitIdle,

    pub const LoadError = error{VulkanNotAvailable};

    /// Load the Vulkan runtime library and resolve all function pointers.
    /// Returns error.VulkanNotAvailable if libvulkan can't be found.
    pub fn init() LoadError!VkLoader {
        // Platform-specific library name
        const lib_name: [:0]const u8 = switch (builtin.os.tag) {
            .windows => "vulkan-1.dll",
            .linux => "libvulkan.so.1",
            else => "libvulkan.so.1", // Android, etc.
        };

        var lib = std.DynLib.open(lib_name) catch return error.VulkanNotAvailable;
        errdefer lib.close();

        // Load the one bootstrap function
        const getProc: PFN_vkGetInstanceProcAddr = lib.lookup(PFN_vkGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse
            return error.VulkanNotAvailable;

        // Pre-instance functions (instance=null)
        const createInstance = resolve(PFN_vkCreateInstance, getProc, null, "vkCreateInstance") orelse
            return error.VulkanNotAvailable;

        // Create a temporary instance to resolve the rest
        const app_info = VkApplicationInfo{
            .pApplicationName = "tomoul-probe",
            .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
            .pEngineName = "tomoul",
            .engineVersion = VK_MAKE_VERSION(0, 0, 1),
            .apiVersion = VK_API_VERSION_1_1,
        };

        const instance_info = VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
        };

        var probe_instance: VkInstance = null;
        if (createInstance(&instance_info, null, &probe_instance) != VK_SUCCESS) {
            return error.VulkanNotAvailable;
        }

        // Resolve all functions against the real instance
        const destroyInstance = resolve(PFN_vkDestroyInstance, getProc, probe_instance, "vkDestroyInstance") orelse {
            // Can't even destroy — just leak and fail
            return error.VulkanNotAvailable;
        };

        // Build the loader with all function pointers BEFORE destroying probe instance
        const loader = VkLoader{
            .lib = lib,
            .vkCreateInstance = createInstance,
            .vkDestroyInstance = destroyInstance,
            .vkEnumeratePhysicalDevices = resolve(PFN_vkEnumeratePhysicalDevices, getProc, probe_instance, "vkEnumeratePhysicalDevices") orelse return error.VulkanNotAvailable,
            .vkGetPhysicalDeviceProperties = resolve(PFN_vkGetPhysicalDeviceProperties, getProc, probe_instance, "vkGetPhysicalDeviceProperties") orelse return error.VulkanNotAvailable,
            .vkGetPhysicalDeviceProperties2 = resolve(PFN_vkGetPhysicalDeviceProperties2, getProc, probe_instance, "vkGetPhysicalDeviceProperties2") orelse return error.VulkanNotAvailable,
            .vkGetPhysicalDeviceMemoryProperties = resolve(PFN_vkGetPhysicalDeviceMemoryProperties, getProc, probe_instance, "vkGetPhysicalDeviceMemoryProperties") orelse return error.VulkanNotAvailable,
            .vkGetPhysicalDeviceQueueFamilyProperties = resolve(PFN_vkGetPhysicalDeviceQueueFamilyProperties, getProc, probe_instance, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return error.VulkanNotAvailable,
            .vkCreateDevice = resolve(PFN_vkCreateDevice, getProc, probe_instance, "vkCreateDevice") orelse return error.VulkanNotAvailable,
            .vkDestroyDevice = resolve(PFN_vkDestroyDevice, getProc, probe_instance, "vkDestroyDevice") orelse return error.VulkanNotAvailable,
            .vkGetDeviceQueue = resolve(PFN_vkGetDeviceQueue, getProc, probe_instance, "vkGetDeviceQueue") orelse return error.VulkanNotAvailable,
            .vkCreateBuffer = resolve(PFN_vkCreateBuffer, getProc, probe_instance, "vkCreateBuffer") orelse return error.VulkanNotAvailable,
            .vkDestroyBuffer = resolve(PFN_vkDestroyBuffer, getProc, probe_instance, "vkDestroyBuffer") orelse return error.VulkanNotAvailable,
            .vkGetBufferMemoryRequirements = resolve(PFN_vkGetBufferMemoryRequirements, getProc, probe_instance, "vkGetBufferMemoryRequirements") orelse return error.VulkanNotAvailable,
            .vkAllocateMemory = resolve(PFN_vkAllocateMemory, getProc, probe_instance, "vkAllocateMemory") orelse return error.VulkanNotAvailable,
            .vkFreeMemory = resolve(PFN_vkFreeMemory, getProc, probe_instance, "vkFreeMemory") orelse return error.VulkanNotAvailable,
            .vkBindBufferMemory = resolve(PFN_vkBindBufferMemory, getProc, probe_instance, "vkBindBufferMemory") orelse return error.VulkanNotAvailable,
            .vkMapMemory = resolve(PFN_vkMapMemory, getProc, probe_instance, "vkMapMemory") orelse return error.VulkanNotAvailable,
            .vkUnmapMemory = resolve(PFN_vkUnmapMemory, getProc, probe_instance, "vkUnmapMemory") orelse return error.VulkanNotAvailable,
            .vkCreateShaderModule = resolve(PFN_vkCreateShaderModule, getProc, probe_instance, "vkCreateShaderModule") orelse return error.VulkanNotAvailable,
            .vkDestroyShaderModule = resolve(PFN_vkDestroyShaderModule, getProc, probe_instance, "vkDestroyShaderModule") orelse return error.VulkanNotAvailable,
            .vkCreateDescriptorSetLayout = resolve(PFN_vkCreateDescriptorSetLayout, getProc, probe_instance, "vkCreateDescriptorSetLayout") orelse return error.VulkanNotAvailable,
            .vkDestroyDescriptorSetLayout = resolve(PFN_vkDestroyDescriptorSetLayout, getProc, probe_instance, "vkDestroyDescriptorSetLayout") orelse return error.VulkanNotAvailable,
            .vkCreatePipelineLayout = resolve(PFN_vkCreatePipelineLayout, getProc, probe_instance, "vkCreatePipelineLayout") orelse return error.VulkanNotAvailable,
            .vkDestroyPipelineLayout = resolve(PFN_vkDestroyPipelineLayout, getProc, probe_instance, "vkDestroyPipelineLayout") orelse return error.VulkanNotAvailable,
            .vkCreateComputePipelines = resolve(PFN_vkCreateComputePipelines, getProc, probe_instance, "vkCreateComputePipelines") orelse return error.VulkanNotAvailable,
            .vkDestroyPipeline = resolve(PFN_vkDestroyPipeline, getProc, probe_instance, "vkDestroyPipeline") orelse return error.VulkanNotAvailable,
            .vkCreateDescriptorPool = resolve(PFN_vkCreateDescriptorPool, getProc, probe_instance, "vkCreateDescriptorPool") orelse return error.VulkanNotAvailable,
            .vkDestroyDescriptorPool = resolve(PFN_vkDestroyDescriptorPool, getProc, probe_instance, "vkDestroyDescriptorPool") orelse return error.VulkanNotAvailable,
            .vkResetDescriptorPool = resolve(PFN_vkResetDescriptorPool, getProc, probe_instance, "vkResetDescriptorPool") orelse return error.VulkanNotAvailable,
            .vkAllocateDescriptorSets = resolve(PFN_vkAllocateDescriptorSets, getProc, probe_instance, "vkAllocateDescriptorSets") orelse return error.VulkanNotAvailable,
            .vkUpdateDescriptorSets = resolve(PFN_vkUpdateDescriptorSets, getProc, probe_instance, "vkUpdateDescriptorSets") orelse return error.VulkanNotAvailable,
            .vkCreateCommandPool = resolve(PFN_vkCreateCommandPool, getProc, probe_instance, "vkCreateCommandPool") orelse return error.VulkanNotAvailable,
            .vkDestroyCommandPool = resolve(PFN_vkDestroyCommandPool, getProc, probe_instance, "vkDestroyCommandPool") orelse return error.VulkanNotAvailable,
            .vkAllocateCommandBuffers = resolve(PFN_vkAllocateCommandBuffers, getProc, probe_instance, "vkAllocateCommandBuffers") orelse return error.VulkanNotAvailable,
            .vkBeginCommandBuffer = resolve(PFN_vkBeginCommandBuffer, getProc, probe_instance, "vkBeginCommandBuffer") orelse return error.VulkanNotAvailable,
            .vkEndCommandBuffer = resolve(PFN_vkEndCommandBuffer, getProc, probe_instance, "vkEndCommandBuffer") orelse return error.VulkanNotAvailable,
            .vkResetCommandBuffer = resolve(PFN_vkResetCommandBuffer, getProc, probe_instance, "vkResetCommandBuffer") orelse return error.VulkanNotAvailable,
            .vkCmdBindPipeline = resolve(PFN_vkCmdBindPipeline, getProc, probe_instance, "vkCmdBindPipeline") orelse return error.VulkanNotAvailable,
            .vkCmdBindDescriptorSets = resolve(PFN_vkCmdBindDescriptorSets, getProc, probe_instance, "vkCmdBindDescriptorSets") orelse return error.VulkanNotAvailable,
            .vkCmdPushConstants = resolve(PFN_vkCmdPushConstants, getProc, probe_instance, "vkCmdPushConstants") orelse return error.VulkanNotAvailable,
            .vkCmdDispatch = resolve(PFN_vkCmdDispatch, getProc, probe_instance, "vkCmdDispatch") orelse return error.VulkanNotAvailable,
            .vkCmdPipelineBarrier = resolve(PFN_vkCmdPipelineBarrier, getProc, probe_instance, "vkCmdPipelineBarrier") orelse return error.VulkanNotAvailable,
            .vkQueueSubmit = resolve(PFN_vkQueueSubmit, getProc, probe_instance, "vkQueueSubmit") orelse return error.VulkanNotAvailable,
            .vkCreateFence = resolve(PFN_vkCreateFence, getProc, probe_instance, "vkCreateFence") orelse return error.VulkanNotAvailable,
            .vkDestroyFence = resolve(PFN_vkDestroyFence, getProc, probe_instance, "vkDestroyFence") orelse return error.VulkanNotAvailable,
            .vkResetFences = resolve(PFN_vkResetFences, getProc, probe_instance, "vkResetFences") orelse return error.VulkanNotAvailable,
            .vkWaitForFences = resolve(PFN_vkWaitForFences, getProc, probe_instance, "vkWaitForFences") orelse return error.VulkanNotAvailable,
            .vkDeviceWaitIdle = resolve(PFN_vkDeviceWaitIdle, getProc, probe_instance, "vkDeviceWaitIdle") orelse return error.VulkanNotAvailable,
        };

        // NOW destroy probe instance — function pointers are already resolved
        destroyInstance(probe_instance, null);

        return loader;
    }

    pub fn deinit(self: *VkLoader) void {
        self.lib.close();
    }

    /// Resolve a Vulkan function pointer via vkGetInstanceProcAddr, cast to target type.
    fn resolve(comptime T: type, getProc: PFN_vkGetInstanceProcAddr, instance: VkInstance, name: [*:0]const u8) ?T {
        const raw = getProc(instance, name) orelse return null;
        return @ptrCast(raw);
    }
};
