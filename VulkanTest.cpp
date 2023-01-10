/* ------------------------------------------------------
 * NOTE: This code is the result of me challenging myself
 * to write a Vulkan app without consulting a tutorial.
 * I typically learn via examples, so limiting myself
 * to using the Vulkan specification was a useful exercise.
 * 
 * This code obviously won't follow best practices, and
 * is just meant to be a testbed for throwaway code. */

#define _CRT_SECURE_NO_WARNINGS
#include <array>
#include <cstdio>
#include <stdexcept>
#include <span>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vk_layer.h>
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#endif
#include "stb/stb_image.h"

// These are core in Vulkan 1.3 but require VK_KHR_synchronization2
// in Vulkan 1.2, which is what I'm targeting, so I need a manual
// lookup.
static PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_LOCAL = nullptr;
static PFN_vkQueueSubmit2KHR vkQueueSubmit2KHR_LOCAL = nullptr;

// Out of principle, I refuse to pull in <algorithm> over max
template<typename T>
inline T max(T a, T b) {
    return a >= b ? a : b;
}
template<typename T>
inline T round_up_to_multiple(T val, T base) {
    return ((val + base - 1) / base) * base;
}
void log_vk_result(VkResult res, const char* ctx) {
    if (res != VK_SUCCESS) {
        fprintf(
            stderr, 
            "[vk] error: %s (context: %s)\n", 
            string_VkResult(res),
            ctx);
    }
}
VkInstance create_instance() {
    // Ensure that the instance supports Vulkan 1.2
    uint32_t apiVersion = 0;
    VkResult enumRes = vkEnumerateInstanceVersion(&apiVersion);
    if (enumRes != VK_SUCCESS || apiVersion < VK_API_VERSION_1_2) {
        log_vk_result(enumRes, "retrieving instance version");
        return nullptr;
    }
    
    const std::array<const char*, 2> neededLayers = {
        /* NOTE: I would not explicitly load the validation layer in a real app */
        "VK_LAYER_KHRONOS_validation",
        // The synchronization2 wrapper layer is for instances, but
        // the actual synchronization2 extension is for devices
        "VK_LAYER_KHRONOS_synchronization2",
    };
    const std::array<const char*, 3> neededExtensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    const VkApplicationInfo applicationInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "Vulkan Test Application",
        .applicationVersion = 0,
        .pEngineName = "Vulkan Test Engine",
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_2
    };
    const VkInstanceCreateInfo instanceSetup = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = neededLayers.size(),
        .ppEnabledLayerNames = neededLayers.data(),
        .enabledExtensionCount = neededExtensions.size(),
        .ppEnabledExtensionNames = neededExtensions.data()
    };
    VkInstance instance = VK_NULL_HANDLE;
    VkResult createRes = vkCreateInstance(
        &instanceSetup, 
        nullptr, 
        &instance);
    log_vk_result(createRes, "instance creation");

    vkCmdPipelineBarrier2KHR_LOCAL = (PFN_vkCmdPipelineBarrier2KHR)vkGetInstanceProcAddr(instance, "vkCmdPipelineBarrier2KHR");
    vkQueueSubmit2KHR_LOCAL = (PFN_vkQueueSubmit2KHR)vkGetInstanceProcAddr(instance, "vkQueueSubmit2KHR");

    return instance;
}

// Details regarding a logical device.
struct LogicalDevice {
    // The device handle.
    VkDevice device;
    // The instance associated with the logical device.
    VkInstance instance;
    // The associated physical device.
    /* ------------------------------------------------
     * NOTE: Yeah, I know you can map multiple physical 
     * devices to a single logical device. */
    VkPhysicalDevice physicalDevice;
    // A graphics + compute + transfer queue
    VkQueue queue;
    // The queue family index of the queue.
    uint32_t queueFamilyIndex;
    // Cached memory properties for this device.
    VkPhysicalDeviceMemoryProperties memoryProperties;
    // Cached properties for this device.
    VkPhysicalDeviceProperties properties;
};

class DescriptorSetLayout {
    const LogicalDevice* mLogicalDevice;
    VkDescriptorSetLayout mHandle;
public:
    DescriptorSetLayout(const LogicalDevice* ldev, std::span<VkDescriptorSetLayoutBinding> bindings)
        : mLogicalDevice(ldev),
        mHandle(VK_NULL_HANDLE)
    {
        const VkDescriptorSetLayoutCreateInfo layoutInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };
        VkResult layoutRes = vkCreateDescriptorSetLayout(
            mLogicalDevice->device,
            &layoutInfo,
            nullptr,
            &mHandle);
        if (layoutRes != VK_SUCCESS) {
            log_vk_result(layoutRes, "create set layout");
            throw std::runtime_error("create set layout");
        }
    }
    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;
    ~DescriptorSetLayout() {
        vkDestroyDescriptorSetLayout(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }
    VkDescriptorSetLayout handle() { return mHandle; }
};

class PipelineLayout {
    const LogicalDevice* mLogicalDevice;
    VkPipelineLayout mHandle;
public:
    PipelineLayout(const LogicalDevice* ldev, std::span<const VkDescriptorSetLayout> sets) 
    : mLogicalDevice(ldev),
    mHandle(VK_NULL_HANDLE) {
        const VkPipelineLayoutCreateInfo layoutInfo = {
           .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
           .pNext = nullptr,
           .flags = 0,
           .setLayoutCount = static_cast<uint32_t>(sets.size()),
           .pSetLayouts = sets.data(),
           .pushConstantRangeCount = 0,
           .pPushConstantRanges = nullptr,
        };
        VkResult layoutRes = vkCreatePipelineLayout(
            mLogicalDevice->device,
            &layoutInfo,
            nullptr,
            &mHandle);
        if (layoutRes != VK_SUCCESS) {
            log_vk_result(layoutRes, "pipeline layout");
            throw std::runtime_error("pipeline layout");
        }
    }
    PipelineLayout(const PipelineLayout&) = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;
    ~PipelineLayout() {
        vkDestroyPipelineLayout(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }

    VkPipelineLayout handle() { return mHandle; }
};

class BasicImageView {
    const LogicalDevice* mLogicalDevice;
    VkImageView mHandle;
public:
    BasicImageView(const LogicalDevice* ldev, VkImage image, VkFormat format)
        : mLogicalDevice(ldev),
        mHandle(VK_NULL_HANDLE)
    {
        const VkImageViewCreateInfo imageViewInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = VkComponentMapping {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY
            },
            .subresourceRange = VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS
            }
        };
        VkResult imageViewRes = vkCreateImageView(
            mLogicalDevice->device,
            &imageViewInfo,
            nullptr,
            &mHandle);
        if (imageViewRes != VK_SUCCESS) {
            log_vk_result(imageViewRes, "create image view");
            throw std::runtime_error("image view");
        }
    }

    BasicImageView(const BasicImageView&) = delete;
    BasicImageView& operator=(const BasicImageView&) = delete;

    ~BasicImageView() {
        vkDestroyImageView(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }

    VkImageView handle() { return mHandle; }
};
class DescriptorPool {
    const LogicalDevice* mLogicalDevice;
    VkDescriptorPool mHandle;
public:
    DescriptorPool(const LogicalDevice* ldev, uint32_t maxSets, std::span<const VkDescriptorPoolSize> poolSizes)
    : mLogicalDevice(ldev),
    mHandle(VK_NULL_HANDLE){
        const VkDescriptorPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };
        VkResult poolRes = vkCreateDescriptorPool(
            mLogicalDevice->device,
            &poolInfo,
            nullptr,
            &mHandle);
        if (poolRes != VK_SUCCESS) {
            log_vk_result(poolRes, "create descriptor pool");
            throw std::runtime_error("descriptor pool");
        }
    }
    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    ~DescriptorPool() {
        vkDestroyDescriptorPool(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }

    VkResult allocate(uint32_t count, const VkDescriptorSetLayout* layouts, VkDescriptorSet* out) {
        const VkDescriptorSetAllocateInfo allocateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = mHandle,
            .descriptorSetCount = count,
            .pSetLayouts = layouts
        };
        return vkAllocateDescriptorSets(
            mLogicalDevice->device,
            &allocateInfo,
            out);
    }
};

class ShaderModule {
    const LogicalDevice* mLogicalDevice;
    VkShaderModule mHandle;

public:
    ShaderModule(const LogicalDevice* ldev, const char* filename) 
        : mLogicalDevice(ldev),
        mHandle(VK_NULL_HANDLE)
    {
        uint32_t codeSize = 0;
        uint32_t* code = nullptr;
        {
            FILE* f = fopen(filename, "rb");
            if (f == nullptr) {
                fprintf(stderr, "[fs] failed to access shader\n");
                throw std::runtime_error("access shader");
            }
            fseek(f, 0, SEEK_END);
            codeSize = ftell(f);
            fseek(f, 0, SEEK_SET);
            code = (uint32_t*)malloc(codeSize); // malloc alignment satisfies uint32_t
            size_t count = fread(code, 4, codeSize / 4, f);
            fclose(f);
            if (count != codeSize / 4) {
                fprintf(stderr, "[fs] failed to load shader from disk\n");
                throw std::runtime_error("load shader");
            }
            
        }
        const VkShaderModuleCreateInfo compressorShaderInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = codeSize,
            .pCode = code // TODO
        };
        VkResult compressorShaderRes = vkCreateShaderModule(
            mLogicalDevice->device,
            &compressorShaderInfo,
            nullptr,
            &mHandle);
        free(code);
        if (compressorShaderRes != VK_SUCCESS) {
            log_vk_result(compressorShaderRes, "compress_bc1: create shader module");
            throw std::runtime_error("create shader module");
        }
    }
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ~ShaderModule() {
        vkDestroyShaderModule(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }

    VkShaderModule handle() { return mHandle; }
};

class ComputePipeline {
    const LogicalDevice* mLogicalDevice;
    VkPipeline mHandle;
public:
    ComputePipeline(const LogicalDevice* ldev, VkPipelineLayout layout, VkShaderModule shader) 
        : mLogicalDevice(ldev),
        mHandle(VK_NULL_HANDLE)
    {
        const VkComputePipelineCreateInfo pipelineInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VkPipelineShaderStageCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
                .pSpecializationInfo = nullptr,
            },
            .layout = layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = 0
        };
        VkResult pipelineRes = vkCreateComputePipelines(
            mLogicalDevice->device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &mHandle);
        if (pipelineRes != VK_SUCCESS) {
            log_vk_result(pipelineRes, "create compute pipeline");
            throw std::runtime_error("compute pipeline");
        }
    }
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ~ComputePipeline() {
        vkDestroyPipeline(
            mLogicalDevice->device,
            mHandle,
            nullptr);
    }

    VkPipeline handle() { return mHandle; }
};

// Creates a logical device. On success, updates `out`. On failure, null-s it.
bool create_logical_device(
    VkInstance instance, 
    LogicalDevice* out,
    VkPhysicalDeviceType preferredType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {

    // Null all out params
    memset(out, 0, sizeof(LogicalDevice));

    // Count physical devices
    /* ------------------------------------------------------------
     * NOTE: If the device count changes in between enumeration and
     * array allocation, you could read invalid data or overflow the
     * buffer. This is admittedly a very unlikely issue. */
    uint32_t deviceCount = 0;
    VkResult deviceCountRes = vkEnumeratePhysicalDevices(
        instance, 
        &deviceCount, 
        nullptr);
    if (deviceCountRes != VK_SUCCESS || deviceCount == 0) {
        log_vk_result(deviceCountRes, "physical device enumeration");
        return false;
    }

    // Retrieve physical devices
    VkPhysicalDevice* devices = new VkPhysicalDevice[deviceCount];
    VkResult deviceRes = vkEnumeratePhysicalDevices(
        instance, 
        &deviceCount,
        devices);
    if (deviceRes != VK_SUCCESS || deviceCount == 0) {
        log_vk_result(deviceRes, "physical device enumeration");
        return false;
    }

    // Iterate through physical devices. 
    // The first GPU matching the preferred type is selected.
    // If no device matches, then the last returned GPU is selected.
    /* ------------------------------------------------------------------
     * NOTE: In production, any device which doesn't support all required 
     * features/limits would be eliminated. If none remain, initialiation
     * would fail.
     * 
     * We would then search for the preferred device type.
     * 
     * Tiebreaking by "first device returned" is still sensible, IMO, but 
     * there are a lot of other possibilities. You could use *desired* 
     * features and limits to tiebreak, but these may vary based on 
     * application-specific details, whereas device selection should
     * probably be implemented in the engine. (Tiebreaking callbacks might
     * help). You could also try ranking GPUs and choose the most powerful
     * one according to a performance estimate. */
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = nullptr,
        // uninitialized...
    };
    for (size_t i = 0; i < deviceCount; i++) {
        vkGetPhysicalDeviceProperties2(devices[i], &props);
        fprintf(
            stderr, 
            "[vk] checking device: %s\n", 
            props.properties.deviceName);
        if (props.properties.apiVersion >= VK_API_VERSION_1_2) {
            // The device is compatible, so select it...
            physicalDevice = devices[i];
            if (props.properties.deviceType == preferredType) {
                // ...and it's the kind we want, so stop searching!
                fprintf(stderr, "     ...selected\n");
                break;
            }
        }
    }
    delete[] devices;

    // Count queue families
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties2(
        physicalDevice,
        &familyCount, 
        nullptr);
    if (familyCount == 0) {
        return false;
    }

    // Enumerate queues and find a graphics/compute/transfer queue.
    /* -------------------------------------------------------------
     * NOTE: In production, I'd want three queues: a graphics queue,
     * a compute queue, and a transfer queue. This way, I could do
     * async compute and async texture / buffer uploads.
     *
     * It would be necessary to handle cases where there aren't
     * three separate queue families and where there aren't enough
     * queues. My Intel iGPU supports _one_ queue family and _one_
     * queue.
     *
     * The "not enough queues" case complicates the engine design.
     * You can't design your engine around N components/threads,
     * each with their own queue; you have to use the queues the
     * hardware supplies. Of course, sharing a queue between threads
     * is always possible. (NVIDIA recommends 5-10 queue submits
     * per frame, so CPU locking stalls don't *seem* disasterous.)
     * You'd lose out on true cross-queue asynchronity (is that a
     * word?) but I've heard that some drivers multiplex the queues
     * within a queue family into a single hardware queue anyways.
     * So as long as forward progress is preserved, it shouldn't
     * be too awful. */
    VkQueueFamilyProperties2* families = new VkQueueFamilyProperties2[familyCount];
    for (uint32_t i = 0; i < familyCount; i++) {
        families[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        families[i].pNext = nullptr;
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(
        physicalDevice,
        &familyCount, 
        families);
    if (familyCount == 0) {
        return false;
    }
    uint32_t selectedFamily = 0;
    for (uint32_t i = 0; i < familyCount; i++) {
        // Find first queue family supporting both graphics and compute operations
        // As per the spec, this queue family will also support transfer operations,
        // even if it is not indicated in the queue flags.
        // 
        // If any queue family supports graphics, there will be a queue family 
        // supporting graphics and compute. (I'm ignoring compute-only devices.)
        VkQueueFlags flags = families[i].queueFamilyProperties.queueFlags;
        if ((flags & VK_QUEUE_GRAPHICS_BIT) && (flags & VK_QUEUE_COMPUTE_BIT)) {
            selectedFamily = i;
            break;
        }
    }
    delete[] families;

    // Create a logical device from the physical device.
    std::array<const char*, 4> neededExtensions = {
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
        VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    const float one = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = selectedFamily,
        .queueCount = 1,
        .pQueuePriorities = &one,
    };
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2feature = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
        .pNext = nullptr,
        .synchronization2 = VK_TRUE,
    };
    const VkPhysicalDeviceVulkan12Features neededVulkan12features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &sync2feature,
        // bunch of false...
        .imagelessFramebuffer = VK_TRUE,
        // bunch of false...
        .timelineSemaphore = VK_TRUE,
        // bunch of false...
    };
    const VkPhysicalDeviceFeatures neededFeatures = {
        // bunch of false...
        .imageCubeArray = VK_TRUE,
        // bunch of false...
        .tessellationShader = VK_TRUE,
        .sampleRateShading = VK_TRUE,
        // bunch of false...
        .multiDrawIndirect = VK_TRUE,
        // bunch of false...
        .fillModeNonSolid = VK_TRUE,
        // bunch of false...
        .alphaToOne = VK_TRUE,
        .multiViewport = VK_TRUE,
        // bunch of false...
        .textureCompressionBC = VK_TRUE,
        // bunch of false...
    };
    const VkDeviceCreateInfo deviceInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &neededVulkan12features,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = neededExtensions.size(),
        .ppEnabledExtensionNames = neededExtensions.data(),
        .pEnabledFeatures = &neededFeatures
    };
    VkDevice device = VK_NULL_HANDLE;
    VkResult createRes = vkCreateDevice(
        physicalDevice, 
        &deviceInfo, 
        nullptr, 
        &device);
    if (createRes != VK_SUCCESS) {
        log_vk_result(createRes, "device creation");
        return false;
    }

    // Retrieve the handle to our queue.
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(
        device, 
        selectedFamily,
        0, 
        &queue);

    // Query memory types.
    VkPhysicalDeviceMemoryProperties2 memProps2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = nullptr,
        // zero out actual properties...
    };
    vkGetPhysicalDeviceMemoryProperties2(
        physicalDevice, 
        &memProps2);

    // Everything has succeeded. Update out params.
    out->device = device;
    out->instance = instance;
    out->physicalDevice = physicalDevice;
    
    out->queue = queue;
    out->queueFamilyIndex = selectedFamily;
    out->memoryProperties = memProps2.memoryProperties;
    out->properties = props.properties;
    return true;
}

uint32_t get_memory_type(const LogicalDevice* ldev, VkMemoryPropertyFlags flags, uint32_t mask) {
    for (uint32_t i = 0; i < ldev->memoryProperties.memoryTypeCount; i++) {
        VkMemoryType cur = ldev->memoryProperties.memoryTypes[i];
        bool matchesFlags = cur.propertyFlags & flags;
        bool matchesMask = (1 << i) & mask;
        if (matchesFlags && matchesMask) {
            return i;
        }
    }
    /* NOTE: In production, I would make this return a proper error code, but 
     * I can't be bothered for test code. */
    fprintf(stderr, "[vk] no matching memory type");
    std::exit(1);
}

namespace compress_bc1 {
    // Returns the size of bytes of a buffer containing tightly backed
    // BC1-compressed texel blocks for an image with the given dimensions.
    VkDeviceSize bc1_buffer_size(uint32_t width, uint32_t height) {
        return 64 // bc1 texel block size in bytes
            * round_up_to_multiple(width, 4u)
            * round_up_to_multiple(height, 4u);
    }

    /* --------------------------------------------------------
     * NOTE: This would certainly be cleaner if I used seperate
     * allocations for the source and destination images. Using
     * VMA would be both cleaner and more performant. That being
     * said, this is a learning project and I want experience 
     * with suballocation, so I'm doing it the ugly way. */
    struct GpuResources {
        const LogicalDevice* mLogicalDevice;
        VkDeviceMemory mAlloc;
        VkImage mSrc;
        VkBuffer mDst;
        VkDeviceSize mDstOffset;

        GpuResources(const LogicalDevice* ldev, uint32_t width, uint32_t height) 
            : mLogicalDevice(ldev), 
            mAlloc(VK_NULL_HANDLE), 
            mSrc(VK_NULL_HANDLE), 
            mDst(VK_NULL_HANDLE),
            mDstOffset(0)
        {
            
            // Create the destination buffer.
            VkDeviceSize dstSize = bc1_buffer_size(width, height);
            const VkBufferCreateInfo dstInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = dstSize,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            };
            VkResult dstRes = vkCreateBuffer(
                mLogicalDevice->device, 
                &dstInfo,
                nullptr, 
                &mDst);
            if (dstRes != VK_SUCCESS) {
                log_vk_result(dstRes, "create dst image");
                throw std::runtime_error("create gpu resources");
            }

            // Create the source image
            const VkImageCreateInfo srcInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent = VkExtent3D {
                    .width = width,
                    .height = height,
                    .depth = 1 
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
            };
            VkResult srcRes = vkCreateImage(
                mLogicalDevice->device, 
                &srcInfo, 
                nullptr, 
                &mSrc);
            if (srcRes != VK_SUCCESS) {
                log_vk_result(srcRes, "create src image");
                throw std::runtime_error("create gpu resources");
            }

            // Retrieve source image memory requirements
            const VkImageMemoryRequirementsInfo2 srcMemInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                .pNext = nullptr,
                .image = mSrc
            };
            VkMemoryRequirements2 srcMemReqs = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
                .pNext = nullptr,
                // uninitialized...
            };
            vkGetImageMemoryRequirements2(
                mLogicalDevice->device, 
                &srcMemInfo, 
                &srcMemReqs);

            // Buffers don't have a required alignment (IIRC), but they 
            // *should* be aligned to optimalBufferCopyOffsetAlignment.
            // Furthermore, since the source image is optimally tiled,
            // the buffer *must* be aligned to bufferImageGranularity.
            // Since both *must* be powers of two, one *must* divide
            // the other, which means we can simply align to the
            // maximum of the two.
            size_t dstOffsetAlign = max(
                mLogicalDevice->properties.limits.bufferImageGranularity,
                mLogicalDevice->properties.limits.optimalBufferCopyOffsetAlignment);
            mDstOffset = round_up_to_multiple(
                srcMemReqs.memoryRequirements.size,
                dstOffsetAlign);

            // Allocate device local memory for the src image and dst buffer
            uint32_t memType = get_memory_type(
                ldev, 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                srcMemReqs.memoryRequirements.memoryTypeBits);
            const VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = mDstOffset + dstSize,
                .memoryTypeIndex = memType
            };
            VkResult allocRes = vkAllocateMemory(
                mLogicalDevice->device,
                &allocInfo,
                nullptr,
                &mAlloc);
            if (allocRes != VK_SUCCESS) {
                log_vk_result(allocRes, "compress_bc1: GPU memory allocation");
                throw std::runtime_error("create gpu resources");
            }

            // Bind source and dest images
            const VkBindImageMemoryInfo srcBindInfo = {
                .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                .pNext = nullptr,
                .image = mSrc,
                .memory = mAlloc,
                .memoryOffset = 0
            };
            VkResult srcBindRes = vkBindImageMemory2(mLogicalDevice->device, 1, &srcBindInfo);
            if (srcBindRes != VK_SUCCESS) {
                log_vk_result(srcBindRes, "compress_bc1: bind src image");
                throw std::runtime_error("create gpu resources");
            }

            // Bind dst image
            const VkBindBufferMemoryInfo dstBindInfo = {
                .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
                .pNext = nullptr,
                .buffer = mDst,
                .memory = mAlloc,
                .memoryOffset = mDstOffset
            };
            VkResult dstBindRes = vkBindBufferMemory2(mLogicalDevice->device, 1, &dstBindInfo);
            if (dstBindRes != VK_SUCCESS) {
                log_vk_result(dstBindRes, "compress_bc1: bind dst image");
                throw std::runtime_error("create gpu resources");
            }

            fprintf(stderr, "[info] reserved GPU resources\n");
        }

        GpuResources(const GpuResources&) = delete;
        GpuResources& operator=(const GpuResources&) = delete;

        ~GpuResources() {
            vkDestroyImage(mLogicalDevice->device, mSrc, nullptr);
            vkDestroyBuffer(mLogicalDevice->device, mDst, nullptr);
            vkFreeMemory(mLogicalDevice->device, mAlloc, nullptr);
            fprintf(stderr, "[info] freed GPU resources\n");
        }

        VkImage src() { return mSrc; }
        VkBuffer dst() { return mDst; }
        VkDeviceSize dst_offset() { return mDstOffset; }
    };

    struct CpuResources {
        const LogicalDevice* mLogicalDevice;
        VkDeviceMemory mAlloc;
        VkBuffer mStaging;
        VkBuffer mOutput;
        VkDeviceSize mOutputOffset;

        CpuResources(const LogicalDevice* ldev, uint32_t width, uint32_t height)
            : mLogicalDevice(ldev), 
            mAlloc(VK_NULL_HANDLE),
            mStaging(VK_NULL_HANDLE),
            mOutput(VK_NULL_HANDLE),
            mOutputOffset(0)
        {
            // Create staging buffer
            VkDeviceSize stagingSize = 4 * width * height; // 4 = RGBA, 1 byte per channel
            const VkBufferCreateInfo stagingInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = stagingSize,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            VkResult stagingRes = vkCreateBuffer(
                mLogicalDevice->device,
                &stagingInfo,
                nullptr,
                &mStaging);
            if (stagingRes != VK_SUCCESS) {
                log_vk_result(stagingRes, "compress_bc1: create staging buffer");
                throw std::runtime_error("create cpu resources");
;            }
            
            // Create output buffer
            mOutputOffset = round_up_to_multiple(
                stagingSize,
                mLogicalDevice->properties.limits.optimalBufferCopyOffsetAlignment);
            VkDeviceSize outputSize = bc1_buffer_size(width, height);
            const VkBufferCreateInfo outputInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = outputSize,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            VkResult outputRes = vkCreateBuffer(
                mLogicalDevice->device,
                &outputInfo,
                nullptr,
                &mOutput
            );
            if (outputRes != VK_SUCCESS) {
                log_vk_result(outputRes, "compress_bc1: create output buffer");
                throw std::runtime_error("create cpu resources");
            }

            // Allocate host-coherent memory for the staging & output buffers
            // TODO: Is this the best choice?
            uint32_t memType = get_memory_type(
                mLogicalDevice, 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                ~0);
            const VkMemoryAllocateInfo allocateInfo = {
               .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
               .pNext = nullptr,
               .allocationSize = mOutputOffset + outputSize,
               .memoryTypeIndex = memType
            };
            VkResult allocRes = vkAllocateMemory(
                mLogicalDevice->device, 
                &allocateInfo,
                nullptr, 
                &mAlloc);
            if (allocRes != VK_SUCCESS) {
                log_vk_result(allocRes, "compress_bc1: CPU memory allocation");
                throw std::runtime_error("create cpu resources");
            }

            // Bind both buffers
            const VkBindBufferMemoryInfo bindInfos[2] = {
                VkBindBufferMemoryInfo {
                    .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
                    .pNext = nullptr,
                    .buffer = mStaging,
                    .memory = mAlloc,
                    .memoryOffset = 0
                },
                VkBindBufferMemoryInfo {
                    .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
                    .pNext = nullptr,
                    .buffer = mOutput,
                    .memory = mAlloc,
                    .memoryOffset = mOutputOffset
                },
            };
            VkResult bindRes = vkBindBufferMemory2(
                mLogicalDevice->device, 
                2, 
                bindInfos);
            if (bindRes != VK_SUCCESS) {
                log_vk_result(bindRes, "compress_bc1: CPU memory binding");
                throw std::runtime_error("create cpu resources");
            }

            fprintf(stderr, "[info] reserved CPU resources\n");
        }

        CpuResources(const CpuResources&) = delete;
        CpuResources& operator=(const CpuResources&) = delete;

        ~CpuResources() {
            vkDestroyBuffer(mLogicalDevice->device, mStaging, nullptr);
            vkDestroyBuffer(mLogicalDevice->device, mOutput, nullptr);
            vkFreeMemory(mLogicalDevice->device, mAlloc, nullptr);
            fprintf(stderr, "[info] freed CPU resources\n");
        }
    };
    struct CommandPool {
        const LogicalDevice* mLogicalDevice;
        VkCommandPool mPool;

        CommandPool(const LogicalDevice* ldev, VkCommandPoolCreateFlags flags)
            : mLogicalDevice(ldev),
            mPool(VK_NULL_HANDLE)
        {
            const VkCommandPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .queueFamilyIndex = mLogicalDevice->queueFamilyIndex
            };
            VkResult poolRes = vkCreateCommandPool(
                mLogicalDevice->device,
                &poolInfo,
                nullptr,
                &mPool);
            if (poolRes != VK_SUCCESS) {
                log_vk_result(poolRes, "create command pool");
                return;
            }
        }
        CommandPool(const CommandPool&) = delete;
        CommandPool& operator=(const CommandPool&) = delete;

        VkResult alloc_primary(uint32_t count, VkCommandBuffer* data) {
            const VkCommandBufferAllocateInfo cmdInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = mPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = count
            };
            return vkAllocateCommandBuffers(
                mLogicalDevice->device,
                &cmdInfo,
                data);
        }

        ~CommandPool() {
            vkDestroyCommandPool(mLogicalDevice->device, mPool, nullptr);
        }
    };
    bool run(const LogicalDevice* ldev, const char* input) {
        // Load the specified image.
        int w, h, origChans;
        stbi_uc* src = stbi_load(
            input,
            &w,
            &h,
            &origChans,
            4);

        if (src == nullptr || w <= 0 || h <= 0) {
            fprintf(stderr, "[img] couldn't load %s\n", input);
            return false;
        }

        uint32_t width = static_cast<uint32_t>(w);
        uint32_t height = static_cast<uint32_t>(h);

        VkDeviceSize inSize = 4 * width * height;
        VkDeviceSize outSize = bc1_buffer_size(width, height);
        CpuResources cpuRz(ldev, width, height);
        GpuResources gpuRz(ldev, width, height);

        // Leaky abstractions. Hopefully, I'll rework this later.
        // Map staging buffer
        void* mapped = nullptr;
        VkResult mappedRes = vkMapMemory(
            ldev->device,
            cpuRz.mAlloc,
            0,
            inSize,
            0,
            &mapped);
        if (mappedRes != VK_SUCCESS) {
            log_vk_result(mappedRes, "compress_bc1: map staging buffer");
            return false;
        }

        // Copy image data into staging buffer.
        // Then free image data and unmap memory.
        memcpy(mapped, src, inSize);
        stbi_image_free(src);
        vkUnmapMemory(ldev->device, cpuRz.mAlloc);
       
        // Create command pool
        CommandPool pool(ldev, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

        // Create command buffer
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkResult cmdRes = pool.alloc_primary(1, &cmd);
        if (cmdRes != VK_SUCCESS) {
            log_vk_result(cmdRes, "compress_bc1: allocate command buffer");
            return false;
        }

        // Begin recording commands
        const VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        // The source image is in the wrong format, we need to perform
        // an image layout change to general.
        // (Why not transfer dst layout? Because we're about to use it
        // as a storage image, so why change it twice.)
        const VkImageMemoryBarrier2KHR layoutChangeBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE_KHR,
            .srcAccessMask = VK_ACCESS_2_NONE_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = ldev->queueFamilyIndex,
            .dstQueueFamilyIndex = ldev->queueFamilyIndex,
            .image = gpuRz.mSrc,
            .subresourceRange = VkImageSubresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS
            }
        };
        const VkDependencyInfoKHR layoutChange = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            // TODO: I don't need a host barrier here because the
            // staging buffer is coherent, right?
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &layoutChangeBarrier
        };
        vkCmdPipelineBarrier2KHR_LOCAL(cmd, &layoutChange);

        // Now, we upload the staging buffer to the texture
        // Do I really have to do this? No, I could just read from the 
        // staging buffer as a storage texel buffer in the compute shader.
        // I'm not doing anything "fancy" (like texture sampling) so it 
        // wouldn't be too awful. I'd lose the cache friendliness of
        // optimal tiling, of course, and CPU memory access is slower
        // then device local on dGPUs, but I'd also eliminate a pipeline
        // barrier and a layout transition, so maybe it would be worth it.
        // Hmm... well, staging buffer uploads is good practice for an
        // actual game engine, so I'll stick with that for now.
        const VkBufferImageCopy stagingToSrcRegion = {
            .bufferOffset = 0,
            .bufferRowLength = 0, // = tightly packed
            .bufferImageHeight = 0, // = tightly packed
            .imageSubresource = VkImageSubresourceLayers {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = VkOffset3D {
                .x = 0,
                .y = 0,
                .z = 0
            },
            .imageExtent = VkExtent3D {
                .width = width,
                .height = height,
                .depth = 1u,
            }
        };
        vkCmdCopyBufferToImage(
            cmd,
            cpuRz.mStaging,
            gpuRz.mSrc,
            VK_IMAGE_LAYOUT_GENERAL,
            1,
            &stagingToSrcRegion);

        // Now, we wait for the image transfer to have been fully completed before
        // we allow the compute shader to execute.
        const VkMemoryBarrier2KHR uploadBarrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR,
        };
        const VkDependencyInfoKHR upload = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 1,
            // TODO: I don't need a host barrier here because the
            // staging buffer is coherent, right?
            .pMemoryBarriers = &uploadBarrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr
        };
        vkCmdPipelineBarrier2KHR_LOCAL(cmd, &upload);

        // Create the pipeline layout for the compute shader.
        // We only need two descriptors: one for the input image,
        // and one for the output buffer. We can put these in the
        // same descriptor set.
        std::array<VkDescriptorSetLayoutBinding,2> compressorSetLayoutBindings = {
            VkDescriptorSetLayoutBinding {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr
            },
            VkDescriptorSetLayoutBinding {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr
            },
        };
        DescriptorSetLayout compressorSetLayout(
            ldev, 
            std::span(compressorSetLayoutBindings));
        VkDescriptorSetLayout compressorSetLayoutHandle = compressorSetLayout.handle();

        PipelineLayout compressorLayout(
            ldev,
            std::span(&compressorSetLayoutHandle, 1));

        // And load the shader module from disk
        ShaderModule compressorShader(ldev, "compress_bc1.spv");

        // Now we create a compute pipeline
        ComputePipeline compressor(
            ldev, 
            compressorLayout.handle(), 
            compressorShader.handle());
        vkCmdBindPipeline(
            cmd, 
            VK_PIPELINE_BIND_POINT_COMPUTE, 
            compressor.handle());

        // Haha but wait, no, we've still gotta do descriptors.
        const std::array<VkDescriptorPoolSize, 2> descriptorPoolSizes = {
            VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
        };
        DescriptorPool descriptorPool(ldev, 1, std::span(descriptorPoolSizes));
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult setRes = descriptorPool.allocate(1, &compressorSetLayoutHandle, &set);
        if (setRes != VK_SUCCESS) {
            log_vk_result(setRes, "create descriptor set");
            return false;
        }

        // Now we have a descriptor set allocated, update it with our resources.
        BasicImageView imageView(ldev, gpuRz.mSrc, VK_FORMAT_R8G8B8A8_UNORM);
        const VkDescriptorImageInfo srcInfo = {
            .sampler = VK_NULL_HANDLE, // storage image
            .imageView = imageView.handle(), // TODO
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL
        };
        const VkDescriptorBufferInfo dstInfo = {
            .buffer = gpuRz.mDst,
            .offset = gpuRz.mDstOffset,
            .range = VK_WHOLE_SIZE
        };
        const VkWriteDescriptorSet setWrites[2] = {
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &srcInfo,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &dstInfo,
                .pTexelBufferView = nullptr
            }
        };
        vkUpdateDescriptorSets(
            ldev->device, 
            2, 
            setWrites, 
            0, 
            nullptr);

        // Bind the descriptor and dispatch the compute shader.
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            compressorLayout.handle(),
            0,
            1,
            &set,
            0,
            nullptr);
        vkCmdDispatch(cmd, (width+3)/4, (height+3)/4, 1);

        // Yet another pipeline barrier -- we wait for the compute shader
        // to finish before we copy the destination buffer to the output buffer.
        const VkMemoryBarrier2KHR downloadBarrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
        };
        const VkDependencyInfoKHR download = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &downloadBarrier,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr
        };
        vkCmdPipelineBarrier2KHR_LOCAL(cmd, &download);

        // At last!
        vkEndCommandBuffer(cmd);

        // Now we submit the queue. We wait on a fence so we know when
        // we can read back from the output buffer.
        const VkFenceCreateInfo fenceInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0
        };
        VkFence fence = VK_NULL_HANDLE;
        VkResult fenceRes = vkCreateFence(
            ldev->device, 
            &fenceInfo, 
            nullptr, 
            &fence);
        if (fenceRes != VK_SUCCESS) {
            log_vk_result(fenceRes, "create fence");
            return false;
        }
        const VkCommandBufferSubmitInfo cmdSubmit = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0
        };
        const VkSubmitInfo2KHR submit = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
            .pNext = nullptr,
            .flags = 0,
            .waitSemaphoreInfoCount = 0,
            .pWaitSemaphoreInfos = 0,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmdSubmit,
            .signalSemaphoreInfoCount = 0,
            .pSignalSemaphoreInfos = 0
        };
        VkResult queueRes = vkQueueSubmit2KHR_LOCAL(ldev->queue, 1, &submit, fence);
        if (queueRes != VK_SUCCESS) {
            log_vk_result(queueRes, "compress_bc1: submit queue");
            vkDestroyFence(ldev->device, fence, nullptr);
            return false;
        }

        // and now, wait for that fence to be signalled.
        VkResult waitRes = vkWaitForFences(
            ldev->device,
            1, 
            &fence, 
            true, 
            ~0);
        vkDestroyFence(ldev->device, fence, nullptr);
        if (waitRes != VK_SUCCESS) {
            log_vk_result(waitRes, "compress_bc1: wait on fence");
            return false;
        }

        return true;
    }
}

int main(int argc, const char** argv) {
    VkInstance instance = create_instance();
    LogicalDevice ldev;
    bool created = create_logical_device(instance, &ldev);
    if (created && argc == 2) {
        printf("[info] compressing %s\n", argv[1]);
        compress_bc1::run(&ldev, argv[1]);
        printf("       ...done\n");
    }
    vkDestroyDevice(ldev.device, nullptr);
    vkDestroyInstance(instance, nullptr);
}