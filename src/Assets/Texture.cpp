#include "Texture.hpp"
#include "Utilities/StbImage.hpp"
#include "Utilities/Exception.hpp"
#include "Common/CoreMinimal.hpp"

#include "Options.hpp"
#include "Runtime/TaskCoordinator.hpp"
#include "TextureImage.hpp"
#include "Runtime/Engine.hpp"
#include "Utilities/FileHelper.hpp"
#include "Vulkan/Device.hpp"
#include "Vulkan/ImageView.hpp"
#include "Vulkan/DescriptorBinding.hpp"
#include "Vulkan/DescriptorSetManager.hpp"
#include "Vulkan/DescriptorSets.hpp"
#include "ThirdParty/lzav/lzav.h"

#include <spdlog/spdlog.h>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <system_error>

#if WITH_KTX2
#include <ktx.h>
#endif

#define M_NEXT_PI 3.14159265358979323846f

namespace
{
    constexpr uint32_t kHdrCacheMagic = 0x48445243; // 'HDRC'
    constexpr uint32_t kHdrCacheVersion = 1;

    struct HdrCacheHeader
    {
        uint32_t magic;
        uint32_t version;
        uint64_t originalSize;
        uint64_t compressedSize;
        uint64_t dataHash;
    };

    uint64_t HashBuffer(const uint8_t* data, size_t size)
    {
        constexpr uint64_t fnvOffset = 1469598103934665603ull;
        constexpr uint64_t fnvPrime = 1099511628211ull;

        uint64_t hash = fnvOffset;
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= static_cast<uint64_t>(data[i]);
            hash *= fnvPrime;
        }

        return hash;
    }
}

namespace Assets
{
    struct TextureTaskContext
    {
        int32_t textureId;
        TextureImage* transferPtr;
        float elapsed;
        bool needFlushHDRSH;
        std::array<char, 256> outputInfo;
    };
    
    void PrefilterEnvironmentMapLevel(const float* sourcePixels, int sourceWidth, int sourceHeight,
                                    float* targetPixels, int targetWidth, int targetHeight, 
                                    float roughness)
    {
        const int sampleCount = std::max(1, static_cast<int>(8 * (1.0f - roughness) + 16 * roughness));
        
        for (int y = 0; y < targetHeight; ++y)
        {
            for (int x = 0; x < targetWidth; ++x)
            {
                // Convert target pixel to direction
                float u = (x + 0.5f) / targetWidth;
                float v = (y + 0.5f) / targetHeight;
                
                float theta = v * M_NEXT_PI;
                float phi = u * 2.0f * M_NEXT_PI;
                
                float sinTheta = std::sin(theta);
                float cosTheta = std::cos(theta);
                float sinPhi = std::sin(phi);
                float cosPhi = std::cos(phi);
                
                // Main reflection direction
                float mainDirX = sinTheta * cosPhi;
                float mainDirY = cosTheta;
                float mainDirZ = sinTheta * sinPhi;
                
                // Build tangent space around main direction
                float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
                if (std::abs(mainDirY) > 0.999f)
                {
                    upX = 1.0f; upY = 0.0f; upZ = 0.0f;
                }
                
                // Tangent vectors
                float tangentX = upY * mainDirZ - upZ * mainDirY;
                float tangentY = upZ * mainDirX - upX * mainDirZ;
                float tangentZ = upX * mainDirY - upY * mainDirX;
                
                float tangentLen = std::sqrt(tangentX * tangentX + tangentY * tangentY + tangentZ * tangentZ);
                tangentX /= tangentLen;
                tangentY /= tangentLen;
                tangentZ /= tangentLen;
                
                float bitangentX = mainDirY * tangentZ - mainDirZ * tangentY;
                float bitangentY = mainDirZ * tangentX - mainDirX * tangentZ;
                float bitangentZ = mainDirX * tangentY - mainDirY * tangentX;
                
                float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
                float totalWeight = 0.0f;
                
                // Monte Carlo sampling
                for (int i = 0; i < sampleCount; ++i)
                {
                    // Generate random numbers (using simple pseudo-random for now)
                    float xi1 = static_cast<float>(i) / sampleCount;
                    float xi2 = static_cast<float>((i * 17 + 13) % sampleCount) / sampleCount;
                    
                    // Importance sampling for GGX distribution
                    float alpha = roughness * roughness;
                    float alpha2 = alpha * alpha;
                    
                    float cosTheta = std::sqrt((1.0f - xi1) / (1.0f + (alpha2 - 1.0f) * xi1));
                    float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
                    float phi = 2.0f * M_NEXT_PI * xi2;
                    
                    // Local sample direction
                    float localX = sinTheta * std::cos(phi);
                    float localY = sinTheta * std::sin(phi);
                    float localZ = cosTheta;
                    
                    // Transform to world space
                    float worldX = localX * tangentX + localY * bitangentX + localZ * mainDirX;
                    float worldY = localX * tangentY + localY * bitangentY + localZ * mainDirY;
                    float worldZ = localX * tangentZ + localY * bitangentZ + localZ * mainDirZ;
                    
                    // Sample environment map
                    float sampleTheta = std::acos(std::clamp(worldY, -1.0f, 1.0f));
                    float samplePhi = std::atan2(worldZ, worldX);
                    if (samplePhi < 0) samplePhi += 2.0f * M_NEXT_PI;
                    
                    float sampleU = samplePhi / (2.0f * M_NEXT_PI);
                    float sampleV = sampleTheta / M_NEXT_PI;
                    
                    int sampleX = static_cast<int>(sampleU * sourceWidth) % sourceWidth;
                    int sampleY = static_cast<int>(sampleV * sourceHeight) % sourceHeight;
                    
                    int sampleIndex = (sampleY * sourceWidth + sampleX) * 4;
                    
                    float weight = 1.0f;
                    colorR += sourcePixels[sampleIndex + 0] * weight;
                    colorG += sourcePixels[sampleIndex + 1] * weight;
                    colorB += sourcePixels[sampleIndex + 2] * weight;
                    totalWeight += weight;
                }
                
                // Normalize and store result
                if (totalWeight > 0.0f)
                {
                    colorR /= totalWeight;
                    colorG /= totalWeight;
                    colorB /= totalWeight;
                }
                
                int targetIndex = (y * targetWidth + x) * 4;
                targetPixels[targetIndex + 0] = colorR;
                targetPixels[targetIndex + 1] = colorG;
                targetPixels[targetIndex + 2] = colorB;
                targetPixels[targetIndex + 3] = 1.0f;
            }
        }
    }

    void PrefilterHdrEnvironmentMap(const float* hdrPixels, int width, int height, 
                             std::vector<std::vector<float>>& mipLevels,
                             std::vector<std::pair<int, int>>& mipDimensions)
    {
        constexpr int maxMipLevels = 8; // Typically 5-8 levels for environment maps
        mipLevels.clear();
        mipDimensions.clear();
        
        // Calculate mip levels
        int currentWidth = width;
        int currentHeight = height;
        
        for (int mipLevel = 0; mipLevel < maxMipLevels; ++mipLevel)
        {
            if (currentWidth < 4 || currentHeight < 4) break;
            
            mipDimensions.push_back({currentWidth, currentHeight});
            mipLevels.emplace_back(currentWidth * currentHeight * 4); // RGBA

            if (mipLevel > 0)
            {
                float roughness = static_cast<float>(mipLevel) / (maxMipLevels - 1);
                PrefilterEnvironmentMapLevel(hdrPixels, width, height, 
                                           mipLevels[mipLevel].data(), 
                                           currentWidth, currentHeight, roughness);
            }
            
            currentWidth = std::max(1, currentWidth / 2);
            currentHeight = std::max(1, currentHeight / 2);
        }
    }

    SphericalHarmonics ProjectHdrToSh(const float* hdrPixels, int width, int height)
    {
        SphericalHarmonics result{};
        
        // Initialize coefficients to zero
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 9; ++j)
                result.coefficients[i][j] = 0.0f;
        
        // SH basis function evaluation constants
        constexpr float shC0 = 0.282095f; // 1/(2*sqrt(π))
        constexpr float shC1 = 0.488603f; // sqrt(3)/(2*sqrt(π))
        constexpr float shC2 = 1.092548f; // sqrt(15)/(2*sqrt(π))
        constexpr float shC3 = 0.315392f; // sqrt(5)/(4*sqrt(π))
        constexpr float shC4 = 0.546274f; // sqrt(15)/(4*sqrt(π))
        
        float weightSum = 0.0f;
        
        // For each pixel in the environment map
        for (int y = 0; y < height; ++y)
        {
            // Calculate spherical coordinates
            float v = (y + 0.5f) / height;
            float theta = v * M_NEXT_PI;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            
            // Pixel solid angle weight (important for correct integration)
            float weight = sinTheta * (M_NEXT_PI / height) * (2.0f * M_NEXT_PI / width);
            
            for (int x = 0; x < width; ++x)
            {
                float u = (x + 0.5f) / width;
                float phi = u * 2.0f * M_NEXT_PI;
                float sinPhi = std::sin(phi);
                float cosPhi = std::cos(phi);
                
                // Convert to direction vector
                float dx = sinTheta * cosPhi;
                float dy = cosTheta;
                float dz = sinTheta * sinPhi;
                
                // Evaluate SH basis functions
                float basis[9];
                // Band 0 (1 coefficient)
                basis[0] = shC0;
                
                // Band 1 (3 coefficients)
                basis[1] = -shC1 * dy;
                basis[2] = shC1 * dz;
                basis[3] = -shC1 * dx;
                
                // Band 2 (5 coefficients)
                basis[4] = shC2 * dx * dy;
                basis[5] = -shC2 * dy * dz;
                basis[6] = shC3 * (3.0f * dy * dy - 1.0f);
                basis[7] = -shC2 * dx * dz;
                basis[8] = shC4 * (dx * dx - dz * dz);
                
                // Get pixel color (RGBA format, we want RGB)
                int pixelIndex = (y * width + x) * 4;
                float r = hdrPixels[pixelIndex + 0];
                float g = hdrPixels[pixelIndex + 1];
                float b = hdrPixels[pixelIndex + 2];
                
                // Project color onto SH basis functions
                for (int i = 0; i < 9; ++i)
                {
                    result.coefficients[0][i] += r * basis[i] * weight;
                    result.coefficients[1][i] += g * basis[i] * weight;
                    result.coefficients[2][i] += b * basis[i] * weight;
                }
                
                weightSum += weight;
            }
        }
                
        return result;
    }

    uint32_t GlobalTexturePool::LoadTexture(const std::string& filename, bool srgb)
    {
        std::vector<uint8_t> data;
        Utilities::Package::FPackageFileSystem::GetInstance().LoadFile(filename, data);
        std::filesystem::path path(filename);
        std::string mime = std::string("image/") + path.extension().string().substr(1);
        return GetInstance()->RequestNewTextureMemAsync(filename, mime, false, data.data(), data.size(),srgb);
    }

    uint32_t GlobalTexturePool::LoadTexture(const std::string& texname, const std::string& mime,
                                            const unsigned char* data, size_t bytelength, bool srgb)
    {
        return GetInstance()->RequestNewTextureMemAsync(texname, mime, false, data, bytelength, srgb);
    }

    uint32_t GlobalTexturePool::LoadHDRTexture(const std::string& filename)
    {
        std::vector<uint8_t> data;
        Utilities::Package::FPackageFileSystem::GetInstance().LoadFile(filename, data);
        return GetInstance()->RequestNewTextureMemAsync(filename, "image/hdr", true, data.data(), data.size(),false);
    }

    TextureImage* GlobalTexturePool::GetTextureImage(uint32_t idx)
    {
        if (GetInstance()->textureImages_.size() > idx)
        {
            return GetInstance()->textureImages_[idx].get();
        }
        return nullptr;
    }

    TextureImage* GlobalTexturePool::GetTextureImageByName(const std::string& name)
    {
        uint32_t id = GetTextureIndexByName(name);
        if (id != -1)
        {
            return GetInstance()->textureImages_[id].get();
        }
        return nullptr;
    }

    uint32_t GlobalTexturePool::GetTextureIndexByName(const std::string& name)
    {
        if (GetInstance()->textureNameMap_.find(name) != GetInstance()->textureNameMap_.end())
        {
            return GetInstance()->textureNameMap_[name].GlobalIdx_;
        }
        return -1;
    }

    GlobalTexturePool::GlobalTexturePool(const Vulkan::Device& device, Vulkan::CommandPool& commandPool,
                                         Vulkan::CommandPool& commandPoolMt) :
        device_(device),
        commandPool_(commandPool),
        mainThreadCommandPool_(commandPoolMt)
    {
        static const uint32_t kMaxBindlessResources = 65535u;// moltenVK returns a invalid value. std::min(65535u, device.DeviceProperties().limits.maxPerStageDescriptorSamplers);
        const std::vector<Vulkan::DescriptorBinding> descriptorBindings =
        {
            {0, kMaxBindlessResources, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_ALL},
            {1, kMaxBindlessResources, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL},
        };
        descriptorSetManager_.reset(new Vulkan::DescriptorSetManager(device, descriptorBindings, 1, true));
        
        // for hdr to bind
        hdrSphericalHarmonics_.resize(100);
        
        GlobalTexturePool::instance_ = this;

        CreateDefaultTextures();
    }

    GlobalTexturePool::~GlobalTexturePool()
    {
        defaultWhiteTexture_.reset();
        textureImages_.clear();
        descriptorSetManager_.reset();
    }

    void GlobalTexturePool::BindTexture(uint32_t textureIdx, const TextureImage& textureImage)
    {
        auto& descriptorSets = descriptorSetManager_->DescriptorSets();
        std::vector<VkWriteDescriptorSet> descriptorWrites =
        {
            descriptorSets.Bind(0, 0, { textureImage.Sampler().Handle(), textureImage.ImageView().Handle(), VK_IMAGE_LAYOUT_GENERAL}, textureIdx, 1),
        };
        descriptorSets.UpdateDescriptors(0, descriptorWrites);
    }

    void GlobalTexturePool::BindStorageTexture(uint32_t textureIdx, const Vulkan::ImageView& textureImage)
    {
        auto& descriptorSets = descriptorSetManager_->DescriptorSets();
        std::vector<VkWriteDescriptorSet> descriptorWrites =
        {
            descriptorSets.Bind(0, 1, {NULL, textureImage.Handle(), VK_IMAGE_LAYOUT_GENERAL}, textureIdx, 1),
        };
        descriptorSets.UpdateDescriptors(0, descriptorWrites);
    }

    uint32_t GlobalTexturePool::TryGetTexureIndex(const std::string& textureName) const
    {
        if (textureNameMap_.find(textureName) != textureNameMap_.end())
        {
            return textureNameMap_.at(textureName).GlobalIdx_;
        }
        return -1;
    }

#if WITH_KTX2
    static VkFormat MapGlFormatToVulkan(uint32_t glFormat, bool srgb)
    {
        switch (glFormat)
        {
        case 0x8058: return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM; // GL_RGBA8
        case 0x8C41: return VK_FORMAT_R8G8B8A8_SRGB; // GL_SRGB8_ALPHA8
        case 0x8D64: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK; // GL_COMPRESSED_RGB8_ETC2
        case 0x83F1: return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK; // GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
        case 0x83F2: return srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK; // GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
        case 0x83F3: return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK; // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
        case 0x8E8C: return VK_FORMAT_BC7_UNORM_BLOCK; // GL_COMPRESSED_RGBA_BPTC_UNORM
        case 0x8E8D: return VK_FORMAT_BC7_SRGB_BLOCK; // GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM
        default: return VK_FORMAT_UNDEFINED;
        }
    }
#endif

    // Map DXGI format to Vulkan format for DDS textures
    static VkFormat MapDxgiFormatToVulkan(uint32_t dxgiFormat, bool srgb, VkComponentMapping& swizzle)
    {
        // Explicit RGBA swizzle instead of identity to be safe
        swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, 
                    VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };

        switch (dxgiFormat)
        {
            // Uncompressed RGBA formats
            case 28: // DXGI_FORMAT_R8G8B8A8_UNORM
                return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            case 29: // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                return VK_FORMAT_R8G8B8A8_SRGB;
            
            // BGRA formats (Vulkan interprets as BGRA natively)
            case 87: // DXGI_FORMAT_B8G8R8A8_UNORM
                return srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
            case 91: // DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
                return VK_FORMAT_B8G8R8A8_SRGB;
            
            // Single and dual channel formats
            case 61: // DXGI_FORMAT_R8_UNORM
                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                return VK_FORMAT_R8_UNORM;
            case 49: // DXGI_FORMAT_R8G8_UNORM
                return VK_FORMAT_R8G8_UNORM;
            
            // BC1 (DXT1)
            case 71: // DXGI_FORMAT_BC1_UNORM
                return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            case 72: // DXGI_FORMAT_BC1_UNORM_SRGB
                return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
            
            // BC2 (DXT3)
            case 74: // DXGI_FORMAT_BC2_UNORM
                return srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
            case 75: // DXGI_FORMAT_BC2_UNORM_SRGB
                return VK_FORMAT_BC2_SRGB_BLOCK;
            
            // BC3 (DXT5)
            case 77: // DXGI_FORMAT_BC3_UNORM
                return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
            case 78: // DXGI_FORMAT_BC3_UNORM_SRGB
                return VK_FORMAT_BC3_SRGB_BLOCK;
            
            // BC4 (single channel)
            case 80: // DXGI_FORMAT_BC4_UNORM
                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                return VK_FORMAT_BC4_UNORM_BLOCK;
            case 81: // DXGI_FORMAT_BC4_SNORM
                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                return VK_FORMAT_BC4_SNORM_BLOCK;
            
            // BC5 (dual channel)
            case 82: // DXGI_FORMAT_BC5_TYPELESS
            case 83: // DXGI_FORMAT_BC5_UNORM
            case 84: // DXGI_FORMAT_BC5_SNORM
                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, 
                            VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE };
                return VK_FORMAT_BC5_UNORM_BLOCK;
            
            // BC6H (HDR)
            case 95: // DXGI_FORMAT_BC6H_UF16
                return VK_FORMAT_BC6H_UFLOAT_BLOCK;
            case 96: // DXGI_FORMAT_BC6H_SF16
                return VK_FORMAT_BC6H_SFLOAT_BLOCK;
            
            // BC7
            case 98: // DXGI_FORMAT_BC7_UNORM
                return srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
            case 99: // DXGI_FORMAT_BC7_UNORM_SRGB
                return VK_FORMAT_BC7_SRGB_BLOCK;

            
            // Floating point formats
            case 2: // DXGI_FORMAT_R32G32B32A32_FLOAT
                return VK_FORMAT_R32G32B32A32_SFLOAT;
            case 10: // DXGI_FORMAT_R16G16B16A16_FLOAT
                return VK_FORMAT_R16G16B16A16_SFLOAT;
            case 41: // DXGI_FORMAT_R32_FLOAT
                return VK_FORMAT_R32_SFLOAT;
            case 54: // DXGI_FORMAT_R16_FLOAT
                return VK_FORMAT_R16_SFLOAT;
            
            // 16-bit formats
            case 56: // DXGI_FORMAT_R16G16B16A16_UNORM
                return VK_FORMAT_R16G16B16A16_UNORM;
            case 57: // DXGI_FORMAT_R16G16B16A16_SNORM
                return VK_FORMAT_R16G16B16A16_SNORM;
            
            default:
                SPDLOG_WARN("Unknown DXGI format: {}, using VK_FORMAT_UNDEFINED", dxgiFormat);
                return VK_FORMAT_UNDEFINED;
        }
    }

#if WITH_KTX2

    static void ProcessKtx(ktxTexture* kTexture, bool srgb, VkFormat& format, uint32_t& miplevel, uint8_t*& pixels, uint32_t& size, int& width, int& height)
    {
        if (kTexture->classId == ktxTexture2_c)
        {
            ktxTexture2* kTex2 = reinterpret_cast<ktxTexture2*>(kTexture);
            if (ktxTexture2_NeedsTranscoding(kTex2))
            {
                ktx_error_code_e result = ktxTexture2_TranscodeBasis(kTex2, KTX_TTF_BC7_RGBA, 0);
                if (result != KTX_SUCCESS) Throw(std::runtime_error("failed to transcode ktx2 texture image "));
                format = srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
                miplevel = 1;
            }
            else
            {
                format = static_cast<VkFormat>(kTex2->vkFormat);
                miplevel = kTex2->numLevels;
            }
        }
        else
        {
            ktxTexture1* kTex1 = reinterpret_cast<ktxTexture1*>(kTexture);
            format = MapGlFormatToVulkan(kTex1->glInternalformat, srgb);
            miplevel = kTex1->numLevels;
        }

        pixels = ktxTexture_GetData(kTexture);

        ktx_size_t offset;
        ktxTexture_GetImageOffset(kTexture, 0, 0, 0, &offset);
        pixels += offset;
        size = static_cast<uint32_t>(ktxTexture_GetImageSize(kTexture, 0));

        width = kTexture->baseWidth;
        height = kTexture->baseHeight;
    }
#endif

    uint32_t GlobalTexturePool::RequestNewTextureMemAsync(const std::string& texname, const std::string& mime, bool hdr,
                                                          const unsigned char* data, size_t bytelength, bool srgb)
    {
        uint32_t newTextureIdx = 0;
        if (textureNameMap_.find(texname) != textureNameMap_.end())
        {
            // 这里要判断一下，如果TextureUnLoaded，重新绑定
            if(textureNameMap_[texname].Status_ == ETextureStatus::ETS_Unloaded)
            {
                textureNameMap_[texname].Status_ = ETextureStatus::ETS_Loaded;
                newTextureIdx = textureNameMap_[texname].GlobalIdx_;
            }
            else
            {
                // 这里要判断一下，如果已经加载了，直接返回
                return textureNameMap_[texname].GlobalIdx_;
            }
        }
        else
        {
            textureImages_.emplace_back(nullptr);
            newTextureIdx = static_cast<uint32_t>(textureImages_.size()) - 1;
            textureNameMap_[texname] = { newTextureIdx, ETextureStatus::ETS_Loaded };
        }

        // load parse bind texture into newTextureIdx with transfer queue

        uint8_t* copyedData = new uint8_t[bytelength];
        memcpy(copyedData, data, bytelength);
        TaskCoordinator::GetInstance()->AddTask(
            [this, hdr, srgb, texname, mime, copyedData, bytelength, newTextureIdx](ResTask& task)
            {
                TextureTaskContext taskContext{};
                const auto timer = std::chrono::high_resolution_clock::now();

                // Load the texture in normal host memory.
                int width = 32;
                int height = 32;
                int channels = 4;
                uint8_t* stbdata = nullptr;
                uint8_t* pixels = nullptr;
                uint32_t size = 0;
                uint32_t miplevel = 1;
                VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
#if WITH_KTX2
                ktxTexture* kTexture = nullptr;
                ktx_error_code_e result;
#endif
                VkComponentMapping swizzle = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
                // load from dds
                bool isDDS = (mime.find("image/dds") != std::string::npos) || (bytelength > 128 && *(uint32_t*)copyedData == 0x20534444);
                if (isDDS)
                {
                    struct DDS_PIXELFORMAT {
                        uint32_t dwSize, dwFlags, dwFourCC, dwRGBBitCount, dwRBitMask, dwGBitMask, dwBBitMask, dwABitMask;
                    };
                    struct DDS_HEADER {
                        uint32_t dwSize, dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount, dwReserved1[11];
                        DDS_PIXELFORMAT ddspf;
                        uint32_t dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2;
                    };

                    auto* header = (DDS_HEADER*)(copyedData + 4);
                    width = header->dwWidth;
                    height = header->dwHeight;
                    
                    // Properly handle mipmap levels
                    // dwFlags & 0x20000 = DDSD_MIPMAPCOUNT
                    miplevel = (header->dwFlags & 0x20000) ? std::max(1u, header->dwMipMapCount) : 1;
                    
                    pixels = copyedData + 4 + header->dwSize;

                    // Process FourCC formats
                    if (header->ddspf.dwFlags & 0x4) { // DDPF_FOURCC
                        switch (header->ddspf.dwFourCC) {
                        case 0x31545844: // DXT1
                            format = srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                            break;
                        case 0x33545844: // DXT3
                            format = srgb ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
                            break;
                        case 0x35545844: // DXT5
                            format = srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
                            break;
                        case 0x55344342: // BC4U
                        case 0x31495441: // ATI1
                            format = VK_FORMAT_BC4_UNORM_BLOCK;
                            swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                            break;
                        case 0x53344342: // BC4S
                            format = VK_FORMAT_BC4_SNORM_BLOCK;
                            swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                            break;
                        case 0x55354342: // BC5U
                        case 0x32495441: // ATI2
                            format = VK_FORMAT_BC5_UNORM_BLOCK;
                            swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE };
                            break;
                        case 0x53354342: // BC5S
                            format = VK_FORMAT_BC5_SNORM_BLOCK;
                            swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE };
                            break;
                        case 0x30315844: // DX10
                        {
                            struct DDS_HEADER_DXT10 {
                                uint32_t dxgiFormat, resourceDimension, miscFlag, arraySize, miscFlags2;
                            };
                            auto* header10 = (DDS_HEADER_DXT10*)pixels;
                            pixels += sizeof(DDS_HEADER_DXT10);
                            
                            format = MapDxgiFormatToVulkan(header10->dxgiFormat, srgb, swizzle);
                            
                            if (format == VK_FORMAT_UNDEFINED) {
                                SPDLOG_ERROR("Unsupported DXGI format {} in DDS file: {}", header10->dxgiFormat, texname);
                                format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                            }
                            break;
                        }
                        default:
                            SPDLOG_WARN("Unknown DDS FourCC: 0x{:08X} in file: {}", header->ddspf.dwFourCC, texname);
                            format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                            break;
                        }
                    }
                    
                    // Process uncompressed and luminance formats
                    // Note: Use independent if/else or flags to avoid skipping Luminance if RGB is set
                    uint32_t pfFlags = header->ddspf.dwFlags;
                    if (!(pfFlags & 0x4)) { // NOT FourCC
                        if (pfFlags & 0x20000) { // DDPF_LUMINANCE
                            if (header->ddspf.dwRGBBitCount == 8) {
                                format = VK_FORMAT_R8_UNORM;
                                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE };
                            } else if (header->ddspf.dwRGBBitCount == 16) {
                                format = VK_FORMAT_R8G8_UNORM;
                                swizzle = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G };
                            }
                            SPDLOG_INFO("DDS Luminance: {}, format: {}", texname, (int)format);
                        }
                        else if (pfFlags & 0x2) { // DDPF_ALPHA
                            format = VK_FORMAT_R8_UNORM;
                            swizzle = { VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_R };
                            SPDLOG_INFO("DDS Alpha-only: {}, format: {}", texname, (int)format);
                        }
                        else if (pfFlags & 0x40) { // DDPF_RGB
                            if (header->ddspf.dwRGBBitCount == 32) {
                                if (header->ddspf.dwRBitMask == 0x00FF0000 && header->ddspf.dwBBitMask == 0x000000FF) {
                                    format = srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
                                } else {
                                    format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                                }
                            } else if (header->ddspf.dwRGBBitCount == 24) {
                                SPDLOG_INFO("Converting 24-bit RGB DDS to 32-bit RGBA: {}", texname);
                                uint32_t pixelCount = width * height;
                                uint32_t newSize = pixelCount * 4;
                                uint8_t* rgbaPixels = new uint8_t[newSize];
                                uint32_t i3 = 0, i4 = 0;
                                for (uint32_t i = 0; i < pixelCount; ++i) {
                                    rgbaPixels[i4] = pixels[i3];
                                    rgbaPixels[i4 + 1] = pixels[i3 + 1];
                                    rgbaPixels[i4 + 2] = pixels[i3 + 2];
                                    rgbaPixels[i4 + 3] = 255;
                                    i3 += 3; i4 += 4;
                                }
                                stbdata = rgbaPixels; // Use stbdata for automatic cleanup later
                                pixels = rgbaPixels;
                                size = newSize;
                                format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                            } else if (header->ddspf.dwRGBBitCount == 16) {
                                format = VK_FORMAT_R5G6B5_UNORM_PACK16;
                            }
                            SPDLOG_INFO("DDS RGB: {}, bitCount: {}, format: {}", texname, header->ddspf.dwRGBBitCount, (int)format);
                        }
                    }
                    
                    // Calculate data size
                    size = bytelength - static_cast<uint32_t>(pixels - copyedData);
                    
                    SPDLOG_INFO("Loaded DDS: {} ({}x{}, {} mips, format: {}, srgb: {}, swizzle: [{},{},{},{}])", 
                                texname, width, height, miplevel, static_cast<int>(format), srgb,
                                (int)swizzle.r, (int)swizzle.g, (int)swizzle.b, (int)swizzle.a);
                }
                // load from ktx inside glb
                else if (mime.find("image/ktx") != std::string::npos)
                {
#if WITH_KTX2
                    result = ktxTexture_CreateFromMemory(copyedData, bytelength, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &kTexture);
                    if (KTX_SUCCESS != result) Throw(std::runtime_error("failed to load ktx texture image "));

                    ProcessKtx(kTexture, srgb, format, miplevel, pixels, size, width, height);
#endif
                }
                else
                {
                    std::hash<std::string> hasher;
                    // load from texture files
                    if (hdr)
                    {
                        std::string cacheFileName = Utilities::CookHelper::GetCookedFileName(fmt::format("{:016x}", hasher(texname)), "texhdr");
                        std::filesystem::path cacheFilePath(cacheFileName);
                        bool cacheLoaded = false;
                        std::vector<std::vector<float>> mipLevels;
                        std::vector<std::pair<int, int>> mipDimensions;

                        if (std::filesystem::exists(cacheFilePath))
                        {
                            std::ifstream cacheFile(cacheFileName, std::ios::binary);
                            if (cacheFile.is_open())
                            {
                                HdrCacheHeader header{};
                                cacheFile.read(reinterpret_cast<char*>(&header), sizeof(header));

                                bool validCache = cacheFile.gcount() == sizeof(header)
                                    && header.magic == kHdrCacheMagic
                                    && header.version == kHdrCacheVersion
                                    && header.originalSize > 0
                                    && header.compressedSize > 0
                                    && header.originalSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())
                                    && header.compressedSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())
                                    && header.originalSize <= static_cast<uint64_t>(std::numeric_limits<int>::max())
                                    && header.compressedSize <= static_cast<uint64_t>(std::numeric_limits<int>::max());

                                if (validCache)
                                {
                                    size_t compressedSize = static_cast<size_t>(header.compressedSize);
                                    size_t originalSize = static_cast<size_t>(header.originalSize);

                                    std::vector<uint8_t> compressedData(compressedSize);
                                    cacheFile.read(reinterpret_cast<char*>(compressedData.data()), compressedSize);
                                    if (!cacheFile)
                                    {
                                        validCache = false;
                                    }
                                    else
                                    {
                                        std::vector<uint8_t> uncompressedData(originalSize);
                                        size_t decompressedSize = lzav_decompress(
                                            compressedData.data(), uncompressedData.data(),
                                            static_cast<int>(compressedSize), static_cast<int>(originalSize));

                                        if (decompressedSize != originalSize
                                            || HashBuffer(uncompressedData.data(), uncompressedData.size()) != header.dataHash)
                                        {
                                            validCache = false;
                                        }
                                        else
                                        {
                                            size_t offset = 0;
                                            auto readFromBuffer = [&](void* dst, size_t readSize) -> bool
                                            {
                                                if (offset + readSize > uncompressedData.size())
                                                {
                                                    return false;
                                                }
                                                std::memcpy(dst, uncompressedData.data() + offset, readSize);
                                                offset += readSize;
                                                return true;
                                            };

                                            SphericalHarmonics sh{};
                                            size_t mipCount = 0;

                                            if (!readFromBuffer(&width, sizeof(int))
                                                || !readFromBuffer(&height, sizeof(int))
                                                || !readFromBuffer(&miplevel, sizeof(uint32_t))
                                                || !readFromBuffer(&sh, sizeof(SphericalHarmonics))
                                                || !readFromBuffer(&mipCount, sizeof(size_t)))
                                            {
                                                validCache = false;
                                            }
                                            else
                                            {
                                                hdrSphericalHarmonics_[newTextureIdx] = sh;
                                                mipDimensions.resize(mipCount);
                                                for (auto& dim : mipDimensions)
                                                {
                                                    if (!readFromBuffer(&dim.first, sizeof(int))
                                                        || !readFromBuffer(&dim.second, sizeof(int)))
                                                    {
                                                        validCache = false;
                                                        break;
                                                    }
                                                }
                                            }

                                            if (validCache)
                                            {
                                                format = VK_FORMAT_R32G32B32A32_SFLOAT;
                                                size = width * height * 4 * sizeof(float);
                                                stbdata = reinterpret_cast<uint8_t*>(malloc(size));
                                                pixels = stbdata;
                                                if (!readFromBuffer(pixels, size))
                                                {
                                                    validCache = false;
                                                }
                                            }

                                            if (validCache)
                                            {
                                                mipLevels.clear();
                                                mipLevels.resize(mipCount);
                                                for (auto& mipData : mipLevels)
                                                {
                                                    size_t mipSize = 0;
                                                    if (!readFromBuffer(&mipSize, sizeof(size_t)))
                                                    {
                                                        validCache = false;
                                                        break;
                                                    }
                                                    mipData.resize(mipSize);
                                                    if (!readFromBuffer(mipData.data(), mipSize * sizeof(float)))
                                                    {
                                                        validCache = false;
                                                        break;
                                                    }
                                                }
                                            }

                                            if (validCache)
                                            {
                                                textureImages_[newTextureIdx] = std::make_unique<TextureImage>(
                                                    commandPool_, width, height, miplevel, format,
                                                    pixels, size, mipLevels, mipDimensions, swizzle);
                                                cacheLoaded = true;
                                            }
                                        }
                                    }
                                }

                                cacheFile.close();
                            }

                            if (!cacheLoaded)
                            {
                                std::error_code removeError;
                                std::filesystem::remove(cacheFilePath, removeError);
                            }
                        }

                        if (!cacheLoaded)
                        {
                            stbdata = reinterpret_cast<uint8_t*>(stbi_loadf_from_memory(
                                copyedData, static_cast<uint32_t>(bytelength), &width, &height, &channels, STBI_rgb_alpha));
                            pixels = stbdata;
                            format = VK_FORMAT_R32G32B32A32_SFLOAT;
                            size = width * height * 4 * sizeof(float);

                            SphericalHarmonics sh = ProjectHdrToSh((float*)pixels, width, height);
                            hdrSphericalHarmonics_[newTextureIdx] = sh;

                            PrefilterHdrEnvironmentMap((float*)pixels, width, height, mipLevels, mipDimensions);
                            miplevel = static_cast<uint32_t>(mipLevels.size());

                            std::vector<uint8_t> uncompressedData;
                            auto writeToBuffer = [&](const void* src, size_t writeSize)
                            {
                                const uint8_t* bytes = static_cast<const uint8_t*>(src);
                                uncompressedData.insert(uncompressedData.end(), bytes, bytes + writeSize);
                            };

                            writeToBuffer(&width, sizeof(int));
                            writeToBuffer(&height, sizeof(int));
                            writeToBuffer(&miplevel, sizeof(uint32_t));
                            writeToBuffer(&sh, sizeof(SphericalHarmonics));

                            size_t mipCount = mipDimensions.size();
                            writeToBuffer(&mipCount, sizeof(size_t));
                            for (const auto& dim : mipDimensions)
                            {
                                writeToBuffer(&dim.first, sizeof(int));
                                writeToBuffer(&dim.second, sizeof(int));
                            }

                            writeToBuffer(pixels, size);

                            for (const auto& mipData : mipLevels)
                            {
                                size_t mipSize = mipData.size();
                                writeToBuffer(&mipSize, sizeof(size_t));
                                writeToBuffer(mipData.data(), mipSize * sizeof(float));
                            }

                            size_t uncompressedSize = uncompressedData.size();
                            if (uncompressedSize <= static_cast<size_t>(std::numeric_limits<int>::max()))
                            {
                                size_t compressedBound = lzav_compress_bound_hi(int(uncompressedSize));
                                std::vector<uint8_t> compressedData(compressedBound);
                                size_t actualCompressedSize = lzav_compress_hi(
                                    uncompressedData.data(), compressedData.data(),
                                    int(uncompressedSize), int(compressedBound));

                                if (actualCompressedSize > 0)
                                {
                                    HdrCacheHeader header{};
                                    header.magic = kHdrCacheMagic;
                                    header.version = kHdrCacheVersion;
                                    header.originalSize = static_cast<uint64_t>(uncompressedSize);
                                    header.compressedSize = static_cast<uint64_t>(actualCompressedSize);
                                    header.dataHash = HashBuffer(uncompressedData.data(), uncompressedData.size());

                                    std::filesystem::path tempCachePath = cacheFilePath;
                                    tempCachePath += ".tmp";

                                    std::ofstream cacheFile(tempCachePath, std::ios::binary | std::ios::trunc);
                                    if (cacheFile.is_open())
                                    {
                                        cacheFile.write(reinterpret_cast<const char*>(&header), sizeof(header));
                                        cacheFile.write(reinterpret_cast<const char*>(compressedData.data()), actualCompressedSize);
                                        cacheFile.flush();
                                        cacheFile.close();

                                        std::error_code removeError;
                                        std::filesystem::remove(cacheFilePath, removeError);

                                        std::error_code renameError;
                                        std::filesystem::rename(tempCachePath, cacheFilePath, renameError);
                                        if (renameError)
                                        {
                                            std::filesystem::remove(tempCachePath);
                                        }
                                    }
                                    else
                                    {
                                        std::error_code removeTempError;
                                        std::filesystem::remove(tempCachePath, removeTempError);
                                    }
                                }
                            }

                            textureImages_[newTextureIdx] = std::make_unique<TextureImage>(
                                commandPool_, width, height, miplevel, format,
                                pixels, size, mipLevels, mipDimensions, swizzle);
                        }
                    }
                    else
                    {
#if WITH_KTX2
                        // ldr texture, try cache fist
                        // hash the texname
                        std::string cacheFileName = Utilities::CookHelper::GetCookedFileName(fmt::format("{:016x}", hasher(texname)), "texktx");
                        if (!std::filesystem::exists(cacheFileName))
                        {
                            // load from stbi and compress to ktx and cache
                            stbdata = stbi_load_from_memory(copyedData, static_cast<uint32_t>(bytelength), &width, &height, &channels, STBI_rgb_alpha);
                            format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
                            size = width * height * 4 * sizeof(uint8_t);

                            ktxTextureCreateInfo createInfo = {
                                0,
                                static_cast<uint32_t>(format),
                                0,
                                static_cast<uint32_t>(width),
                                static_cast<uint32_t>(height),
                                1, 2, 1, 1, 1,KTX_FALSE,KTX_FALSE
                            };

                            result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, reinterpret_cast<ktxTexture2**>(&kTexture));
                            if (result != KTX_SUCCESS) Throw(std::runtime_error("failed to create ktx2 image "));

                            std::memcpy(ktxTexture_GetData(kTexture), stbdata, size);

                            ktxBasisParams params = {};
                            params.structSize = sizeof(params);
                            params.uastc = KTX_TRUE;
                            params.compressionLevel = 2;
                            params.qualityLevel = 128;
                            params.threadCount = 12;
                            result = ktxTexture2_CompressBasisEx(reinterpret_cast<ktxTexture2*>(kTexture), &params);
                            if (KTX_SUCCESS != result) Throw(std::runtime_error("failed to compress ktx2 image "));
                            // save to cache
                            ktxTexture_WriteToNamedFile(kTexture, cacheFileName.c_str());
                        }
                        else
                        {
                            result = ktxTexture_CreateFromNamedFile(cacheFileName.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &kTexture);
                            if (result != KTX_SUCCESS) Throw(std::runtime_error("failed to load ktx image "));
                        }

                        ProcessKtx(kTexture, srgb, format, miplevel, pixels, size, width, height);
#endif
                    }
                }

                // load from memory using stb
                if (stbdata)
                {
                    SPDLOG_INFO("Loaded STB from memory: {} ({}x{}, format: {}, srgb: {})", 
                                texname, width, height, static_cast<int>(format), srgb);
                }
                else if (kTexture)
                {
                    SPDLOG_INFO("Loaded KTX from memory: {} ({}x{}, mips: {}, format: {})", 
                                texname, width, height, miplevel, static_cast<int>(format));
                }

                // create texture image
                if (!hdr)
                {
#if WITH_KTX2
                    if (kTexture) // Only attempt to read swizzle if kTexture was loaded/created
                    {
                        ktx_uint8_t* swizzlePtr;
                        unsigned int swizzleLen;
                        if (ktxHashList_FindValue(&kTexture->kvDataHead, KTX_SWIZZLE_KEY, &swizzleLen, (void**)&swizzlePtr) == KTX_SUCCESS) {
                            auto parse = [](char c) {
                                if (c == 'r') return VK_COMPONENT_SWIZZLE_R;
                                if (c == 'g') return VK_COMPONENT_SWIZZLE_G;
                                if (c == 'b') return VK_COMPONENT_SWIZZLE_B;
                                if (c == 'a') return VK_COMPONENT_SWIZZLE_A;
                                if (c == '0') return VK_COMPONENT_SWIZZLE_ZERO;
                                if (c == '1') return VK_COMPONENT_SWIZZLE_ONE;
                                return VK_COMPONENT_SWIZZLE_IDENTITY;
                            };
                            if (swizzleLen >= 4) {
                                swizzle.r = parse(swizzlePtr[0]);
                                swizzle.g = parse(swizzlePtr[1]);
                                swizzle.b = parse(swizzlePtr[2]);
                                swizzle.a = parse(swizzlePtr[3]);
                            }
                        }
                    }
#endif
                if (isDDS && miplevel > 1)
                {
                    // Create image with all mips
                    textureImages_[newTextureIdx] = std::make_unique<TextureImage>(commandPool_, width, height, miplevel, format, nullptr, 0, swizzle);
                    
                    // Upload all mips
                    uint32_t currWidth = width;
                    uint32_t currHeight = height;
                    const uint8_t* currData = pixels;
                    
                    for (uint32_t level = 0; level < miplevel; ++level)
                    {
                        uint32_t mipSize = 0;
                        if (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK) {
                            uint32_t blockWidth = (currWidth + 3) / 4;
                            uint32_t blockHeight = (currHeight + 3) / 4;
                            uint32_t blockSize = (format == VK_FORMAT_BC1_RGB_UNORM_BLOCK || format == VK_FORMAT_BC1_RGB_SRGB_BLOCK || 
                                                 format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK ||
                                                 format == VK_FORMAT_BC4_UNORM_BLOCK || format == VK_FORMAT_BC4_SNORM_BLOCK) ? 8 : 16;
                            mipSize = blockWidth * blockHeight * blockSize;
                        } else {
                            // Assume 4 bytes per pixel for uncompressed (RGBA8/BGRA8)
                            mipSize = currWidth * currHeight * 4;
                        }
                        
                        // Safety check
                        if (currData + mipSize <= copyedData + bytelength + (stbdata != nullptr ? width*height*4 : 0)) {
                            textureImages_[newTextureIdx]->UpdateDataMainThread(commandPool_, 0, 0, currWidth, currHeight, currWidth, currHeight, currData, mipSize, level);
                        }
                        
                        currData += mipSize;
                        currWidth = std::max(1u, currWidth / 2);
                        currHeight = std::max(1u, currHeight / 2);
                    }
                }
                else
                {
                    textureImages_[newTextureIdx] = std::make_unique<TextureImage>(commandPool_, width, height, miplevel, format, pixels, size, swizzle);
                }
            }

        BindTexture(newTextureIdx, *(textureImages_[newTextureIdx]));

        // clean up
        if (stbdata) stbi_image_free(stbdata);
        
#if WITH_KTX2
        if (kTexture) ktxTexture_Destroy(kTexture);
#endif
        
        // transfer
        taskContext.textureId = newTextureIdx;
        taskContext.needFlushHDRSH = hdr;
        taskContext.elapsed = std::chrono::duration<float, std::chrono::seconds::period>(
            std::chrono::high_resolution_clock::now() - timer).count();
        std::string info = fmt::format("loaded {} ({} x {} x {}) in {:.2f}ms", texname, width, height, miplevel,
                                       taskContext.elapsed * 1000.f);
        std::copy(info.begin(), info.end(), taskContext.outputInfo.data());
        task.SetContext(taskContext);
    }, [this, copyedData](ResTask& task)
    {
        TextureTaskContext taskContext{};
        task.GetContext(taskContext);
        textureImages_[taskContext.textureId]->MainThreadPostLoading(mainThreadCommandPool_);
        //SPDLOG_INFO("{}", taskContext.outputInfo.data());
        delete[] copyedData;

        if (taskContext.needFlushHDRSH)
        {
            NextEngine::GetInstance()->GetScene().UpdateHDRSH();
        }
    }, 0);

    return newTextureIdx;
}

    void GlobalTexturePool::FreeNonSystemTextures()
    {
        // make sure the binded image not in use
        device_.WaitIdle();
        
        for( int i = 0; i < textureImages_.size(); ++i)
        {
            if( i > 10 )
            {
                // free up TextureImage;, rebind with a default texture sampler
                textureImages_[i].reset();
                BindTexture(i, *defaultWhiteTexture_);
            }
        }

        for( auto& textureGroup : textureNameMap_ )
        {
            if( textureGroup.second.GlobalIdx_ > 10 )
            {
                textureGroup.second.Status_ = ETextureStatus::ETS_Unloaded;
            }
        }
    }

    void GlobalTexturePool::CreateDefaultTextures()
    {
        defaultWhiteTexture_ = std::make_unique<TextureImage>(commandPool_, 16, 16, 1, VK_FORMAT_R8G8B8A8_UNORM, nullptr, 0);
    }

    GlobalTexturePool* GlobalTexturePool::instance_ = nullptr;
}
