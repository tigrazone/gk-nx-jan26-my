#pragma once

#include "Vulkan.hpp"

namespace Vulkan
{
	class Device;

	class ImageView final
	{
	public:

		VULKAN_NON_COPIABLE(ImageView)

		explicit ImageView(const Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, const uint32_t miplevel = 1, VkComponentMapping swizzle = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY });
		~ImageView();

		const class Device& Device() const { return device_; }


	private:

		const class Device& device_;

		VULKAN_HANDLE(VkImageView, imageView_)
	};

}
