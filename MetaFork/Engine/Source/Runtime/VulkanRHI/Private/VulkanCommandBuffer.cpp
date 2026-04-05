// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	VulkanCommandBuffer.cpp: Vulkan device RHI implementation.
=============================================================================*/

#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanRenderpass.h"
#include "VulkanDescriptorSets.h"
#include "VulkanMemory.h"
#include "VulkanRayTracing.h"


#define CMD_BUFFER_TIME_TO_WAIT_BEFORE_DELETING		10


FVulkanCommandBuffer::FVulkanCommandBuffer(FVulkanDevice& InDevice, FVulkanCommandBufferPool& InCommandBufferPool)
	: Device(InDevice)
	, CommandBufferPool(InCommandBufferPool)
#if RHI_NEW_GPU_PROFILER
	, EventStream(InCommandBufferPool.GetQueue().GetProfilerQueue())
#endif
{
	FScopeLock ScopeLock(CommandBufferPool.GetCS());
	AllocMemory();
}

void FVulkanCommandBuffer::AllocMemory()
{
	// Assumes we are inside a lock for the pool
	check(State == EState::NotAllocated);
	CurrentViewports.Empty();
	CurrentScissors.Empty();

	VkCommandBufferAllocateInfo CreateCmdBufInfo;
	ZeroVulkanStruct(CreateCmdBufInfo, VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
	CreateCmdBufInfo.level = (GetCommandBufferType() == EVulkanCommandBufferType::Primary) ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
	CreateCmdBufInfo.commandBufferCount = 1;
	CreateCmdBufInfo.commandPool = CommandBufferPool.GetHandle();

	VERIFYVULKANRESULT(VulkanRHI::vkAllocateCommandBuffers(Device.GetInstanceHandle(), &CreateCmdBufInfo, &CommandBufferHandle));

	bNeedsDynamicStateSet = 1;
	bHasPipeline = 0;
	bHasViewport = 0;
	bHasScissor = 0;
	bHasStencilRef = 0;
	State = EState::ReadyForBegin;

	INC_DWORD_STAT(STAT_VulkanNumCmdBuffers);
}

FVulkanCommandBuffer::~FVulkanCommandBuffer()
{
	if (State != EState::NotAllocated)
	{
		FreeMemory();
	}
}

void FVulkanCommandBuffer::FreeMemory()
{
	// Assumes we are inside a lock for the pool
	check(State != EState::NotAllocated);
	check(CommandBufferHandle != VK_NULL_HANDLE);
	VulkanRHI::vkFreeCommandBuffers(Device.GetInstanceHandle(), CommandBufferPool.GetHandle(), 1, &CommandBufferHandle);
	CommandBufferHandle = VK_NULL_HANDLE;

	DEC_DWORD_STAT(STAT_VulkanNumCmdBuffers);
	State = EState::NotAllocated;
}

void FVulkanCommandBuffer::EndRenderPass()
{
	checkf(IsInsideRenderPass(), TEXT("Can't EndRP as we're NOT inside one! CmdBuffer 0x%p State=%d"), CommandBufferHandle, (int32)State);
#if VULKAN_SUPPORTS_RENDERPASS2
	if (Device.GetOptionalExtensions().HasKHRRenderPass2)
	{
		VkSubpassEndInfo SubpassInfo;
		ZeroVulkanStruct(SubpassInfo, VK_STRUCTURE_TYPE_SUBPASS_END_INFO);

		VkOffset2D Offsets[2];
		VkSubpassFragmentDensityMapOffsetEndInfoQCOM offsetInfo;
		if (Device.GetOptionalExtensions().HasQcomFragmentDensityMapOffset && GVulkanUseQcomFragmentDensityMapOffsets && RenderPassProperties.bHasFragmentDensityMap)
		{
			Offsets[0].x = (GVulkanQcomFragmentDensityMapOffsets[0].X / GRHIVariableRateShadingImageOffsetGranularity.X) * GRHIVariableRateShadingImageOffsetGranularity.X;
			Offsets[0].y = (GVulkanQcomFragmentDensityMapOffsets[0].Y / GRHIVariableRateShadingImageOffsetGranularity.Y) * GRHIVariableRateShadingImageOffsetGranularity.Y;
			Offsets[1].x = (GVulkanQcomFragmentDensityMapOffsets[1].X / GRHIVariableRateShadingImageOffsetGranularity.X) * GRHIVariableRateShadingImageOffsetGranularity.X;
			Offsets[1].y = (GVulkanQcomFragmentDensityMapOffsets[1].Y / GRHIVariableRateShadingImageOffsetGranularity.Y) * GRHIVariableRateShadingImageOffsetGranularity.Y;

			ZeroVulkanStruct(offsetInfo, VK_STRUCTURE_TYPE_SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM);
			offsetInfo.sType = (VkStructureType)VK_STRUCTURE_TYPE_SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM;
			offsetInfo.pFragmentDensityOffsets = Offsets;
			offsetInfo.fragmentDensityOffsetCount = 2;

			SubpassInfo.pNext = &offsetInfo;
		}
		VulkanRHI::vkCmdEndRenderPass2KHR(CommandBufferHandle, &SubpassInfo);
	}
	else
#endif // VULKAN_SUPPORTS_RENDERPASS2
	{
	VulkanRHI::vkCmdEndRenderPass(CommandBufferHandle);
	}

	State = EState::IsInsideBegin;
	// BEGIN META SECTION - Fix Multiview Queries
	CurrentMultiViewCount = 0;
	// END META SECTION - Fix Multiview Queries
	RenderPassProperties = {};
}

// BEGIN META SECTION - Multi-View Per View Viewports / Render Areas
void FVulkanCommandBuffer::BeginRenderPass(const FVulkanBeginRenderPassInfo& BeginRenderPassInfo, const VkClearValue* AttachmentClearValues, const VkRect2D* RenderAreas, const uint32 NumRenderAreas )
{
// END META SECTION - Multi-View Per View Viewports / Render Areas
	checkf(IsOutsideRenderPass(), TEXT("Can't BeginRP as already inside one! CmdBuffer 0x%p State=%d"), CommandBufferHandle, (int32)State);

	const FVulkanRenderTargetLayout& Layout = BeginRenderPassInfo.RenderPass.GetLayout();

	VkRenderPassBeginInfo Info;
	ZeroVulkanStruct(Info, VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO);
	Info.renderPass = BeginRenderPassInfo.RenderPass.GetHandle();
	Info.framebuffer = BeginRenderPassInfo.Framebuffer.GetHandle();
	Info.renderArea = BeginRenderPassInfo.Framebuffer.GetRenderArea();
// BEGIN META SECTION - Multi-View Per View Viewports / Render Areas
	Info.renderArea = RenderAreas[0];
// END META SECTION - Multi-View Per View Viewports / Render Areas
	Info.clearValueCount = Layout.GetNumUsedClearValues();
	Info.pClearValues = AttachmentClearValues;

	const VkSubpassContents SubpassContents = BeginRenderPassInfo.bIsParallelRenderPass ?
		VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE;

	// BEGIN META SECTION - Multi-View Per View Viewports / Render Areas
#if VULKAN_SUPPORTS_MULTIVIEW_PER_VIEW_RENDER_AREAS
	if (NumRenderAreas > 1)
	{
		if (RenderAreas[0].offset.x != RenderAreas[1].offset.x ||
			RenderAreas[0].offset.y != RenderAreas[1].offset.y ||
			RenderAreas[0].extent.width != RenderAreas[1].extent.width ||
			RenderAreas[0].extent.height != RenderAreas[1].extent.height)
		{
			ZeroVulkanStruct(RenderAreaInfo, (VkStructureType)VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO_QCOM);

			RenderAreaRects = { RenderAreas[0], RenderAreas[1] };

			RenderAreaInfo.perViewRenderAreaCount = 2;
			RenderAreaInfo.pPerViewRenderAreas = RenderAreaRects.GetData();

			RenderAreaInfo.pNext = Info.pNext;
			Info.pNext = &RenderAreaInfo;

			// Update Info.renderArea to the union of the two render areas
			// Assuming first render area is left of second and they do not overlap
			VkRect2D renderAreaUnion = RenderAreas[0];
			renderAreaUnion.extent.width = (RenderAreas[1].offset.x - renderAreaUnion.offset.x) + RenderAreas[1].extent.width;
			renderAreaUnion.extent.height = (RenderAreas[1].offset.y - renderAreaUnion.offset.y) + RenderAreas[1].extent.height;

			Info.renderArea = renderAreaUnion;
		}
	}
#endif
	// END META SECTION - Multi-View Per View Viewports / Render Areas

	if (Device.GetOptionalExtensions().HasKHRRenderPass2)
	{
		VkSubpassBeginInfo SubpassInfo;
		ZeroVulkanStruct(SubpassInfo, VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO);
		SubpassInfo.contents = SubpassContents;
		VulkanRHI::vkCmdBeginRenderPass2KHR(CommandBufferHandle, &Info, &SubpassInfo);
	}
	else
	{
		VulkanRHI::vkCmdBeginRenderPass(CommandBufferHandle, &Info, SubpassContents);
	}

	RenderPassProperties.bHasFragmentDensityMap = Layout.GetHasFragmentDensityAttachment();
	// BEGIN META SECTION - Fix Multiview Queries
	CurrentMultiViewCount = BeginRenderPassInfo.RenderPass.GetLayout().GetMultiViewCount();
	// END META SECTION - Fix Multiview Queries
	State = EState::IsInsideRenderPass;

	// Acquire a descriptor pool set on a first render pass
	if (CurrentDescriptorPoolSetContainer == nullptr)
	{
		AcquirePoolSetContainer();
	}
}

void FVulkanCommandBuffer::End(FVulkanQueryPool* QueryPool)
{
	checkf(IsOutsideRenderPass() || (GetCommandBufferType() == EVulkanCommandBufferType::Secondary), 
		TEXT("Can't End as we're inside a render pass! CmdBuffer 0x%p State=%d"), CommandBufferHandle, (int32)State);

	// Reset barrier events for next use
	for (VkEvent BarrierEvent : EndedBarrierEvents)
	{
		VulkanRHI::vkCmdResetEvent(GetHandle(), BarrierEvent, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
	}

#if RHI_NEW_GPU_PROFILER
	if (QueryPool)
	{
		auto& Event = EmplaceProfilerEvent<UE::RHI::GPUProfiler::FEvent::FEndWork>(0);
		const uint32 IndexInPool = QueryPool->ReserveQuery(&Event.GPUTimestampBOP);
		VulkanRHI::vkCmdWriteTimestamp(GetHandle(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, QueryPool->GetHandle(), IndexInPool);
	}
#else
	if (QueryPool)
	{
		const uint32 IndexInPool = QueryPool->ReserveQuery(&EndTimestamp);
		VulkanRHI::vkCmdWriteTimestamp(GetHandle(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, QueryPool->GetHandle(), IndexInPool);
	}
#endif

	VERIFYVULKANRESULT(VulkanRHI::vkEndCommandBuffer(GetHandle()));
	State = EState::HasEnded;
}

void FVulkanCommandBuffer::Begin(FVulkanQueryPool* QueryPool, VkRenderPass RenderPassHandle)
{
	checkf((GetCommandBufferType() == EVulkanCommandBufferType::Primary) || (RenderPassHandle != VK_NULL_HANDLE),
		TEXT("Secondary command buffers require the render pass handle!"));

	{
		FScopeLock ScopeLock(CommandBufferPool.GetCS());
		if (State == EState::NeedReset)
		{
			VulkanRHI::vkResetCommandBuffer(CommandBufferHandle, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
		}
		else
		{
			checkf(State == EState::ReadyForBegin, TEXT("Can't Begin as we're NOT ready! CmdBuffer 0x%p State=%d"), CommandBufferHandle, (int32)State);
		}
		State = (RenderPassHandle != VK_NULL_HANDLE) ? EState::IsInsideRenderPass : EState::IsInsideBegin;
	}

	VkCommandBufferBeginInfo BeginInfo;
	ZeroVulkanStruct(BeginInfo, VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
	BeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VkCommandBufferInheritanceInfo InheritanceInfo;
	if (GetCommandBufferType() == EVulkanCommandBufferType::Secondary)
	{
		BeginInfo.flags |= VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		BeginInfo.pInheritanceInfo = &InheritanceInfo;

		ZeroVulkanStruct(InheritanceInfo, VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO);
		InheritanceInfo.renderPass = RenderPassHandle;
	}

	VERIFYVULKANRESULT(VulkanRHI::vkBeginCommandBuffer(CommandBufferHandle, &BeginInfo));

	check(!CurrentDescriptorPoolSetContainer);

	if (Device.SupportsBindless())
	{
		FVulkanBindlessDescriptorManager* BindlessDescriptorManager = Device.GetBindlessDescriptorManager();
		const VkPipelineStageFlags SupportedStages = CommandBufferPool.GetQueue().GetSupportedStageBits();
		BindlessDescriptorManager->BindDescriptorBuffers(CommandBufferHandle, SupportedStages);
	}

	bNeedsDynamicStateSet = true;

#if RHI_NEW_GPU_PROFILER
	if (QueryPool)
	{
		auto& Event = EmplaceProfilerEvent<UE::RHI::GPUProfiler::FEvent::FBeginWork>(0, UINT64_MAX);
		const uint32 IndexInPool = QueryPool->ReserveQuery(&Event.GPUTimestampTOP);
		VulkanRHI::vkCmdWriteTimestamp(GetHandle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, QueryPool->GetHandle(), IndexInPool);
	}
#else
	if (QueryPool)
	{
		StartTimestamp = 0;
		EndTimestamp = 0;
		const uint32 IndexInPool = QueryPool->ReserveQuery(&StartTimestamp);
		VulkanRHI::vkCmdWriteTimestamp(GetHandle(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, QueryPool->GetHandle(), IndexInPool);
	}
#endif
}

void FVulkanCommandBuffer::AcquirePoolSetContainer()
{
	check(!CurrentDescriptorPoolSetContainer);
	CurrentDescriptorPoolSetContainer = &Device.GetDescriptorPoolsManager().AcquirePoolSetContainer();
	ensure(TypedDescriptorPoolSets.Num() == 0);
}

bool FVulkanCommandBuffer::AcquirePoolSetAndDescriptorsIfNeeded(const class FVulkanDescriptorSetsLayout& Layout, bool bNeedDescriptors, VkDescriptorSet* OutDescriptors)
{
	//#todo-rco: This only happens when we call draws outside a render pass...
	if (!CurrentDescriptorPoolSetContainer)
	{
		AcquirePoolSetContainer();
	}

	const uint32 Hash = VULKAN_HASH_POOLS_WITH_TYPES_USAGE_ID ? Layout.GetTypesUsageID() : GetTypeHash(Layout);
	FVulkanTypedDescriptorPoolSet*& FoundTypedSet = TypedDescriptorPoolSets.FindOrAdd(Hash);

	if (!FoundTypedSet)
	{
		FoundTypedSet = CurrentDescriptorPoolSetContainer->AcquireTypedPoolSet(Layout);
		bNeedDescriptors = true;
	}

	if (bNeedDescriptors)
	{
		return FoundTypedSet->AllocateDescriptorSets(Layout, OutDescriptors);
	}

	return false;
}

void FVulkanCommandBuffer::Reset()
{
	// Reset the secondary command buffers we executed from this one
	for (FVulkanCommandBuffer* SecondaryCommand : ExecutedSecondaryCommandBuffers)
	{
		SecondaryCommand->Reset();
	}
	ExecutedSecondaryCommandBuffers.Empty();

	// Hold lock while State is altered
	FScopeLock ScopeLock(CommandBufferPool.GetCS());
	if (State == EState::Submitted)
	{
		bHasPipeline = false;
		bHasViewport = false;
		bHasScissor = false;
		bHasStencilRef = false;

		CurrentViewports.Empty();
		CurrentScissors.Empty();
		CurrentStencilRef = 0;

		if (CurrentDescriptorPoolSetContainer)
		{
			//#todo-rco: Reset here?
			TypedDescriptorPoolSets.Reset();
			Device.GetDescriptorPoolsManager().ReleasePoolSet(*CurrentDescriptorPoolSetContainer);
			CurrentDescriptorPoolSetContainer = nullptr;
		}
		else
		{
			check(TypedDescriptorPoolSets.Num() == 0);
		}

		for (VkEvent BarrierEvent : EndedBarrierEvents)
		{
			Device.ReleaseBarrierEvent(BarrierEvent);
		}
		EndedBarrierEvents.Reset();

		// Change state at the end to be safe
		State = EState::NeedReset;
	}
}

void FVulkanCommandBuffer::SetSubmitted()
{
	for (FVulkanCommandBuffer* SecondaryCommand : ExecutedSecondaryCommandBuffers)
	{
		SecondaryCommand->SetSubmitted();
	}

	FScopeLock Lock(CommandBufferPool.GetCS());
	State = FVulkanCommandBuffer::EState::Submitted;
	SubmittedTime = FPlatformTime::Seconds();
}

void FVulkanCommandBuffer::BeginSplitBarrier(VkEvent BarrierEvent, const VkDependencyInfo& DependencyInfo)
{
	VulkanRHI::vkCmdSetEvent2KHR(GetHandle(), BarrierEvent, &DependencyInfo);
}

void FVulkanCommandBuffer::EndSplitBarrier(VkEvent BarrierEvent, const VkDependencyInfo& DependencyInfo)
{
	VulkanRHI::vkCmdWaitEvents2KHR(GetHandle(), 1, &BarrierEvent, &DependencyInfo);
	EndedBarrierEvents.Add(BarrierEvent);
}


FVulkanCommandBufferPool::FVulkanCommandBufferPool(FVulkanDevice& InDevice, FVulkanQueue& InQueue, EVulkanCommandBufferType InCommandBufferType)
	: Device(InDevice)
	, Queue(InQueue)
	, CommandBufferType(InCommandBufferType)
{
	VkCommandPoolCreateInfo CmdPoolInfo;
	ZeroVulkanStruct(CmdPoolInfo, VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
	CmdPoolInfo.queueFamilyIndex = InQueue.GetFamilyIndex();
	CmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // :todo: Investigate use of VK_COMMAND_POOL_CREATE_TRANSIENT_BIT?
	VERIFYVULKANRESULT(VulkanRHI::vkCreateCommandPool(Device.GetInstanceHandle(), &CmdPoolInfo, VULKAN_CPU_ALLOCATOR, &Handle));
}

FVulkanCommandBufferPool::~FVulkanCommandBufferPool()
{
	for (int32 Index = 0; Index < CmdBuffers.Num(); ++Index)
	{
		FVulkanCommandBuffer* CmdBuffer = CmdBuffers[Index];
		delete CmdBuffer;
	}

	for (int32 Index = 0; Index < FreeCmdBuffers.Num(); ++Index)
	{
		FVulkanCommandBuffer* CmdBuffer = FreeCmdBuffers[Index];
		delete CmdBuffer;
	}

	VulkanRHI::vkDestroyCommandPool(Device.GetInstanceHandle(), Handle, VULKAN_CPU_ALLOCATOR);
	Handle = VK_NULL_HANDLE;
}

void FVulkanCommandBufferPool::FreeUnusedCmdBuffers(FVulkanQueue* InQueue, bool bTrimMemory)
{
#if VULKAN_DELETE_STALE_CMDBUFFERS
	FScopeLock ScopeLock(&CS);
	
	if (bTrimMemory)
	{
		VulkanRHI::vkTrimCommandPool(Device.GetInstanceHandle(), Handle, 0);
		return;
	}

	const double CurrentTime = FPlatformTime::Seconds();

	for (int32 Index = CmdBuffers.Num() - 1; Index >= 0; --Index)
	{
		FVulkanCommandBuffer* CmdBuffer = CmdBuffers[Index];
		if ((CmdBuffer->State == FVulkanCommandBuffer::EState::ReadyForBegin || CmdBuffer->State == FVulkanCommandBuffer::EState::NeedReset) &&
			((CurrentTime - CmdBuffer->SubmittedTime) > CMD_BUFFER_TIME_TO_WAIT_BEFORE_DELETING))
		{
			CmdBuffer->FreeMemory();
			CmdBuffers.RemoveAtSwap(Index, EAllowShrinking::No);
			FreeCmdBuffers.Add(CmdBuffer);
		}
	}
#endif
}

FVulkanCommandBuffer* FVulkanCommandBufferPool::Create()
{
	// Assumes we are inside a lock for the pool
	if (FreeCmdBuffers.Num())
	{
		FVulkanCommandBuffer* CmdBuffer = FreeCmdBuffers[0];
		FreeCmdBuffers.RemoveAtSwap(0);
		CmdBuffer->AllocMemory();
		CmdBuffers.Add(CmdBuffer);
		return CmdBuffer;
	}

	FVulkanCommandBuffer* CmdBuffer = new FVulkanCommandBuffer(Device, *this);
	CmdBuffers.Add(CmdBuffer);
	check(CmdBuffer);
	return CmdBuffer;
}

