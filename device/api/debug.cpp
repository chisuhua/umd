/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "libhsakmt.h"
#include "cmdio.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool *is_device_debugged;


int debug_get_reg_status(uint32_t node_id, bool *is_debugged);
uint32_t *convert_queue_ids(uint32_t NumQueues, HSA_QUEUEID *Queues);

device_status_t init_device_debugging_memory(unsigned int NumNodes)
{
	unsigned int i;

	is_device_debugged = (bool*)malloc(NumNodes * sizeof(bool));
	if (!is_device_debugged)
		return DEVICE_STATUS_NO_MEMORY;

	for (i = 0; i < NumNodes; i++)
		is_device_debugged[i] = false;

	return DEVICE_STATUS_SUCCESS;
}

void destroy_device_debugging_memory(void)
{
	if (is_device_debugged) {
		free(is_device_debugged);
		is_device_debugged = NULL;
	}
}

device_status_t DEVICEAPI DeviceDbgRegister(uint32_t NodeId)
{
	device_status_t result;
	uint32_t gpu_id;

	if (!is_device_debugged)
		return DEVICE_STATUS_NO_MEMORY;

	result = validate_nodeid(NodeId, &gpu_id);
	if (result != DEVICE_STATUS_SUCCESS)
		return result;

	struct ioctl_dbg_register_args args = {0};

	args.gpu_id = gpu_id;

	long err = cmd_dbg_register(&args);

	if (err == 0)
		result = DEVICE_STATUS_SUCCESS;
	else
		result = DEVICE_STATUS_ERROR;

	return result;
}

device_status_t DEVICEAPI DeviceDbgUnregister(uint32_t NodeId)
{
	uint32_t gpu_id;
	device_status_t result;


	if (!is_device_debugged)
		return DEVICE_STATUS_NO_MEMORY;

	result = validate_nodeid(NodeId, &gpu_id);
	if (result != DEVICE_STATUS_SUCCESS)
		return result;

	struct ioctl_dbg_unregister_args args = {0};

	args.gpu_id = gpu_id;
	long err = cmd_dbg_unregister(&args);

	if (err)
		return DEVICE_STATUS_ERROR;

	return DEVICE_STATUS_SUCCESS;
}

device_status_t DEVICEAPI DeviceDbgWavefrontControl(uint32_t NodeId,
						  HSA_DBG_WAVEOP Operand,
						  HSA_DBG_WAVEMODE Mode,
						  uint32_t TrapId,
						  HsaDbgWaveMessage *DbgWaveMsgRing)
{
	device_status_t result;
	uint32_t gpu_id;

	struct ioctl_dbg_wave_control_args *args;


	result = validate_nodeid(NodeId, &gpu_id);
	if (result != DEVICE_STATUS_SUCCESS)
		return result;


/* Determine Size of the ioctl buffer */
	uint32_t buff_size = sizeof(Operand) + sizeof(Mode) + sizeof(TrapId) +
			     sizeof(DbgWaveMsgRing->DbgWaveMsg) +
			     sizeof(DbgWaveMsgRing->MemoryVA) + sizeof(*args);

	args = (struct ioctl_dbg_wave_control_args *)malloc(buff_size);
	if (!args)
		return DEVICE_STATUS_ERROR;

	memset(args, 0, buff_size);

	args->gpu_id = gpu_id;
	args->buf_size_in_bytes = buff_size;

	/* increment pointer to the start of the non fixed part */
	unsigned char *run_ptr = (unsigned char *)args + sizeof(*args);

	/* save variable content pointer for kfd */
	args->content_ptr = (uint64_t)run_ptr;

	/* insert items, and increment pointer accordingly */
	*((HSA_DBG_WAVEOP *)run_ptr) = Operand;
	run_ptr += sizeof(Operand);

	*((HSA_DBG_WAVEMODE *)run_ptr) = Mode;
	run_ptr += sizeof(Mode);

	*((uint32_t *)run_ptr) = TrapId;
	run_ptr += sizeof(TrapId);

	*((HsaDbgWaveMessageAMD *)run_ptr) = DbgWaveMsgRing->DbgWaveMsg;
	run_ptr += sizeof(DbgWaveMsgRing->DbgWaveMsg);

	*((void **)run_ptr) = DbgWaveMsgRing->MemoryVA;
	run_ptr += sizeof(DbgWaveMsgRing->MemoryVA);

	/* send to kernel */
	long err = cmd_dbg_wave_control(args);

	free(args);

	if (err)
		return DEVICE_STATUS_ERROR;

	return DEVICE_STATUS_SUCCESS;
}

device_status_t DEVICEAPI DeviceDbgAddressWatch(uint32_t NodeId,
					      uint32_t NumWatchPoints,
					      HSA_DBG_WATCH_MODE WatchMode[],
					      void *WatchAddress[],
					      uint64_t WatchMask[],
					      HsaEvent *WatchEvent[])
{
	device_status_t result;
	uint32_t gpu_id;

	/* determine the size of the watch mask and event buffers
	 * the value is NULL if and only if no vector data should be attached
	 */
	uint32_t watch_mask_items = WatchMask[0] > 0 ? NumWatchPoints:1;
	uint32_t watch_event_items = WatchEvent != NULL ? NumWatchPoints:0;

	struct ioctl_dbg_address_watch_args *args;
	uint32_t		 i = 0;


	result = validate_nodeid(NodeId, &gpu_id);
	if (result != DEVICE_STATUS_SUCCESS)
		return result;

	if (NumWatchPoints > MAX_ALLOWED_NUM_POINTS)
		return DEVICE_STATUS_INVALID_PARAMETER;

	/* Size and structure of the ioctl buffer is dynamic in this case
	 * Here we calculate the buff size.
	 */
	uint32_t buff_size = sizeof(NumWatchPoints) +
		(sizeof(WatchMode[0]) + sizeof(WatchAddress[0])) *
			NumWatchPoints +
		watch_mask_items * sizeof(uint64_t) +
		watch_event_items * sizeof(HsaEvent *) + sizeof(*args);

	args = (struct ioctl_dbg_address_watch_args *) malloc(buff_size);
	if (!args)
		return DEVICE_STATUS_ERROR;

	memset(args, 0, buff_size);

	args->gpu_id = gpu_id;
	args->buf_size_in_bytes = buff_size;


	/* increment pointer to the start of the non fixed part */
	unsigned char *run_ptr = (unsigned char *)args + sizeof(*args);

	/* save variable content pointer for kfd */
	args->content_ptr = (uint64_t)run_ptr;
	/* insert items, and increment pointer accordingly */

	*((uint32_t *)run_ptr) = NumWatchPoints;
	run_ptr += sizeof(NumWatchPoints);

	for (i = 0; i < NumWatchPoints; i++) {
		*((HSA_DBG_WATCH_MODE *)run_ptr) = WatchMode[i];
		run_ptr += sizeof(WatchMode[i]);
	}

	for (i = 0; i < NumWatchPoints; i++) {
		*((void **)run_ptr) = WatchAddress[i];
		run_ptr += sizeof(WatchAddress[i]);
	}

	for (i = 0; i < watch_mask_items; i++) {
		*((uint64_t *)run_ptr) = WatchMask[i];
		run_ptr += sizeof(WatchMask[i]);
	}

	for (i = 0; i < watch_event_items; i++)	{
		*((HsaEvent **)run_ptr) = WatchEvent[i];
		run_ptr += sizeof(WatchEvent[i]);
	}

	/* send to kernel */
	long err = cmd_dbg_address_watch(args);

	free(args);

	if (err)
		return DEVICE_STATUS_ERROR;
	return DEVICE_STATUS_SUCCESS;
}

int debug_get_reg_status(uint32_t node_id, bool *is_debugged)
{
	*is_debugged = NULL;
	if (!is_device_debugged)
		return -1;

	*is_debugged = is_device_debugged[node_id];
	return 0;
}

static device_status_t debug_trap(uint32_t NodeId,
				uint32_t op,
				uint32_t data1,
				uint32_t data2,
				uint32_t data3,
				uint32_t pid,
				uint64_t pointer,
				struct ioctl_dbg_trap_args *argout)
{
	uint32_t gpu_id;
	device_status_t result;
	HsaCoreProperties* NodeProperties;
	struct ioctl_dbg_trap_args args = {0};

	if (op == KFD_IOC_DBG_TRAP_NODE_SUSPEND ||
			op == KFD_IOC_DBG_TRAP_NODE_RESUME ||
			op == KFD_IOC_DBG_TRAP_GET_VERSION ||
			op == KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT) {
		if  (NodeId != INVALID_NODEID)
			return DEVICE_STATUS_INVALID_HANDLE;

		// gpu_id is ignored for suspend/resume queues.
		gpu_id = INVALID_NODEID;
	} else {
		if (validate_nodeid(NodeId, &gpu_id) != DEVICE_STATUS_SUCCESS)
			return DEVICE_STATUS_INVALID_HANDLE;

		result = DeviceGetNodeProperties(NodeId, &NodeProperties);

		if (result != DEVICE_STATUS_SUCCESS)
			return result;

		if (!NodeProperties->Capability.ui32.DebugTrapSupported)
			return DEVICE_STATUS_NOT_SUPPORTED;
	}

	if (pid == INVALID_PID) {
		pid = (uint32_t) getpid();
	}

	memset(&args, 0x00, sizeof(args));
	args.gpu_id = gpu_id;
	args.op = op;
	args.data1 = data1;
	args.data2 = data2;
	args.data3 = data3;
	args.pid = pid;
	args.ptr = pointer;

	long err = cmd_dbg_trap(&args);

	if (argout)
		*argout = args;

	if ((op == KFD_IOC_DBG_TRAP_NODE_SUSPEND ||
			op == KFD_IOC_DBG_TRAP_NODE_RESUME) && err >= 0 &&
				err <= args.data2)
		result = DEVICE_STATUS_SUCCESS;
	else if (err == 0)
		result = DEVICE_STATUS_SUCCESS;
	else
		result = DEVICE_STATUS_ERROR;

	return result;
}

device_status_t DEVICEAPI DeviceEnableDebugTrapWithPollFd(uint32_t   NodeId,
							HSA_QUEUEID QueueId,
							HSAint32 *PollFd) //OUT
{
	device_status_t result;
	struct ioctl_dbg_trap_args argout = {0};

	if (QueueId != INVALID_QUEUEID)
		return DEVICE_STATUS_NOT_SUPPORTED;

	result =  debug_trap(NodeId,
				KFD_IOC_DBG_TRAP_ENABLE,
				1,
				QueueId,
				0,
				INVALID_PID,
				0,
				&argout);

	*PollFd = argout.data3;

	return result;
}

device_status_t DEVICEAPI DeviceEnableDebugTrap(uint32_t   NodeId,
					      HSA_QUEUEID QueueId)
{
	HSAint32 PollFd = 0;
	device_status_t status = DeviceEnableDebugTrapWithPollFd(NodeId,
							       QueueId,
							       &PollFd);

	if (status == DEVICE_STATUS_SUCCESS)
		close(PollFd);
	return status;
}



device_status_t DEVICEAPI DeviceDisableDebugTrap(uint32_t NodeId)
{
	return  debug_trap(NodeId,
			KFD_IOC_DBG_TRAP_ENABLE,
			0,
			0,
			0,
			INVALID_PID,
			0,
			NULL);
}

device_status_t DEVICEAPI DeviceSetWaveLaunchTrapOverride(
					uint32_t NodeId,
					HSA_DBG_TRAP_OVERRIDE TrapOverride,
					HSA_DBG_TRAP_MASK     TrapMask)
{
	if (TrapOverride >= HSA_DBG_TRAP_OVERRIDE_NUM)
		return DEVICE_STATUS_INVALID_PARAMETER;

	return debug_trap(NodeId,
				KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE,
				TrapOverride,
				TrapMask,
				0,
				INVALID_PID,
				0,
				NULL);
}

device_status_t DEVICEAPI DeviceSetWaveLaunchMode(
				uint32_t NodeId,
				HSA_DBG_WAVE_LAUNCH_MODE WaveLaunchMode)
{
	return debug_trap(NodeId,
				KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE,
				WaveLaunchMode,
				0,
				0,
				INVALID_PID,
				0,
				NULL);
}

/**
 *   Suspend the execution of a set of queues. A queue that is suspended
 *   allows the wave context save state to be inspected and modified. If a
 *   queue is already suspended it remains suspended. A suspended queue
 *   can be resumed by DeviceDbgQueueResume().
 *
 *   For each node that has a queue suspended, a sequentially consistent
 *   system scope release will be performed that synchronizes with a
 *   sequentially consistent system scope acquire performed by this
 *   call. This ensures any memory updates performed by the suspended
 *   queues are visible to the thread calling this operation.
 *
 *   Pid is the process that owns the queues that are to be supended or
 *   resumed. If the value is -1 then the Pid of the process calling
 *   DeviceQueueSuspend or DeviceQueueResume is used.
 *
 *   NumQueues is the number of queues that are being requested to
 *   suspend or resume.
 *
 *   Queues is a pointer to an array with NumQueues entries of
 *   HSA_QUEUEID. The queues in the list must be for queues the exist
 *   for Pid, and can be a mixture of queues for different nodes.
 *
 *   GracePeriod is the number of milliseconds  to wait after
 *   initialiating context save before forcing waves to context save. A
 *   value of 0 indicates no grace period. It is ignored by
 *   DeviceQueueResume.
 *
 *   Flags is a bit set of the values defined by HSA_DBG_NODE_CONTROL.
 *   Returns:
 *    - DEVICE_STATUS_SUCCESS if successful.
 *    - DEVICE_STATUS_INVALID_HANDLE if any QueueId is invalid for Pid.
 */

device_status_t
DEVICEAPI
DeviceQueueSuspend(
		uint32_t    Pid,         // IN
		uint32_t    NumQueues,   // IN
		HSA_QUEUEID *Queues,      // IN
		uint32_t    GracePeriod, // IN
		uint32_t    Flags)       // IN
{
	device_status_t result;
	uint32_t *queue_ids_ptr;


	queue_ids_ptr = convert_queue_ids(NumQueues, Queues);
	if (!queue_ids_ptr)
		return DEVICE_STATUS_NO_MEMORY;

	result = debug_trap(INVALID_NODEID,
			KFD_IOC_DBG_TRAP_NODE_SUSPEND,
			Flags,
			NumQueues,
			GracePeriod,
			Pid,
			(uint64_t)queue_ids_ptr,
			NULL);

	free(queue_ids_ptr);
	return result;
}
/**
 *   Resume the execution of a set of queues. If a queue is not
 *   suspended by DeviceDbgQueueSuspend() then it remains executing. Any
 *   changes to the wave state data will be used when the waves are
 *   restored. Changes to the control stack data will have no effect.
 *
 *   For each node that has a queue resumed, a sequentially consistent
 *   system scope release will be performed that synchronizes with a
 *   sequentially consistent system scope acquire performed by all
 *   queues being resumed. This ensures any memory updates performed by
 *   the thread calling this operation are visible to the resumed
 *   queues.
 *
 *   For each node that has a queue resumed, the instruction cache will
 *   be invalidated. This ensures any instruction code updates performed
 *   by the thread calling this operation are visible to the resumed
 *   queues.
 *
 *   Pid is the process that owns the queues that are to be supended or
 *   resumed. If the value is -1 then the Pid of the process calling
 *   DeviceQueueSuspend or DeviceQueueResume is used.
 *
 *   NumQueues is the number of queues that are being requested to
 *   suspend or resume.
 *
 *   Queues is a pointer to an array with NumQueues entries of
 *   HSA_QUEUEID. The queues in the list must be for queues the exist
 *   for Pid, and can be a mixture of queues for different nodes.
 *
 *   Flags is a bit set of the values defined by HSA_DBG_NODE_CONTROL.
 *   Returns:
 *    - DEVICE_STATUS_SUCCESS if successful
 *    - DEVICE_STATUS_INVALID_HANDLE if any QueueId is invalid.
 */

device_status_t
DEVICEAPI
DeviceQueueResume(
		uint32_t    Pid,         // IN
		uint32_t    NumQueues,   // IN
		HSA_QUEUEID *Queues,      // IN
		uint32_t    Flags)       // IN
{
	device_status_t result;
	uint32_t *queue_ids_ptr;


	queue_ids_ptr = convert_queue_ids(NumQueues, Queues);
	if (!queue_ids_ptr)
		return DEVICE_STATUS_NO_MEMORY;

	result = debug_trap(INVALID_NODEID,
			KFD_IOC_DBG_TRAP_NODE_RESUME,
			Flags,
			NumQueues,
			0,
			Pid,
			(uint64_t)queue_ids_ptr,
			NULL);
	free(queue_ids_ptr);
	return result;
}

device_status_t
DEVICEAPI
DeviceQueryDebugEvent(
		uint32_t		NodeId, //IN
		uint32_t		Pid, // IN
		uint32_t		*QueueId, // IN/OUT
		bool			ClearEvents, // IN
		DEBUG_EVENT_TYPE	*EventsReceived, // OUT
		bool			*IsSuspended, // OUT
		bool			*IsNew // OUT
		)
{
	device_status_t result;
	struct ioctl_dbg_trap_args argout = {0};
	uint32_t flags = 0;

	if (ClearEvents)
		flags |= KFD_DBG_EV_FLAG_CLEAR_STATUS;

	result = debug_trap(NodeId,
			    KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT,
			    *QueueId,
			    flags,
			    0,
			    Pid,
			    0,
			    &argout);

	if (result)
		return result;

	*QueueId = argout.data1;
	*EventsReceived = (DEBUG_EVENT_TYPE)(argout.data3 & (KFD_DBG_EV_STATUS_TRAP | KFD_DBG_EV_STATUS_VMFAULT));
	*IsSuspended = argout.data3 & KFD_DBG_EV_STATUS_SUSPENDED;
	*IsNew = argout.data3 & KFD_DBG_EV_STATUS_NEW_QUEUE;

	return result;
}

/**
 * Get the major and minor version of the kernel debugger support.
 *
 * Returns:
 *   - DEVICE_STATUS_SUCCESS if successful.
 *
 *   - DEVICE_STATUS_INVALID_HANDLE if NodeId is invalid.
 *
 *   - DEVICE_STATUS_NOT_SUPPORTED if debug trap not supported for NodeId.
*/
device_status_t
DEVICEAPI
DeviceGetKernelDebugTrapVersionInfo(
    uint32_t *Major,  //Out
    uint32_t *Minor   //Out
)
{
	device_status_t result;
	struct ioctl_dbg_trap_args argout = {0};

	result =  debug_trap(INVALID_NODEID,
				KFD_IOC_DBG_TRAP_GET_VERSION,
				0,
				0,
				0,
				INVALID_PID,
				0,
				&argout);

	*Major = argout.data1;
	*Minor = argout.data2;
	return result;
}

/**
 * Get the major and minor version of the Thunk debugger support.
*/
void
DEVICEAPI
DeviceGetThunkDebugTrapVersionInfo(
    uint32_t *Major,  //Out
    uint32_t *Minor   //Out
)
{
	*Major = KFD_IOCTL_DBG_MAJOR_VERSION;
	*Minor = KFD_IOCTL_DBG_MINOR_VERSION;
}

device_status_t
DEVICEAPI
DeviceGetQueueSnapshot(
		uint32_t	NodeId, //IN
		uint32_t	Pid, // IN
		bool		ClearEvents, //IN
		void		*SnapshotBuf, //IN
		uint32_t	*QssEntries //IN/OUT
		)
{
	device_status_t result;
	struct ioctl_dbg_trap_args argout = {0};
	uint32_t flags = 0;

	if (ClearEvents)
		flags |= KFD_DBG_EV_FLAG_CLEAR_STATUS;

	result = debug_trap(NodeId,
			    KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT,
			    flags,
			    *QssEntries,
			    0,
			    Pid,
			    (uint64_t)SnapshotBuf,
			    &argout);

	if (result)
		return result;

	*QssEntries = argout.data2;

	return DEVICE_STATUS_SUCCESS;
}

device_status_t DEVICEAPI DeviceSetAddressWatch(
		uint32_t		NodeId,
		uint32_t		Pid,
		HSA_DBG_WATCH_MODE	WatchMode,
		void			*WatchAddress,
		uint64_t		WatchAddrMask,
		uint32_t		*WatchId
		)
{

	device_status_t result;
	uint32_t     TruncatedWatchAddressMask;
	struct ioctl_dbg_trap_args argout = {0};

	/* Right now we only support 32 bit watch address masks, so we need
	 * to check that we aren't losing data when we truncate the mask
	 * to be passed to the kernel.
	 */
	if (WatchAddrMask > (uint64_t) UINT_MAX)
	{
		return DEVICE_STATUS_INVALID_PARAMETER;
	}
	TruncatedWatchAddressMask = (uint32_t) WatchAddrMask;

	if (WatchId == NULL)
		return DEVICE_STATUS_INVALID_PARAMETER;

	result = debug_trap(NodeId,
			KFD_IOC_DBG_TRAP_SET_ADDRESS_WATCH, // op
			*WatchId,
			WatchMode,
			TruncatedWatchAddressMask,
			Pid,
			(uint64_t) WatchAddress,
			&argout);
	*WatchId = argout.data1;

	return result;
}

device_status_t DEVICEAPI DeviceClearAddressWatch(
		uint32_t NodeId,
		uint32_t Pid,
		uint32_t WatchId
		)
{

	device_status_t result;

	result = debug_trap(NodeId,
			KFD_IOC_DBG_TRAP_CLEAR_ADDRESS_WATCH, // op
			WatchId,
			0,
			0,
			Pid,
			0,
			NULL);
	return result;
}
