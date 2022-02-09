/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA Foundation for a copy.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include "inc/pps.h"
#include "inc/pps_executable.h"
#include "inc/runtime.h"

// #define GridDimX 1024*1024
#define GridDimX 1
//#define BlockDimX 512
#define BlockDimX 64

// Code object filename.
#define FILENAME "vector_copy.hsaco"

// Argument alignment.
#define HSA_ARGUMENT_ALIGN_BYTES 16

// Alignment attribute.
#if defined(_MSC_VER)
  #define ALIGNED_(x) __declspec(align(x))
#else
  #if defined(__GNUC__)
    #define ALIGNED_(x) __attribute__ ((aligned(x)))
  #endif // __GNUC__
#endif // _MSC_VER

namespace hip_impl {
    void hip_throw(const std::exception& ex) {
       #if defined(__cpp_exceptions)
          throw ex;
       #else
          std::cerr << ex.what() << std::endl;
          std::terminate();
       #endif
    }
}
/*
hsa_executable_symbol_t find_kernel_by_name(hsa_executable_t executable, const char* kname) {
      using namespace hip_impl;

      pair<const char*, hsa_executable_symbol_t> r{kname, {}};

      hsa_executable_iterate_agent_symbols(
          executable, this_agent(),
          [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t x, void* s) {
              auto p = static_cast<pair<const char*, hsa_executable_symbol_t>*>(s);

              if (type(x) != HSA_SYMBOL_KIND_KERNEL) {
                  return HSA_STATUS_SUCCESS;
              }
              if (name(x) != p->first) return HSA_STATUS_SUCCESS;

              p->second = x;

              return HSA_STATUS_INFO_BREAK;
          },
          &r);

      return r.second;
  }
*/


// Finds GPU device.
hsa_status_t FindGpuDevice(hsa_agent_t agent, void *data) {
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(
    agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type
  );

  if (hsa_error_code != HSA_STATUS_SUCCESS) {
    return hsa_error_code;
  }

  if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
    bool is_apu_node;
    hsa_error_code = hsa_agent_get_info(
        agent, HSA_AGENT_INFO_IS_APU_NODE, &is_apu_node
    );
    if (hsa_error_code != HSA_STATUS_SUCCESS) {
        return hsa_error_code;
    }
    if (is_apu_node == true) {
        *((hsa_agent_t*)data) = agent;
    }
  }

  return HSA_STATUS_SUCCESS;
}


/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
        hsa_region_t* ret = (hsa_region_t*) data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}


/*
 * Determines if a memory region can be used for device
 * allocations.  On APU this is host memory, on a dGPU
 * this is the coarse-grained non-system memory:
 */
static hsa_status_t get_device_memory_region(hsa_region_t region, void* data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
        return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if ((flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) ||
        (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED))
    {
        if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED)
           printf ( "found FINE GRAINED device region, flags=%x\n", flags);
        else
           printf ( "found COURSE GRAINED device region, flags=%x\n", flags);
        hsa_region_t* ret = (hsa_region_t*) data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

int main() {
  // Initialize hsa runtime.
  hcs::runtime_init();
  hsa_status_t hsa_status = hsa_init();
  assert(HSA_STATUS_SUCCESS == hsa_status);

  //std::string file(FILENAME);
  // Open file.
  std::ifstream file(FILENAME, std::ios::in | std::ios::binary);
  assert(file.is_open() && file.good());
  // Find out file size.
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  // Allocate memory for raw code object.
  void *raw_code_object = malloc(size);
  assert(raw_code_object);

  // Read file contents.
  file.read((char*)raw_code_object, size);

  // Close file.
  file.close();
  std::string image {reinterpret_cast<const char*>(raw_code_object), size};

/*
  // Deserialize code object.
  hsa_code_object_t code_object = {0};
  hsa_status = hsa_code_object_deserialize(raw_code_object, size, NULL, &code_object);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  assert(0 != code_object.handle);

  // Free raw code object memory.
  free(raw_code_object);
*/

 /*
  static const auto cor_deleter = [](hsa_code_object_reader_t* p) {
      if (!p) return;
      hsa_code_object_reader_destroy(*p);
      delete p;
  };

  using RAII_code_reader =
      std::unique_ptr<hsa_code_object_reader_t, decltype(cor_deleter)>;

  RAII_code_reader code_reader{new hsa_code_object_reader_t, cor_deleter};
  hsa_code_object_reader_create_from_memory(
         file.data(), file.size(), code_reader.get());

*/






  // Find GPU device.
  hsa_agent_t device = {0};
  hsa_status = hsa_iterate_agents(FindGpuDevice, &device);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  assert(0 != device.handle);

  // Print out device name.
  char device_name[64] = {0};
  hsa_status = hsa_agent_get_info(device, HSA_AGENT_INFO_NAME, device_name);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  std::cout << "Using <" << device_name << ">" << std::endl;

  // Get queue size.
  uint32_t queue_size = 0;
  hsa_status = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // Create command queue.
  hsa_queue_t* commandQueue;
  hsa_status = hsa_queue_create(device, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                                0, 0, &commandQueue);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // Create executable.
  hsa_executable_t hsaExecutable;
  // hsa_status = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &hsaExecutable);
  hsa_status = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &hsaExecutable);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  hsaExecutable = hip_impl::load_executable(image, hsaExecutable, device);
/*
  // Load code object.
  hsa_status = hsa_executable_load_agent_code_object(hsaExecutable, device, *code_reader, NULL, NULL);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // Freeze executable.
  hsa_status = hsa_executable_freeze(hsaExecutable, NULL);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  */

  // Get symbol handle.
  hsa_executable_symbol_t kernelSymbol;
  // hsa_status = hsa_executable_get_symbol(hsaExecutable, NULL, "vector_copy", device, 0, &kernelSymbol);
  hsa_status = hsa_executable_get_symbol_by_name(hsaExecutable, "vector_copy.kd", &device, &kernelSymbol);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // Get code handle.
  uint64_t codeHandle;
  hsa_status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &codeHandle);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  uint64_t kernel_ctrl;
  hsa_status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CTRL, &kernel_ctrl);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  uint64_t kernel_mode;
  hsa_status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_MODE, &kernel_mode);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  //===--------------------------------------------------------------------===//

  // Get a signal.
  hsa_signal_t signal;
  hsa_status = hsa_signal_create(1, 0, NULL, &signal);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // Setup dispatch packet.
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  // Setup dispatch size and fences.
  const int kNumDimension = 1;
  aql.completion_signal = signal;
  aql.setup = kNumDimension << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  aql.workgroup_size_x = (uint16_t) BlockDimX;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = (uint32_t) GridDimX;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  aql.group_segment_size = 0;
  aql.private_segment_size = 0;
  aql.kernel_ctrl = kernel_ctrl;
  aql.kernel_mode = kernel_mode;

  hsa_region_t kernarg_region;
  kernarg_region.handle=(uint64_t)-1;
  hsa_agent_iterate_regions(device, get_kernarg_memory_region, &kernarg_region);

  hsa_region_t device_region;  // a region of memory accessible from the device:
  device_region.handle=(uint64_t)-1;
  hsa_agent_iterate_regions(device, get_device_memory_region, &device_region);

  // Setup kernel arguments.
  uint32_t* in = (uint32_t*)calloc(GridDimX *BlockDimX, sizeof(uint32_t));
  uint32_t* out = (uint32_t*)calloc(GridDimX * BlockDimX, sizeof(uint32_t));

  size_t sizeBytes = GridDimX * BlockDimX * sizeof(uint32_t);
  /*
  uint32_t* in_d, *out_d;
  hsa_status = hsa_memory_allocate(device_region, sizeBytes, (void**)&in_d);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  hsa_status = hsa_memory_allocate(device_region, sizeBytes, (void**)&out_d);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  */

  // Init host memory:
  for (uint32_t i = 0; i < GridDimX * BlockDimX; ++i) {
    in[i] = i;
    out[i] = 0xff;
  }

/*
  hsa_status = hsa_memory_copy(in_d, in, sizeBytes);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  */


  struct ALIGNED_(HSA_ARGUMENT_ALIGN_BYTES) args_t {
    void* arg0;
    void* arg1;
  } args;

  // args.arg0 = in_d;
  // args.arg1 = out_d;
  args.arg0 = in;
  args.arg1 = out;

  /*
   * Allocate the kernel argument buffer from the correct region.
   */
  /*
  void* kernarg_address = NULL;
  // hsa_status = hsa_memory_allocate(kernarg_region, sizeof(args), &kernarg_address);
  hsa_status = hsa_memory_allocate(device_region, sizeof(args), &kernarg_address);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  */



  // Copy args from CPU stack to the kernarg region where they can be accessed by the GPU:
  // memcpy(kernarg_address, &args, sizeof(args));
  // hsa_status = hsa_memory_copy(kernarg_address, &args, sizeof(args));


  // Bind kernel arguments and kernel code.
  aql.kernel_object = codeHandle;
  // aql.kernarg_address = kernarg_address;
  aql.kernarg_address = &args;



  const uint32_t queueSize = commandQueue->size;
  const uint32_t queueMask = queueSize - 1;

  // Write to command queue.
  // for(int i=0; i<1000; i++)
  {
    uint64_t index = hsa_queue_load_write_index_relaxed(commandQueue);
    ((hsa_kernel_dispatch_packet_t*)(commandQueue
                                         ->base_address))[index & queueMask] =
        aql;
    hsa_queue_store_write_index_relaxed(commandQueue, index + 1);

    // Ringdoor bell.
    hsa_signal_store_relaxed(commandQueue->doorbell_signal, index);

    if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                HSA_WAIT_STATE_ACTIVE) != 0) {
      printf("Signal wait returned unexpected value\n");
      exit(0);
    }

    hsa_signal_store_relaxed(signal, 1);
  }


  // Copy back to host:
  /*
  hsa_status = hsa_memory_copy(out, out_d, sizeBytes);
  assert(HSA_STATUS_SUCCESS == hsa_status);
  */

  // Validate.
  bool valid = true;
  int failIndex = 0;
  for (int i = 0; i < GridDimX *BlockDimX ; i++) {
    printf("%2x:%2x, ", in[i], out[i]);
    if ((i % 16) == 0) printf("\n");
    if (in[i] != out[i]) {
      failIndex = i;
      valid = false;
      break;
    }
  }
  if (valid) {
    printf("passed validation\n");
  } else {
    printf("VALIDATION FAILED!\nBad index: %d, ref(in):%d, computed(out):%d\n", failIndex, in[failIndex], out[failIndex]);
  }

  // Cleanup.
  hsa_status = hsa_signal_destroy(signal);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  hsa_status = hsa_executable_destroy(hsaExecutable);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  // hsa_status = hsa_code_object_destroy(code_object);
  // assert(HSA_STATUS_SUCCESS == hsa_status);

  hsa_status = hsa_queue_destroy(commandQueue);
  assert(HSA_STATUS_SUCCESS == hsa_status);

  hsa_status = hsa_shut_down();
  assert(HSA_STATUS_SUCCESS == hsa_status);

  free(in);
  free(out);

  return EXIT_SUCCESS;
}
