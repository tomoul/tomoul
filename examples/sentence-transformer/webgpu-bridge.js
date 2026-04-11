/**
 * WebGPU Bridge for Tomoul WASM
 *
 * Implements the extern "webgpu" functions that the Zig WASM module imports.
 * Provides GPU buffer management, pipeline compilation, and compute dispatch
 * using the browser's WebGPU API.
 *
 * Usage:
 *   const bridge = new TomoulWebGpuBridge();
 *   await bridge.init();
 *   const wasm = await bridge.loadWasm('tomoul_sentence_transformer.wasm');
 *   // Now WASM can call WebGPU functions through the bridge
 *
 * Async Design:
 *   - Buffer creation, upload, pipeline creation are sync from WASM perspective
 *   - Dispatch recording is sync (command encoder recording)
 *   - Submit queues work but doesn't wait (sync from WASM)
 *   - Readback is async — the bridge's embed() wrapper handles this
 */

export class TomoulWebGpuBridge {
    constructor() {
        /** @type {GPUDevice|null} */
        this.device = null;
        /** @type {GPUQueue|null} */
        this.queue = null;
        /** @type {GPUAdapter|null} */
        this.adapter = null;

        // Handle registries (u32 ID → object)
        /** @type {Map<number, {buffer: GPUBuffer, size: number, usage: number}>} */
        this.buffers = new Map();
        /** @type {Map<number, {pipeline: GPUComputePipeline, bindGroupLayout: GPUBindGroupLayout, numStorage: number, hasUniform: boolean}>} */
        this.pipelines = new Map();
        this.nextBufferHandle = 1;
        this.nextPipelineHandle = 1;

        // Command recording state
        /** @type {GPUCommandEncoder|null} */
        this.commandEncoder = null;
        /** @type {GPUComputePassEncoder|null} */
        this.computePass = null;
        this.currentBindings = new Map();  // slot → buffer handle
        this.currentUniform = null;        // {data: Uint8Array}

        // Pending readback state
        this.pendingReadbacks = [];

        // WASM memory reference (set after loading)
        /** @type {WebAssembly.Memory|null} */
        this.wasmMemory = null;

        // Device info
        this.deviceName = 'WebGPU Device';
    }

    /**
     * Initialize WebGPU device and adapter.
     * @returns {Promise<boolean>}
     */
    async init() {
        if (!navigator.gpu) {
            console.error('WebGPU not supported in this browser');
            return false;
        }

        this.adapter = await navigator.gpu.requestAdapter();

        if (!this.adapter) {
            console.error('No WebGPU adapter found');
            return false;
        }

        // Request device with maximum buffer size and workgroup storage
        const requiredLimits = {};
        const adapterLimits = this.adapter.limits;
        requiredLimits.maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
        requiredLimits.maxBufferSize = adapterLimits.maxBufferSize;
        requiredLimits.maxComputeWorkgroupsPerDimension = adapterLimits.maxComputeWorkgroupsPerDimension;
        requiredLimits.maxComputeWorkgroupStorageSize = adapterLimits.maxComputeWorkgroupStorageSize;

        this.device = await this.adapter.requestDevice({
            requiredLimits,
        });

        this.queue = this.device.queue;
        this.deviceName = this.adapter.info?.device || this.adapter.info?.description || 'WebGPU Device';

        this.device.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
        });

        return true;
    }

    /**
     * Get the WASM import object containing extern "webgpu" functions.
     * @returns {object} Import object for WebAssembly.instantiate
     */
    getImportObject() {
        const bridge = this;

        return {
            webgpu: {
                wgpuInit: () => {
                    return bridge.device ? 1 : 0;
                },

                wgpuCreateBuffer: (size, writable) => {
                    if (!bridge.device) return 0xFFFFFFFF;

                    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
                    if (writable) {
                        usage |= GPUBufferUsage.COPY_SRC;
                    }

                    try {
                        const buffer = bridge.device.createBuffer({ size, usage });
                        const handle = bridge.nextBufferHandle++;
                        bridge.buffers.set(handle, { buffer, size, usage });
                        return handle;
                    } catch (e) {
                        console.error('createBuffer failed:', e);
                        return 0xFFFFFFFF;
                    }
                },

                wgpuUploadToBuffer: (handle, ptr, size) => {
                    const entry = bridge.buffers.get(handle);
                    if (!entry) return;

                    const data = new Uint8Array(bridge.wasmMemory.buffer, ptr, size);
                    bridge.queue.writeBuffer(entry.buffer, 0, data);
                },

                wgpuReadbackFromBuffer: (handle, ptr, size) => {
                    // Queue readback request — fulfilled after submit completes
                    bridge.pendingReadbacks.push({ handle, ptr, size });
                },

                wgpuDestroyBuffer: (handle) => {
                    const entry = bridge.buffers.get(handle);
                    if (entry) {
                        entry.buffer.destroy();
                        bridge.buffers.delete(handle);
                    }
                },

                wgpuCreatePipeline: (wgslPtr, wgslLen, entryPtr, entryLen, numStorage, hasUniform) => {
                    if (!bridge.device) return 0xFFFFFFFF;

                    try {
                        const wgsl = new TextDecoder().decode(
                            new Uint8Array(bridge.wasmMemory.buffer, wgslPtr, wgslLen)
                        );
                        const entryPoint = new TextDecoder().decode(
                            new Uint8Array(bridge.wasmMemory.buffer, entryPtr, entryLen)
                        );

                        const shaderModule = bridge.device.createShaderModule({ code: wgsl });

                        const pipeline = bridge.device.createComputePipeline({
                            layout: 'auto',
                            compute: {
                                module: shaderModule,
                                entryPoint: entryPoint,
                            },
                        });

                        const bindGroupLayout = pipeline.getBindGroupLayout(0);
                        const pipeHandle = bridge.nextPipelineHandle++;
                        bridge.pipelines.set(pipeHandle, {
                            pipeline,
                            bindGroupLayout,
                            numStorage,
                            hasUniform: hasUniform !== 0,
                        });
                        return pipeHandle;
                    } catch (e) {
                        console.error('createPipeline failed:', e);
                        return 0xFFFFFFFF;
                    }
                },

                wgpuDestroyPipeline: (handle) => {
                    bridge.pipelines.delete(handle);
                },

                wgpuBeginCommandBuffer: () => {
                    bridge.commandEncoder = bridge.device.createCommandEncoder();
                    bridge.currentBindings.clear();
                    bridge.currentUniform = null;
                    bridge.pendingReadbacks = [];
                },

                wgpuSetBinding: (slot, handle) => {
                    bridge.currentBindings.set(slot, handle);
                },

                wgpuSetUniform: (ptr, size) => {
                    // Copy uniform data from WASM memory
                    const data = new Uint8Array(size);
                    data.set(new Uint8Array(bridge.wasmMemory.buffer, ptr, size));
                    bridge.currentUniform = data;
                },

                wgpuDispatch: (pipelineHandle, groupsX, groupsY, groupsZ) => {
                    const pipeEntry = bridge.pipelines.get(pipelineHandle);
                    if (!pipeEntry || !bridge.commandEncoder) return;

                    // Build bind group entries
                    const entries = [];
                    const sortedSlots = Array.from(bridge.currentBindings.keys()).sort((a, b) => a - b);

                    for (const slot of sortedSlots) {
                        const bufHandle = bridge.currentBindings.get(slot);
                        const bufEntry = bridge.buffers.get(bufHandle);
                        if (bufEntry) {
                            entries.push({
                                binding: slot,
                                resource: { buffer: bufEntry.buffer },
                            });
                        }
                    }

                    // Add uniform buffer if needed
                    if (pipeEntry.hasUniform && bridge.currentUniform) {
                        const uniformSlot = sortedSlots.length > 0 ? Math.max(...sortedSlots) + 1 : 0;

                        // Create/reuse uniform buffer
                        const uniformSize = Math.max(bridge.currentUniform.byteLength, 16);
                        // Align to 16 bytes (WebGPU requirement for uniform buffers)
                        const alignedSize = Math.ceil(uniformSize / 16) * 16;
                        const uniformBuffer = bridge.device.createBuffer({
                            size: alignedSize,
                            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                        });
                        bridge.queue.writeBuffer(uniformBuffer, 0, bridge.currentUniform);

                        entries.push({
                            binding: uniformSlot,
                            resource: { buffer: uniformBuffer },
                        });
                    }

                    const bindGroup = bridge.device.createBindGroup({
                        layout: pipeEntry.bindGroupLayout,
                        entries,
                    });

                    // Record compute pass
                    const pass = bridge.commandEncoder.beginComputePass();
                    pass.setPipeline(pipeEntry.pipeline);
                    pass.setBindGroup(0, bindGroup);
                    pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
                    pass.end();

                    // Clear per-dispatch state
                    bridge.currentBindings.clear();
                    bridge.currentUniform = null;
                },

                wgpuSubmit: () => {
                    if (!bridge.commandEncoder) return;

                    // Create staging buffers for readbacks
                    for (const rb of bridge.pendingReadbacks) {
                        const bufEntry = bridge.buffers.get(rb.handle);
                        if (bufEntry) {
                            const stagingBuffer = bridge.device.createBuffer({
                                size: rb.size,
                                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                            });
                            bridge.commandEncoder.copyBufferToBuffer(
                                bufEntry.buffer, 0, stagingBuffer, 0, rb.size
                            );
                            rb.stagingBuffer = stagingBuffer;
                        }
                    }

                    const commandBuffer = bridge.commandEncoder.finish();
                    bridge.queue.submit([commandBuffer]);
                    bridge.commandEncoder = null;
                },

                wgpuGetDeviceName: (ptr, maxLen) => {
                    const name = bridge.deviceName;
                    const encoded = new TextEncoder().encode(name);
                    const len = Math.min(encoded.length, maxLen);
                    new Uint8Array(bridge.wasmMemory.buffer, ptr, len).set(encoded.subarray(0, len));
                    return len;
                },
            },
        };
    }

    /**
     * Fulfill pending readbacks after GPU work completes.
     * Must be called after WASM submit returns and before results are used.
     * @returns {Promise<void>}
     */
    async fulfillReadbacks() {
        for (const rb of this.pendingReadbacks) {
            if (!rb.stagingBuffer) continue;

            await rb.stagingBuffer.mapAsync(GPUMapMode.READ);
            const mapped = new Uint8Array(rb.stagingBuffer.getMappedRange());
            const dest = new Uint8Array(this.wasmMemory.buffer, rb.ptr, rb.size);
            dest.set(mapped);
            rb.stagingBuffer.unmap();
            rb.stagingBuffer.destroy();
        }
        this.pendingReadbacks = [];
    }

    /**
     * Load a Tomoul WASM module with WebGPU bridge.
     * @param {string|URL} wasmUrl - URL to the .wasm file
     * @returns {Promise<WebAssembly.Instance>}
     */
    async loadWasm(wasmUrl) {
        const importObject = this.getImportObject();

        // Add stub "env" imports if needed by the WASM module
        importObject.env = importObject.env || {};

        const response = await fetch(wasmUrl);
        const bytes = await response.arrayBuffer();
        const { instance } = await WebAssembly.instantiate(bytes, importObject);

        // Capture WASM memory for buffer transfers
        this.wasmMemory = instance.exports.memory;

        return instance;
    }

    /**
     * High-level: embed text using the WASM model with WebGPU acceleration.
     *
     * @param {WebAssembly.Instance} instance - WASM instance from loadWasm()
     * @param {string} text - Text to embed
     * @returns {Promise<Float32Array>} - 384-dimensional embedding vector
     */
    async embed(instance, text) {
        const exports = instance.exports;
        const memory = exports.memory;

        // Write text to input buffer
        const inputPtr = exports.get_input_buffer_ptr();
        const encoded = new TextEncoder().encode(text);
        const maxInput = exports.get_max_input_bytes();
        const textLen = Math.min(encoded.length, maxInput);
        new Uint8Array(memory.buffer, inputPtr, textLen).set(encoded.subarray(0, textLen));

        // Run embedding (triggers GPU dispatches via bridge)
        const dim = exports.embed(textLen);
        if (dim === 0) {
            throw new Error('Embedding failed');
        }

        // Fulfill GPU readbacks
        await this.fulfillReadbacks();

        // Read output
        const outputPtr = exports.get_output_buffer_ptr();
        return new Float32Array(memory.buffer, outputPtr, dim).slice();
    }
}
