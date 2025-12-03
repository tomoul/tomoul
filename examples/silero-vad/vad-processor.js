/**
 * AudioWorklet Processor for VAD
 *
 * This runs in a separate audio thread and collects audio samples,
 * then sends them to the main thread for VAD processing.
 *
 * AudioWorklet is the modern replacement for ScriptProcessorNode
 * and doesn't suffer from garbage collection issues.
 */
class VADProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(0);
        this.chunkSize = 512;
        this.targetSampleRate = 16000;
        this.isActive = true;

        // Handle messages from main thread
        this.port.onmessage = (event) => {
            if (event.data.type === 'stop') {
                this.isActive = false;
            } else if (event.data.type === 'config') {
                this.chunkSize = event.data.chunkSize || 512;
                this.targetSampleRate = event.data.targetSampleRate || 16000;
            }
        };
    }

    /**
     * Downsample audio from native sample rate to target rate
     * Uses simple decimation - good enough for VAD
     */
    downsample(inputData, inputSampleRate, targetSampleRate) {
        if (targetSampleRate >= inputSampleRate) {
            return inputData;
        }

        const ratio = inputSampleRate / targetSampleRate;
        const newLength = Math.floor(inputData.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            result[i] = inputData[Math.floor(i * ratio)];
        }

        return result;
    }

    /**
     * Process audio - called by the audio worklet system
     * @param inputs - Array of inputs, each input is array of channels
     * @param outputs - Array of outputs (we don't use these)
     * @returns true to keep processor alive
     */
    process(inputs, outputs) {
        if (!this.isActive) {
            return false; // Stop the processor
        }

        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }

        const channelData = input[0]; // First channel (mono)
        if (!channelData || channelData.length === 0) {
            return true;
        }

        // Downsample to target rate
        // sampleRate is a global in AudioWorkletGlobalScope
        const downsampled = this.downsample(channelData, sampleRate, this.targetSampleRate);

        // Accumulate in buffer
        const newBuffer = new Float32Array(this.buffer.length + downsampled.length);
        newBuffer.set(this.buffer);
        newBuffer.set(downsampled, this.buffer.length);
        this.buffer = newBuffer;

        // Send complete chunks to main thread
        while (this.buffer.length >= this.chunkSize) {
            const chunk = this.buffer.slice(0, this.chunkSize);
            this.buffer = this.buffer.slice(this.chunkSize);

            // Send chunk to main thread for VAD processing
            this.port.postMessage({
                type: 'audioChunk',
                samples: chunk
            });
        }

        return true; // Keep processor alive
    }
}

registerProcessor('vad-processor', VADProcessor);
