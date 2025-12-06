// src/core/audio.zig
// Audio processing for Whisper: Mel spectrogram computation
//
// Implements the full audio preprocessing pipeline:
// 1. Audio loading (via libsndfile FFI or raw PCM)
// 2. STFT with Hann window
// 3. Mel filterbank projection
// 4. Log scaling
//
// Whisper parameters:
// - Sample rate: 16,000 Hz
// - N_FFT: 400 (window size)
// - Hop length: 160 (~10ms per frame)
// - N_MELS: 80 (or 128 for large-v3)
// - Chunk: 30 seconds = 480,000 samples = 3,000 frames

const std = @import("std");
const math = std.math;
const Tensor = @import("tensor.zig").Tensor;

// ============================================================================
// Constants
// ============================================================================

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS_DEFAULT: usize = 80;
pub const N_MELS_LARGE: usize = 128;
pub const CHUNK_LENGTH: usize = 30; // seconds
pub const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_LENGTH; // 480,000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 (Whisper uses N_SAMPLES / HOP_LENGTH)
pub const N_FFT_BINS: usize = N_FFT / 2 + 1; // 201

// ============================================================================
// Complex number operations
// ============================================================================

pub const Complex = struct {
    re: f32,
    im: f32,

    pub fn add(a: Complex, b: Complex) Complex {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }

    pub fn sub(a: Complex, b: Complex) Complex {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }

    pub fn mul(a: Complex, b: Complex) Complex {
        return .{
            .re = a.re * b.re - a.im * b.im,
            .im = a.re * b.im + a.im * b.re,
        };
    }

    pub fn magnitude(self: Complex) f32 {
        return @sqrt(self.re * self.re + self.im * self.im);
    }

    pub fn magnitudeSquared(self: Complex) f32 {
        return self.re * self.re + self.im * self.im;
    }

    pub fn fromPolar(r: f32, theta: f32) Complex {
        return .{ .re = r * @cos(theta), .im = r * @sin(theta) };
    }
};

// ============================================================================
// FFT Implementation (Cooley-Tukey radix-2 DIT)
// ============================================================================

/// Bit-reverse an index for FFT
fn bitReverse(x: usize, log2n: u6) usize {
    var result: usize = 0;
    var val = x;
    for (0..log2n) |_| {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

/// In-place Cooley-Tukey FFT (radix-2 decimation-in-time)
/// Input length must be a power of 2
pub fn fft(data: []Complex) void {
    const n = data.len;
    if (n <= 1) return;

    // Compute log2(n)
    const log2n: u6 = @intCast(math.log2_int(usize, n));

    // Bit-reversal permutation
    for (0..n) |i| {
        const j = bitReverse(i, log2n);
        if (i < j) {
            const tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    // Cooley-Tukey iterative FFT
    var size: usize = 2;
    while (size <= n) : (size *= 2) {
        const half_size = size / 2;
        const angle_step = -2.0 * math.pi / @as(f32, @floatFromInt(size));

        var k: usize = 0;
        while (k < n) : (k += size) {
            var w = Complex{ .re = 1.0, .im = 0.0 };
            const w_step = Complex.fromPolar(1.0, angle_step);

            for (0..half_size) |j| {
                const u = data[k + j];
                const t = w.mul(data[k + j + half_size]);
                data[k + j] = u.add(t);
                data[k + j + half_size] = u.sub(t);
                w = w.mul(w_step);
            }
        }
    }
}

// ============================================================================
// Optimized 400-point DFT using Bluestein's algorithm with cached tables
// ============================================================================

// For N=400, M = smallest power of 2 >= 2*400-1 = 799, so M = 1024
const BLUESTEIN_M: usize = 1024;

/// Cached Bluestein tables for N=400
const BluesteinCache = struct {
    chirp: []Complex, // [N_FFT] chirp factors
    b_fft: []Complex, // [BLUESTEIN_M] pre-computed FFT of b sequence
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) !BluesteinCache {
        var chirp = try allocator.alloc(Complex, N_FFT);
        errdefer allocator.free(chirp);

        var b_fft = try allocator.alloc(Complex, BLUESTEIN_M);
        errdefer allocator.free(b_fft);

        const n_f: f32 = @floatFromInt(N_FFT);

        // Compute chirp: W^(k²/2) where W = exp(-2πi/N)
        for (0..N_FFT) |k| {
            const k_f: f32 = @floatFromInt(k);
            const angle = -math.pi * k_f * k_f / n_f;
            chirp[k] = .{ .re = @cos(angle), .im = @sin(angle) };
        }

        // Compute b sequence and its FFT
        @memset(b_fft, Complex{ .re = 0, .im = 0 });
        b_fft[0] = chirp[0];
        for (1..N_FFT) |k| {
            b_fft[k] = chirp[k];
            b_fft[BLUESTEIN_M - k] = chirp[k];
        }

        // FFT of b (in-place)
        fft(b_fft);

        return .{
            .chirp = chirp,
            .b_fft = b_fft,
            .allocator = allocator,
        };
    }

    fn deinit(self: *BluesteinCache) void {
        self.allocator.free(self.chirp);
        self.allocator.free(self.b_fft);
    }
};

var bluestein_cache: ?BluesteinCache = null;

fn getBluesteinCache(allocator: std.mem.Allocator) !*BluesteinCache {
    if (bluestein_cache == null) {
        bluestein_cache = try BluesteinCache.init(allocator);
    }
    return &bluestein_cache.?;
}

/// Optimized 400-point DFT using pre-computed Bluestein tables
fn rfft400(allocator: std.mem.Allocator, input: []const f32) ![]Complex {
    const cache = try getBluesteinCache(allocator);

    // Allocate working buffer for convolution
    var a = try allocator.alloc(Complex, BLUESTEIN_M);
    defer allocator.free(a);

    // a[k] = x[k] * conj(chirp[k]) for k < N, else 0
    @memset(a, Complex{ .re = 0, .im = 0 });
    for (0..N_FFT) |k| {
        const x = if (k < input.len) input[k] else 0.0;
        const c = cache.chirp[k];
        a[k] = .{
            .re = x * c.re,
            .im = -x * c.im,
        };
    }

    // FFT of a
    fft(a);

    // Pointwise multiply with pre-computed FFT(b)
    for (0..BLUESTEIN_M) |k| {
        const ar = a[k].re;
        const ai = a[k].im;
        const br = cache.b_fft[k].re;
        const bi = cache.b_fft[k].im;
        a[k] = .{
            .re = ar * br - ai * bi,
            .im = ar * bi + ai * br,
        };
    }

    // IFFT (conjugate, FFT, conjugate, scale)
    for (a) |*v| {
        v.im = -v.im;
    }
    fft(a);
    const scale: f32 = 1.0 / @as(f32, @floatFromInt(BLUESTEIN_M));
    for (a) |*v| {
        v.re = v.re * scale;
        v.im = -v.im * scale;
    }

    // Extract result: X[k] = conj(chirp[k]) * conv[k]
    var result = try allocator.alloc(Complex, N_FFT_BINS);
    for (0..N_FFT_BINS) |k| {
        const conv_val = a[k];
        const c = cache.chirp[k];
        result[k] = .{
            .re = conv_val.re * c.re + conv_val.im * c.im,
            .im = conv_val.im * c.re - conv_val.re * c.im,
        };
    }

    return result;
}

/// DFT for real input - returns only positive frequencies (N/2 + 1 bins)
/// Uses optimized Bluestein for N=400, radix-2 FFT for power-of-2
pub fn rfft(allocator: std.mem.Allocator, input: []const f32, n_fft: usize) ![]Complex {
    // Special case for Whisper's N_FFT=400 - use optimized path
    if (n_fft == N_FFT) {
        return rfft400(allocator, input);
    }

    const n_bins = n_fft / 2 + 1;
    var result = try allocator.alloc(Complex, n_bins);
    errdefer allocator.free(result);

    // Check if power of 2 - use fast radix-2 FFT
    const is_power_of_2 = (n_fft & (n_fft - 1)) == 0;

    if (is_power_of_2) {
        // Use radix-2 FFT
        var complex_buf = try allocator.alloc(Complex, n_fft);
        defer allocator.free(complex_buf);

        for (0..n_fft) |i| {
            complex_buf[i] = .{
                .re = if (i < input.len) input[i] else 0.0,
                .im = 0.0,
            };
        }

        fft(complex_buf);
        @memcpy(result[0..n_bins], complex_buf[0..n_bins]);
    } else {
        // Fallback to naive DFT for other non-power-of-2 sizes
        const n_f: f32 = @floatFromInt(n_fft);
        for (0..n_bins) |k| {
            var sum_re: f32 = 0.0;
            var sum_im: f32 = 0.0;
            const k_f: f32 = @floatFromInt(k);
            for (0..n_fft) |n| {
                const x = if (n < input.len) input[n] else 0.0;
                const n_f2: f32 = @floatFromInt(n);
                const angle = -2.0 * math.pi * k_f * n_f2 / n_f;
                sum_re += x * @cos(angle);
                sum_im += x * @sin(angle);
            }
            result[k] = .{ .re = sum_re, .im = sum_im };
        }
    }

    return result;
}

// ============================================================================
// Hann Window
// ============================================================================

/// Generate Hann window coefficients (periodic, matching PyTorch)
/// PyTorch uses: 0.5 - 0.5 * cos(2 * pi * n / N) where N is window size
pub fn hannWindow(allocator: std.mem.Allocator, size: usize) ![]f32 {
    var window = try allocator.alloc(f32, size);
    const n: f32 = @floatFromInt(size);

    for (0..size) |i| {
        const x: f32 = @floatFromInt(i);
        // Periodic Hann window (matches torch.hann_window)
        window[i] = 0.5 - 0.5 * @cos(2.0 * math.pi * x / n);
    }

    return window;
}

// Pre-computed Hann window for N_FFT=400 (periodic, matching PyTorch)
var hann_400_storage: [N_FFT]f32 = undefined;
var hann_400_initialized: bool = false;

pub fn getHannWindow400() *const [N_FFT]f32 {
    if (!hann_400_initialized) {
        const n: f32 = @floatFromInt(N_FFT);
        for (0..N_FFT) |i| {
            const x: f32 = @floatFromInt(i);
            // Periodic Hann window (matches torch.hann_window)
            hann_400_storage[i] = 0.5 - 0.5 * @cos(2.0 * math.pi * x / n);
        }
        hann_400_initialized = true;
    }
    return &hann_400_storage;
}

// ============================================================================
// STFT (Short-Time Fourier Transform)
// ============================================================================

/// Compute STFT magnitude spectrum
/// Returns: [n_frames, n_fft_bins] power spectrum
pub fn stft(
    allocator: std.mem.Allocator,
    audio: []const f32,
    n_fft: usize,
    hop_length: usize,
) !Tensor {
    const n_samples = audio.len;
    const n_frames = (n_samples - n_fft) / hop_length + 1;
    const n_bins = n_fft / 2 + 1;

    // Allocate output
    var shape = [_]usize{ n_frames, n_bins };
    var output = try Tensor.init(allocator, &shape);
    errdefer output.deinit();

    // Get Hann window
    var hann: []f32 = undefined;
    var hann_allocated = false;
    if (n_fft == N_FFT) {
        hann = @constCast(getHannWindow400()[0..N_FFT]);
    } else {
        hann = try hannWindow(allocator, n_fft);
        hann_allocated = true;
    }
    defer if (hann_allocated) allocator.free(hann);

    // Allocate windowed frame buffer
    var windowed = try allocator.alloc(f32, n_fft);
    defer allocator.free(windowed);

    // Process each frame
    for (0..n_frames) |frame| {
        const start = frame * hop_length;

        // Apply window
        for (0..n_fft) |i| {
            windowed[i] = audio[start + i] * hann[i];
        }

        // Compute FFT
        var fft_result = try rfft(allocator, windowed, n_fft);
        defer allocator.free(fft_result);

        // Store power spectrum (magnitude squared)
        const row_start = frame * n_bins;
        for (0..n_bins) |i| {
            output.data[row_start + i] = fft_result[i].magnitudeSquared();
        }
    }

    return output;
}

// ============================================================================
// Mel Filterbank
// ============================================================================

/// Convert frequency in Hz to mel scale
pub fn hzToMel(hz: f32) f32 {
    return 2595.0 * math.log10(1.0 + hz / 700.0);
}

/// Convert mel scale to frequency in Hz
pub fn melToHz(mel: f32) f32 {
    return 700.0 * (math.pow(f32, 10.0, mel / 2595.0) - 1.0);
}

/// Generate mel filterbank matrix
/// Returns: [n_mels, n_fft_bins] filterbank
pub fn melFilterbank(
    allocator: std.mem.Allocator,
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
) !Tensor {
    const n_bins = n_fft / 2 + 1;
    const sr_f: f32 = @floatFromInt(sample_rate);
    const nyquist = sr_f / 2.0;

    // Mel range
    const mel_min = hzToMel(0.0);
    const mel_max = hzToMel(nyquist);

    // Create n_mels + 2 equally spaced points in mel space
    var mel_points = try allocator.alloc(f32, n_mels + 2);
    defer allocator.free(mel_points);

    for (0..n_mels + 2) |i| {
        const ratio: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_mels + 1));
        mel_points[i] = mel_min + ratio * (mel_max - mel_min);
    }

    // Convert to Hz and then to FFT bin indices
    var bin_indices = try allocator.alloc(usize, n_mels + 2);
    defer allocator.free(bin_indices);

    const n_fft_f: f32 = @floatFromInt(n_fft);
    for (0..n_mels + 2) |i| {
        const hz = melToHz(mel_points[i]);
        bin_indices[i] = @intFromFloat(@floor((n_fft_f + 1.0) * hz / sr_f));
    }

    // Create filterbank matrix
    var shape = [_]usize{ n_mels, n_bins };
    var filterbank = try Tensor.init(allocator, &shape);
    @memset(filterbank.data, 0.0);

    // Build triangular filters
    for (0..n_mels) |m| {
        const left = bin_indices[m];
        const center = bin_indices[m + 1];
        const right = bin_indices[m + 2];

        // Rising edge
        if (center > left) {
            for (left..center) |k| {
                const weight = @as(f32, @floatFromInt(k - left)) / @as(f32, @floatFromInt(center - left));
                filterbank.data[m * n_bins + k] = weight;
            }
        }

        // Falling edge
        if (right > center) {
            for (center..right) |k| {
                const weight = @as(f32, @floatFromInt(right - k)) / @as(f32, @floatFromInt(right - center));
                filterbank.data[m * n_bins + k] = weight;
            }
        }
    }

    return filterbank;
}

// ============================================================================
// Mel Spectrogram
// ============================================================================

/// Compute log-mel spectrogram from audio samples
/// Returns: [n_mels, n_frames] tensor
pub fn melSpectrogram(
    allocator: std.mem.Allocator,
    audio: []const f32,
    n_mels: usize,
) !Tensor {
    // Compute STFT power spectrum
    var power_spec = try stft(allocator, audio, N_FFT, HOP_LENGTH);
    defer power_spec.deinit();

    const n_frames = power_spec.shape[0];
    const n_bins = power_spec.shape[1];

    // Get or compute mel filterbank
    var filterbank = try melFilterbank(allocator, n_mels, N_FFT, SAMPLE_RATE);
    defer filterbank.deinit();

    // Apply mel filterbank: mel_spec = power_spec @ filterbank.T
    // power_spec: [n_frames, n_bins]
    // filterbank: [n_mels, n_bins]
    // result: [n_frames, n_mels] -> transpose to [n_mels, n_frames]
    var shape = [_]usize{ n_mels, n_frames };
    var mel_spec = try Tensor.init(allocator, &shape);
    errdefer mel_spec.deinit();

    // Matrix multiply: for each frame, dot product with each mel filter
    for (0..n_frames) |frame| {
        for (0..n_mels) |mel| {
            var sum: f32 = 0.0;
            for (0..n_bins) |bin| {
                sum += power_spec.data[frame * n_bins + bin] * filterbank.data[mel * n_bins + bin];
            }
            mel_spec.data[mel * n_frames + frame] = sum;
        }
    }

    // Log scaling (Whisper uses log10 with clipping)
    const log_offset: f32 = 1e-10;
    const max_log: f32 = 0.0;
    const min_log: f32 = -8.0; // ~80 dB dynamic range

    for (mel_spec.data) |*val| {
        var log_val = math.log10(val.* + log_offset);
        log_val = @max(log_val, min_log);
        log_val = @min(log_val, max_log);
        // Normalize to [-1, 1] range (approximately)
        val.* = (log_val - min_log) / (max_log - min_log) * 2.0 - 1.0;
    }

    return mel_spec;
}

/// Load pre-computed Whisper mel filterbank from file
/// Returns: [n_mels, n_fft_bins] filterbank matching librosa/Slaney normalization
pub fn loadWhisperMelFilterbank(allocator: std.mem.Allocator, n_mels: usize) !Tensor {
    const n_bins = N_FFT_BINS; // 201

    // Load from binary file
    const filename = if (n_mels == 80) "models/mel_filterbank_80.bin" else "models/mel_filterbank_128.bin";

    const file = std.fs.cwd().openFile(filename, .{}) catch |err| {
        std.debug.print("Warning: Could not load {s}: {}, using generated filterbank\n", .{ filename, err });
        return melFilterbank(allocator, n_mels, N_FFT, SAMPLE_RATE);
    };
    defer file.close();

    const expected_size = n_mels * n_bins * @sizeOf(f32);
    const data = try allocator.alloc(u8, expected_size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != expected_size) {
        std.debug.print("Warning: Filterbank file size mismatch, using generated filterbank\n", .{});
        return melFilterbank(allocator, n_mels, N_FFT, SAMPLE_RATE);
    }

    // Create tensor and copy data
    var shape = [_]usize{ n_mels, n_bins };
    const filterbank = try Tensor.init(allocator, &shape);

    const float_ptr: [*]const f32 = @ptrCast(@alignCast(data.ptr));
    @memcpy(filterbank.data, float_ptr[0 .. n_mels * n_bins]);

    return filterbank;
}

/// Compute mel spectrogram with Whisper's exact normalization
/// This matches OpenAI Whisper's log_mel_spectrogram function
pub fn whisperMelSpectrogram(
    allocator: std.mem.Allocator,
    audio: []const f32,
    n_mels: usize,
) !Tensor {
    // PyTorch STFT center padding: n_fft // 2 on each side with REFLECT padding
    const stft_pad = N_FFT / 2; // 200 samples

    // First, create the 30-second audio buffer
    const audio_30s = try allocator.alloc(f32, N_SAMPLES);
    defer allocator.free(audio_30s);
    @memset(audio_30s, 0.0);
    const audio_to_copy = @min(audio.len, N_SAMPLES);
    @memcpy(audio_30s[0..audio_to_copy], audio[0..audio_to_copy]);

    // Now apply REFLECT padding (PyTorch's default for STFT center=True)
    // Reflect padding: pad[i] = audio[stft_pad - 1 - i] for left side
    //                  pad[i] = audio[N - 2 - i] for right side
    const padded_audio_len = N_SAMPLES + 2 * stft_pad; // 480400 samples
    const padded_audio = try allocator.alloc(f32, padded_audio_len);
    defer allocator.free(padded_audio);

    // Left reflect padding: indices go stft_pad-1, stft_pad-2, ..., 0 reflected
    // Reflect means audio[1], audio[2], ..., audio[stft_pad] (in reverse order)
    for (0..stft_pad) |i| {
        padded_audio[i] = audio_30s[stft_pad - i];
    }

    // Copy main audio
    @memcpy(padded_audio[stft_pad .. stft_pad + N_SAMPLES], audio_30s);

    // Right reflect padding
    for (0..stft_pad) |i| {
        padded_audio[stft_pad + N_SAMPLES + i] = audio_30s[N_SAMPLES - 2 - i];
    }

    // Compute STFT on the full padded buffer
    var power_spec = try stft(allocator, padded_audio, N_FFT, HOP_LENGTH);
    defer power_spec.deinit();

    const n_frames_raw = power_spec.shape[0];
    const n_bins = power_spec.shape[1];

    // We expect 3001 frames, drop the last to get exactly 3000
    const n_frames = N_FRAMES; // Always output exactly 3000 frames
    if (n_frames_raw < n_frames) {
        std.debug.print("Warning: STFT produced {d} frames, expected >= {d}\n", .{ n_frames_raw, n_frames });
    }

    // Load Whisper's pre-computed mel filterbank (Slaney normalized)
    var filterbank = try loadWhisperMelFilterbank(allocator, n_mels);
    defer filterbank.deinit();

    // Apply filterbank: mel_spec = filterbank @ power_spec.T
    // filterbank: [n_mels, n_bins], power_spec: [n_frames, n_bins]
    // result: [n_mels, n_frames]
    var shape = [_]usize{ n_mels, n_frames };
    var mel_spec = try Tensor.init(allocator, &shape);
    errdefer mel_spec.deinit();

    for (0..n_frames) |frame| {
        for (0..n_mels) |mel| {
            var sum: f32 = 0.0;
            for (0..n_bins) |bin| {
                sum += power_spec.data[frame * n_bins + bin] * filterbank.data[mel * n_bins + bin];
            }
            mel_spec.data[mel * n_frames + frame] = sum;
        }
    }

    // Whisper's log scaling: log10(max(mel, 1e-10))
    for (mel_spec.data) |*val| {
        val.* = math.log10(@max(val.*, 1e-10));
    }

    // Find max for normalization
    var max_val: f32 = mel_spec.data[0];
    for (mel_spec.data) |val| {
        max_val = @max(max_val, val);
    }

    // Whisper normalization: clamp to (max - 8.0), then (x + 4.0) / 4.0
    for (mel_spec.data) |*val| {
        val.* = @max(val.*, max_val - 8.0); // Clamp to 80dB below max
        val.* = (val.* + 4.0) / 4.0; // Normalize to roughly [-1, 1]
    }

    return mel_spec;
}

// ============================================================================
// Audio File Loading (Pure Zig WAV loader)
// ============================================================================

/// Audio file info
pub const AudioInfo = struct {
    sample_rate: usize,
    channels: usize,
    frames: usize,
};

/// Audio load result
pub const AudioLoadResult = struct {
    samples: []f32,
    info: AudioInfo,
};

/// WAV file header structures (extern for byte-level layout)
const WavHeader = extern struct {
    riff_magic: [4]u8, // "RIFF"
    file_size: u32, // File size - 8
    wave_magic: [4]u8, // "WAVE"
};

const WavChunkHeader = extern struct {
    chunk_id: [4]u8,
    chunk_size: u32,
};

const WavFmtChunk = extern struct {
    audio_format: u16, // 1 = PCM, 3 = IEEE float
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
};

/// Read u16 from bytes (little-endian)
fn readU16(data: []const u8) u16 {
    return std.mem.readInt(u16, data[0..2], .little);
}

/// Read u32 from bytes (little-endian)
fn readU32(data: []const u8) u32 {
    return std.mem.readInt(u32, data[0..4], .little);
}

/// Load WAV file and convert to 16kHz mono f32
pub fn loadWavFile(allocator: std.mem.Allocator, path: []const u8) !AudioLoadResult {
    // Null-terminate path for std.fs
    const path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z, path);

    const file = try std.fs.openFileAbsoluteZ(path_z.ptr, .{});
    defer file.close();

    // Read entire file into memory for simplicity
    const file_size = try file.getEndPos();
    const file_data = try allocator.alloc(u8, file_size);
    defer allocator.free(file_data);
    _ = try file.readAll(file_data);

    // Parse RIFF header (12 bytes)
    if (file_data.len < 12) {
        return error.InvalidWavFile;
    }

    // Check "RIFF" magic
    if (!std.mem.eql(u8, file_data[0..4], "RIFF")) {
        return error.InvalidWavFile;
    }
    // Check "WAVE" magic
    if (!std.mem.eql(u8, file_data[8..12], "WAVE")) {
        return error.InvalidWavFile;
    }

    // Parse chunks (byte-by-byte to avoid alignment issues)
    var fmt_chunk: ?WavFmtChunk = null;
    var data_start: usize = 0;
    var data_size: u32 = 0;
    var pos: usize = 12; // After RIFF header

    while (pos + 8 <= file_data.len) {
        const chunk_id = file_data[pos..][0..4];
        const chunk_size = readU32(file_data[pos + 4 ..]);
        pos += 8;

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (pos + 16 > file_data.len) break;
            fmt_chunk = WavFmtChunk{
                .audio_format = readU16(file_data[pos..]),
                .num_channels = readU16(file_data[pos + 2 ..]),
                .sample_rate = readU32(file_data[pos + 4 ..]),
                .byte_rate = readU32(file_data[pos + 8 ..]),
                .block_align = readU16(file_data[pos + 12 ..]),
                .bits_per_sample = readU16(file_data[pos + 14 ..]),
            };
            pos += chunk_size;
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            data_start = pos;
            data_size = chunk_size;
            break;
        } else {
            pos += chunk_size;
        }
    }

    const fmt = fmt_chunk orelse return error.MissingFmtChunk;

    // Validate format
    if (fmt.audio_format != 1 and fmt.audio_format != 3) {
        return error.UnsupportedAudioFormat;
    }

    const bytes_per_sample = fmt.bits_per_sample / 8;
    const frame_size = fmt.num_channels * bytes_per_sample;
    const frames: usize = data_size / frame_size;
    const channels: usize = @intCast(fmt.num_channels);
    const sample_rate: usize = @intCast(fmt.sample_rate);

    // Allocate and convert samples
    const samples = try allocator.alloc(f32, frames * channels);
    defer allocator.free(samples);

    const audio_data = file_data[data_start..][0..data_size];

    if (fmt.audio_format == 3 and fmt.bits_per_sample == 32) {
        // IEEE float 32-bit
        const float_ptr: [*]const f32 = @ptrCast(@alignCast(audio_data.ptr));
        @memcpy(samples, float_ptr[0..samples.len]);
    } else if (fmt.audio_format == 1 and fmt.bits_per_sample == 16) {
        // PCM 16-bit
        for (0..frames * channels) |i| {
            const offset = i * 2;
            const sample_i16 = std.mem.readInt(i16, audio_data[offset..][0..2], .little);
            samples[i] = @as(f32, @floatFromInt(sample_i16)) / 32768.0;
        }
    } else if (fmt.audio_format == 1 and fmt.bits_per_sample == 24) {
        // PCM 24-bit
        for (0..frames * channels) |i| {
            const offset = i * 3;
            const b = audio_data[offset..][0..3];
            const sample_i32: i32 = @as(i32, b[0]) |
                (@as(i32, b[1]) << 8) |
                (@as(i32, @as(i8, @bitCast(b[2]))) << 16);
            samples[i] = @as(f32, @floatFromInt(sample_i32)) / 8388608.0;
        }
    } else if (fmt.audio_format == 1 and fmt.bits_per_sample == 32) {
        // PCM 32-bit
        for (0..frames * channels) |i| {
            const offset = i * 4;
            const sample_i32 = std.mem.readInt(i32, audio_data[offset..][0..4], .little);
            samples[i] = @as(f32, @floatFromInt(sample_i32)) / 2147483648.0;
        }
    } else {
        return error.UnsupportedBitDepth;
    }

    // Convert to mono
    var mono: []f32 = undefined;
    if (channels == 1) {
        mono = try allocator.alloc(f32, frames);
        @memcpy(mono, samples[0..frames]);
    } else {
        mono = try allocator.alloc(f32, frames);
        for (0..frames) |i| {
            var sum: f32 = 0.0;
            for (0..channels) |ch| {
                sum += samples[i * channels + ch];
            }
            mono[i] = sum / @as(f32, @floatFromInt(channels));
        }
    }
    errdefer allocator.free(mono);

    // Resample to 16kHz if needed
    if (sample_rate != SAMPLE_RATE) {
        const resampled = try resample(allocator, mono, sample_rate, SAMPLE_RATE);
        allocator.free(mono);
        mono = resampled;
    }

    return AudioLoadResult{
        .samples = mono,
        .info = AudioInfo{
            .sample_rate = SAMPLE_RATE,
            .channels = 1,
            .frames = mono.len,
        },
    };
}

/// Load audio file (auto-detect format)
/// Currently supports: WAV
pub fn loadAudioFile(allocator: std.mem.Allocator, path: []const u8) !AudioLoadResult {
    // Check file extension
    if (std.mem.endsWith(u8, path, ".wav") or std.mem.endsWith(u8, path, ".WAV")) {
        return loadWavFile(allocator, path);
    }

    // Default to WAV for now
    return loadWavFile(allocator, path);
}

/// Simple linear interpolation resampler
fn resample(allocator: std.mem.Allocator, input: []const f32, in_rate: usize, out_rate: usize) ![]f32 {
    const in_len = input.len;
    const out_len = (in_len * out_rate + in_rate - 1) / in_rate;

    var output = try allocator.alloc(f32, out_len);
    errdefer allocator.free(output);

    const ratio = @as(f64, @floatFromInt(in_rate)) / @as(f64, @floatFromInt(out_rate));

    for (0..out_len) |i| {
        const src_pos = @as(f64, @floatFromInt(i)) * ratio;
        const src_idx: usize = @intFromFloat(src_pos);
        const frac: f32 = @floatCast(src_pos - @as(f64, @floatFromInt(src_idx)));

        if (src_idx + 1 < in_len) {
            output[i] = input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac;
        } else if (src_idx < in_len) {
            output[i] = input[src_idx];
        } else {
            output[i] = 0.0;
        }
    }

    return output;
}

// ============================================================================
// Tests
// ============================================================================

test "complex operations" {
    const a = Complex{ .re = 1.0, .im = 2.0 };
    const b = Complex{ .re = 3.0, .im = 4.0 };

    const sum = a.add(b);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), sum.re, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), sum.im, 0.001);

    const prod = a.mul(b);
    try std.testing.expectApproxEqAbs(@as(f32, -5.0), prod.re, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), prod.im, 0.001);
}

test "bit reverse" {
    try std.testing.expectEqual(@as(usize, 0), bitReverse(0, 3));
    try std.testing.expectEqual(@as(usize, 4), bitReverse(1, 3));
    try std.testing.expectEqual(@as(usize, 2), bitReverse(2, 3));
    try std.testing.expectEqual(@as(usize, 6), bitReverse(3, 3));
}

test "fft basic" {
    var data = [_]Complex{
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 2.0, .im = 0.0 },
        .{ .re = 3.0, .im = 0.0 },
        .{ .re = 4.0, .im = 0.0 },
    };

    fft(&data);

    // DC component should be sum of inputs
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), data[0].re, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0].im, 0.001);
}

test "hann window" {
    const allocator = std.testing.allocator;
    const window = try hannWindow(allocator, 4);
    defer allocator.free(window);

    // Hann window should be 0 at edges, 1 at center
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), window[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), window[3], 0.001);
}

test "hz to mel conversion" {
    // 1000 Hz should be about 1000 mels (by design of mel scale)
    const mel_1000 = hzToMel(1000.0);
    try std.testing.expect(mel_1000 > 900.0 and mel_1000 < 1100.0);

    // Round trip
    const hz_back = melToHz(mel_1000);
    try std.testing.expectApproxEqAbs(@as(f32, 1000.0), hz_back, 1.0);
}

test "mel filterbank shape" {
    const allocator = std.testing.allocator;
    var fb = try melFilterbank(allocator, 80, 400, 16000);
    defer fb.deinit();

    try std.testing.expectEqual(@as(usize, 80), fb.shape[0]);
    try std.testing.expectEqual(@as(usize, 201), fb.shape[1]);
}
