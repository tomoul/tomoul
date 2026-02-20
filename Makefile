# Makefile for zblas benchmarking in tomoul

.PHONY: bench test clean

# Benchmark zblas performance in tomoul
bench-zblas:
	cd /home/oolurin/Projects/Tomoul/tomoul && \
	zig build -Dmodel=whisper-tiny -Doptimize=ReleaseFast -Dzblas=true 2>&1 && \
	echo "=== zblas (8x8 kernel + Phase 8 packing) ===" && \
	for i in 1 2 3; do \
		./zig-out/bin/tomoul_whisper-tiny transcribe models/english_man.wav 2>&1 | grep -E "Encoder|Decoder|Total"; \
	done

# Alias for bench
test: bench

# Clean build artifacts
clean:
	cd /home/oolurin/Projects/Tomoul/tomoul && rm -rf zig-out zig-cache