# Punctuation Restoration Demo

Interactive web demo for the Tomoul XLM-RoBERTa punctuation restoration model.

## Features

- 🎯 **Real-time punctuation restoration** - Add proper punctuation to unpunctuated text
- ✨ **Capitalization** - Automatic sentence capitalization
- 💡 **Example inputs** - Pre-loaded examples to try
- 📊 **Statistics** - Track characters, words, and punctuation added
- 🎨 **Modern UI** - Clean, responsive interface

## Quick Start

```bash
cd examples/punctuation-demo
npm install
npm start
```

Then open **http://localhost:8001/** in your browser!

### System Requirements

- **RAM**: At least 8GB recommended (model uses ~3-4GB during inference)
- **Disk**: 2.2GB for model weights
- **CPU**: Multi-core recommended for better performance
- **Node.js**: v14.0.0 or higher

**Note:** The first startup will take a moment as it loads the large 2.1GB model into memory.

**Troubleshooting:**
If you get "Processing failed" errors, it may be due to insufficient memory. Consider:
1. Using the smaller "base" model variant (768 hidden dim instead of 1024)
2. Running on a system with more available RAM
3. Closing other memory-intensive applications

## How It Works

The demo provides a user interface for the punctuation restoration model. The actual model implementation:

- **Model**: XLM-RoBERTa Large (24 layers, 1024 hidden dimension)
- **Size**: 2.1GB weights file
- **Tokenizer**: SentencePiece (250k vocabulary)
- **Architecture**: Transformer-based token classification
- **Labels**: 6 punctuation classes (O, COMMA, PERIOD, QUESTION, PERIOD_U, COMMA_U)

### Native Library Integration

The punctuation model is built as native libraries for production use:

```
release/lib/
├── libtomoul_fullstop-punctuation-multilang-large_linux_x86_64.so
├── libtomoul_fullstop-punctuation-multilang-large_linux_aarch64.so
├── libtomoul_fullstop-punctuation-multilang-large_mac_x86_64.dylib
└── libtomoul_fullstop-punctuation-multilang-large_mac_aarch64.dylib
```

### C API Example

```c
#include "tomoul_fullstop-punctuation-multilang-large.h"

// Initialize model
if (tomoul_xlm_roberta_punctuation_init("weights.tl", "vocab.txt") != 0) {
    // Handle error
}

// Process text
char input[] = "hello world how are you";
char output[1024];
int len = tomoul_xlm_roberta_punctuation_process(
    input, strlen(input), output, sizeof(output)
);

// output now contains: "Hello world, how are you?"

// Cleanup
tomoul_xlm_roberta_punctuation_destroy();
```

## Example Inputs

Try these examples:

1. **Simple**: `hello world how are you doing today`
2. **Introduction**: `my name is john i live in new york city i work as a software engineer`
3. **Questions**: `what time is it can you tell me please i need to catch my train`
4. **Complex**: `this is a test however i am not sure if it works correctly we will find out soon`

## Model Details

- **Base Model**: oliverguhr/fullstop-punctuation-multilang-large
- **Languages**: Multilingual (English, German, French, Italian, and more)
- **Task**: Token classification for punctuation restoration
- **Framework**: Custom Zig implementation with zero dependencies

## Files

- `index.html` - Main demo interface
- `server.js` - Node.js server (API + static files)
- `package.json` - Node.js dependencies
- `README.md` - This file

## Architecture

The demo uses Node.js with FFI (Foreign Function Interface) to call the native C library:

```
Browser
  ↓ HTTP
Node.js Server (port 8001)
  ↓ FFI (ffi-napi)
C Library (.so/.dylib)
  ↓ Native code
Zig Implementation
  ↓ Inference
XLM-RoBERTa Model (2.1GB)
```

## License

Part of the Tomoul project. See the main repository for license details.
