// compile_shaders.js — Compile GLSL compute shaders to SPIR-V using @webgpu/glslang (WASM)
// Usage: node compile_shaders.js [shader_names...]
// If no names given, compiles all .comp files without matching .spv

const fs = require('fs');
const path = require('path');

const SHADER_DIR = path.join(__dirname, 'src', 'gpu', 'shaders', 'vulkan');

async function main() {
    const glslang = await require('@webgpu/glslang')();

    let names = process.argv.slice(2);
    if (names.length === 0) {
        // Compile all .comp files
        names = fs.readdirSync(SHADER_DIR)
            .filter(f => f.endsWith('.comp'))
            .map(f => f.replace('.comp', ''));
    }

    for (const name of names) {
        const compPath = path.join(SHADER_DIR, `${name}.comp`);
        const spvPath = path.join(SHADER_DIR, `${name}.spv`);

        if (!fs.existsSync(compPath)) {
            console.error(`  SKIP ${name}.comp (not found)`);
            continue;
        }

        const glsl = fs.readFileSync(compPath, 'utf8');
        try {
            const spirv = glslang.compileGLSL(glsl, 'compute');
            // spirv is a Uint32Array — write raw bytes
            const buf = Buffer.from(spirv.buffer, spirv.byteOffset, spirv.byteLength);
            fs.writeFileSync(spvPath, buf);
            console.log(`  OK  ${name}.comp -> ${name}.spv (${buf.length} bytes)`);
        } catch (e) {
            console.error(`  ERR ${name}.comp: ${e.message || e}`);
            process.exitCode = 1;
        }
    }
}

main();
