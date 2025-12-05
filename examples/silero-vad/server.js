#!/usr/bin/env node
/**
 * Silero VAD WebAssembly Demo Server
 *
 * Serves the example demo and zig-out/ directory for WebAssembly development.
 * Adds proper CORS and MIME type headers for Wasm files.
 *
 * Usage:
 *   node examples/silero-vad/server.js [port]
 *   npm start (from examples/silero-vad/)
 *
 * Default port is 8000.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// MIME types
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// Find project root (examples/silero-vad/ -> examples/ -> project root)
const scriptDir = __dirname;
const projectRoot = path.resolve(scriptDir, '..', '..');

class TomoulServer {
  constructor(port = 8000) {
    this.port = port;
    this.projectRoot = projectRoot;
  }

  /**
   * Get MIME type from file extension
   */
  getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    return MIME_TYPES[ext] || 'application/octet-stream';
  }

  /**
   * Resolve file path with fallback logic
   */
  resolveFilePath(requestPath) {
    // Remove query string and decode URI
    const cleanPath = decodeURIComponent(requestPath.split('?')[0]);

    // Try direct path from project root
    let filePath = path.join(this.projectRoot, cleanPath);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      return filePath;
    }

    // Try with examples/silero-vad/ prefix
    filePath = path.join(this.projectRoot, 'examples', 'silero-vad', cleanPath);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      return filePath;
    }

    // If it's a directory, try index.html
    const dirPath = path.join(this.projectRoot, 'examples', 'silero-vad', cleanPath);
    if (fs.existsSync(dirPath) && fs.statSync(dirPath).isDirectory()) {
      const indexPath = path.join(dirPath, 'index.html');
      if (fs.existsSync(indexPath)) {
        return indexPath;
      }
    }

    // Default index.html for root
    if (cleanPath === '/' || cleanPath === '') {
      const indexPath = path.join(this.projectRoot, 'examples', 'silero-vad', 'index.html');
      if (fs.existsSync(indexPath)) {
        return indexPath;
      }
    }

    return null;
  }

  /**
   * Add CORS and security headers
   */
  addHeaders(res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', '*');
    // Required for SharedArrayBuffer (if needed later)
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  }

  /**
   * Handle incoming requests
   */
  handleRequest(req, res) {
    const parsedUrl = url.parse(req.url);
    const requestPath = parsedUrl.pathname;

    // Handle OPTIONS for CORS preflight
    if (req.method === 'OPTIONS') {
      this.addHeaders(res);
      res.writeHead(200);
      res.end();
      return;
    }

    // Only handle GET requests
    if (req.method !== 'GET') {
      res.writeHead(405, { 'Content-Type': 'text/plain' });
      res.end('Method Not Allowed');
      return;
    }

    // Resolve file path
    const filePath = this.resolveFilePath(requestPath);

    if (!filePath) {
      console.log(`404: ${requestPath}`);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('404 Not Found');
      return;
    }

    // Read and serve file
    fs.readFile(filePath, (err, data) => {
      if (err) {
        console.error(`Error reading ${filePath}:`, err);
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('500 Internal Server Error');
        return;
      }

      const mimeType = this.getMimeType(filePath);
      this.addHeaders(res);
      res.writeHead(200, { 'Content-Type': mimeType });
      res.end(data);

      console.log(`200: ${requestPath} → ${path.relative(this.projectRoot, filePath)}`);
    });
  }

  /**
   * Start the server
   */
  start() {
    const server = http.createServer((req, res) => this.handleRequest(req, res));

    server.listen(this.port, () => {
      console.log(`Silero VAD Demo Server
  → http://localhost:${this.port}
  → Serving: examples/silero-vad/
  → Press Ctrl+C to stop`);
    });

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      console.log('\nServer stopped.');
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      console.log('\nServer stopped.');
      process.exit(0);
    });
  }
}

// Main entry point
if (require.main === module) {
  const port = parseInt(process.argv[2]) || 8000;
  const server = new TomoulServer(port);
  server.start();
}

module.exports = TomoulServer;
