#!/usr/bin/env node
/**
 * bench-llamacpp.mjs
 * OpenAI 兼容端点流式测速：TTFT（首字延迟）/ decode 速度（tok/s）
 *
 * 用法:
 *   node bench-llamacpp.mjs --base http://<SERVER_IP>:8080/v1 --model qwen3.6-vision --text "你好" --max-tokens 256
 *   node bench-llamacpp.mjs --base ... --image /path/to/pic.png --text "描述这张图"
 *   node bench-llamacpp.mjs --base ... --text-file long.md --max-tokens 128
 */

import fs from 'node:fs';

function arg(name, def = null) {
  const i = process.argv.indexOf('--' + name);
  if (i === -1) return def;
  const v = process.argv[i + 1];
  return v && !v.startsWith('--') ? v : true;
}
const num = (name, def) => {
  const v = arg(name);
  return v === null || v === true ? def : Number(v);
};

const base = String(arg('base', 'http://127.0.0.1:8080/v1')).replace(/\/+$/, '');
const model = String(arg('model', 'qwen3.6-vision'));
const maxTokens = num('max-tokens', 256);
const apiKey = String(arg('key', 'none'));
const timeoutMs = num('timeout', 600000);

let prompt = arg('text', '用一句话介绍你自己。');
if (prompt === true) prompt = '用一句话介绍你自己。';
const textFile = arg('text-file');
if (textFile && typeof textFile === 'string') prompt = fs.readFileSync(textFile, 'utf8');

const imagePath = arg('image');
let content;
if (imagePath && typeof imagePath === 'string') {
  const b64 = fs.readFileSync(imagePath).toString('base64');
  const mime = imagePath.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
  content = [
    { type: 'text', text: prompt },
    { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
  ];
} else {
  content = [{ type: 'text', text: prompt }];
}

const ac = new AbortController();
const timer = setTimeout(() => ac.abort(), timeoutMs);

const t0 = performance.now();
let tFirstChunk = null;
let tFirstToken = null;
let tDone = null;
let usage = null;
let chunkCount = 0;
let charCount = 0;
let first = '';

try {
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    signal: ac.signal,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content }],
      max_tokens: maxTokens,
      temperature: 0,
      stream: true,
      stream_options: { include_usage: true },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error(JSON.stringify({ ok: false, status: res.status, error: err }, null, 2));
    process.exit(1);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const now = performance.now();
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';

    for (const line of lines) {
      const s = line.trim();
      if (!s.startsWith('data:')) continue;
      const payload = s.slice(5).trim();
      if (!payload || payload === '[DONE]') continue;
      let obj;
      try { obj = JSON.parse(payload); } catch { continue; }

      chunkCount++;
      if (tFirstChunk === null) tFirstChunk = now;

      const delta = obj.choices?.[0]?.delta;
      const piece = delta?.content ?? obj.choices?.[0]?.text ?? '';
      if (piece) {
        if (tFirstToken === null) tFirstToken = now;
        charCount += piece.length;
        first += piece;
      }
      if (obj.usage) usage = obj.usage;
    }
  }
  tDone = performance.now();
} catch (e) {
  clearTimeout(timer);
  console.error(JSON.stringify({ ok: false, error: String(e?.message || e) }, null, 2));
  process.exit(1);
} finally {
  clearTimeout(timer);
}

const genTokens = usage?.completion_tokens ?? null;
const promptTokens = usage?.prompt_tokens ?? null;
const ttftMs = tFirstToken !== null ? +(tFirstToken - t0).toFixed(1) : null;
const totalMs = tDone !== null ? +(tDone - t0).toFixed(1) : null;
const decodeMs = tFirstToken !== null && tDone !== null ? +(tDone - tFirstToken).toFixed(1) : null;

const report = {
  ok: true,
  endpoint: `${base}/chat/completions`,
  model,
  hasImage: !!(imagePath && typeof imagePath === 'string'),
  promptChars: prompt.length,
  promptTokens,
  generationTokens: genTokens,
  chunks: chunkCount,
  outputChars: charCount,
  ttftMs,
  totalMs,
  decodeMs,
  decodeTokensPerSec: genTokens && decodeMs ? +(genTokens / (decodeMs / 1000)).toFixed(2) : null,
  overallTokensPerSec: genTokens && totalMs ? +(genTokens / (totalMs / 1000)).toFixed(2) : null,
  rawUsage: usage ?? null,
  sampleOutput: first.slice(0, 300),
};

console.log(JSON.stringify(report, null, 2));
