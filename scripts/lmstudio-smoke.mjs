#!/usr/bin/env node
// Embeds one string through the built LM Studio embedder and prints the
// dimension, vector norm and latency. Run `pnpm build` first.
//
//   LMSTUDIO_MODEL=text-embedding-embeddinggemma-300m node scripts/lmstudio-smoke.mjs ["text"]
//
// LMSTUDIO_BASE_URL, LMSTUDIO_BATCH_SIZE, LMSTUDIO_TIMEOUT_MS etc. apply as in the server.

import { resolveEmbedBackend } from '../dist/cli/options.js'
import { LmStudioEmbedder } from '../dist/embedder/lmstudio-embedder.js'

const text = process.argv[2] ?? 'The quick brown fox jumps over the lazy dog.'

try {
  const { lmstudio, warnings } = resolveEmbedBackend({ ...process.env, EMBED_BACKEND: 'lmstudio' })
  for (const warning of warnings) {
    console.error(`Warning: ${warning}`)
  }
  const embedder = new LmStudioEmbedder(lmstudio)
  for (const warning of embedder.modelWarnings) {
    console.error(`Warning: ${warning}`)
  }

  console.log(`endpoint: ${lmstudio.baseUrl}/embeddings`)
  console.log(`model:    ${lmstudio.model}`)

  // The first call may include model load time, so time a second call too.
  for (const label of ['first call', 'second call']) {
    const started = performance.now()
    const vector = await embedder.embed(text, 'document')
    const latencyMs = performance.now() - started
    const norm = Math.hypot(...vector)
    console.log(
      `${label}: dims=${vector.length} norm=${norm.toFixed(6)} latency=${latencyMs.toFixed(0)}ms`
    )
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
}
