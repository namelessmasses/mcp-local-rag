// Opt-in check against a real LM Studio server: LMSTUDIO_LIVE=1 with
// LMSTUDIO_MODEL set (and LMSTUDIO_BASE_URL unless it is the default).
// Skipped otherwise, so the suite never needs LM Studio.

import { describe, expect, it } from 'vitest'
import { resolveEmbedBackend } from '../../cli/options.js'
import { LmStudioEmbedder } from '../lmstudio-embedder.js'

const live = process.env['LMSTUDIO_LIVE'] === '1'

describe.skipIf(!live)('LM Studio live', () => {
  const resolved = resolveEmbedBackend({ ...process.env, EMBED_BACKEND: 'lmstudio' })
  const settings = resolved.lmstudio

  it('embeds queries and documents into consistent unit vectors', async () => {
    if (settings === undefined) {
      throw new Error('LMSTUDIO_MODEL must be set for the live test')
    }
    const embedder = new LmStudioEmbedder(settings)

    const query = await embedder.embed('How do I water tomatoes?', 'query')
    const documents = await embedder.embedBatch(
      ['Tomatoes need regular watering.', 'Sourdough needs a starter.'],
      'document'
    )

    expect(query.length).toBeGreaterThan(0)
    for (const vector of [query, ...documents]) {
      expect(vector).toHaveLength(query.length)
      expect(Math.hypot(...vector)).toBeCloseTo(1, 4)
    }
    const dot = (a: number[], b: number[]): number =>
      a.reduce((sum, v, i) => sum + v * (b[i] ?? 0), 0)
    expect(dot(query, documents[0] ?? [])).toBeGreaterThan(dot(query, documents[1] ?? []))
  }, 180_000)
})
