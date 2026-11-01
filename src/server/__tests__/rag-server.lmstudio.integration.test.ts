// End-to-end through RAGServer with EMBED_BACKEND=lmstudio against a mock
// LM Studio. The parser, chunker and LanceDB store are real; only the embedding
// HTTP endpoint is mocked.

import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  type MockLmStudio,
  requestInputs,
  startMockLmStudio,
} from '../../__tests__/lmstudio-mock-server.js'
import { parseJson, privateMembers } from '../../__tests__/test-doubles.js'
import { embeddingSnapshotPath } from '../../embedder/embedding-snapshot.js'
import { LmStudioEmbedder } from '../../embedder/lmstudio-embedder.js'
import { RAGServer } from '../index.js'
import type { SyncStatusResult } from '../types.js'

const MODEL = 'text-embedding-embeddinggemma-300m'

const GUIDE = `# Garden Guide

## Tomatoes

Tomatoes need full sun and regular watering. Plant tomato seedlings after the last frost and stake them early so the vines stay off the ground.

## Compost

Compost turns kitchen scraps into soil. Mix green scraps such as vegetable peelings with brown material such as dry leaves, and turn the pile every week.
`

const NOTES = `# Sourdough Notes

Sourdough bread rises slowly from a wild yeast starter. Feed the starter daily with flour and water, and bake the loaf in a hot covered pot for a crisp crust.
`

describe('RAGServer with the LM Studio backend', () => {
  const dbPath = resolve('./tmp/test-lancedb-lmstudio')
  const dataDir = resolve('./tmp/test-data-lmstudio')
  const guidePath = resolve(dataDir, 'guide.md')
  const notesPath = resolve(dataDir, 'notes.md')
  let mock: MockLmStudio
  let server: RAGServer
  /** Inputs sent while ingesting, kept before later tests clear the request log. */
  let ingestInputs: string[] = []

  beforeAll(async () => {
    rmSync(dbPath, { recursive: true, force: true })
    rmSync(dataDir, { recursive: true, force: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(dataDir, { recursive: true })
    writeFileSync(guidePath, GUIDE)
    mock = await startMockLmStudio()
    server = new RAGServer({
      dbPath,
      // Would fail to load if the transformers path were taken.
      modelName: 'does-not-exist/never-downloaded',
      cacheDir: resolve('./tmp/test-lmstudio-no-model-cache'),
      baseDir: dataDir,
      maxFileSize: 100 * 1024 * 1024,
      lmstudio: {
        baseUrl: mock.baseUrl,
        model: MODEL,
        batchSize: 4,
        maxInputTokens: 2048,
        timeoutMs: 5000,
      },
    })
    await server.initialize()
  })

  afterAll(async () => {
    await server.close()
    await mock.close()
    rmSync(dbPath, { recursive: true, force: true })
    rmSync(dataDir, { recursive: true, force: true })
  })

  it('uses the LM Studio embedder, not the transformers one', () => {
    const { embedder } = privateMembers<{ embedder: object }>(server)

    expect(embedder).toBeInstanceOf(LmStudioEmbedder)
    expect(existsSync(resolve('./tmp/test-lmstudio-no-model-cache'))).toBe(false)
  })

  it('ingests a markdown file and finds it with query_documents', async () => {
    const ingested = await server.handleIngestFile({ filePath: guidePath })
    expect(
      parseJson<{ chunkCount: number }>(ingested.content[0]?.text ?? '').chunkCount
    ).toBeGreaterThan(0)

    ingestInputs = mock.requests.flatMap((request) => requestInputs(request))
    mock.requests.length = 0
    const result = await server.handleQueryDocuments({ query: 'compost leaves', limit: 3 })
    const hits = parseJson<{ filePath: string; text: string }[]>(result.content[0]?.text ?? '')

    expect(hits.length).toBeGreaterThan(0)
    expect(hits[0]?.filePath).toBe(guidePath)
    expect(hits[0]?.text).toContain('brown material')
    // The query went out with the EmbeddingGemma query prompt.
    expect(requestInputs(mock.requests.at(-1))).toEqual([
      'task: search result | query: compost leaves',
    ])
  })

  it('embeds stored chunks with the document prompt and writes no embedding snapshot', async () => {
    expect(ingestInputs.length).toBeGreaterThan(0)

    // Chunks carry the document prompt; sentence grouping embeds raw sentences.
    expect(ingestInputs.some((text) => text.startsWith('title: none | text: '))).toBe(true)
    expect(ingestInputs.some((text) => !text.startsWith('title: none | text: '))).toBe(true)
    expect(existsSync(embeddingSnapshotPath(dbPath, guidePath))).toBe(false)
  })

  it('reports the backend and model in status', async () => {
    const result = await server.handleStatus()
    const status = parseJson<{
      documentCount: number
      chunkCount: number
      embedder: { backend: string; model: string }
    }>(result.content[0]?.text ?? '')

    expect(status.documentCount).toBe(1)
    expect(status.chunkCount).toBeGreaterThan(0)
    expect(status.embedder).toEqual({ backend: 'lmstudio', model: MODEL })
  })

  it('lists the ingested file', async () => {
    const result = await server.handleListFiles()
    const listed = parseJson<{ files: { filePath: string; ingested: boolean }[] }>(
      result.content[0]?.text ?? ''
    )

    expect(listed.files.find((file) => file.filePath === guidePath)?.ingested).toBe(true)
  })

  it('syncs a new file in through sync_start and sync_status', async () => {
    writeFileSync(notesPath, NOTES)
    const started = await server.handleSyncStart({})
    const { jobId } = parseJson<{ jobId: string }>(started.content[0]?.text ?? '')

    let state: SyncStatusResult | undefined
    const deadline = Date.now() + 8000
    while (Date.now() < deadline) {
      const polled = await server.handleSyncStatus({ jobId })
      state = parseJson<SyncStatusResult>(polled.content[0]?.text ?? '')
      if (state.state !== 'running') {
        break
      }
      await new Promise((resolveTick) => setImmediate(resolveTick))
    }
    expect(state?.state).toBe('succeeded')

    const result = await server.handleQueryDocuments({ query: 'sourdough starter', limit: 3 })
    const hits = parseJson<{ filePath: string }[]>(result.content[0]?.text ?? '')
    expect(hits[0]?.filePath).toBe(notesPath)
  })

  it('deletes a file and stops returning it', async () => {
    await server.handleDeleteFile({ filePath: notesPath })

    const listed = parseJson<{ files: { filePath: string; ingested: boolean }[] }>(
      (await server.handleListFiles()).content[0]?.text ?? ''
    )
    expect(listed.files.find((file) => file.filePath === notesPath)?.ingested).toBe(false)

    const result = await server.handleQueryDocuments({ query: 'sourdough starter', limit: 3 })
    const hits = parseJson<{ filePath: string }[]>(result.content[0]?.text ?? '')
    expect(hits.every((hit) => hit.filePath !== notesPath)).toBe(true)
  })

  it('surfaces a clear error when LM Studio is down during a query', async () => {
    const down = await startMockLmStudio()
    const baseUrl = down.baseUrl
    await down.close()
    const offline = new RAGServer({
      dbPath,
      modelName: 'unused',
      cacheDir: './tmp/unused',
      baseDir: dataDir,
      maxFileSize: 1024,
      lmstudio: { baseUrl, model: MODEL, batchSize: 4, maxInputTokens: 2048, timeoutMs: 2000 },
    })
    await offline.initialize()

    await expect(offline.handleQueryDocuments({ query: 'compost' })).rejects.toThrow(
      `LM Studio is not running at ${baseUrl}`
    )
    await offline.close()
  })
})
