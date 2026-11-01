import { createServer } from 'node:net'
import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import {
  type MockLmStudio,
  mockVector,
  requestInputs,
  startMockLmStudio,
} from '../../__tests__/lmstudio-mock-server.js'
import { expectError } from '../../__tests__/test-doubles.js'
import { EmbeddingError } from '../index.js'
import { LmStudioEmbedder, type LmStudioEmbedderConfig } from '../lmstudio-embedder.js'

const QWEN = 'text-embedding-qwen3-embedding-0.6b'
const GEMMA = 'text-embedding-embeddinggemma-300m'
const QWEN_QUERY_PROMPT =
  'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

let mock: MockLmStudio

beforeAll(async () => {
  mock = await startMockLmStudio()
})

afterAll(async () => {
  await mock.close()
})

beforeEach(() => {
  mock.requests.length = 0
  mock.setHandler(undefined)
})

function create(overrides: Partial<LmStudioEmbedderConfig> = {}): LmStudioEmbedder {
  return new LmStudioEmbedder({
    baseUrl: mock.baseUrl,
    model: GEMMA,
    batchSize: 16,
    maxInputTokens: 2048,
    timeoutMs: 5000,
    ...overrides,
  })
}

function norm(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
}

/** The `input` array of the nth request. */
function bodyInput(index: number): string[] {
  return requestInputs(mock.requests[index])
}

describe('LmStudioEmbedder requests', () => {
  it('posts the model and an input array to /v1/embeddings', async () => {
    await create().embedBatch(['alpha', 'beta'])

    expect(mock.requests).toHaveLength(1)
    expect(mock.requests[0]?.method).toBe('POST')
    expect(mock.requests[0]?.url).toBe('/v1/embeddings')
    expect(mock.requests[0]?.body).toEqual({ model: GEMMA, input: ['alpha', 'beta'] })
  })

  it('embeds a single text through the same endpoint', async () => {
    const vector = await create().embed('alpha')

    expect(bodyInput(0)).toEqual(['alpha'])
    expect(vector).toHaveLength(32)
  })

  it('tolerates a trailing slash in the base URL', async () => {
    await create({ baseUrl: `${mock.baseUrl}/` }).embed('alpha')

    expect(mock.requests[0]?.url).toBe('/v1/embeddings')
  })

  it('splits inputs into requests of at most batchSize, keeping order', async () => {
    const texts = ['one', 'two', 'three', 'four', 'five']

    const vectors = await create({ batchSize: 2 }).embedBatch(texts)

    expect(mock.requests.map((_request, i) => bodyInput(i).length)).toEqual([2, 2, 1])
    expect(vectors).toHaveLength(5)
    for (const [i, text] of texts.entries()) {
      const expected = mockVector(text)
      const scale = norm(expected)
      expect(vectors[i]?.[0]).toBeCloseTo((expected[0] ?? 0) / scale, 6)
    }
  })

  it('returns [] for no texts without calling LM Studio', async () => {
    expect(await create().embedBatch([])).toEqual([])
    expect(mock.requests).toHaveLength(0)
  })

  it('rejects empty text like the stock embedder', async () => {
    const error = expectError(
      await create()
        .embedBatch(['ok', ''])
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toBe('Cannot generate embedding for empty text')
    expect(mock.requests).toHaveLength(0)
  })
})

describe('LmStudioEmbedder prompts', () => {
  it('prefixes Qwen3 queries with the instruction and leaves documents raw', async () => {
    const embedder = create({ model: QWEN })

    await embedder.embed('how?', 'query')
    await embedder.embedBatch(['doc'], 'document')
    await embedder.embedBatch(['plain'])

    expect(bodyInput(0)).toEqual([`${QWEN_QUERY_PROMPT}how?`])
    expect(bodyInput(1)).toEqual(['doc'])
    expect(bodyInput(2)).toEqual(['plain'])
    expect(embedder.modelWarnings).toEqual([])
    expect(await embedder.getDocumentPrompt()).toBe('')
  })

  it('applies EmbeddingGemma query and document templates', async () => {
    const embedder = create({ model: GEMMA })

    await embedder.embed('how?', 'query')
    await embedder.embedBatch(['doc'], 'document')
    await embedder.embedBatch(['plain'])

    expect(bodyInput(0)).toEqual(['task: search result | query: how?'])
    expect(bodyInput(1)).toEqual(['title: none | text: doc'])
    expect(bodyInput(2)).toEqual(['plain'])
    expect(embedder.modelWarnings).toEqual([])
    expect(await embedder.getDocumentPrompt()).toBe('title: none | text: ')
  })

  it('embeds other models raw and warns that no prompt is configured', async () => {
    const embedder = create({ model: 'nomic-embed-text-v1.5' })

    await embedder.embed('how?', 'query')
    await embedder.embedBatch(['doc'], 'document')

    expect(bodyInput(0)).toEqual(['how?'])
    expect(bodyInput(1)).toEqual(['doc'])
    expect(embedder.modelWarnings).toHaveLength(1)
    expect(embedder.modelWarnings[0]).toContain('No embedding prompt is configured')
    expect(embedder.modelWarnings[0]).toContain('nomic-embed-text-v1.5')
  })

  it('lets overrides replace the built-in prompts and clear the warning', async () => {
    const embedder = create({
      model: 'custom-model',
      queryPrompt: 'Q: ',
      documentPrompt: 'D: ',
    })

    await embedder.embed('x', 'query')
    await embedder.embedBatch(['y'], 'document')

    expect(bodyInput(0)).toEqual(['Q: x'])
    expect(bodyInput(1)).toEqual(['D: y'])
    expect(embedder.modelWarnings).toEqual([])
    expect(await embedder.getDocumentPrompt()).toBe('D: ')
  })
})

describe('LmStudioEmbedder responses', () => {
  it('L2-normalizes returned vectors', async () => {
    const [vector] = await create().embedBatch(['alpha beta'])

    expect(norm(vector ?? [])).toBeCloseTo(1, 6)
  })

  it('orders rows by data[i].index when LM Studio returns them shuffled', async () => {
    mock.setHandler((_request, response) => {
      response.setHeader('Content-Type', 'application/json')
      response.end(
        JSON.stringify({
          data: [
            { index: 1, embedding: [0, 2] },
            { index: 0, embedding: [3, 0] },
          ],
        })
      )
      return true
    })

    const vectors = await create().embedBatch(['a', 'b'])

    expect(vectors).toEqual([
      [1, 0],
      [0, 1],
    ])
  })

  it('falls back to array order when index is absent', async () => {
    mock.setHandler((_request, response) => {
      response.end(JSON.stringify({ data: [{ embedding: [2, 0] }, { embedding: [0, 4] }] }))
      return true
    })

    expect(await create().embedBatch(['a', 'b'])).toEqual([
      [1, 0],
      [0, 1],
    ])
  })

  it.each([
    ['a wrong count', { data: [{ index: 0, embedding: [1, 0] }] }, /1 embeddings for 2 inputs/],
    ['no data array', { error: 'x' }, /no embeddings for 2 inputs/],
    [
      'duplicate indexes',
      {
        data: [
          { index: 0, embedding: [1, 0] },
          { index: 0, embedding: [0, 1] },
        ],
      },
      /unexpected indexes/,
    ],
    [
      'a non-numeric value',
      { data: [{ embedding: [1, 'x'] }, { embedding: [1, 0] }] },
      /non-numeric or non-finite/,
    ],
    [
      'a null value',
      { data: [{ embedding: [1, null] }, { embedding: [1, 0] }] },
      /non-numeric or non-finite/,
    ],
    ['a missing embedding', { data: [{ index: 0 }, { index: 1 }] }, /without an embedding array/],
    ['a zero vector', { data: [{ embedding: [0, 0] }, { embedding: [1, 0] }] }, /zero vector/],
    [
      'inconsistent dimensions',
      { data: [{ embedding: [1, 0] }, { embedding: [1, 0, 0] }] },
      /inconsistent dimensions/,
    ],
  ])('rejects %s', async (_name, payload, message) => {
    mock.setHandler((_request, response) => {
      response.end(JSON.stringify(payload))
      return true
    })

    const error = expectError(
      await create()
        .embedBatch(['a', 'b'])
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toMatch(message)
    expect(error.message).toContain(mock.baseUrl)
    expect(error.message).toContain(GEMMA)
  })

  it('rejects non-finite values that JSON cannot carry as numbers', async () => {
    mock.setHandler((_request, response) => {
      // JSON has no Infinity; 1e999 parses to it.
      response.end('{"data":[{"embedding":[1e999,0]}]}')
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error.message).toMatch(/non-numeric or non-finite/)
  })

  it('rejects dimensions that differ between HTTP batches', async () => {
    let call = 0
    mock.setHandler((_request, response) => {
      call += 1
      response.end(JSON.stringify({ data: [{ embedding: call === 1 ? [1, 0] : [1, 0, 0] }] }))
      return true
    })

    const error = expectError(
      await create({ batchSize: 1 })
        .embedBatch(['a', 'b'])
        .catch((e: unknown) => e)
    )

    expect(error.message).toMatch(/inconsistent dimensions/)
  })

  it('rejects a body that is not JSON', async () => {
    mock.setHandler((_request, response) => {
      response.end('<html>')
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error.message).toMatch(/not valid JSON/)
  })
})

describe('LmStudioEmbedder failures', () => {
  it('names the model when LM Studio answers 404', async () => {
    mock.setHandler((_request, response) => {
      response.statusCode = 404
      response.end(JSON.stringify({ error: { message: 'Not found' } }))
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toContain(`"${GEMMA}"`)
    expect(error.message).toContain('HTTP 404')
    expect(error.message).toContain(mock.baseUrl)
  })

  it('names the model when a 400 says the model is not loaded', async () => {
    mock.setHandler((_request, response) => {
      response.statusCode = 400
      response.end(JSON.stringify({ error: "Failed to load model: no such model 'x'" }))
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error.message).toContain(`embedding model "${GEMMA}"`)
  })

  it('reports other 4xx responses with their status and body', async () => {
    mock.setHandler((_request, response) => {
      response.statusCode = 422
      response.end('bad input shape')
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error.message).toContain('HTTP 422')
    expect(error.message).toContain('bad input shape')
    expect(error.message).toContain(GEMMA)
  })

  it('reports 5xx responses with the endpoint and model', async () => {
    mock.setHandler((_request, response) => {
      response.statusCode = 500
      response.end('boom')
      return true
    })

    const error = expectError(
      await create()
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toContain('HTTP 500')
    expect(error.message).toContain('boom')
    expect(error.message).toContain(`${mock.baseUrl}/embeddings`)
    expect(error.message).toContain(GEMMA)
  })

  it('says LM Studio is not running when the connection is refused', async () => {
    // Find a free port, then release it so nothing listens there.
    const probe = createServer()
    await new Promise<void>((resolveListen) => probe.listen(0, '127.0.0.1', resolveListen))
    const address = probe.address()
    const port = typeof address === 'object' && address !== null ? address.port : 0
    await new Promise<void>((resolveClose) => probe.close(() => resolveClose()))
    const baseUrl = `http://127.0.0.1:${port}/v1`

    const error = expectError(
      await create({ baseUrl })
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toContain(`LM Studio is not running at ${baseUrl}`)
    expect(error.message).toContain(GEMMA)
  })

  it('times out when LM Studio does not answer', async () => {
    mock.setHandler(() => new Promise<boolean>(() => {}))

    const error = expectError(
      await create({ timeoutMs: 150 })
        .embed('a')
        .catch((e: unknown) => e)
    )

    expect(error).toBeInstanceOf(EmbeddingError)
    expect(error.message).toContain('150ms')
    expect(error.message).toContain('LMSTUDIO_TIMEOUT_MS')
    expect(error.message).toContain(GEMMA)
  })
})

describe('LmStudioEmbedder token estimates', () => {
  it('estimates UTF-8 bytes / 2, rounded up', async () => {
    const embedder = create({ model: 'custom-model', queryPrompt: '', documentPrompt: '' })

    // ASCII: 5 bytes -> 3. CJK: 3 characters x 3 bytes = 9 bytes -> 5.
    expect(await embedder.countTokens(['hello', '日本語'])).toEqual([3, 5])
  })

  it('counts the prompt of the requested role, matching what embedBatch sends', async () => {
    const embedder = create({ model: GEMMA })
    const text = 'hello'

    const [plain] = await embedder.countTokens([text])
    const [query] = await embedder.countTokens([text], 'query')
    const [document] = await embedder.countTokens([text], 'document')
    await embedder.embedBatch([text], 'query')
    await embedder.embedBatch([text], 'document')

    expect(plain).toBe(3)
    expect(query).toBe(Math.ceil(Buffer.byteLength(bodyInput(0)[0] ?? '') / 2))
    expect(document).toBe(Math.ceil(Buffer.byteLength(bodyInput(1)[0] ?? '') / 2))
    expect(mock.requests).toHaveLength(2)
  })

  it('counts without calling LM Studio and reports the configured limit', async () => {
    const embedder = create({ maxInputTokens: 8192 })

    await embedder.countTokens(['x'])

    expect(mock.requests).toHaveLength(0)
    expect(await embedder.getTokenLimit()).toBe(8192)
  })

  it('exposes the prefix flags and a no-op dispose, and omits a computation identity', async () => {
    const embedder = create({ titlePrefix: true })

    expect(embedder.titlePrefix).toBe(true)
    expect(embedder.headingPrefix).toBe(false)
    await expect(embedder.dispose()).resolves.toBeUndefined()
    expect('getComputationIdentity' in embedder).toBe(false)
  })
})
