// Mock LM Studio server for tests: serves POST /v1/embeddings over loopback.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'

export interface RecordedRequest {
  method: string | undefined
  url: string | undefined
  body: unknown
}

/** Replaces the default reply; return `true` once the response has been written. */
export type MockHandler = (
  request: RecordedRequest,
  response: ServerResponse
) => boolean | Promise<boolean>

export interface MockLmStudio {
  /** Base URL including `/v1`, as `LMSTUDIO_BASE_URL` takes it. */
  baseUrl: string
  requests: RecordedRequest[]
  /** Replaces the handler; `undefined` restores the default embedding reply. */
  setHandler(handler: MockHandler | undefined): void
  close(): Promise<void>
}

const DIMENSION = 32

/** Deterministic unnormalized bag-of-words vector, scaled so normalization is observable. */
export function mockVector(text: string, dimension: number = DIMENSION): number[] {
  const vector = new Array<number>(dimension).fill(0)
  for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
    let hash = 0
    for (const char of word) {
      hash = (hash * 31 + (char.codePointAt(0) ?? 0)) % 1_000_003
    }
    vector[hash % dimension] = (vector[hash % dimension] ?? 0) + 3
  }
  // Never all-zero, so inputs without letters or digits stay embeddable.
  vector[0] = (vector[0] ?? 0) + 0.5
  return vector
}

async function readBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []
  for await (const chunk of request) {
    chunks.push(Buffer.from(chunk))
  }
  const text = Buffer.concat(chunks).toString('utf8')
  return text === '' ? undefined : JSON.parse(text)
}

/** The `input` strings of a recorded embeddings request. */
export function requestInputs(request: RecordedRequest | undefined): string[] {
  const body = request?.body
  const input = typeof body === 'object' && body !== null ? Reflect.get(body, 'input') : undefined
  if (typeof input === 'string') {
    return [input]
  }
  return Array.isArray(input) ? input.map(String) : []
}

export async function startMockLmStudio(): Promise<MockLmStudio> {
  const requests: RecordedRequest[] = []
  let handler: MockHandler | undefined

  const handle = async (request: IncomingMessage, response: ServerResponse): Promise<void> => {
    const recorded: RecordedRequest = {
      method: request.method,
      url: request.url,
      body: await readBody(request),
    }
    requests.push(recorded)
    if (handler !== undefined && (await handler(recorded, response))) {
      return
    }
    const data = requestInputs(recorded).map((text, index) => ({
      object: 'embedding',
      index,
      embedding: mockVector(text),
    }))
    response.setHeader('Content-Type', 'application/json')
    response.end(JSON.stringify({ object: 'list', data, usage: { total_tokens: 0 } }))
  }

  const server: Server = createServer((request, response) => {
    handle(request, response).catch((error: unknown) => {
      response.statusCode = 500
      response.end(String(error))
    })
  })
  await new Promise<void>((resolveListen) => server.listen(0, '127.0.0.1', resolveListen))
  const address = server.address()
  const port = typeof address === 'object' && address !== null ? address.port : 0

  return {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    requests,
    setHandler(next) {
      handler = next
    },
    close: () =>
      new Promise<void>((resolveClose) => {
        server.closeAllConnections()
        server.close(() => resolveClose())
      }),
  }
}
