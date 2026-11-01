// Embedder backed by LM Studio's OpenAI-compatible /v1/embeddings endpoint

import { toError } from '../utils/errors.js'
import { isObjectLike } from '../utils/type-guards.js'
import { EmbeddingError, type EmbeddingRole } from './index.js'

// ============================================
// Type Definitions
// ============================================

/** LM Studio backend settings, already validated by the config parser. */
export interface LmStudioEmbedderConfig {
  /** Base URL including `/v1`, e.g. `http://127.0.0.1:1234/v1`. */
  baseUrl: string
  /** Model id as LM Studio lists it. */
  model: string
  /** Maximum inputs per HTTP request. */
  batchSize: number
  /** Token cap reported to the chunker. */
  maxInputTokens: number
  /** Per-request budget in milliseconds. */
  timeoutMs: number
  /** Overrides the built-in query prompt, which is prepended verbatim. */
  queryPrompt?: string
  /** Overrides the built-in document prompt, which is prepended verbatim. */
  documentPrompt?: string
  titlePrefix?: boolean
  headingPrefix?: boolean
}

interface RolePrompts {
  query: string
  document: string
}

// ============================================
// Model Prompts
// ============================================

/** Qwen3-Embedding instructs queries only; documents are embedded raw. */
const QWEN3_PROMPTS: RolePrompts = {
  query:
    'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:',
  document: '',
}

/** EmbeddingGemma's retrieval templates, with the document title unknown. */
const EMBEDDING_GEMMA_PROMPTS: RolePrompts = {
  query: 'task: search result | query: ',
  document: 'title: none | text: ',
}

/** Prompts for the model ids this tool knows, or `null` for any other id. */
function builtInPrompts(model: string): RolePrompts | null {
  const id = model.toLowerCase()
  if (id.includes('qwen3-embedding')) {
    return QWEN3_PROMPTS
  }
  if (id.includes('embeddinggemma')) {
    return EMBEDDING_GEMMA_PROMPTS
  }
  return null
}

// ============================================
// Embedder Class
// ============================================

/** Embeds through a running LM Studio server; nothing is loaded in-process. */
export class LmStudioEmbedder {
  private readonly config: LmStudioEmbedderConfig
  private readonly prompts: RolePrompts
  private readonly warnings: string[] = []

  constructor(config: LmStudioEmbedderConfig) {
    this.config = config
    const builtIn = builtInPrompts(config.model)
    if (
      builtIn === null &&
      (config.queryPrompt === undefined || config.documentPrompt === undefined)
    ) {
      this.warnings.push(
        `No embedding prompt is configured for LM Studio model "${config.model}"; text is embedded as-is. Set LMSTUDIO_QUERY_PROMPT / LMSTUDIO_DOCUMENT_PROMPT if the model expects prompts.`
      )
    }
    this.prompts = {
      query: config.queryPrompt ?? builtIn?.query ?? '',
      document: config.documentPrompt ?? builtIn?.document ?? '',
    }
  }

  get headingPrefix(): boolean {
    return this.config.headingPrefix ?? false
  }

  get titlePrefix(): boolean {
    return this.config.titlePrefix ?? false
  }

  /** Model settings this tool cannot honor. */
  get modelWarnings(): readonly string[] {
    return this.warnings
  }

  /** Nothing is held in-process, so there is nothing to release. */
  async dispose(): Promise<void> {}

  /** The configured cap; LM Studio does not report the model's own limit. */
  async getTokenLimit(): Promise<number> {
    return this.config.maxInputTokens
  }

  /**
   * Estimated token length of each text as embedded for `role`, prompt
   * included. UTF-8 bytes / 2 overestimates English and is safe for CJK.
   */
  async countTokens(texts: string[], role?: EmbeddingRole): Promise<number[]> {
    return this.withPrompt(texts, role).map((text) =>
      Math.ceil(Buffer.byteLength(text, 'utf8') / 2)
    )
  }

  /** The prompt documents are embedded behind, or `''`. */
  async getDocumentPrompt(): Promise<string> {
    return this.prompts.document
  }

  private withPrompt(texts: string[], role: EmbeddingRole | undefined): string[] {
    const prompt = role === undefined ? '' : this.prompts[role]
    return prompt === '' ? texts : texts.map((text) => prompt + text)
  }

  /** Single-text embedding; delegates to {@link embedBatch}. */
  async embed(text: string, role?: EmbeddingRole): Promise<number[]> {
    const embeddings = await this.embedBatch([text], role)
    const embedding = embeddings[0]
    if (embedding === undefined) {
      throw new EmbeddingError('Missing embedder batch output row')
    }
    return embedding
  }

  /** Batched embedding, split into requests of at most `batchSize` inputs. */
  async embedBatch(texts: string[], role?: EmbeddingRole): Promise<number[][]> {
    if (texts.length === 0) {
      return []
    }
    if (texts.some((text) => text.length === 0)) {
      throw new EmbeddingError('Cannot generate embedding for empty text')
    }

    const promptedTexts = this.withPrompt(texts, role)
    const embeddings: number[][] = []
    for (let i = 0; i < promptedTexts.length; i += this.config.batchSize) {
      const batch = promptedTexts.slice(i, i + this.config.batchSize)
      embeddings.push(...(await this.requestEmbeddings(batch)))
    }

    const dimension = embeddings[0]?.length
    if (embeddings.some((embedding) => embedding.length !== dimension)) {
      throw new EmbeddingError(this.describe('returned vectors of inconsistent dimensions'))
    }
    return embeddings
  }

  private get endpoint(): string {
    return `${this.config.baseUrl.replace(/\/+$/, '')}/embeddings`
  }

  /** Names the endpoint and model, which every failure here involves. */
  private describe(problem: string): string {
    return `LM Studio at ${this.endpoint} (model "${this.config.model}") ${problem}`
  }

  private async requestEmbeddings(inputs: string[]): Promise<number[][]> {
    const body = await this.post({ model: this.config.model, input: inputs })
    return this.parseVectors(body, inputs.length)
  }

  private async post(payload: { model: string; input: string[] }): Promise<unknown> {
    let response: Response
    try {
      response = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(this.config.timeoutMs),
      })
    } catch (error) {
      throw this.connectionError(error)
    }

    let text: string
    try {
      text = await response.text()
    } catch (error) {
      throw this.connectionError(error)
    }
    if (!response.ok) {
      throw this.httpError(response.status, text)
    }
    try {
      const parsed: unknown = JSON.parse(text)
      return parsed
    } catch (error) {
      throw new EmbeddingError(this.describe('returned a response that is not valid JSON'), {
        cause: toError(error),
      })
    }
  }

  private connectionError(error: unknown): EmbeddingError {
    const cause = toError(error)
    if (cause.name === 'TimeoutError') {
      return new EmbeddingError(
        this.describe(
          `did not respond within ${this.config.timeoutMs}ms (LMSTUDIO_TIMEOUT_MS). The first request may be loading the model`
        ),
        { cause }
      )
    }
    // fetch wraps the socket error as `cause` on a generic TypeError.
    const code = isObjectLike(cause.cause) ? cause.cause['code'] : undefined
    if (code === 'ECONNREFUSED') {
      return new EmbeddingError(
        `LM Studio is not running at ${this.config.baseUrl}. Start its local server and load model "${this.config.model}"`,
        { cause }
      )
    }
    return new EmbeddingError(this.describe(`request failed: ${cause.message}`), { cause })
  }

  private httpError(status: number, bodyText: string): EmbeddingError {
    const detail = bodyText.trim().slice(0, 300)
    const suffix = detail === '' ? '' : `: ${detail}`
    const modelProblem =
      status === 404 ||
      (status >= 400 && status < 500 && /model/i.test(detail) && /(not|no|unknown)/i.test(detail))
    if (modelProblem) {
      return new EmbeddingError(
        `LM Studio at ${this.endpoint} does not have an embedding model "${this.config.model}" available (HTTP ${status}). Download it and load it as an embedding model${suffix}`
      )
    }
    return new EmbeddingError(this.describe(`returned HTTP ${status}${suffix}`))
  }

  /** Validates the response and returns L2-normalized vectors in input order. */
  private parseVectors(body: unknown, expected: number): number[][] {
    const data = isObjectLike(body) ? body['data'] : undefined
    if (!Array.isArray(data) || data.length !== expected) {
      const got = Array.isArray(data) ? data.length : 'no'
      throw new EmbeddingError(this.describe(`returned ${got} embeddings for ${expected} inputs`))
    }

    const rows = data.map((item: unknown, position: number) => this.toRow(item, position))
    rows.sort((a, b) => a.index - b.index)
    const indexes = new Set(rows.map((row) => row.index))
    if (indexes.size !== expected || rows.some((row, i) => row.index !== i)) {
      throw new EmbeddingError(this.describe('returned embeddings with unexpected indexes'))
    }
    return rows.map((row) => row.vector)
  }

  private toRow(item: unknown, position: number): { index: number; vector: number[] } {
    const embedding = isObjectLike(item) ? item['embedding'] : undefined
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new EmbeddingError(this.describe('returned an item without an embedding array'))
    }
    const rawIndex = isObjectLike(item) ? item['index'] : undefined
    const index = typeof rawIndex === 'number' ? rawIndex : position
    return { index, vector: this.toVector(embedding) }
  }

  private toVector(values: unknown[]): number[] {
    let sumOfSquares = 0
    const vector: number[] = []
    for (const value of values) {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new EmbeddingError(this.describe('returned a non-numeric or non-finite value'))
      }
      sumOfSquares += value * value
      vector.push(value)
    }
    const norm = Math.sqrt(sumOfSquares)
    if (norm === 0) {
      throw new EmbeddingError(this.describe('returned a zero vector'))
    }
    return vector.map((value) => value / norm)
  }
}
