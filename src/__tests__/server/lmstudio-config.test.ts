import { describe, expect, it } from 'vitest'
import { createEmbedder } from '../../cli/common.js'
import { parseLmStudioBaseUrl, resolveEmbedBackend } from '../../cli/options.js'
import { Embedder } from '../../embedder/index.js'
import { LmStudioEmbedder } from '../../embedder/lmstudio-embedder.js'
import { resolveServerConfig } from '../../server-main.js'

const BASE = { EMBED_BACKEND: 'lmstudio', LMSTUDIO_MODEL: 'text-embedding-embeddinggemma-300m' }

describe('resolveEmbedBackend', () => {
  it.each([undefined, '', '  ', 'transformers', 'TRANSFORMERS'])(
    'keeps the default embedder for EMBED_BACKEND=%j',
    (value) => {
      const env = value === undefined ? {} : { EMBED_BACKEND: value, LMSTUDIO_MODEL: 'ignored' }

      expect(resolveEmbedBackend(env)).toEqual({ warnings: [] })
    }
  )

  it('applies the documented defaults', () => {
    expect(resolveEmbedBackend(BASE)).toEqual({
      lmstudio: {
        baseUrl: 'http://127.0.0.1:1234/v1',
        model: 'text-embedding-embeddinggemma-300m',
        batchSize: 16,
        maxInputTokens: 2048,
        timeoutMs: 120_000,
      },
      warnings: [],
    })
  })

  it('reads every setting', () => {
    const resolved = resolveEmbedBackend({
      ...BASE,
      EMBED_BACKEND: ' LMStudio ',
      LMSTUDIO_BASE_URL: 'http://localhost:5555/v1/',
      LMSTUDIO_BATCH_SIZE: '4',
      LMSTUDIO_MAX_INPUT_TOKENS: '8192',
      LMSTUDIO_TIMEOUT_MS: '30000',
      LMSTUDIO_QUERY_PROMPT: 'q: ',
      LMSTUDIO_DOCUMENT_PROMPT: 'd: ',
    })

    expect(resolved.warnings).toEqual([])
    expect(resolved.lmstudio).toEqual({
      baseUrl: 'http://localhost:5555/v1',
      model: 'text-embedding-embeddinggemma-300m',
      batchSize: 4,
      maxInputTokens: 8192,
      timeoutMs: 30_000,
      queryPrompt: 'q: ',
      documentPrompt: 'd: ',
    })
  })

  it('requires LMSTUDIO_MODEL instead of falling back to transformers', () => {
    expect(() => resolveEmbedBackend({ EMBED_BACKEND: 'lmstudio' })).toThrow(
      /requires LMSTUDIO_MODEL/
    )
    expect(() => resolveEmbedBackend({ EMBED_BACKEND: 'lmstudio', LMSTUDIO_MODEL: '  ' })).toThrow(
      /requires LMSTUDIO_MODEL/
    )
  })

  it('rejects an unknown EMBED_BACKEND instead of silently using transformers', () => {
    expect(() => resolveEmbedBackend({ EMBED_BACKEND: 'lmstudo' })).toThrow(
      /Invalid EMBED_BACKEND value: "lmstudo".*"transformers" or "lmstudio"/
    )
  })

  it.each([
    ['LMSTUDIO_BATCH_SIZE', '0', 'batchSize', 16],
    ['LMSTUDIO_BATCH_SIZE', 'abc', 'batchSize', 16],
    ['LMSTUDIO_BATCH_SIZE', '2.5', 'batchSize', 16],
    ['LMSTUDIO_BATCH_SIZE', '99999', 'batchSize', 16],
    ['LMSTUDIO_MAX_INPUT_TOKENS', '-1', 'maxInputTokens', 2048],
    ['LMSTUDIO_MAX_INPUT_TOKENS', 'lots', 'maxInputTokens', 2048],
    ['LMSTUDIO_TIMEOUT_MS', '5', 'timeoutMs', 120_000],
    ['LMSTUDIO_TIMEOUT_MS', '1e3', 'timeoutMs', 120_000],
  ] as const)('warns and uses the default for %s=%s', (name, value, field, fallback) => {
    const resolved = resolveEmbedBackend({ ...BASE, [name]: value })

    expect(resolved.lmstudio?.[field]).toBe(fallback)
    expect(resolved.warnings).toHaveLength(1)
    expect(resolved.warnings[0]).toContain(`Invalid ${name} value: "${value}"`)
    expect(resolved.warnings[0]).toContain(`Using default (${fallback})`)
  })

  it('treats empty prompt overrides as unset', () => {
    const resolved = resolveEmbedBackend({
      ...BASE,
      LMSTUDIO_QUERY_PROMPT: '',
      LMSTUDIO_DOCUMENT_PROMPT: '',
    })

    expect(resolved.lmstudio).not.toHaveProperty('queryPrompt')
    expect(resolved.lmstudio).not.toHaveProperty('documentPrompt')
  })
})

describe('parseLmStudioBaseUrl', () => {
  it('defaults when unset or blank', () => {
    expect(parseLmStudioBaseUrl(undefined)).toEqual({ value: 'http://127.0.0.1:1234/v1' })
    expect(parseLmStudioBaseUrl('  ')).toEqual({ value: 'http://127.0.0.1:1234/v1' })
  })

  it.each(['not a url', 'ftp://host/v1', '127.0.0.1:1234/v1'])('warns for %j', (value) => {
    const parsed = parseLmStudioBaseUrl(value)

    expect(parsed.value).toBe('http://127.0.0.1:1234/v1')
    expect(parsed.warning).toContain('Invalid LMSTUDIO_BASE_URL')
  })
})

describe('resolveServerConfig with EMBED_BACKEND', () => {
  it('leaves the default configuration without lmstudio settings', async () => {
    const config = await resolveServerConfig({}, process.cwd())

    expect(config).not.toHaveProperty('lmstudio')
    expect(config.modelName).toBe('Xenova/all-MiniLM-L6-v2')
  })

  it('carries lmstudio settings and their warnings', async () => {
    const config = await resolveServerConfig(
      { ...BASE, LMSTUDIO_BATCH_SIZE: 'nope' },
      process.cwd()
    )

    expect(config.lmstudio?.model).toBe('text-embedding-embeddinggemma-300m')
    expect(config.lmstudio?.batchSize).toBe(16)
    expect(config.configWarnings?.join('\n')).toContain('Invalid LMSTUDIO_BATCH_SIZE')
  })

  it('fails to start without LMSTUDIO_MODEL', async () => {
    await expect(resolveServerConfig({ EMBED_BACKEND: 'lmstudio' }, process.cwd())).rejects.toThrow(
      /requires LMSTUDIO_MODEL/
    )
  })
})

describe('CLI createEmbedder', () => {
  const global = {
    dbPath: './lancedb/',
    cacheDir: './models/',
    modelName: 'Xenova/all-MiniLM-L6-v2',
  }

  it('builds the transformers embedder by default', () => {
    expect(createEmbedder(global)).toBeInstanceOf(Embedder)
  })

  it('builds the LM Studio embedder when the config carries lmstudio settings', () => {
    const resolved = resolveEmbedBackend(BASE)

    expect(createEmbedder({ ...global, ...resolved })).toBeInstanceOf(LmStudioEmbedder)
  })
})
