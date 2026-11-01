// Per-repo configuration file: `.mcp-local-rag.json` supplies defaults for the
// documented environment variables, so each repository can pick its own
// backend, model and database without every client repeating them.

import { readFileSync, statSync } from 'node:fs'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { AppError, toError } from './errors.js'
import { isObjectLike } from './type-guards.js'

/** File name searched from the working directory upward. */
export const CONFIG_FILE_NAME = '.mcp-local-rag.json'

/** Points at a config file explicitly, for clients that start the server elsewhere. */
const CONFIG_PATH_ENV = 'MCP_LOCAL_RAG_CONFIG'

const MAX_CONFIG_FILE_BYTES = 64 * 1024

/** Settings a config file may set: the documented variables that choose where and how to index. */
const ALLOWED_KEYS: ReadonlySet<string> = new Set([
  'BASE_DIR',
  'BASE_DIRS',
  'IGNORE_PATHS',
  'DB_PATH',
  'CACHE_DIR',
  'MAX_FILE_SIZE',
  'MODEL_NAME',
  'CHUNK_MIN_LENGTH',
  'EMBED_TITLE_PREFIX',
  'EMBED_HEADING_PREFIX',
  'EMBED_BACKEND',
  'LMSTUDIO_BASE_URL',
  'LMSTUDIO_MODEL',
  'LMSTUDIO_BATCH_SIZE',
  'LMSTUDIO_MAX_INPUT_TOKENS',
  'LMSTUDIO_TIMEOUT_MS',
  'LMSTUDIO_QUERY_PROMPT',
  'LMSTUDIO_DOCUMENT_PROMPT',
  'STORE_IMAGES',
  'RAG_DEVICE',
  'RAG_DTYPE',
  'RAG_MAX_DISTANCE',
  'RAG_GROUPING',
  'RAG_MAX_FILES',
  'RAG_HYBRID_WEIGHT',
])

/**
 * Environment-only: a file in a cloned repository must not be able to make the
 * server run a command, or redirect where models are downloaded from.
 */
const ENV_ONLY_KEYS: ReadonlySet<string> = new Set([
  'RAG_RERANK_CMD',
  'RAG_RERANK_TIMEOUT_MS',
  'HF_ENDPOINT',
])

/** Variables holding paths; relative values resolve against the config file's folder. */
const PATH_KEYS: ReadonlySet<string> = new Set(['DB_PATH', 'CACHE_DIR', 'BASE_DIR'])

/** Variables that accept a JSON array in the file. */
const LIST_KEYS: ReadonlySet<string> = new Set(['BASE_DIRS', 'IGNORE_PATHS'])

/** Both set the document roots; the file must not add a second source beside the environment. */
const ROOT_KEYS = ['BASE_DIR', 'BASE_DIRS'] as const

export class ConfigFileError extends AppError {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, 'config', 'config', options)
    this.name = 'ConfigFileError'
  }
}

export interface ConfigFileResult {
  /** The file that was read, or undefined when none applies. */
  path?: string
  /** Variables set from the file, because the environment did not set them. */
  applied: string[]
  warnings: string[]
}

function isUnset(env: NodeJS.ProcessEnv, key: string): boolean {
  return (env[key] ?? '') === ''
}

/** Walks from `start` toward the filesystem root for the config file. */
function findConfigFile(start: string): string | undefined {
  for (let directory = start; ; directory = dirname(directory)) {
    const candidate = join(directory, CONFIG_FILE_NAME)
    try {
      if (statSync(candidate).isFile()) {
        return candidate
      }
    } catch {
      // Not here; keep walking up.
    }
    if (dirname(directory) === directory) {
      return undefined
    }
  }
}

/** The explicit file when `MCP_LOCAL_RAG_CONFIG` is set, otherwise the nearest one. */
function locate(env: NodeJS.ProcessEnv, cwd: string): string | undefined {
  const explicit = env[CONFIG_PATH_ENV]
  if (explicit === undefined || explicit.trim() === '') {
    return findConfigFile(resolve(cwd))
  }
  const path = resolve(cwd, explicit.trim())
  try {
    if (statSync(path).isFile()) {
      return path
    }
  } catch {
    // Reported below with the path.
  }
  throw new ConfigFileError(`${CONFIG_PATH_ENV} points to "${path}", which is not a readable file.`)
}

function readJsonObject(path: string): Record<string, unknown> {
  let text: string
  try {
    if (statSync(path).size > MAX_CONFIG_FILE_BYTES) {
      throw new ConfigFileError(
        `Config file ${path} is larger than ${MAX_CONFIG_FILE_BYTES} bytes.`
      )
    }
    text = readFileSync(path, 'utf8')
  } catch (error) {
    if (error instanceof ConfigFileError) {
      throw error
    }
    throw new ConfigFileError(`Cannot read config file ${path}: ${toError(error).message}`, {
      cause: toError(error),
    })
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new ConfigFileError(`Config file ${path} is not valid JSON: ${toError(error).message}`, {
      cause: toError(error),
    })
  }
  if (!isObjectLike(parsed) || Array.isArray(parsed)) {
    throw new ConfigFileError(`Config file ${path} must contain a JSON object.`)
  }
  return parsed
}

/** Path values are resolved against the config file, so the cwd of the launcher does not matter. */
function resolveAgainst(baseDirectory: string, value: string): string {
  return isAbsolute(value) ? value : resolve(baseDirectory, value)
}

/** The environment-variable form of one config value, or an error naming the key. */
function toEnvValue(key: string, value: unknown, baseDirectory: string, path: string): string {
  if (LIST_KEYS.has(key) && Array.isArray(value)) {
    if (!value.every((item): item is string => typeof item === 'string')) {
      throw new ConfigFileError(`${key} in ${path} must be an array of strings.`)
    }
    const items =
      key === 'BASE_DIRS' ? value.map((item) => resolveAgainst(baseDirectory, item)) : value
    return JSON.stringify(items)
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    const text = String(value)
    return PATH_KEYS.has(key) && text !== '' ? resolveAgainst(baseDirectory, text) : text
  }
  throw new ConfigFileError(
    `${key} in ${path} must be a string, number or boolean${LIST_KEYS.has(key) ? ', or an array of strings' : ''}.`
  )
}

/** Sorts file keys into settings to apply and warnings for the rest. */
function selectKeys(
  config: Record<string, unknown>,
  path: string
): { keys: string[]; warnings: string[] } {
  const keys: string[] = []
  const warnings: string[] = []
  for (const key of Object.keys(config)) {
    if (ALLOWED_KEYS.has(key)) {
      keys.push(key)
    } else if (ENV_ONLY_KEYS.has(key)) {
      warnings.push(
        `${key} in ${path} was ignored: it can only be set through the environment, so a repository file cannot run commands or change download hosts.`
      )
    } else {
      warnings.push(`Unknown setting "${key.slice(0, 100)}" in ${path} was ignored.`)
    }
  }
  return { keys, warnings }
}

/**
 * Fills unset environment variables from the nearest `.mcp-local-rag.json`.
 * A variable the environment already sets always wins, key by key; the
 * document roots are the exception, treated as one group so the file cannot
 * add a root source beside the environment's. A malformed file throws rather
 * than falling back, which would index with a different model or database.
 */
export function applyConfigFile(env: NodeJS.ProcessEnv, cwd: string): ConfigFileResult {
  const path = locate(env, cwd)
  if (path === undefined) {
    return { applied: [], warnings: [] }
  }
  const config = readJsonObject(path)
  const { keys, warnings } = selectKeys(config, path)
  const rootsFromEnv = ROOT_KEYS.some((key) => !isUnset(env, key))
  const applied: string[] = []
  for (const key of keys) {
    const skipped = ROOT_KEYS.some((rootKey) => rootKey === key) ? rootsFromEnv : !isUnset(env, key)
    if (!skipped) {
      env[key] = toEnvValue(key, config[key], dirname(path), path)
      applied.push(key)
    }
  }
  return { path, applied, warnings }
}
