// Single construction point for the configured embedding backend

import type { PersistentEmbeddingProvider } from './embedding-snapshot.js'
import { Embedder, type EmbedderConfig, type EmbeddingRole } from './index.js'
import { LmStudioEmbedder, type LmStudioEmbedderConfig } from './lmstudio-embedder.js'

/** LM Studio settings from the environment; the prefix flags come from `EmbedderConfig`. */
export type LmStudioSettings = Omit<LmStudioEmbedderConfig, 'titlePrefix' | 'headingPrefix'>

/** What the server and CLI need from any embedding backend. */
export interface EmbeddingProvider extends PersistentEmbeddingProvider {
  embed(text: string, role?: EmbeddingRole): Promise<number[]>
  dispose(): Promise<void>
  readonly modelWarnings: readonly string[]
}

/**
 * Builds the Transformers.js embedder, or the LM Studio one when `lmstudio` is
 * set. The LM Studio path never constructs an `Embedder`, so no model is
 * downloaded or loaded.
 */
export function createEmbeddingProvider(
  transformers: EmbedderConfig,
  lmstudio?: LmStudioSettings
): EmbeddingProvider {
  if (lmstudio === undefined) {
    return new Embedder(transformers)
  }
  return new LmStudioEmbedder({
    ...lmstudio,
    titlePrefix: transformers.titlePrefix === true,
    headingPrefix: transformers.headingPrefix === true,
  })
}
