// Smoke tests for the top-level CLI entry routing in src/index.ts.
// These spawn the entry as a real subprocess via tsx so we observe the
// actual exit codes and stderr surface that users see.

import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const PROJECT_ROOT = resolve(__dirname, '../../..')
const ENTRY = resolve(PROJECT_ROOT, 'src/index.ts')
const TSX_LOADER = import.meta.resolve('tsx')

interface RunResult {
  status: number | null
  stderr: string
  stdout: string
}

function runCli(args: string[]): RunResult {
  const result = spawnSync(process.execPath, ['--import', 'tsx', ENTRY, ...args], {
    encoding: 'utf-8',
    cwd: PROJECT_ROOT,
    timeout: 15000,
  })
  return {
    status: result.status,
    stderr: result.stderr ?? '',
    stdout: result.stdout ?? '',
  }
}

describe('CLI entry routing', () => {
  it('rejects global CLI flags on the bare server launch and points at env vars', () => {
    const { status, stderr } = runCli(['--db-path', '/tmp/should-not-apply'])

    expect(status).toBe(1)
    expect(stderr).toContain('Global CLI options are not supported')
    expect(stderr).toContain('DB_PATH')
    expect(stderr).toContain('MODEL_NAME')
  })

  it('errors on unknown subcommands and lists the available commands', () => {
    const { status, stderr } = runCli(['definitely-not-a-command'])

    expect(status).toBe(1)
    expect(stderr).toContain('Unknown command:')
    expect(stderr).toContain('skills')
    expect(stderr).toContain('ingest')
    expect(stderr).toContain('read-neighbors')
    expect(stderr).toContain('relocate')
    expect(stderr).toContain('sync')
  })

  // Dispatch is proven through runSync's own stderr rather than a module mock:
  // `sync --help` is an output signature no other code path produces, so a
  // missing switch arm shows up as a silent exit-0 run with no usage text.
  it('routes sync to the sync command and forwards the arguments after it', () => {
    const { status, stderr } = runCli(['sync', '--help'])

    expect(status).toBe(0)
    expect(stderr).toContain('Usage: mcp-local-rag [global-options] sync [options] [path]')
    expect(
      stderr.split('Usage: mcp-local-rag [global-options] sync [options] [path]')
    ).toHaveLength(2)
  })

  it('routes relocate to its command help and forwards the arguments after it', () => {
    const { status, stderr } = runCli(['relocate', '--help'])

    expect(status).toBe(0)
    expect(stderr).toContain(
      'Usage: mcp-local-rag [global-options] relocate --from <old-absolute-directory> --to <new-absolute-directory>'
    )
  })

  it('applies global options placed before sync to the sync run', () => {
    const { status, stderr } = runCli(['--db-path', '/etc/mcp-local-rag-should-not-apply', 'sync'])

    expect(status).toBe(1)
    expect(stderr).toContain('sensitive system path')
    expect(stderr).toContain('--db-path')
  })

  it('strips ANSI escape and control characters from the echoed unknown command', () => {
    const evil = '[31mboom[0m\r\n--inject'
    const { status, stderr } = runCli([evil])

    expect(status).toBe(1)
    expect(stderr).toContain('Unknown command:')
    // None of the control characters from the input should reach stderr verbatim.
    expect(stderr).not.toContain('[31m')
    expect(stderr).not.toContain('[0m')
  })
})

describe('per-repo config file at the entry point', () => {
  function runIn(
    directory: string,
    env: NodeJS.ProcessEnv,
    args: string[]
  ): { status: number | null; stderr: string } {
    // Absolute loader URL: `--import tsx` would resolve from the temp cwd.
    const result = spawnSync(process.execPath, ['--import', TSX_LOADER, ENTRY, ...args], {
      encoding: 'utf-8',
      cwd: directory,
      timeout: 15000,
      env: { ...process.env, EMBED_BACKEND: '', LMSTUDIO_MODEL: '', ...env },
    })
    return { status: result.status, stderr: result.stderr ?? '' }
  }

  function withConfig(content: string, run: (directory: string) => void): void {
    const directory = mkdtempSync(join(tmpdir(), 'rag-entry-'))
    try {
      writeFileSync(join(directory, '.mcp-local-rag.json'), content)
      run(directory)
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  }

  it('applies .mcp-local-rag.json before the CLI reads its configuration', () => {
    withConfig('{"EMBED_BACKEND":"lmstudio"}', (directory) => {
      const { status, stderr } = runIn(directory, {}, ['list'])

      expect(status).toBe(1)
      expect(stderr).toContain('EMBED_BACKEND=lmstudio requires LMSTUDIO_MODEL')
    })
  })

  it('lets a real environment variable override the file', () => {
    withConfig('{"EMBED_BACKEND":"lmstudio"}', (directory) => {
      const { stderr } = runIn(directory, { EMBED_BACKEND: 'transformers' }, ['list', '--help'])

      expect(stderr).not.toContain('requires LMSTUDIO_MODEL')
    })
  })

  it('exits with a clear error for a malformed file', () => {
    withConfig('{ nope', (directory) => {
      const { status, stderr } = runIn(directory, {}, ['list'])

      expect(status).toBe(1)
      expect(stderr).toContain('is not valid JSON')
    })
  })
})
