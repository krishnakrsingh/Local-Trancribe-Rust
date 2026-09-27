# The model manifest

`registry.json` is the list of models VaaniDesk knows how to find, verify and
install. It is **data, not code**: adding a model is a text edit here, and the
loader validates the whole file on every start. Nothing in `src/` needs to change
to add a model.

The compiled-in copy is this file (`include_str!` in `src/manifest.rs`). To ship
a different catalogue without a new build, point the app at your own file:

```rust
ModelManager::from_manifest_file(std::path::Path::new(r"C:\models\registry.json"))?
```

## The rules the loader enforces

Every one of these is a load *failure*, not a warning, and the error names the
model and the field:

| rule | why |
|---|---|
| `schema_version` must be `1` | a newer manifest read by an older build would skip a checksum |
| unknown keys are rejected | a misspelled `sze` is otherwise a silent default, and the model loads with no size at all |
| `id` must be unique | it is the primary key and the CLI's `--model` argument |
| `dir` must be unique | two models with one directory would collide on disk |
| `license.spdx` and `license.url` are required | shipping a model whose terms nobody has seen is a legal problem, not a UX one |
| `size_bytes` must equal the sum of `files` | the UI shows the total before downloading; a second code path to get it wrong is a second thing to get wrong |
| every `sha256` is 64 hex characters | upper case is accepted and normalised, because a human pasting from `certutil -hashfile` will paste upper case |
| every `path` is relative, `/`-separated, with no `..` | a manifest is data; a path that escapes its model directory is an attack |
| at least one file is `required` | otherwise nothing could ever be verified |

One model is refused if **any** of them fails. A catalogue that loads "the models
that happened to parse" is how a model ships without a licence.

## Adding a model

1. Create `<model root>/<dir>/` somewhere, or let the app create it.
2. For each file, record its byte count and its SHA-256:

   ```powershell
   Get-FileHash -Algorithm SHA256 <file>
   ```

   `scripts/a7/hash_artifacts.ps1` does a whole directory at once and prints the
   JSON fragment this file needs.
3. Add a `models[]` entry. A minimal one:

   ```json
   {
     "id": "my-model-1.0",
     "engine_id": "my-model",
     "dir": "my-model",
     "version": "1.0",
     "display_name": "My Model 1.0",
     "description": "One paragraph a user can read before downloading anything.",
     "task": "asr",
     "architecture": "conformer-rnnt",
     "access": "open",
     "license": {
       "spdx": "Apache-2.0",
       "name": "Apache License 2.0",
       "url": "https://www.apache.org/licenses/LICENSE-2.0"
     },
     "source": {
       "kind": "huggingface",
       "repo": "org/name",
       "revision": "main",
       "homepage": "https://huggingface.co/org/name"
     },
     "languages": ["en"],
     "sample_rates": [16000],
     "size_bytes": 1234,
     "files": [
       {
         "path": "model.onnx",
         "role": "weights",
         "bytes": 1234,
         "sha256": "…64 hex characters…",
         "required": true
       }
     ]
   }
   ```

### Fields

| field | required | notes |
|---|---|---|
| `id` | yes | stable across releases; the CLI's `--model` |
| `dir` | no | defaults to `id`; one path component, not a path |
| `engine_id` | yes | the key in `vaani_asr::EngineRegistry`, e.g. `sravaani` |
| `version` | yes | of the **weights**, independent of `APP_VERSION` |
| `display_name` | yes | what the picker shows |
| `description` | yes | one paragraph |
| `task` | yes | `asr`, `vad` or `diarization` |
| `architecture` | yes | slug, e.g. `fastconformer-tdt-rnnt` |
| `architecture_params` | no | free-form `{"enc_d_model": "1024", …}` |
| `access` | yes | `open`, `gated-token` or `request-access` |
| `license` | yes | `{spdx, name, url}`; `url` must be `http(s)` |
| `source` | yes | see below |
| `languages` | yes | BCP-47 or ISO-639-3; at least one |
| `sample_rates` | yes | Hz; at least one, none of them zero |
| `size_bytes` | yes | must equal the sum of `files` |
| `resources` | no | `min_vram_bytes`, `rec_vram_bytes`, `min_ram_bytes`, `rec_ram_bytes` |
| `gpu` | no | `compute_capability_min` (`"8.6"`), `vram_headroom_bytes` |
| `runtime` | no | `opset`, `onnxruntime`, `cudnn`, `cuda_redist` |
| `files` | yes | at least one, at least one of them `required` |

### `source`

| `kind` | required | URL built from |
|---|---|---|
| `huggingface` | `repo` as `org/name` | `https://huggingface.co/{repo}/resolve/{revision}/{path}` |
| `github-releases` | `url_template` | the template, with `{path}` substituted |
| `direct` | `url_template` | the template, with `{path}` substituted |
| `bundled` | nothing | no URL: there is nothing to download |

`revision` defaults to `main`. A `huggingface` source needs a bearer token for
every request, and the token is sent **only** to `huggingface.co` — never
forwarded to the CDN host a redirect points at. Other providers are never sent
one.

### `files[].role`

The crate interprets six values and accepts any other string unchanged:
`weights`, `config`, `tokenizer`, `frontend`, `provenance`, `reference`. An
unknown role is carried through rather than refused, because the next model to be
added will have a role nobody has thought of yet and refusing to describe it would
turn a documentation change into a Rust change.

### `files[].required`

`false` means the model is fully usable without the file, and its absence is
never reported as a fault. `SraVaani`'s fp32 `encoder.onnx` is `false`: it is
1.65 GiB of the 2.5 GiB total, the app never opens it, and it exists so the
fp16 graph can be compared against its own reference. Marking it optional is what
stops discovery reporting a perfectly good install as broken — and it is why it
must not be deleted either.

## TOML also works

The format is sniffed from the first non-whitespace character. TOML is supported
because a registry a human edits by hand wants comments, and JSON is supported
because a registry a tool generates wants a stable schema. The loader accepts
both and validates them identically.

## Where models are looked for

| order | root | on Windows |
|---|---|---|
| 1 | user | `%LOCALAPPDATA%\VaaniDesk\models` |
| 2 | bundled | the dev/export `artifacts/` tree |
| 3 | configured | `VAANI_MODEL_DIRS`, `;`-separated |

A model is at `<root>/<dir>/<file.path>`. Downloads always go to the **user**
root; `artifacts/` belongs to `scripts/export_sravaani_onnx.py` and is read-only
in an installed app.

| variable | effect |
|---|---|
| `VAANI_MODELS_DIR` | replaces the user root |
| `VAANI_ARTIFACTS` | the model directory `vaani-asr` uses; its **parent** becomes the bundled root |
| `VAANI_ARTIFACTS_ROOT` | replaces the bundled root outright |
| `VAANI_MODEL_DIRS` | extra configured roots, `;`-separated |
