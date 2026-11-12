# Remote File Upload API

Transfer inferred Deep Zoom Image (DZI) tiles and thumbnails from a campus
cluster (or any remote machine) onto the CISL Viewer server, register the same
database rows the website inference worker would write, and place results in a
chosen project (including newly created experiment projects).

**Base URL (hosted):** `https://cisl-inference.beckman.illinois.edu`

---

## Overview

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/remote-token` | GET | Admin browser UI to mint/revoke tokens (easiest) |
| `/api/remote/auth/session` | POST | Mint token using website login session (no GIS JWT) |
| `/api/remote/auth` | POST | Exchange a Google ID token for a long-lived API token |
| `/api/remote/tokens` | GET | List current admin's tokens (prefix only) |
| `/api/remote/tokens/<id>` | DELETE | Revoke a token by id (website session) |
| `/api/remote/token` | DELETE | Revoke an API token (Bearer cisl-remote) |
| `/api/remote/projects` | POST | Create an empty experiment project |
| `/api/remote/upload` | POST | Upload DZI archive + thumbnail + InferenceRun metadata |

Authentication is **Google Sign-In** (Identity Services ID token), or a
server-minted **`cisl-remote`** API token obtained after Google Sign-In.

Only **authorized admin** users (same bar as `/api/inference/submit`) may use
these endpoints.

---

## Authentication

### Easiest: mint from the website (no GIS JWT)

Campus cluster is CLI-only — you never Google-sign-in there.

1. On a laptop/desktop browser, open  
   `https://cisl-inference.beckman.illinois.edu/login`  
   and sign in with Google (allowlisted **admin**).
2. Open **Campus cluster API token**:  
   `https://cisl-inference.beckman.illinois.edu/remote-token`  
   (also in the admin nav — key icon).
3. Click **Generate token**, copy the `crr_…` value (shown once).
4. On the cluster:

```bash
export CISLVIEWER_REMOTE_TOKEN='crr_…'
export CISLVIEWER_SITE='https://cisl-inference.beckman.illinois.edu'
```

That page calls `POST /api/remote/auth/session`, which uses your existing
Flask login cookie — **no Google GIS credential interception**.

### Alternative: mint via API while already logged in

```bash
# From a machine where you have a browser session cookie is awkward;
# use the /remote-token page instead. Programmatic equivalent:
curl -sS -X POST "$SITE/api/remote/auth/session" \
  -H 'Content-Type: application/json' \
  -b 'tissue_viewer_session=…' \
  -d '{"label":"campus-cluster","ttl_days":90}'
```

### Legacy: exchange a Google ID token

Only needed for non-browser automation that already has a GIS JWT:

```bash
curl -sS -X POST "$SITE/api/remote/auth" \
  -H 'Content-Type: application/json' \
  -d "{\"credential\": \"$CREDENTIAL\", \"label\": \"campus-cluster\", \"ttl_days\": 90}"
```

Response (token shown **once**):

```json
{
  "success": true,
  "token": "crr_…",
  "authorization_header": "Bearer cisl-remote crr_…",
  "token_prefix": "crr_…",
  "expires_at": "2026-12-21T00:00:00Z",
  "user": {"id": 1, "email": "you@illinois.edu", "role": "admin"},
  "note": "Store this token as CISLVIEWER_REMOTE_TOKEN. It is shown once…"
}
```

### Authorize subsequent cluster requests

```http
Authorization: Bearer cisl-remote $CISLVIEWER_REMOTE_TOKEN
```

You may also send a fresh Google ID token as `Authorization: Bearer <jwt>`
(no `cisl-remote` prefix). That is convenient for short scripts, not long jobs.

### Revoke a token

From the `/remote-token` page (Revoke button), or:

```bash
curl -sS -X DELETE "$SITE/api/remote/token" \
  -H "Authorization: Bearer cisl-remote $CISLVIEWER_REMOTE_TOKEN"
```

Or by id while logged into the website: `DELETE /api/remote/tokens/<id>`.

### Obtaining a Google ID token for scripts (optional)

Prefer `/remote-token` above. If you still need a raw GIS JWT:

1. **Browser helper** — sign in on the viewer login page, then use a small
   page/console snippet that prints `credential` from Google Identity Services
   and POST it to `/api/remote/auth` (then discard the JWT; keep only the
   `crr_` token).
2. **Installed-app / device OAuth** — use a Google OAuth client that can obtain
   an ID token whose `aud` matches the server’s `GOOGLE_CLIENT_ID`. The server
   verifies audience strictly; mismatched client IDs are rejected.

The server never accepts a shared static password for this API.

---

## Create a project

Experiments should usually land in a **new project** (not `main`).

```bash
curl -sS -X POST "$SITE/api/remote/projects" \
  -H "Authorization: Bearer cisl-remote $CISLVIEWER_REMOTE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "campus_exp_2026q3",
    "display_name": "Campus Exp 2026 Q3",
    "description": "Cluster FP16 runs"
  }'
```

Project names: lowercase `a-z`, `0-9`, `_`, `-` only. Reserved names
(`api`, `dzi`, `login`, …) are rejected.

Alternatively, set `"create_project": true` on upload (see below) to create the
project on first upload.

---

## Cluster client: prefer infer → DZI → upload (no PNG)

Campus jobs should **not** write a whole-slide PNG/JPEG only to re-encode it as
DZI. The viewer already serves Deep Zoom tiles; the hot path matches the
inference-server flow (`uint8` HWC canvas → `pyvips.Image.new_from_memory` →
`dzsave`).

| Mode | Entry | When to use |
|------|-------|-------------|
| **Preferred** | `REMOTE_UPLOAD=1 sbatch jobs/cislnet/wsi/wsi.sh` → `scripts/remote_wsi_upload.py --infer_cislnet` | New WSI runs: GPU stitch → DZI → `/api/remote/upload` |
| Legacy | `jobs/cislnet/wsi/remote_upload_vs13.sh` → `--wsi_pred_dir` | Re-upload existing `results/.../wsi_pred` PNGs |

Shared writers live in [`util/dzi_write.py`](util/dzi_write.py):
`write_dzi_from_array` (hot path) and `write_dzi_from_file` (legacy).

Example (direct, same as `REMOTE_UPLOAD=1` on `wsi.sh`):

```bash
source /u/akshahi2/.secrets.sh
python scripts/remote_wsi_upload.py \
  --infer_cislnet \
  --name c13 \
  --checkpoints_dir results/cislnet \
  --which_epoch 20 \
  --parent_folder input/IR \
  --input_nc 13 --num_ir_channels 10 \
  --project campus_c13_wsi \
  --output_type CISLNet-c13-e20 \
  --model 1 \
  --create_project --ensure_project --overwrite
```

---

## Upload inferred DZI + thumbnail

### Multipart fields

| Field | Required | Description |
|-------|----------|-------------|
| `metadata` | yes | JSON string (or form field) with InferenceRun + placement fields |
| `dzi_archive` | yes | `.tar`, `.tar.gz` / `.tgz`, or `.zip` containing `{sample}.dzi` and `{sample}_files/` |
| `thumbnail` | yes | JPEG thumbnail (`{sample}.jpg`) |

### Archive layout

Either flat:

```text
6233-1.dzi
6233-1_files/
  0/
  1/
  …
```

or wrapped in one top-level folder:

```text
bundle/
  6233-1.dzi
  6233-1_files/
    …
```

On the server these become:

```text
/mnt/files/<project>/dzi/<output_type>/<sample_id>.dzi
/mnt/files/<project>/dzi/<output_type>/<sample_id>_files/…
/mnt/files/<project>/thumbnail/<output_type>/<sample_id>.jpg
```

This matches the layout produced by local `inference-project` `create_dzi()`.

### Metadata fields

Fields mirror what the website inference pipeline records on `InferenceRun`
(and what `_register_completed_run_available` needs for the viewer).

| Field | Required | Notes |
|-------|----------|-------|
| `sample_id` | **yes** | Basename used for DZI / thumbnail files |
| `output_type` | **yes** | Modality folder, e.g. `IR(VS)`, `IR+VIS(VS)`, custom tag |
| `project_name` | no | Default `main`; prefer a dedicated experiment project |
| `create_project` | no | If true, create missing non-`main` project |
| `project_display_name` | no | Used when creating a project |
| `project_description` | no | Used when creating a project |
| `overwrite` | no | Replace existing DZI/thumbnail/run with same ids |
| `source_path` | no | Cluster path string (stored for history); default `remote://{sample_id}` |
| `run_type` | no | Default `inference` (`convert_ir`, `convert_tiff`, `convert_incoming`, `remote_upload` also allowed) |
| `input_type` | no | `IR` / `VIS` / `HE` (default `IR`) |
| `file_format` | no | Default `.bsq` |
| `model_type` | no | `0` (10ch) or `1` (13ch) |
| `use_tensorrt` | no | bool |
| `tensorrt_precision` | no | `fp16` / `fp32` / null |
| `bsq_load_time` | no | seconds (float) |
| `inference_time` | no | seconds |
| `write_time` | no | seconds |
| `total_time` | no | seconds |
| `progress_message` | no | default remote-completed message |
| `error_message` | no | optional text |
| `terminal_output` | no | optional stdout capture (size-capped) |
| `started_at` / `completed_at` | no | ISO-8601 (e.g. `2026-09-22T18:00:00Z`) |
| `run_uuid` | no | Client-supplied UUID; server generates one if omitted |
| `resolution` / `dimensions` | no | Optional; otherwise parsed from the DZI XML |

Successful uploads always create/update an `InferenceRun` with
`status=completed`, register/update `AvailableImage`, and sync the `Project`
sample list (same outcome as a successful local worker finalize).

### Example: pack and upload

```bash
SAMPLE=6233-1
OUT_TYPE='IR(VS)'
PROJECT=campus_exp_2026q3

# After cluster inference wrote local dzi/thumb next to the sample:
tar -czf "${SAMPLE}_dzi.tar.gz" -C /path/to/out \
  "${SAMPLE}.dzi" "${SAMPLE}_files"

META=$(cat <<EOF
{
  "sample_id": "$SAMPLE",
  "project_name": "$PROJECT",
  "create_project": true,
  "output_type": "$OUT_TYPE",
  "source_path": "/scratch/you/samples/$SAMPLE",
  "run_type": "inference",
  "input_type": "IR",
  "file_format": ".bsq",
  "model_type": 0,
  "use_tensorrt": true,
  "tensorrt_precision": "fp16",
  "bsq_load_time": 12.4,
  "inference_time": 88.1,
  "write_time": 9.2,
  "total_time": 109.7,
  "progress_message": "Completed successfully",
  "terminal_output": "…optional…",
  "overwrite": false
}
EOF
)

curl -sS -X POST "$CISLVIEWER_SITE/api/remote/upload" \
  -H "Authorization: Bearer cisl-remote $CISLVIEWER_REMOTE_TOKEN" \
  -F "metadata=$META;type=application/json" \
  -F "dzi_archive=@${SAMPLE}_dzi.tar.gz" \
  -F "thumbnail=@/path/to/out/${SAMPLE}.jpg;type=image/jpeg"
```

Success response includes `run`, `viewer_url`, and `thumbnail_url`.

### Limits

| Resource | Limit |
|----------|-------|
| DZI archive | 2 GiB |
| Unpacked archive | 3 GiB / 200k members |
| Thumbnail | 10 MiB JPEG |
| Token TTL | 1–365 days (default 90) |

---

## Security audit (pre-exposure)

Performed against the implementation in `viewer/remote_upload.py` and the
`/api/remote/*` routes in `viewer/app.py` before documenting public use.

### Threat model

| Threat | Mitigation |
|--------|------------|
| Unauthenticated upload | Google JWT signature + audience check, or hashed `cisl-remote` token |
| Non-allowlisted Google account | Same allowlist as `/login` (`User.authorized` / bootstrap admin) |
| Guest / non-admin upload | Requires `role=admin` (parity with `/api/inference/submit`) |
| Stolen browser session alone | Endpoints do not rely on session cookies; Bearer token required |
| Token leakage | Token stored as SHA-256 only; revoke via DELETE; expiry enforced |
| Zip-slip / path traversal | Archive members validated; `..`, absolute paths, symlinks/hardlinks rejected |
| Absolute path members rewriting to relative | Rejected before `lstrip` (regression covered by unit test) |
| Overwriting unrelated projects | `project_name` / `sample_id` / `output_type` sanitized; reserved names blocked |
| Oversized DoS | Content-length check + streaming size caps on archive and thumbnail |
| Non-image “thumbnail” | JPEG magic-byte check |
| Status spoofing to hijack queue | Upload path forces `status=completed`; does not enqueue local GPU work |
| Audit gaps | `CREATE`/`UPDATE` audit logs on tokens, projects, and inference runs; `login_user` so `user_id` is recorded |

### Residual risks / operator checklist

1. **Treat `crr_` tokens like passwords** — store in cluster secrets (not in git or world-readable home files). Revoke on machine compromise.
2. **TLS only** — call the HTTPS hosted site; do not send tokens over plain HTTP.
3. **Admin scope** — any admin token can write to any project name the API allows. Prefer dedicated experiment projects and rotate tokens per campaign.
4. **No per-IP rate limit** in-app yet — rely on nginx / fail2ban if abuse appears; archive size caps bound disk fill rate per request.
5. **CORS is wide on the Flask app** — acceptable because auth is Bearer (not cookies), but do not embed API tokens in front-end JavaScript on untrusted pages.
6. **Google Client ID** is public by design; authorization is allowlist + role, not Client ID secrecy (see `SECURITY.md`).

### Verdict

**Safe to expose** on the hosted HTTPS site to allowlisted admins, provided
tokens are handled as secrets and nginx continues to terminate TLS. Unit tests
in `viewer/test_remote_upload.py` cover auth header parsing, metadata
validation, zip-slip rejection, symlink policy, and JPEG checks.

---

## Implementation map

| Piece | Location |
|-------|----------|
| Cluster client (infer→DZI→upload) | `scripts/remote_wsi_upload.py` (`--infer_cislnet`) |
| DZI writers (array + file) | `util/dzi_write.py` |
| Slurm: remote path (no PNG) | `jobs/cislnet/wsi/wsi.sh` with `REMOTE_UPLOAD=1` |
| Slurm: legacy pred→DZI | `jobs/cislnet/wsi/remote_upload_vs13.sh` |
| Helpers (auth parse, archive safety, metadata) | `viewer/remote_upload.py` (hosted site) |
| `RemoteApiToken` model + routes | `viewer/app.py` |
| Tests | `viewer/test_remote_upload.py` |
| Local finalize analogue | `_register_completed_run_available` / inference worker `_finalize` |

After upload, open `/{project_name}` on the viewer to browse the new sample.
