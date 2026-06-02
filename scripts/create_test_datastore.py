#!/usr/bin/env python3
r"""Provision a Vertex AI Search data store from a local JSONL file.

Provisions the GCS bucket, uploads the JSONL, creates a CONTENT_REQUIRED
structured data store, imports the documents, and waits for indexing.

Supports two JSONL formats:

  Fixture format (integration tests):
      {"id": "x", "text": "...", "title": "...", "category": "..."}

  T&C ingest format (from knowledge_base/ingest.py):
      {"clause_text": "...", "section": "...", "page_number": 1, "source": "..."}

Usage
-----
    # Authenticate first (or rely on the GCE service account in CI)
    gcloud auth application-default login

    # T&C datastore (run from the repo root)
    python -m scripts.create_test_datastore \
        --bucket <globally-unique-bucket-name> \
        --jsonl ITSM-UseCase/data/processed/tnc_chunks.jsonl \
        [--project agentic-ai-evaluation-bootcamp] \
        [--datastore-id itsm-tnc-datastore]

    # Integration test datastore (original behaviour)
    python -m scripts.create_test_datastore \
        --bucket <globally-unique-bucket-name> \
        [--project agentic-ai-evaluation-bootcamp] \
        [--datastore-id vertex-search-integration-test]

After the script finishes it prints the VERTEX_AI_DATASTORE_ID value
to add to your .env file.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests


DISCOVERY_ENGINE_BASE = "https://discoveryengine.googleapis.com/v1"
STORAGE_BASE = "https://storage.googleapis.com/storage/v1"
STORAGE_UPLOAD_BASE = "https://storage.googleapis.com/upload/storage/v1"

# Default JSONL paths
_DEFAULT_FIXTURE = Path(__file__).parent.parent / "aieng-eval-agents" / "tests" / "fixtures" / "vertex_test_data.jsonl"
_DEFAULT_TNC = Path(__file__).parent.parent / "ITSM-UseCase" / "data" / "processed" / "tnc_chunks.jsonl"


def get_session() -> google.auth.transport.requests.AuthorizedSession:
    """Return an authorised requests session using Application Default Credentials."""
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return google.auth.transport.requests.AuthorizedSession(credentials)


def create_bucket(session, project: str, bucket: str) -> None:
    """Create a GCS bucket in us-central1, skipping if it already exists."""
    url = f"{STORAGE_BASE}/b?project={project}"
    body = {"name": bucket, "location": "us-central1", "storageClass": "STANDARD"}
    resp = session.post(url, json=body)
    if resp.status_code == 409:
        print(f"  Bucket gs://{bucket} already exists — skipping creation.")
    elif resp.status_code in (200, 201):
        print(f"  Created bucket gs://{bucket}")
    else:
        print(f"  Error creating bucket: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()


def transform_to_content_required(source_path: Path) -> bytes:
    """Transform a JSONL file to Discovery Engine CONTENT_REQUIRED format.

    Handles two input formats automatically:

    Fixture format:
        {"id": "x", "text": "...", "title": "...", "category": "..."}

    T&C ingest format (from knowledge_base/ingest.py):
        {"clause_text": "...", "section": "...", "page_number": 1, "source": "..."}

    Output format:
        {
            "id": "x",
            "content": {"mimeType": "text/plain", "rawBytes": "<base64>"},
            "structData": {...}
        }
    """
    if not source_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {source_path}")

    output_lines = []
    for i, raw_line in enumerate(source_path.read_text(encoding="utf-8").strip().splitlines()):
        row = json.loads(raw_line)

        if "clause_text" in row:
            # T&C ingest format — generate a sequential id
            doc_id = f"tnc-{i:04d}"
            text = row.pop("clause_text")
        else:
            # Fixture format — id and text are explicit fields
            doc_id = row.pop("id")
            text = row.pop("text", "")

        doc = {
            "id": doc_id,
            "content": {
                "mimeType": "text/plain",
                "rawBytes": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            },
            "structData": row,
        }
        output_lines.append(json.dumps(doc))

    return "\n".join(output_lines).encode("utf-8")


def upload_fixture(session, bucket: str, object_name: str, jsonl_path: Path) -> None:
    """Transform and upload a JSONL file to GCS in CONTENT_REQUIRED format."""
    payload = transform_to_content_required(jsonl_path)
    url = f"{STORAGE_UPLOAD_BASE}/b/{bucket}/o?uploadType=media&name={object_name}"
    resp = session.post(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    print(f"  Transformed and uploaded {jsonl_path.name} → gs://{bucket}/{object_name}")


def create_datastore(session, project: str, datastore_id: str, display_name: str) -> None:
    """Create a CONTENT_REQUIRED structured search data store, skipping if it exists."""
    url = (
        f"{DISCOVERY_ENGINE_BASE}/projects/{project}/locations/global"
        f"/collections/default_collection/dataStores?dataStoreId={datastore_id}"
    )
    body = {
        "displayName": display_name,
        "industryVertical": "GENERIC",
        "contentConfig": "CONTENT_REQUIRED",
        "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
    }
    resp = session.post(url, json=body)
    if resp.status_code == 409:
        print(f"  Data store '{datastore_id}' already exists — skipping creation.")
    elif resp.status_code in (200, 201):
        print(f"  Created data store '{datastore_id}'")
        # Allow a moment for the data store to become fully ready
        time.sleep(5)
    else:
        print(f"  Error creating data store: {resp.status_code} {resp.text}", file=sys.stderr)
        resp.raise_for_status()


def import_documents(session, project: str, datastore_id: str, gcs_uri: str) -> str:
    """Trigger an async document import from GCS. Returns the operation name."""
    url = (
        f"{DISCOVERY_ENGINE_BASE}/projects/{project}/locations/global"
        f"/collections/default_collection/dataStores/{datastore_id}"
        f"/branches/default_branch/documents:import"
    )
    body = {
        "gcsSource": {
            "inputUris": [gcs_uri],
            # "document" matches our JSONL format:
            # {id, content:{mimeType,rawBytes}, structData:{...}}
            "dataSchema": "document",
        },
        # FULL replaces all existing documents, keeping the test store deterministic
        "reconciliationMode": "FULL",
    }
    resp = session.post(url, json=body)
    resp.raise_for_status()
    operation_name = resp.json()["name"]
    print(f"  Import operation started: {operation_name}")
    return operation_name


def wait_for_operation(
    session,
    operation_name: str,
    timeout_sec: int = 600,
    poll_interval: int = 15,
) -> dict:
    """Poll the operation until it is done or the timeout is reached."""
    url = f"{DISCOVERY_ENGINE_BASE}/{operation_name}"
    start = time.time()
    deadline = start + timeout_sec

    while time.time() < deadline:
        resp = session.get(url)
        resp.raise_for_status()
        op = resp.json()

        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"Import operation failed: {op['error']}")
            # Check per-document failure count in metadata
            metadata = op.get("metadata", {})
            failure_count = int(metadata.get("failureCount", 0))
            total_count = int(metadata.get("totalCount", 0))
            if failure_count > 0:
                samples = op.get("response", {}).get("errorSamples", [])
                sample_msg = samples[0]["message"] if samples else "unknown error"
                raise RuntimeError(
                    f"Import completed but {failure_count}/{total_count} documents failed. First error: {sample_msg}"
                )
            print(f"  Indexing complete — {total_count} documents imported.")
            return op

        elapsed = int(time.time() - start)
        print(f"  Indexing in progress… ({elapsed}s elapsed, checking again in {poll_interval}s)")
        time.sleep(poll_interval)

    raise TimeoutError(f"Operation did not complete within {timeout_sec}s: {operation_name}")


def main() -> None:
    """Parse CLI arguments and provision the Vertex AI Search data store."""
    parser = argparse.ArgumentParser(
        description="Provision a Vertex AI Search data store from a local JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        default="agentic-ai-evaluation-bootcamp",
        help="GCP project ID (default: agentic-ai-evaluation-bootcamp)",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name for staging the import file (must be globally unique)",
    )
    parser.add_argument(
        "--datastore-id",
        default="itsm-tnc-datastore",
        help="Vertex AI Search data store ID (default: itsm-tnc-datastore)",
    )
    parser.add_argument(
        "--jsonl",
        default=str(_DEFAULT_TNC),
        help="Path to the JSONL file to upload (default: ITSM-UseCase/data/processed/tnc_chunks.jsonl)",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    gcs_object = f"itsm-tnc/{jsonl_path.name}"
    gcs_uri = f"gs://{args.bucket}/{gcs_object}"
    datastore_resource = (
        f"projects/{args.project}/locations/global/collections/default_collection/dataStores/{args.datastore_id}"
    )
    display_name = f"ITSM T&C Datastore ({args.datastore_id})"

    print("Vertex AI Search — data store provisioning")
    print("=" * 55)
    print(f"  Project:    {args.project}")
    print(f"  Bucket:     gs://{args.bucket}")
    print(f"  Data store: {datastore_resource}")
    print(f"  JSONL:      {jsonl_path}")
    print()

    session = get_session()

    print("Step 1/5  Creating GCS bucket…")
    create_bucket(session, args.project, args.bucket)

    print("Step 2/5  Uploading JSONL to GCS…")
    upload_fixture(session, args.bucket, gcs_object, jsonl_path)

    print("Step 3/5  Creating Vertex AI Search data store…")
    create_datastore(session, args.project, args.datastore_id, display_name)

    print("Step 4/5  Importing documents…")
    operation_name = import_documents(session, args.project, args.datastore_id, gcs_uri)

    print("Step 5/5  Waiting for indexing (may take several minutes)…")
    wait_for_operation(session, operation_name)

    print()
    print("=" * 55)
    print("Done! Add this to your .env file:")
    print()
    print(f'VERTEX_AI_DATASTORE_ID="{datastore_resource}"')
    print("=" * 55)


if __name__ == "__main__":
    main()
