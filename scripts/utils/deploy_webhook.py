"""
Lightweight GitHub webhook receiver for automated CI/CD deployment.
Listens on port 9000, validates webhook signatures, and triggers deploy.
"""

import hashlib
import hmac
import logging
import os
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ── Config ───────────────────────────────────────────────────────

ENV_FILE = Path("/opt/trading-desk/.env.webhook")
REPO_DIR = "/opt/trading-desk"
DEPLOY_SCRIPT = "/opt/trading-desk/scripts/utils/deploy.sh"

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/opt/trading-desk/logs/deploy_webhook.log"),
    ],
)
logger = logging.getLogger("deploy_webhook")

# ── Load secrets ─────────────────────────────────────────────────


def _load_env() -> dict[str, str]:
    """Load key=value pairs from .env.webhook file."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip("'\"")
    return env


_env = _load_env()
WEBHOOK_SECRET = _env.get("GITHUB_WEBHOOK_SECRET", "")
DEPLOY_TOKEN = _env.get("DEPLOY_TOKEN", "")

if not WEBHOOK_SECRET:
    logger.warning("GITHUB_WEBHOOK_SECRET not set in %s", ENV_FILE)

# ── App ──────────────────────────────────────────────────────────

app = FastAPI(title="Deploy Webhook", docs_url=None, redoc_url=None)

_last_deploy: dict = {"time": None, "status": None, "trigger": None}


def _verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub HMAC SHA256 webhook signature."""
    if not WEBHOOK_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _run_deploy(trigger: str) -> dict:
    """Pull latest code and run deploy script."""
    logger.info("Deploy triggered by: %s", trigger)
    results = {"trigger": trigger, "steps": []}

    # Git pull
    try:
        pull_result = subprocess.run(
            ["git", "-C", REPO_DIR, "pull", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        results["steps"].append({
            "step": "git_pull",
            "returncode": pull_result.returncode,
            "stdout": pull_result.stdout.strip(),
            "stderr": pull_result.stderr.strip(),
        })
        logger.info("git pull: rc=%d %s", pull_result.returncode, pull_result.stdout.strip())
        if pull_result.returncode != 0:
            logger.error("git pull failed: %s", pull_result.stderr.strip())
            results["status"] = "failed"
            _last_deploy.update(time=time.time(), status="failed", trigger=trigger)
            return results
    except subprocess.TimeoutExpired:
        logger.error("git pull timed out")
        results["steps"].append({"step": "git_pull", "error": "timeout"})
        results["status"] = "failed"
        _last_deploy.update(time=time.time(), status="failed", trigger=trigger)
        return results

    # Deploy script
    try:
        deploy_result = subprocess.run(
            ["bash", DEPLOY_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        results["steps"].append({
            "step": "deploy_script",
            "returncode": deploy_result.returncode,
            "stdout": deploy_result.stdout.strip()[-500:],
            "stderr": deploy_result.stderr.strip()[-500:],
        })
        logger.info("deploy.sh: rc=%d", deploy_result.returncode)
        if deploy_result.returncode != 0:
            logger.error("deploy.sh failed: %s", deploy_result.stderr.strip()[-200:])
    except subprocess.TimeoutExpired:
        logger.error("deploy.sh timed out")
        results["steps"].append({"step": "deploy_script", "error": "timeout"})

    results["status"] = "success" if all(
        s.get("returncode", 1) == 0 for s in results["steps"]
    ) else "partial"
    _last_deploy.update(time=time.time(), status=results["status"], trigger=trigger)
    logger.info("Deploy complete: %s", results["status"])
    return results


# ── Routes ───────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "deploy-webhook",
        "last_deploy": _last_deploy,
    }


@app.post("/webhook")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
    x_github_event: str = Header(None, alias="X-GitHub-Event"),
):
    body = await request.body()
    logger.info("Webhook received: event=%s", x_github_event)

    # Validate signature
    if not x_hub_signature_256:
        logger.warning("Missing webhook signature")
        raise HTTPException(status_code=401, detail="Missing signature")

    if not _verify_signature(body, x_hub_signature_256):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Only deploy on push to main
    if x_github_event != "push":
        logger.info("Ignoring event: %s", x_github_event)
        return {"status": "ignored", "event": x_github_event}

    payload = await request.json()
    ref = payload.get("ref", "")
    if ref != "refs/heads/main":
        logger.info("Ignoring push to %s", ref)
        return {"status": "ignored", "ref": ref}

    pusher = payload.get("pusher", {}).get("name", "unknown")
    logger.info("Push to main by %s, deploying...", pusher)

    result = _run_deploy(trigger=f"github-push:{pusher}")
    return result


@app.post("/deploy")
async def manual_deploy(
    request: Request,
    authorization: str = Header(None),
):
    if not DEPLOY_TOKEN:
        raise HTTPException(status_code=503, detail="Deploy token not configured")

    token = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, DEPLOY_TOKEN):
        logger.warning("Invalid deploy token from %s", request.client.host)
        raise HTTPException(status_code=401, detail="Invalid token")

    logger.info("Manual deploy triggered from %s", request.client.host)
    result = _run_deploy(trigger=f"manual:{request.client.host}")
    return result


# ── Entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
