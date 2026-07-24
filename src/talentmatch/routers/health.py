from fastapi import APIRouter, Response
from prometheus_client import Counter, generate_latest

router = APIRouter(tags=["health"])

HEALTH_COUNTER = Counter("health_checks_total", "Total health checks")


@router.get("/health")
async def health():
    HEALTH_COUNTER.inc()
    return {"status": "ok"}


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
