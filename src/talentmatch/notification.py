"""Notification module — sends emails in live mode or logs them in dry_run.

Default mode is dry_run: emails are logged to MongoDB email_log collection
but not actually sent. Set EMAIL_MODE=live in .env to enable real sending.
"""

import logging
from datetime import datetime

from talentmatch.config import settings
from talentmatch.models.email_log import EmailLog
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.notification")


async def send_notification(
    recipient: str,
    subject: str,
    body: str,
) -> dict:
    """Send an email notification or log it as dry_run.

    In dry_run mode (default), creates an EmailLog record with status=dry_run
    without actually sending. In live mode, would integrate with an SMTP/API
    sender (not yet implemented — logs warning and marks as dry_run).

    Args:
        recipient: Email address of the recipient.
        subject: Email subject line.
        body: Email body text.

    Returns:
        Dict with recipient, subject, mode, status for logging in graph state.
    """
    mode = settings.email_mode

    log = EmailLog(
        recipient=recipient,
        subject=subject,
        body=body,
        mode=mode,
    )

    if mode == "live":
        # TODO: integrate with SMTP or email API (SendGrid, SES, etc.)
        logger.warning(
            "Live email not yet implemented, falling back to dry_run: to=%s",
            recipient,
            extra={"trace_id": get_trace_id()},
        )
        log.status = "dry_run"
        log.mode = "dry_run"
    else:
        log.status = "dry_run"
        logger.info(
            "Email dry_run: to=%s subject=%s",
            recipient,
            subject,
            extra={"trace_id": get_trace_id()},
        )

    log.sent_at = datetime.utcnow()
    await log.insert()

    return {
        "recipient": recipient,
        "subject": subject,
        "mode": log.mode,
        "status": log.status,
    }
