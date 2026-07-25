"""EmailLog Beanie document for tracking sent/dry-run emails.

Stores every email attempt (live or dry_run) with recipient, subject,
body, mode, and delivery status.
"""

from datetime import datetime
from typing import Literal

from beanie import Document
from pydantic import Field


class EmailLog(Document):
    """Record of an email sent or dry-run logged during matching workflows.

    Attributes:
        recipient: Email address of the recipient.
        subject: Email subject line.
        body: Email body text.
        mode: Whether email was actually sent (live) or just logged (dry_run).
        status: Delivery status (pending, sent, failed, dry_run).
        sent_at: Timestamp of when the email was sent/logged.
    """

    recipient: str
    subject: str
    body: str
    mode: Literal["dry_run", "live"] = "dry_run"
    status: str = "pending"
    sent_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "email_log"
