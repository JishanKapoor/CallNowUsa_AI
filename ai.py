from __future__ import annotations

import json
import logging
import os
import re
import signal
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pytz
from dateutil.parser import parse as parse_datetime
from openai import AzureOpenAI
from callnowusa import Client as CallNowUSAClient
from dotenv import load_dotenv
load_dotenv()

# The rest of your imports and code
import os
# Environment variables and defaults
CALLNOWUSA_NUMBER: "default"
ACCOUNT_SID: str = "SID_d5cf1823-5664-42cc-b6b6-fb10bcdaec56"
AUTH_TOKEN: str = "AUTH_aaa784bc-a599-499f-946b-ba7115c59726"
AZURE_API_KEY: str = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT: str = os.getenv("AZURE_ENDPOINT")

TIMEZONE: str = os.getenv("ASSISTANT_TZ", "US/Eastern")
TZ = pytz.timezone(TIMEZONE)

DB_PATH: Path = Path("contacts.db")
POLL_INTERVAL_SEC: int = 30
LLM_MODEL: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Supported labels for message classification
LABELS = [
    "horny", "funny", "meeting", "business", "personal", "romantic",
    "scary", "flirty", "angry", "high priority", "medium priority", "low priority"
]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sms_assistant")

@dataclass
class Message:
    """Represents an SMS message with sender, body, timestamp, direction, and label."""
    sender: str
    body: str
    timestamp: datetime
    direction: str
    label: str = "other"

    @classmethod
    def from_raw(
        cls, sender: str, body: str, ts_text: str, direction: str, label: str = "other"
    ) -> "Message":
        """Create a Message from raw data, parsing timestamp and applying timezone."""
        ts = parse_datetime(ts_text)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=TZ)
        return cls(sender, body.strip(), ts, direction, label)

@dataclass
class Rule:
    """Represents a rule for automatic SMS actions (e.g., reply, forward)."""
    id_: int
    from_contact: Optional[str]
    from_number: Optional[str]
    action_type: str
    reply_message: Optional[str]
    forward_to_contact: Optional[str]
    forward_to_number: Optional[str]
    condition: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    timeout: Optional[int]
    is_one_time: bool
    created_at: datetime

    def active_now(self, now: datetime) -> bool:
        """Check if the rule is active based on start and end times."""
        if self.start_time and self.end_time:
            s = datetime.combine(now.date(), datetime.strptime(self.start_time, "%H:%M").time()).replace(tzinfo=TZ)
            e = datetime.combine(now.date(), datetime.strptime(self.end_time, "%H:%M").time()).replace(tzinfo=TZ)
            if e < s:
                e += timedelta(days=1)
            return s <= now <= e
        return True

@dataclass
class Condition:
    """Represents a conditional SMS forwarding rule."""
    id_: int
    type_: str  # e.g., "priority", "sender", "keyword"
    value: str  # e.g., "high priority", "+1234567890", "meeting"
    action: str  # e.g., "forward"
    target: str  # e.g., phone number or alias
    start_time: Optional[str]  # e.g., "21:00"
    end_time: Optional[str]  # e.g., "01:00"
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(TZ))

    def matches(self, message: Message, now: datetime) -> bool:
        """Check if a message matches this condition and is within time range."""
        if self.start_time and self.end_time:
            try:
                s = datetime.combine(now.date(), datetime.strptime(self.start_time, "%H:%M").time()).replace(tzinfo=TZ)
                e = datetime.combine(now.date(), datetime.strptime(self.end_time, "%H:%M").time()).replace(tzinfo=TZ)
                if e < s:
                    e += timedelta(days=1)
                if not (s <= now <= e):
                    return False
            except ValueError as e:
                logger.error("Error parsing condition times: %s", e)
                return False
        if self.type_ == "priority":
            return message.label == self.value
        elif self.type_ == "sender":
            return message.sender == self.value
        elif self.type_ == "keyword":
            return self.value.lower() in message.body.lower()
        return False

@dataclass
class ScheduledSMS:
    """Represents a scheduled SMS to be sent at a specific time or after a delay."""
    id_: int
    recipient: str
    message: str
    send_time: datetime
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(TZ))

class SQLiteStore:
    """Manages persistent storage for contacts, rules, messages, conditions, and scheduled SMS."""
    _lock = threading.RLock()

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("SQLite schema ready at %s", db_path)

    def _init_schema(self) -> None:
        """Initialize or update database schema."""
        with self.conn:
            # Create tables if they don't exist
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS contacts (
                    alias TEXT PRIMARY KEY,
                    phone_number TEXT
                );
                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_contact TEXT,
                    from_number TEXT,
                    action_type TEXT,
                    reply_message TEXT,
                    forward_to_contact TEXT,
                    forward_to_number TEXT,
                    condition TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    timeout INTEGER,
                    is_one_time INTEGER,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS message_log (
                    sender TEXT,
                    message TEXT,
                    timestamp TEXT,
                    direction TEXT,
                    replied INTEGER,
                    label TEXT DEFAULT 'other'
                );
                CREATE TABLE IF NOT EXISTS conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    value TEXT,
                    action TEXT,
                    target TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS scheduled_sms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recipient TEXT,
                    message TEXT,
                    send_time TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT
                );
                """
            )
            # Check for missing columns in conditions table and add them
            cur = self.conn.execute("PRAGMA table_info(conditions)")
            columns = [row["name"] for row in cur.fetchall()]
            if "start_time" not in columns:
                self.conn.execute("ALTER TABLE conditions ADD COLUMN start_time TEXT")
            if "end_time" not in columns:
                self.conn.execute("ALTER TABLE conditions ADD COLUMN end_time TEXT")

    def add_contact(self, alias: str, number: str) -> None:
        """Add or update a contact."""
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO contacts (alias, phone_number) VALUES (?, ?)",
                (alias.lower(), number),
            )

    def delete_contact(self, alias: str) -> bool:
        """Delete a contact."""
        with self._lock, self.conn:
            cur = self.conn.execute("DELETE FROM contacts WHERE alias = ?", (alias.lower(),))
            return cur.rowcount > 0

    def resolve(self, alias: str) -> Optional[str]:
        """Resolve an alias to a phone number."""
        alias = alias.strip().lower()
        cur = self.conn.execute("SELECT phone_number FROM contacts WHERE alias = ?", (alias,))
        row = cur.fetchone()
        return row[0] if row else None

    def update_contact(self, alias: str, new_number: str) -> bool:
        """Update a contact's phone number."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "UPDATE contacts SET phone_number = ? WHERE alias = ?",
                (new_number, alias.lower()),
            )
            return cur.rowcount > 0

    def list_contacts(self) -> List[Tuple[str, str]]:
        """List all contacts."""
        cur = self.conn.execute("SELECT alias, phone_number FROM contacts ORDER BY alias")
        return [(r["alias"], r["phone_number"]) for r in cur.fetchall()]

    def add_rule(self, **kwargs: Any) -> int:
        """Add a rule."""
        cols = (
            "from_contact, from_number, action_type, reply_message, forward_to_contact, forward_to_number, "
            "condition, start_time, end_time, timeout, is_one_time, created_at"
        ).split(", ")
        vals = [kwargs.get(c) for c in cols]
        with self._lock, self.conn:
            cur = self.conn.execute(
                f"INSERT INTO rules ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})",
                vals,
            )
            return cur.lastrowid

    def list_rules(self) -> List[Rule]:
        """List all active rules."""
        cur = self.conn.execute("SELECT * FROM rules ORDER BY id")
        return [self._row_to_rule(r) for r in cur.fetchall()]

    def delete_rule(self, rid: Optional[int] = None):
        """Delete a rule or all rules."""
        with self._lock, self.conn:
            if rid is None:
                self.conn.execute("DELETE FROM rules")
            else:
                self.conn.execute("DELETE FROM rules WHERE id = ?", (rid,))

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> Rule:
        """Convert a database row to a Rule object."""
        return Rule(
            id_=row["id"],
            from_contact=row["from_contact"],
            from_number=row["from_number"],
            action_type=row["action_type"],
            reply_message=row["reply_message"],
            forward_to_contact=row["forward_to_contact"],
            forward_to_number=row["forward_to_number"],
            condition=row["condition"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            timeout=row["timeout"],
            is_one_time=bool(row["is_one_time"]),
            created_at=parse_datetime(row["created_at"]),
        )

    def add_condition(self, type_: str, value: str, action: str, target: str, start_time: Optional[str] = None,
                     end_time: Optional[str] = None) -> int:
        """Add a conditional forwarding rule."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO conditions (type, value, action, target, start_time, end_time, active, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (type_, value, action, target, start_time, end_time, datetime.now(TZ).isoformat()),
            )
            return cur.lastrowid

    def list_conditions(self) -> List[Condition]:
        """List all active conditional forwarding rules."""
        cur = self.conn.execute(
            "SELECT id, type, value, action, target, start_time, end_time, active, created_at FROM conditions WHERE active = 1 ORDER BY id")
        return [
            Condition(
                id_=row["id"],
                type_=row["type"],
                value=row["value"],
                action=row["action"],
                target=row["target"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                active=bool(row["active"]),
                created_at=parse_datetime(row["created_at"])
            )
            for row in cur.fetchall()
        ]

    def stop_condition(self, condition_id: int) -> bool:
        """Deactivate a conditional forwarding rule."""
        with self._lock, self.conn:
            cur = self.conn.execute("UPDATE conditions SET active = 0 WHERE id = ?", (condition_id,))
            return cur.rowcount > 0

    def add_scheduled_sms(self, recipient: str, message: str, send_time: datetime) -> int:
        """Add a scheduled SMS."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO scheduled_sms (recipient, message, send_time, active, created_at) VALUES (?, ?, ?, 1, ?)",
                (recipient, message, send_time.isoformat(), datetime.now(TZ).isoformat()),
            )
            return cur.lastrowid

    def list_scheduled_sms(self) -> List[ScheduledSMS]:
        """List all active scheduled SMS."""
        cur = self.conn.execute(
            "SELECT id, recipient, message, send_time, active, created_at FROM scheduled_sms WHERE active = 1 ORDER BY send_time")
        return [
            ScheduledSMS(
                id_=row["id"],
                recipient=row["recipient"],
                message=row["message"],
                send_time=parse_datetime(row["send_time"]),
                active=bool(row["active"]),
                created_at=parse_datetime(row["created_at"])
            )
            for row in cur.fetchall()
        ]

    def mark_scheduled_sms_sent(self, sms_id: int) -> bool:
        """Mark a scheduled SMS as sent (deactivate it)."""
        with self._lock, self.conn:
            cur = self.conn.execute("UPDATE scheduled_sms SET active = 0 WHERE id = ?", (sms_id,))
            return cur.rowcount > 0

    def log_msg(self, sender: str, message: str, ts: datetime, direction: str, label: str = "other"):
        """Log a message to the database."""
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO message_log (sender, message, timestamp, direction, replied, label) VALUES (?, ?, ?, ?, 0, ?)",
                (sender, message, ts.isoformat(), direction, label)
            )

    def mark_replied(self, sender: str, ts: datetime):
        """Mark a message as replied."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE message_log SET replied = 1 WHERE sender = ? AND timestamp = ?",
                (sender, ts.isoformat()),
            )

    def was_replied(self, sender: str, ts: datetime) -> bool:
        """Check if a message was replied to."""
        cur = self.conn.execute(
            "SELECT replied FROM message_log WHERE sender = ? AND timestamp = ?",
            (sender, ts.isoformat()),
        )
        row = cur.fetchone()
        return bool(row and row[0])

    def get_message(self, sender: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific message."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "SELECT sender, message, timestamp, direction, replied, label FROM message_log WHERE sender = ? AND timestamp = ?",
                (sender, timestamp)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_recent_messages(self, limit: int = 100) -> List[Message]:
        """Retrieve recent messages."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "SELECT sender, message, timestamp, direction, replied, label FROM message_log "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            messages = []
            for row in cur.fetchall():
                try:
                    ts = parse_datetime(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=TZ)
                    messages.append(
                        Message(
                            sender=row["sender"],
                            body=row["message"],
                            timestamp=ts,
                            direction=row["direction"],
                            label=row["label"]
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to parse message: %s", e)
            return messages

callnow_client = CallNowUSAClient(
    account_sid=ACCOUNT_SID,
    auth_token=AUTH_TOKEN,
    phone_number=CALLNOWUSA_NUMBER,
)

openai_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-02-15-preview",
)

PHONE_RE = re.compile(r"\+\d{10,15}")

def normalize_phone_number(num: Optional[str]) -> Optional[str]:
    """Normalize phone numbers to include +1 for US numbers if missing."""
    if not num:
        return None
    num = num.strip()
    if num.startswith("+"):
        return num
    if len(num) == 10 and num.isdigit():
        return f"+1{num}"
    return num

def is_valid_number(num: Optional[str]) -> bool:
    """Validate phone numbers."""
    if not num:
        return False
    num = normalize_phone_number(num)
    if not num:
        return False
    if num.startswith("+1"):
        return len(num) == 12 and num[2:].isdigit()
    elif num.startswith("+"):
        return 11 <= len(num) <= 16 and num[1:].isdigit()
    return False

def parse_time(time_str: str, base_date: datetime) -> datetime:
    """Parse a time string (e.g., '7 am' or '30 seconds later') into a timezone-aware datetime object."""
    time_str = time_str.lower().strip()
    logger.debug("Parsing time string: '%s' with base_date: %s", time_str, base_date)
    try:
        # Ensure base_date is timezone-aware
        if base_date.tzinfo is None:
            base_date = TZ.localize(base_date)

        # Handle delay formats (e.g., '30 seconds later', 'in the next 1 minute')
        delay_match = re.match(r"^(?:in\s*the\s*next\s*)?(\d+)\s*(second|minute|hour)s?\s*(?:later)?$", time_str)
        if delay_match:
            amount, unit = delay_match.groups()
            amount = int(amount)
            logger.debug("Matched delay: amount=%d, unit=%s", amount, unit)
            if unit == "second":
                result = base_date + timedelta(seconds=amount)
            elif unit == "minute":
                result = base_date + timedelta(minutes=amount)
            elif unit == "hour":
                result = base_date + timedelta(hours=amount)
            else:
                raise ValueError(f"Invalid time unit: {unit}")
            return result.astimezone(TZ)

        # Handle relative dates like "tomorrow" or "next week"
        relative_match = re.match(r"^(tomorrow|next week)(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*(am|pm)?))?$", time_str)
        if relative_match:
            relative, time_part, period = relative_match.groups()
            logger.debug("Matched relative date: %s, time_part=%s, period=%s", relative, time_part, period)
            base = base_date.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(TZ)
            if relative == "tomorrow":
                base = base + timedelta(days=1)
            else:  # next week
                base = base + timedelta(days=(7 - base_date.weekday()))
            if time_part:
                time_match = re.match(r"(\d{1,2})(?::(\d{2}))?", time_part)
                if time_match:
                    hour, minute = time_match.groups()
                    hour = int(hour)
                    minute = int(minute) if minute else 0
                    if period:
                        if period == "pm" and hour != 12:
                            hour += 12
                        elif period == "am" and hour == 12:
                            hour = 0
                    if not (0 <= hour <= 23):
                        raise ValueError(f"Invalid hour: {hour} (must be 0-23)")
                    if not (0 <= minute <= 59):
                        raise ValueError(f"Invalid minute: {minute} (must be 0-59)")
                    base = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return base.astimezone(TZ)

        # Handle specific times like "7 am" or "12:04 pm"
        time_match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", time_str)
        if time_match:
            hour, minute, period = time_match.groups()
            hour = int(hour)
            minute = int(minute) if minute else 0
            logger.debug("Matched time: hour=%d, minute=%d, period=%s", hour, minute, period)
            if period:
                if period == "pm" and hour != 12:
                    hour += 12
                elif period == "am" and hour == 12:
                    hour = 0
            if not (0 <= hour <= 23):
                raise ValueError(f"Invalid hour: {hour} (must be 0-23)")
            if not (0 <= minute <= 59):
                raise ValueError(f"Invalid minute: {minute} (must be 0-59)")
            # Use today's date with the specified time, adjusted for TZ
            today = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
            result = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If the time is earlier than or equal to now, assume it's for tomorrow
            if result <= datetime.now(TZ):
                result += timedelta(days=1)
            return result.astimezone(TZ)

        raise ValueError(f"Invalid time format: {time_str}")
    except Exception as e:
        logger.error("Failed to parse time '%s': %s", time_str, e)
        raise

class Classifier:
    """Classifies SMS messages into priority levels and other categories."""
    _priority_prompt = (
        "Classify this SMS as one of: high priority, medium priority, low priority.\n\n"
        "- High priority: Urgent, critical, or time-sensitive messages (e.g., 'ASAP', 'emergency', scheduled events, meetings, or tasks).\n"
        "- Medium priority: Casual or non-urgent messages (e.g., 'hey', 'hi', 'wanna go out tonight').\n"
        "- Low priority: Scam messages with suspicious intent (e.g., 'free', 'win', 'prize', phishing links).\n\n"
        "Return JSON: {\"label\": <label>} only."
    )

    _other_prompt = (
        "Classify this SMS into one of the following labels: "
        + ", ".join([label for label in LABELS if label not in ["high priority", "medium priority", "low priority"]])
        + ". Return JSON: {\"label\": <label>} only."
    )

    def priority_label(self, body: str) -> str:
        """Classify message priority based on content."""
        body_lower = body.lower()
        scam_keywords = [
            "win", "free", "prize", "click here", "urgent action",
            "verify your account", "http", ".com", "claim now"
        ]
        if any(keyword in body_lower for keyword in scam_keywords):
            return "low priority"
        high_priority_keywords = [
            "urgent", "asap", "critical", "emergency", "immediately",
            "meeting", "schedule", "appointment", "deadline"
        ]
        if any(keyword in body_lower for keyword in high_priority_keywords):
            return "high priority"
        return "medium priority"

    def other_label(self, body: str) -> str:
        """Classify non-priority labels."""
        body_lower = body.lower()
        if any(keyword in body_lower for keyword in ["meeting", "schedule", "appointment"]) or \
                ("urgent" in body_lower and "text me" in body_lower):
            return "meeting"
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._other_prompt},
                    {"role": "user", "content": body[:500]},
                ],
                temperature=0,
            )
            data = json.loads(response.choices[0].message.content.strip().replace("'", '"'))
            label = data.get("label", "other")
            valid_other_labels = [
                label for label in LABELS
                if label not in ["high priority", "medium priority", "low priority"]
            ]
            return label if label in valid_other_labels else "other"
        except Exception as exc:
            logger.debug("Other classifier error: %s", exc)
            return "other"

classifier = Classifier()

class InboxMonitor(threading.Thread):
    """Monitors the inbox and handles scheduled SMS."""
    def __init__(self, store: SQLiteStore, interval: int = POLL_INTERVAL_SEC):
        super().__init__(daemon=True)
        self.store = store
        self.interval = int(interval)
        self._cache: List[Message] = []
        self._last_fetch: Optional[datetime] = None
        self._task_lock = threading.Lock()
        self._scheduled: Dict[int, List[threading.Timer]] = defaultdict(list)
        self._scheduled_sms_timers: Dict[int, threading.Timer] = {}
        self._stop_event = threading.Event()

    def stop(self):
        """Stop the monitor thread and cancel scheduled SMS timers."""
        self._stop_event.set()
        for timer in self._scheduled_sms_timers.values():
            timer.cancel()
        self._scheduled_sms_timers.clear()

    def run(self):
        """Run the monitor loop."""
        logger.info("InboxMonitor running every %d s", int(self.interval))
        self._schedule_pending_sms()
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._poll()
            except Exception:
                logger.exception("Error in monitor loop")
            dt = time.perf_counter() - t0
            time.sleep(max(0, int(self.interval - dt)))

    def _poll(self):
        """Poll the inbox and process new messages."""
        raw = callnow_client.check_inbox(from_=CALLNOWUSA_NUMBER).fetch().get("status", "")
        logger.info("Raw inbox data: %s", raw)
        messages = self._parse_messages(raw)
        new_messages = []
        for m in messages:
            existing = self.store.get_message(m.sender, m.timestamp.isoformat())
            if existing:
                m.label = existing["label"]
                non_priority_label = classifier.other_label(m.body)
                if non_priority_label != "other":
                    m.label = non_priority_label
            else:
                m.label = classifier.priority_label(m.body)
                self.store.log_msg(m.sender, m.body, m.timestamp, m.direction, m.label)
                new_messages.append(m)
        self._cache = messages
        self._last_fetch = datetime.now(TZ)
        logger.info("Fetched %d messages (%d new)", len(messages), len(new_messages))
        self._apply_rules(new_messages)
        self._apply_conditions(new_messages)

    def _apply_rules(self, messages: List[Message]):
        """Apply existing rules to messages."""
        rules = self.store.list_rules()
        now = datetime.now(TZ)
        for msg in messages:
            if msg.direction != "received":
                continue
            for rule in rules:
                if not rule.active_now(now):
                    continue
                if rule.condition and rule.condition.lower() not in msg.body.lower():
                    continue
                sender_match = False
                if rule.from_number and rule.from_number == msg.sender:
                    sender_match = True
                elif rule.from_contact and self.store.resolve(rule.from_contact) == msg.sender:
                    sender_match = True
                if not sender_match:
                    continue
                if rule.timeout:
                    self._schedule(rule, msg)
                else:
                    self._execute(rule, msg)

    def _apply_conditions(self, messages: List[Message]):
        """Apply conditional forwarding rules to new messages."""
        conditions = self.store.list_conditions()
        now = datetime.now(TZ)
        for msg in messages:
            if msg.direction != "received":
                continue
            for condition in conditions:
                if condition.matches(msg, now):
                    self._execute_condition(condition, msg)

    def _schedule(self, rule: Rule, msg: Message):
        """Schedule a rule to execute after a timeout."""
        def wrapper():
            self._execute(rule, msg)
            self._scheduled[rule.id_].remove(timer)
        timer = threading.Timer(rule.timeout, wrapper)
        self._scheduled[rule.id_].append(timer)
        timer.start()

    def _execute(self, rule: Rule, msg: Message):
        """Execute a rule's action."""
        if self.store.was_replied(msg.sender, msg.timestamp):
            return
        if rule.action_type == "reply" and rule.reply_message:
            try:
                callnow_client.messages.create(
                    to=msg.sender,
                    from_=CALLNOWUSA_NUMBER,
                    body=rule.reply_message,
                )
                self.store.mark_replied(msg.sender, msg.timestamp)
            except Exception:
                logger.exception("Auto-reply failed")
        elif rule.action_type == "forward" and rule.forward_to_number:
            try:
                callnow_client.messages.create(
                    to=rule.forward_to_number,
                    from_=CALLNOWUSA_NUMBER,
                    body=f"Forwarded from {msg.sender}: {msg.body}",
                )
                self.store.mark_replied(msg.sender, msg.timestamp)
            except Exception:
                logger.exception("Forward failed")
        if rule.is_one_time:
            self.store.delete_rule(rule.id_)

    def _execute_condition(self, condition: Condition, msg: Message):
        """Execute a condition's action."""
        if self.store.was_replied(msg.sender, msg.timestamp):
            return
        if condition.action == "forward":
            target = normalize_phone_number(condition.target) if is_valid_number(condition.target) else self.store.resolve(
                condition.target)
            if not target:
                logger.error("Invalid target for condition %d: %s", condition.id_, condition.target)
                return
            try:
                callnow_client.messages.create(
                    to=target,
                    from_=CALLNOWUSA_NUMBER,
                    body=f"Forwarded: {msg.body} (from {msg.sender})",
                )
                self.store.mark_replied(msg.sender, msg.timestamp)
                logger.info("Forwarded message from %s to %s for condition %d", msg.sender, target, condition.id_)
            except Exception:
                logger.exception("Forward failed for condition %d", condition.id_)

    def _schedule_pending_sms(self):
        """Schedule all active SMS from the database."""
        for sms in self.store.list_scheduled_sms():
            self._schedule_sms(sms)

    def _schedule_sms(self, sms: ScheduledSMS):
        """Schedule an SMS to be sent at the specified time."""
        now = datetime.now(TZ)
        if not sms.active or sms.send_time <= now:
            return
        delay = (sms.send_time - now).total_seconds()
        if delay <= 0:
            return

        def send_sms():
            try:
                callnow_client.messages.create(
                    to=sms.recipient,
                    from_=CALLNOWUSA_NUMBER,
                    body=sms.message,
                )
                self.store.mark_scheduled_sms_sent(sms.id_)
                logger.info("Sent scheduled SMS %d to %s: %s", sms.id_, sms.recipient, sms.message)
            except Exception:
                logger.exception("Failed to send scheduled SMS %d", sms.id_)
            finally:
                self._scheduled_sms_timers.pop(sms.id_, None)

        timer = threading.Timer(delay, send_sms)
        self._scheduled_sms_timers[sms.id_] = timer
        timer.start()
        logger.info("Scheduled SMS %d to %s at %s", sms.id_, sms.recipient, sms.send_time)

    def _parse_messages(self, raw: str) -> List[Message]:
        """Parse raw inbox data into Message objects."""
        msgs: List[Message] = []
        if not raw or not isinstance(raw, str) or not raw.strip():
            logger.warning("Empty or invalid inbox data: %s", raw)
            return msgs
        cleaned_raw = raw.replace('\u202a', '').replace('\u202c', '')
        entries = cleaned_raw.split(',')
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            m = re.match(r'(\[?(\+\d+)\]?):"(.*?)"\s+(sent|received)\s+([\d\s\-:]+)', entry)
            if m:
                _, sender, body, direction, ts = m.groups()
                try:
                    msg = Message.from_raw(normalize_phone_number(sender), body, ts.strip(), direction.strip())
                    msgs.append(msg)
                except Exception as e:
                    logger.warning("Parse failed for entry '%s': %s", entry, e)
            else:
                logger.warning("Regex failed to match entry: %s", entry)
        return msgs

class CommandParser:
    """Parses user commands into actionable instructions."""
    _sys_prompt = (
        "You are an expert NLP command parser for an SMS assistant.\n"
        "Input is a casual English sentence from the user.\n"
        "Your job is to interpret the user's intent and output a JSON list of commands.\n"
        "Each command is a dictionary with an 'action' key and relevant parameters.\n"
        "\n"
        "Always return a valid JSON list of commands, even if there is only one.\n"
        "Never output anything other than a JSON list.\n"
        "If you cannot identify any actionable command or required parameters are missing, output a single command with action 'chat' and the original message.\n"
        "\n"
        "Valid actions include:\n"
        "  - send_sms\n"
        "  - add_contact\n"
        "  - forward_messages\n"
        "  - sms_forward_stop\n"
        "  - show_messages\n"
        "  - list_contacts\n"
        "  - contact_exists\n"
        "  - update_contact\n"
        "  - delete_contact\n"
        "  - show_filtered\n"
        "  - show_meetings\n"
        "  - show_forwarding\n"
        "  - add_condition\n"
        "  - list_conditions\n"
        "  - stop_condition\n"
        "  - modify_condition\n"
        "  - schedule_sms\n"
        "  - list_scheduled_sms\n"
        "  - poll\n"
        "  - chat\n"
        "\n"
        "For 'send_sms', include:\n"
        "  - 'recipient': phone number or alias\n"
        "  - 'message': the message text to send (exclude words like 'send' or 'text')\n"
        "\n"
        "For 'add_contact', include:\n"
        "  - 'alias': contact name\n"
        "  - 'number': phone number\n"
        "\n"
        "For 'forward_messages', include:\n"
        "  - 'from': sender alias or number\n"
        "  - 'to': recipient alias or number\n"
        "  - Only trigger for explicit phrases like 'forward messages', 'set up forwarding', or 'forward from X to Y' without time ranges.\n"
        "\n"
        "For 'sms_forward_stop', include:\n"
        "  - 'all': boolean, set to true to stop all forwarding sessions\n"
        "  - or 'source': sender alias or number, and 'destination': recipient alias or number\n"
        "  - or 'index': zero-based index of the forwarding session to stop\n"
        "\n"
        "For 'show_messages', include:\n"
        "  - optional 'contact': alias or number (normalize to include +1 for US numbers)\n"
        "  - optional 'limit': number of messages (default 100)\n"
        "  - optional 'sort': 'newest' or 'oldest' (default 'newest')\n"
        "  - optional 'direction': 'sent', 'received', or 'all' (default 'all')\n"
        "\n"
        "For 'show_filtered', include:\n"
        "  - 'label': filter label (e.g., 'urgent', 'funny', 'romantic', 'meeting', 'high priority')\n"
        "\n"
        "For 'show_forwarding', include:\n"
        "  - no parameters; displays all active forwarding sessions\n"
        "\n"
        "For 'add_condition', include:\n"
        "  - 'type': condition type (e.g., 'priority', 'sender', 'keyword')\n"
        "  - 'value': condition value (e.g., 'high priority', '+1234567890', 'meeting')\n"
        "  - 'condition_action': action to take (e.g., 'forward')\n"
        "  - 'target': target number or alias\n"
        "  - optional 'start_time': start time in 24-hour format (e.g., '21:00') or 'now'\n"
        "  - optional 'end_time': end time in 24-hour format (e.g., '01:00') or relative (e.g., '30 seconds later')\n"
        "  - Note: 'forward' is NOT a valid top-level action; use 'add_condition' for any forwarding with time ranges.\n"
        "\n"
        "For 'modify_condition', include:\n"
        "  - 'id': condition ID to modify\n"
        "  - optional 'type': new condition type\n"
        "  - optional 'value': new condition value\n"
        "  - optional 'condition_action': new action\n"
        "  - optional 'target': new target number or alias\n"
        "  - optional 'start_time': new start time\n"
        "  - optional 'end_time': new end time\n"
        "\n"
        "For 'list_conditions', include:\n"
        "  - no parameters; displays all active conditional forwarding rules\n"
        "\n"
        "For 'stop_condition', include:\n"
        "  - 'id': condition ID to stop\n"
        "\n"
        "For 'schedule_sms', include:\n"
        "  - 'recipient': phone number or alias\n"
        "  - 'message': message to send (if missing, return 'chat')\n"
        "  - 'send_time': when to send (e.g., 'tomorrow at 7 am', '30 seconds later', 'in the next 30 seconds')\n"
        "\n"
        "For 'list_scheduled_sms', include:\n"
        "  - no parameters; displays all active scheduled SMS\n"
        "\n"
        "Map 'urgent', 'important', 'critical', 'essential' to 'high priority' for 'show_filtered' and 'add_condition'.\n"
        "Recognize phrases like 'check inbox', 'recheck inbox', 'check it again', 'see inbox again', 'reload inbox' as 'poll'.\n"
        "Recognize 'show me my scheduled messages' or 'list scheduled sms' as 'list_scheduled_sms'.\n"
        "Recognize 'schedule sms forwarding from X to Y from START to END', 'start sms forwarding from X to Y from START to END', 'forward (all) sms from X to Y from START to END', or 'send all sms from X to Y until END' as 'add_condition' with type='sender', value='X', condition_action='forward', target='Y', start_time='now' or 'START', end_time='END'. Convert times to 24-hour format (e.g., '12:54 pm' to '12:54').\n"
        "Recognize 'start sms forwarding from X to Y for the next Z seconds/minutes/hours' or 'forward (all) sms from X to Y for the next Z seconds/minutes/hours' as 'add_condition' with type='sender', value='X', condition_action='forward', target='Y', start_time='now', end_time='Z seconds/minutes/hours later'.\n"
        "Recognize 'cancel the schedule' or 'stop the schedule' as 'stop_condition' with the ID of the last condition.\n"
        "Recognize 'change the schedule to forward from X to Y from START to END' as 'modify_condition' with updated parameters.\n"
        "For 'show_messages', recognize 'sent messages from X' as direction='sent', 'received messages from X' as direction='received', and 'all messages from X' or 'messages from X' as direction='all'.\n"
        "For 'add_condition', recognize 'send me only sms if i receive from X' as type='sender', value='X', condition_action='forward', target='me'.\n"
        "Recognize 'send an sms to X in Y saying Z' or variations (e.g., 'schedule sms to X in Y saying Z', 'send an sms to X at Y saying Z') as 'schedule_sms' with recipient='X', send_time='Y', message='Z'. If message is missing, return 'chat'.\n"
        "Normalize phone numbers to include +1 for US numbers (e.g., '4372392448' becomes '+14372392448').\n"
        "Resolve 'me' to the user's phone number (CALLNOWUSA_NUMBER).\n"
        "Break down compound instructions into multiple commands.\n"
        "Resolve pronouns like 'him', 'her', 'them' to the last mentioned contact.\n"
        "\n"
        "Examples:\n"
        "- Input: \"send sms from Alice to Bob saying hi\"\n"
        "  Output: [{\"action\": \"send_sms\", \"recipient\": \"Bob\", \"message\": \"hi\"}]\n"
        "- Input: \"set up sms forwarding from Alice to Bob\"\n"
        "  Output: [{\"action\": \"forward_messages\", \"from\": \"Alice\", \"to\": \"Bob\"}]\n"
        "- Input: \"show urgent messages\"\n"
        "  Output: [{\"action\": \"show_filtered\", \"label\": \"high priority\"}]\n"
        "- Input: \"check inbox again\"\n"
        "  Output: [{\"action\": \"poll\"}]\n"
        "- Input: \"do i have any sent messages from 4372392448\"\n"
        "  Output: [{\"action\": \"show_messages\", \"contact\": \"+14372392448\", \"direction\": \"sent\"}]\n"
        "- Input: \"add a condition, send me only sms if i receive from +14372392441\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"+14372392441\", \"condition_action\": \"forward\", \"target\": \"" + CALLNOWUSA_NUMBER + "\"}]\n"
        "- Input: \"add a condition, only essential messages from 9pm to 1 am\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"priority\", \"value\": \"high priority\", \"condition_action\": \"forward\", \"target\": \"" + CALLNOWUSA_NUMBER + "\", \"start_time\": \"21:00\", \"end_time\": \"01:00\"}]\n"
        "- Input: \"schedule sms forwarding from Mike to Jonathan from 9pm to 1 am\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"Mike\", \"condition_action\": \"forward\", \"target\": \"Jonathan\", \"start_time\": \"21:00\", \"end_time\": \"01:00\"}]\n"
        "- Input: \"start sms forwarding from mike to navya from 12:54 pm to 12:55 pm\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"mike\", \"condition_action\": \"forward\", \"target\": \"navya\", \"start_time\": \"12:54\", \"end_time\": \"12:55\"}]\n"
        "- Input: \"forward all sms from mike to navya for the next 30 seconds\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"mike\", \"condition_action\": \"forward\", \"target\": \"navya\", \"start_time\": \"now\", \"end_time\": \"30 seconds later\"}]\n"
        "- Input: \"send all sms from navya to jishan until 1:10 pm\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"navya\", \"condition_action\": \"forward\", \"target\": \"jishan\", \"start_time\": \"now\", \"end_time\": \"13:10\"}]\n"
        "- Input: \"forward all sms from mike to navya until 2 pm\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"mike\", \"condition_action\": \"forward\", \"target\": \"navya\", \"start_time\": \"now\", \"end_time\": \"14:00\"}]\n"
        "- Input: \"start sms forwarding from mike to navya for the next 1 minute\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"mike\", \"condition_action\": \"forward\", \"target\": \"navya\", \"start_time\": \"now\", \"end_time\": \"1 minute later\"}]\n"
        "- Input: \"change the schedule to forward from Mike to Jonathan from 8pm to 2 am\"\n"
        "  Output: [{\"action\": \"modify_condition\", \"id\": \"<last_condition_id>\", \"type\": \"sender\", \"value\": \"Mike\", \"condition_action\": \"forward\", \"target\": \"Jonathan\", \"start_time\": \"20:00\", \"end_time\": \"02:00\"}]\n"
        "- Input: \"show active conditions\"\n"
        "  Output: [{\"action\": \"list_conditions\"}]\n"
        "- Input: \"stop the first one\"\n"
        "  Output: [{\"action\": \"stop_condition\", \"id\": \"<first_active_condition_id>\"}]\n"
        "- Input: \"cancel the schedule\"\n"
        "  Output: [{\"action\": \"stop_condition\", \"id\": \"<last_condition_id>\"}]\n"
        "- Input: \"show me my scheduled messages\"\n"
        "  Output: [{\"action\": \"list_scheduled_sms\"}]\n"
        "- Input: \"schedule an sms to Mike for tomorrow saying where did you get that watch from?\"\n"
        "  Output: [{\"action\": \"schedule_sms\", \"recipient\": \"Mike\", \"message\": \"where did you get that watch from?\", \"send_time\": \"tomorrow\"}]\n"
        "- Input: \"send an sms to +14372392448 in the next 30 seconds saying hey\"\n"
        "  Output: [{\"action\": \"schedule_sms\", \"recipient\": \"+14372392448\", \"message\": \"hey\", \"send_time\": \"30 seconds later\"}]\n"
        "- Input: \"send an sms to navya at 12:55 pm\"\n"
        "  Output: [{\"action\": \"chat\", \"message\": \"send an sms to navya at 12:55 pm\"}]\n"
        "- Input: \"send an sms to navya at 12:55 pm saying hi\"\n"
        "  Output: [{\"action\": \"schedule_sms\", \"recipient\": \"navya\", \"message\": \"hi\", \"send_time\": \"12:55 pm\"}]\n"
        "- Input: \"schedule sms forwarding from 12:34 am to 12:35 am from jishan to floxy\"\n"
        "  Output: [{\"action\": \"add_condition\", \"type\": \"sender\", \"value\": \"jishan\", \"condition_action\": \"forward\", \"target\": \"floxy\", \"start_time\": \"00:34\", \"end_time\": \"00:35\"}]\n"
        "Always respond with a JSON list of commands."
    )

    def __init__(self):
        self.last_cmd: Optional[Dict[str, Any]] = None
        self.last_contact: Optional[str] = None
        self.last_action: Optional[str] = None
        self.last_condition_id: Optional[int] = None
        self.forwarding_sessions: List[Dict[str, Any]] = []

    def _ordinal_to_index(self, ordinal: str) -> int:
        """Convert ordinal text to zero-based index."""
        ordinal_map = {
            "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
            "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9
        }
        if ordinal in ordinal_map:
            return ordinal_map[ordinal]
        if ordinal.endswith(("th", "st", "nd", "rd")):
            try:
                return int(ordinal.rstrip("stndrh")) - 1
            except ValueError:
                raise ValueError(f"Invalid ordinal: {ordinal}")
        raise ValueError(f"Unknown ordinal: {ordinal}")

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Parse user input into a list of commands."""
        text_low = text.lower().strip()
        logger.debug("Parsing input: %s", text)

        # Handle repeat commands
        if text_low in {"do it again", "again", "repeat"} and self.last_cmd:
            logger.debug("Repeating last command: %s", self.last_cmd)
            return [self.last_cmd]

        # Regex for stopping specific forwarding condition with sender= format, including time ranges
        m = re.match(
            r"(?:stop|remove)\s*(?:forwarding\s*from\s*sender\s*=\s*(\w+|\+\d{10,15})\s*→\s*(\w+|\+\d{10,15})(?:\s*from\s*(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2}))?)",
            text_low
        )
        if m:
            sender, target, start_time, end_time = m.groups()
            sender = normalize_phone_number(sender) if sender and is_valid_number(sender) else sender
            target = normalize_phone_number(target) if target and is_valid_number(target) else target
            conditions = SQLiteStore(DB_PATH).list_conditions()
            matching_condition = None
            for cond in conditions:
                if (cond.type_ == "sender" and
                        cond.value.lower() == sender.lower() and
                        cond.target.lower() == target.lower() and
                        (not start_time or cond.start_time == start_time) and
                        (not end_time or cond.end_time == end_time)):
                    matching_condition = cond
                    break
            if matching_condition:
                logger.debug("Matched stop_condition: id=%d", matching_condition.id_)
                return [{"action": "stop_condition", "id": matching_condition.id_}]
            else:
                logger.debug("No matching condition found for sender=%s, target=%s, start_time=%s, end_time=%s",
                             sender, target, start_time, end_time)
                print(f" No matching forwarding condition found for sender={sender}, target={target}")
                return [{"action": "chat", "message": text}]

        # Regex for stopping forwarding by sender and target
        m = re.match(
            r"(?:stop|remove)\s*(?:sms\s*)?forwarding\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})",
            text_low
        )
        if m:
            sender, target = m.groups()
            sender = normalize_phone_number(sender) if is_valid_number(sender) else sender
            target = normalize_phone_number(target) if is_valid_number(target) else target
            conditions = SQLiteStore(DB_PATH).list_conditions()
            matching_condition = None
            for cond in conditions:
                if cond.type_ == "sender" and cond.value.lower() == sender.lower() and cond.target.lower() == target.lower():
                    matching_condition = cond
                    break
            if matching_condition:
                logger.debug("Matched stop_condition: id=%d", matching_condition.id_)
                return [{"action": "stop_condition", "id": matching_condition.id_}]
            else:
                logger.debug("No matching condition found for sender=%s, target=%s", sender, target)
                return [{"action": "sms_forward_stop", "source": sender, "destination": target}]

        # Regex for stopping condition by ID or ordinal
        m = re.match(r"(?:stop|remove)\s*(?:the\s*)?(first|second|third|fourth|fifth|condition\s*(\d+))", text_low)
        if m:
            if m.group(1) in ["first", "second", "third", "fourth", "fifth"]:
                ordinal_map = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
                index = ordinal_map[m.group(1)]
                conditions = SQLiteStore(DB_PATH).list_conditions()
                if conditions and index < len(conditions):
                    logger.debug("Matched stop_condition: id=%d", conditions[index].id_)
                    return [{"action": "stop_condition", "id": conditions[index].id_}]
                else:
                    print(f" No condition at position {m.group(1)}")
                    return [{"action": "chat", "message": text}]
            else:
                logger.debug("Matched stop_condition: id=%s", m.group(2))
                return [{"action": "stop_condition", "id": int(m.group(2))}]

        # Regex for scheduling SMS forwarding with absolute times
        m = re.match(
            r"(?:schedule|start|forward\s*(?:all\s*)?sms)\s*forwarding\s*from\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})",
            text_low
        )
        if m:
            start_time, end_time, from_contact, to_contact = m.groups()
            try:
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        start_dt = datetime.strptime(start_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid start time format: {start_time}")
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        end_dt = datetime.strptime(end_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid end time format: {end_time}")
                start_time = start_dt.strftime("%H:%M")
                end_time = end_dt.strftime("%H:%M")
            except ValueError as e:
                logger.error("Failed to normalize time formats '%s' to '%s': %s", start_time, end_time, e)
                print(f" Invalid time format: {start_time} to {end_time}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched scheduled forwarding: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            cmd = {
                "action": "add_condition",
                "type": "sender",
                "value": normalize_phone_number(from_contact) if is_valid_number(from_contact) else from_contact,
                "condition_action": "forward",
                "target": normalize_phone_number(to_contact) if is_valid_number(to_contact) else to_contact,
                "start_time": start_time,
                "end_time": end_time
            }
            self.last_cmd = cmd
            self.last_contact = to_contact
            self.last_action = "add_condition"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for scheduling SMS forwarding with relative time
        m = re.match(
            r"(?:start|forward\s*(?:all\s*)?sms)\s*forwarding\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*for\s*the\s*next\s*(\d+)\s*(second|minute|hour)s?",
            text_low
        )
        if m:
            from_contact, to_contact, amount, unit = m.groups()
            if int(amount) == 0:
                print(" Forwarding duration must be greater than 0.")
                return [{"action": "chat", "message": text}]
            start_time = "now"
            end_time = f"{amount} {unit}s later"
            logger.debug("Matched relative time forwarding: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            cmd = {
                "action": "add_condition",
                "type": "sender",
                "value": normalize_phone_number(from_contact) if is_valid_number(from_contact) else from_contact,
                "condition_action": "forward",
                "target": normalize_phone_number(to_contact) if is_valid_number(to_contact) else to_contact,
                "start_time": start_time,
                "end_time": end_time
            }
            self.last_cmd = cmd
            self.last_contact = to_contact
            self.last_action = "add_condition"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for scheduling SMS forwarding with "until <time>"
        m = re.match(
            r"(?:send|forward\s*(?:all\s*)?sms)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*until\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            text_low
        )
        if m:
            from_contact, to_contact, end_time = m.groups()
            try:
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        end_dt = datetime.strptime(end_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid end time format: {end_time}")
                end_time = end_dt.strftime("%H:%M")
            except ValueError as e:
                logger.error("Failed to normalize end time '%s': %s", end_time, e)
                print(f" Invalid end time format: {end_time}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched until forwarding: from=%s, to=%s, start_time=now, end_time=%s",
                         from_contact, to_contact, end_time)
            cmd = {
                "action": "add_condition",
                "type": "sender",
                "value": normalize_phone_number(from_contact) if is_valid_number(from_contact) else from_contact,
                "condition_action": "forward",
                "target": normalize_phone_number(to_contact) if is_valid_number(to_contact) else to_contact,
                "start_time": "now",
                "end_time": end_time
            }
            self.last_cmd = cmd
            self.last_contact = to_contact
            self.last_action = "add_condition"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for indefinite forwarding with "until stopped"
        m = re.match(
            r"(?:send|forward\s*(?:all\s*)?sms)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*until\s*stopped",
            text_low
        )
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched indefinite forwarding: from=%s, to=%s, start_time=now, end_time=None",
                         from_contact, to_contact)
            cmd = {
                "action": "add_condition",
                "type": "sender",
                "value": normalize_phone_number(from_contact) if is_valid_number(from_contact) else from_contact,
                "condition_action": "forward",
                "target": normalize_phone_number(to_contact) if is_valid_number(to_contact) else to_contact,
                "start_time": "now",
                "end_time": None
            }
            self.last_cmd = cmd
            self.last_contact = to_contact
            self.last_action = "add_condition"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for scheduling SMS commands
        m = re.match(
            r"(?:schedule|send\s*(?:an\s*)?sms)\s*to\s*(\w+|\+\d{10,15})\s*(?:at\s*|in\s*the\s*next\s*(\d+\s*(?:second|minute|hour)s?)|tomorrow(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?|next\s*week(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?|(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))\s*(?:saying\s*(.+))?",
            text_low
        )
        if m:
            recipient, relative_time, tomorrow_time, next_week_time, absolute_time, message = m.groups()
            recipient = normalize_phone_number(recipient) if is_valid_number(recipient) else recipient
            if not message:
                logger.debug("No message provided for schedule_sms")
                print(" Please provide a message to send (e.g., 'saying hello').")
                return [{"action": "chat", "message": text}]
            time_str = relative_time or tomorrow_time or next_week_time or absolute_time
            if relative_time:
                time_match = re.match(r"(\d+)\s*(second|minute|hour)s?", relative_time)
                if time_match:
                    amount, unit = time_match.groups()
                    time_str = f"{amount} {unit}s later"
            try:
                parse_time(time_str, datetime.now(TZ))
            except ValueError as e:
                logger.error("Failed to parse send time '%s': %s", time_str, e)
                print(f" Invalid send time format: {time_str}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched schedule_sms: recipient=%s, message=%s, send_time=%s", recipient, message, time_str)
            cmd = {
                "action": "schedule_sms",
                "recipient": recipient,
                "message": message.strip(),
                "send_time": time_str
            }
            self.last_cmd = cmd
            self.last_contact = recipient
            self.last_action = "schedule_sms"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for modifying scheduled forwarding
        m = re.match(
            r"change the schedule to forward from (\w+) to (\w+) from (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            text_low
        )
        if m:
            from_contact, to_contact, start_time, end_time = m.groups()
            try:
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        start_dt = datetime.strptime(start_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid start time format: {start_time}")
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        end_dt = datetime.strptime(end_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid end time format: {end_time}")
                start_time = start_dt.strftime("%H:%M")
                end_time = end_dt.strftime("%H:%M")
            except ValueError as e:
                logger.error("Failed to normalize time formats '%s' to '%s': %s", start_time, end_time, e)
                print(f" Invalid time format: {start_time} to {end_time}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched modify_condition: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            cmd = {
                "action": "modify_condition",
                "id": "<last_condition_id>",
                "type": "sender",
                "value": from_contact,
                "condition_action": "forward",
                "target": to_contact,
                "start_time": start_time,
                "end_time": end_time
            }
            self.last_cmd = cmd
            self.last_contact = to_contact
            self.last_action = "modify_condition"
            logger.debug("Set last_cmd: %s, last_contact: %s, last_action: %s", cmd, self.last_contact,
                         self.last_action)
            return [cmd]

        # Regex for cancel all scheduled SMS
        if text_low == "remove all scheduled sms":
            logger.debug("Matched cancel_all_scheduled_sms")
            return [{"action": "cancel_all_scheduled_sms"}]

        # Regex for cancel schedule
        if text_low in ["cancel the schedule", "stop the schedule"]:
            if self.last_condition_id is not None:
                logger.debug("Matched stop_condition: id=%d", self.last_condition_id)
                return [{"action": "stop_condition", "id": self.last_condition_id}]
            return [{"action": "chat", "message": text}]

        # Regex for list scheduled SMS
        if text_low in ["show me my scheduled messages", "list scheduled sms", "show scheduled sms"]:
            logger.debug("Matched list_scheduled_sms")
            return [{"action": "list_scheduled_sms"}]

        # Regex for list conditions
        if text_low in ["show active conditions", "show sms conditions"]:
            logger.debug("Matched list_conditions")
            return [{"action": "list_conditions"}]

        # Regex for add contact
        m = re.match(r"add (\w+),\s*(?:number is\s*)?(\+?\d{10,15})", text_low)
        if m:
            alias, number = m.groups()
            logger.debug("Matched add_contact: alias=%s, number=%s", alias, number)
            return [{"action": "add_contact", "alias": alias, "number": number}]

        # Regex for send SMS
        m = re.search(
            r"(?:send\s*(?:sms|text|message)\s*(?:from\s*(\w+)\s*)?to\s*(\w+|\+?\d{10,15})\s*(?:saying|:)\s*(.+))",
            text_low
        )
        if m:
            _, recipient, message = m.groups()
            logger.debug("Matched send_sms: recipient=%s, message=%s", recipient, message)
            return [{"action": "send_sms", "recipient": recipient, "message": message.strip()}]

        # Regex for inbox recheck
        if any(phrase in text_low for phrase in [
            "check inbox", "recheck inbox", "check it again", "see inbox again",
            "reload inbox", "show messages again"
        ]):
            logger.debug("Matched poll command")
            return [{"action": "poll"}]

        # Regex for sent/received/all messages
        m = re.search(r"(?:do\s*i\s*have\s*any\s*(sent|received|all)?\s*messages\s*from\s*(\+?\d{10,15}|\w+))",
                      text_low)
        if m:
            direction, contact = m.groups()
            direction = direction or "all"
            contact = normalize_phone_number(contact) if is_valid_number(contact) else contact
            logger.debug("Matched show_messages: contact=%s, direction=%s", contact, direction)
            return [{"action": "show_messages", "contact": contact, "direction": direction, "sort": "newest"}]

        # Regex for ordinal-based forwarding stop (already handled above in condition regex)
        m = re.search(r"(?:stop|remove)\s+the\s+(\w+)(?:st|nd|rd|th)?\s+forwarding", text_low)
        if m:
            ordinal_str = m.group(1)
            try:
                index = self._ordinal_to_index(ordinal_str)
                logger.debug("Matched sms_forward_stop: index=%d", index)
                return [{"action": "sms_forward_stop", "index": index}]
            except ValueError:
                logger.debug("Invalid ordinal: %s", ordinal_str)
                return [{"action": "chat", "message": text}]

        # Regex for conditional forwarding with show
        m = re.match(
            r"show.*forwarding.*if.*none.*add.*forward.*from (\w+).*to (\w+).*then show.*forwarding",
            text_low
        )
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched conditional forward: from=%s, to=%s", from_contact, to_contact)
            return [
                {"action": "show_forwarding"},
                {"action": "forward_messages", "from": from_contact, "to": to_contact, "conditional": "if_none"},
                {"action": "show_forwarding"}
            ]

        # Regex for update contact
        m = re.search(r"change (it|number)?\s*(?:of\s*)?(\w+)\s*(?:to)?\s*(\+\d{10,15})", text_low)
        if m:
            _, alias, number = m.groups()
            logger.debug("Matched update_contact: alias=%s, number=%s", alias, number)
            return [{"action": "update_contact", "alias": alias, "number": number}]

        # Regex for delete contact
        m = re.search(r"(?:delete|remove)\s*(?:contact\s*)?(\w+)", text_low)
        if m:
            alias = m.group(1)
            if alias.lower() not in ["forwarding", "condition", "sender", "sms", "1st", "2nd", "3rd", "4th", "5th"]:
                logger.debug("Matched delete_contact: alias=%s", alias)
                return [{"action": "delete_contact", "alias": alias}]
            logger.debug("Skipping delete_contact for reserved keyword: %s", alias)
            return [{"action": "chat", "message": text}]

        # Priority mapping
        priority_map = {
            "urgent": "high priority",
            "important": "high priority",
            "critical": "high priority",
            "emergency": "high priority",
            "essential": "high priority",
            "high priority": "high priority",
            "medium priority": "medium priority",
            "low priority": "low priority",
            "needs.*attention": "high priority"
        }
        for term, label in priority_map.items():
            if re.search(term, text_low):
                date_filter = None
                if "today" in text_low:
                    date_filter = "today"
                elif "tomorrow" in text_low:
                    date_filter = "tomorrow"
                elif "next week" in text_low:
                    date_filter = "next week"
                elif "upcoming" in text_low:
                    date_filter = "upcoming"
                cmd = {"action": "show_filtered", "label": label}
                if date_filter:
                    cmd["date_filter"] = date_filter
                logger.debug("Matched show_filtered: label=%s, date_filter=%s", label, date_filter)
                return [cmd]

        # Schedule/meeting mapping
        if re.search(r"schedule|meeting", text_low) and not text_low.startswith(
                ("schedule an sms", "schedule sms forwarding", "start sms forwarding", "forward all sms",
                 "send all sms")):
            date_filter = None
            if "today" in text_low:
                date_filter = "today"
            elif "tomorrow" in text_low:
                date_filter = "tomorrow"
            elif "next week" in text_low:
                date_filter = "next week"
            elif "upcoming" in text_low:
                date_filter = "upcoming"
            cmd = {"action": "show_meetings"}
            if date_filter:
                cmd["date_filter"] = date_filter
            logger.debug("Matched show_meetings: date_filter=%s", date_filter)
            return [cmd]

        # Forwarding setup
        m = re.search(r"(?:set\s*up\s*(?:sms\s*)?forwarding|forward\s*messages)\s*from\s*(\w+)\s*to\s*(\w+)",
                      text_low)
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched forward_messages: from=%s, to=%s", from_contact, to_contact)
            return [{"action": "forward_messages", "from": from_contact, "to": to_contact}]

        # Show forwarding
        if "active" in text_low and "forward" in text_low:
            logger.debug("Matched show_forwarding")
            return [{"action": "show_forwarding"}]

        # Stop all forwarding
        if "stop all" in text_low and "forward" in text_low:
            logger.debug("Matched sms_forward_stop: all=true")
            return [{"action": "sms_forward_stop", "all": True}]

        # Fallback to LLM parsing
        try:
            rsp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._sys_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.4,
            )
            data = json.loads(rsp.choices[0].message.content.strip().replace("'", '"'))
            cmds = data if isinstance(data, list) else [data]
            logger.debug("LLM parsed commands: %s", cmds)
        except Exception as exc:
            logger.debug("Parser LLM fallback: %s", exc)
            cmds = self._regex_fallback(text)

        # Update state for non-chat commands
        if cmds and cmds[0].get("action") != "chat":
            self.last_cmd = cmds[0]
            if cmds[0].get("action") == "add_condition":
                self.last_condition_id = None
            if cmds[0].get("alias"):
                self.last_contact = cmds[0]["alias"]
            elif cmds[0].get("recipient"):
                self.last_contact = cmds[0]["recipient"]
            elif cmds[0].get("contact"):
                self.last_contact = cmds[0]["contact"]
            elif cmds[0].get("value"):
                self.last_contact = cmds[0]["value"]
            self.last_action = cmds[0]["action"]
            logger.debug("Updated last_cmd: %s, last_contact: %s, last_action: %s, last_condition_id: %s",
                         self.last_cmd, self.last_contact, self.last_action, self.last_condition_id)

        return cmds

    def _regex_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback regex parsing for commands."""
        text_low = text.lower().strip()
        logger.debug("Regex fallback for: %s", text_low)

        # Handle stop specific forwarding condition
        m = re.match(
            r"stop\s*(?:sender\s*=\s*(\w+|\+\d{10,15}),\s*)?forward\s*(?:to\s*(\w+|\+\d{10,15}))(?:\s*from\s*(\w+|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\w+|\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d+\s*(?:second|minute|hour)s?\s*later))?",
            text_low
        )
        if m:
            sender, target, start_time, end_time = m.groups()
            sender = normalize_phone_number(sender) if sender and is_valid_number(sender) else sender
            target = normalize_phone_number(target) if target and is_valid_number(target) else target
            conditions = SQLiteStore(DB_PATH).list_conditions()
            matching_condition = None
            for cond in conditions:
                if (cond.type_ == "sender" and
                        cond.value == sender and
                        cond.target == target and
                        (not start_time or cond.start_time == start_time or (
                                start_time == "now" and cond.start_time == datetime.now(TZ).strftime("%H:%M"))) and
                        (not end_time or cond.end_time == end_time or (
                                end_time and "later" in end_time and cond.end_time == parse_time(end_time, datetime.now(
                            TZ)).strftime("%H:%M")))):
                    matching_condition = cond
                    break
            if matching_condition:
                logger.debug("Matched stop_condition: id=%d", matching_condition.id_)
                return [{"action": "stop_condition", "id": matching_condition.id_}]
            else:
                logger.debug("No matching condition found for sender=%s, target=%s, start_time=%s, end_time=%s",
                             sender, target, start_time, end_time)
                print(f" No matching forwarding condition found for sender={sender}, target={target}")
                return [{"action": "chat", "message": text}]

        # Handle add condition for sender-based forwarding
        m = re.match(r"add a condition,\s*send me only sms if i receive from (\+?\d{10,15}|\w+)", text_low)
        if m:
            value = normalize_phone_number(m.group(1)) if is_valid_number(m.group(1)) else m.group(1)
            logger.debug("Matched add_condition: type=sender, value=%s, condition_action=forward, target=%s", value,
                         CALLNOWUSA_NUMBER)
            return [{"action": "add_condition", "type": "sender", "value": value, "condition_action": "forward",
                     "target": CALLNOWUSA_NUMBER}]

        # Handle add condition for priority-based forwarding
        m = re.match(
            r"add a condition,\s*only (essential|urgent|important|critical) messages(?: from (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?",
            text_low)
        if m:
            priority, start_time, end_time = m.groups()
            cmd = {
                "action": "add_condition",
                "type": "priority",
                "value": "high priority",
                "condition_action": "forward",
                "target": CALLNOWUSA_NUMBER
            }
            if start_time and end_time:
                cmd["start_time"] = start_time
                cmd["end_time"] = end_time
            logger.debug(
                "Matched add_condition: type=priority, value=high priority, condition_action=forward, target=%s, start_time=%s, end_time=%s",
                CALLNOWUSA_NUMBER, start_time, end_time)
            return [cmd]

        # Handle scheduled SMS forwarding with absolute times
        m = re.match(
            r"(?:schedule|start|forward\s*(?:all\s*)?sms)\s*forwarding\s*from\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})",
            text_low
        )
        if m:
            start_time, end_time, from_contact, to_contact = m.groups()
            try:
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        start_dt = datetime.strptime(start_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid start time format: {start_time}")
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        end_dt = datetime.strptime(end_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid end time format: {end_time}")
                start_time = start_dt.strftime("%H:%M")
                end_time = end_dt.strftime("%H:%M")
            except ValueError as e:
                logger.error("Failed to normalize time formats '%s' to '%s': %s", start_time, end_time, e)
                print(f" Invalid time format: {start_time} to {end_time}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched scheduled forwarding: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            return [{"action": "add_condition", "type": "sender", "value": from_contact, "condition_action": "forward",
                     "target": to_contact, "start_time": start_time, "end_time": end_time}]

        # Handle scheduled SMS forwarding with relative time
        m = re.match(
            r"(?:start|forward\s*(?:all\s*)?sms)\s*forwarding\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*for\s*the\s*next\s*(\d+)\s*(second|minute|hour)s?",
            text_low
        )
        if m:
            from_contact, to_contact, amount, unit = m.groups()
            if int(amount) == 0:
                print(" Forwarding duration must be greater than 0.")
                return [{"action": "chat", "message": text}]
            start_time = "now"
            end_time = f"{amount} {unit}s later"
            logger.debug("Matched relative time forwarding: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            return [{"action": "add_condition", "type": "sender", "value": from_contact, "condition_action": "forward",
                     "target": to_contact, "start_time": start_time, "end_time": end_time}]

        # Handle scheduled SMS forwarding with "until <time>"
        m = re.match(
            r"(?:send|forward\s*(?:all\s*)?sms)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*until\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            text_low
        )
        if m:
            from_contact, to_contact, end_time = m.groups()
            try:
                for fmt in ["%I:%M%p", "%I:%M %p", "%I%p", "%I %p"]:
                    try:
                        end_dt = datetime.strptime(end_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid end time format: {end_time}")
                end_time = end_dt.strftime("%H:%M")
            except ValueError as e:
                logger.error("Failed to normalize end time '%s': %s", end_time, e)
                print(f" Invalid end time format: {end_time}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched until forwarding: from=%s, to=%s, start_time=now, end_time=%s",
                         from_contact, to_contact, end_time)
            return [{"action": "add_condition", "type": "sender", "value": from_contact, "condition_action": "forward",
                     "target": to_contact, "start_time": "now", "end_time": end_time}]

        # Handle indefinite forwarding with "until stopped"
        m = re.match(
            r"(?:send|forward\s*(?:all\s*)?sms)\s*from\s*(\w+|\+\d{10,15})\s*to\s*(\w+|\+\d{10,15})\s*until\s*stopped",
            text_low
        )
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched indefinite forwarding: from=%s, to=%s, start_time=now, end_time=None",
                         from_contact, to_contact)
            return [{"action": "add_condition", "type": "sender", "value": from_contact, "condition_action": "forward",
                     "target": to_contact, "start_time": "now", "end_time": None}]

        # Handle schedule SMS with time phrases
        m = re.match(
            r"(?:schedule|send\s*(?:an\s*)?sms)\s*to\s*(\w+|\+\d{10,15})\s*(?:at\s*|in\s*the\s*next\s*(\d+\s*(?:second|minute|hour)s?)|tomorrow(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?|next\s*week(?:\s*at\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?|(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))\s*(?:saying\s*(.+))?",
            text_low
        )
        if m:
            recipient, relative_time, tomorrow_time, next_week_time, absolute_time, message = m.groups()
            recipient = normalize_phone_number(recipient) if is_valid_number(recipient) else recipient
            if not message:
                logger.debug("No message provided for schedule_sms")
                print(" Please provide a message to send (e.g., 'saying hello').")
                return [{"action": "chat", "message": text}]
            time_str = relative_time or tomorrow_time or next_week_time or absolute_time
            if relative_time:
                time_match = re.match(r"(\d+)\s*(second|minute|hour)s?", relative_time)
                if time_match:
                    amount, unit = time_match.groups()
                    time_str = f"{amount} {unit}s later"
            try:
                parse_time(time_str, datetime.now(TZ))
            except ValueError as e:
                logger.error("Failed to parse send time '%s': %s", time_str, e)
                print(f" Invalid send time format: {time_str}")
                return [{"action": "chat", "message": text}]
            logger.debug("Matched schedule_sms: recipient=%s, message=%s, send_time=%s", recipient, message, time_str)
            return [
                {"action": "schedule_sms", "recipient": recipient, "message": message.strip(), "send_time": time_str}]

        # ... (rest of the method remains unchanged)

        # Handle modify scheduled forwarding
        m = re.match(
            r"change the schedule to forward from (\w+) to (\w+) from (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*to\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
            text_low)
        if m:
            from_contact, to_contact, start_time, end_time = m.groups()
            logger.debug("Matched modify_condition: from=%s, to=%s, start_time=%s, end_time=%s",
                         from_contact, to_contact, start_time, end_time)
            return [{"action": "modify_condition", "id": "<last_condition_id>", "type": "sender", "value": from_contact,
                     "condition_action": "forward", "target": to_contact, "start_time": start_time,
                     "end_time": end_time}]

        # Handle cancel all scheduled SMS
        if text_low == "remove all scheduled sms":
            logger.debug("Matched cancel_all_scheduled_sms")
            return [{"action": "cancel_all_scheduled_sms"}]

        # Handle cancel schedule
        if text_low in ["cancel the schedule", "stop the schedule"]:
            if self.last_condition_id is not None:
                logger.debug("Matched stop_condition: id=%d", self.last_condition_id)
                return [{"action": "stop_condition", "id": self.last_condition_id}]
            return [{"action": "chat", "message": text}]

        # Handle list scheduled SMS
        if text_low in ["show me my scheduled messages", "list scheduled sms", "show scheduled sms"]:
            logger.debug("Matched list_scheduled_sms")
            return [{"action": "list_scheduled_sms"}]

        # Handle list conditions
        if text_low == "show active conditions":
            logger.debug("Matched list_conditions")
            return [{"action": "list_conditions"}]

        # Handle stop condition
        m = re.match(r"stop (?:the )?(first|condition (\d+))", text_low)
        if m:
            if m.group(1) == "first":
                conditions = SQLiteStore(DB_PATH).list_conditions()
                if conditions:
                    logger.debug("Matched stop_condition: id=%d", conditions[0].id_)
                    return [{"action": "stop_condition", "id": conditions[0].id_}]
            else:
                logger.debug("Matched stop_condition: id=%s", m.group(2))
                return [{"action": "stop_condition", "id": int(m.group(2))}]
            return [{"action": "chat", "message": text}]

        # Handle add contact
        m = re.match(r"add (\w+),\s*(?:number is\s*)?(\+?\d{10,15})", text_low)
        if m:
            alias, number = m.groups()
            logger.debug("Matched add_contact: alias=%s, number=%s", alias, number)
            return [{"action": "add_contact", "alias": alias, "number": number}]

        # Handle send sms
        m = re.search(
            r"(?:send\s*(?:sms|text|message)\s*(?:from\s*(\w+)\s*)?to\s*(\w+|\+?\d{10,15})\s*(?:saying|:)\s*(.+))",
            text_low)
        if m:
            _, recipient, message = m.groups()
            logger.debug("Matched send_sms: recipient=%s, message=%s", recipient, message)
            return [{"action": "send_sms", "recipient": recipient, "message": message.strip()}]

        # Handle inbox recheck
        if any(phrase in text_low for phrase in [
            "check inbox", "recheck inbox", "check it again", "see inbox again",
            "reload inbox", "show messages again"
        ]):
            logger.debug("Matched poll command")
            return [{"action": "poll"}]

        # Handle sent/received/all messages
        m = re.search(r"(?:do\s*i\s*have\s*any\s*(sent|received|all)?\s*messages\s*from\s*(\+?\d{10,15}|\w+))",
                      text_low)
        if m:
            direction, contact = m.groups()
            direction = direction or "all"
            contact = normalize_phone_number(contact) if is_valid_number(contact) else contact
            logger.debug("Matched show_messages: contact=%s, direction=%s", contact, direction)
            return [{"action": "show_messages", "contact": contact, "direction": direction, "sort": "newest"}]

        # Handle ordinal-based forwarding stop
        m = re.search(r"(?:stop|remove)\s+the\s+(\w+)(?:st|nd|rd|th)?\s+forwarding", text_low)
        if m:
            ordinal_str = m.group(1)
            try:
                index = self._ordinal_to_index(ordinal_str)
                logger.debug("Matched sms_forward_stop: index=%d", index)
                return [{"action": "sms_forward_stop", "index": index}]
            except ValueError:
                logger.debug("Invalid ordinal: %s", ordinal_str)
                return [{"action": "chat", "message": text}]

        # Handle conditional forwarding
        m = re.match(
            r"show.*forwarding.*if.*none.*add.*forward.*from (\w+).*to (\w+).*then show.*forwarding",
            text_low)
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched conditional forward: from=%s, to=%s", from_contact, to_contact)
            return [
                {"action": "show_forwarding"},
                {"action": "forward_messages", "from": from_contact, "to": to_contact, "conditional": "if_none"},
                {"action": "show_forwarding"}
            ]

        # Handle update contact
        m = re.search(r"change (it|number)?\s*(?:of\s*)?(\w+)\s*(?:to)?\s*(\+\d{10,15})", text_low)
        if m:
            _, alias, number = m.groups()
            logger.debug("Matched update_contact: alias=%s, number=%s", alias, number)
            return [{"action": "update_contact", "alias": alias, "number": number}]

        # Handle delete contact
        m = re.search(r"(?:delete|remove)\s*(?:contact\s*)?(\w+)", text_low)
        if m:
            alias = m.group(1)
            logger.debug("Matched delete_contact: alias=%s", alias)
            return [{"action": "delete_contact", "alias": alias}]

        # Priority mapping
        priority_map = {
            "urgent": "high priority",
            "important": "high priority",
            "critical": "high priority",
            "emergency": "high priority",
            "essential": "high priority",
            "high priority": "high priority",
            "medium priority": "medium priority",
            "low priority": "low priority",
            "needs.*attention": "high priority"
        }
        for term, label in priority_map.items():
            if re.search(term, text_low):
                date_filter = None
                if "today" in text_low:
                    date_filter = "today"
                elif "tomorrow" in text_low:
                    date_filter = "tomorrow"
                elif "next week" in text_low:
                    date_filter = "next week"
                elif "upcoming" in text_low:
                    date_filter = "upcoming"
                cmd = {"action": "show_filtered", "label": label}
                if date_filter:
                    cmd["date_filter"] = date_filter
                logger.debug("Matched show_filtered: label=%s, date_filter=%s", label, date_filter)
                return [cmd]

        # Schedule/meeting mapping
        if re.search(r"schedule|meeting", text_low) and not text_low.startswith(
                ("schedule an sms", "schedule sms forwarding", "start sms forwarding", "forward all sms",
                 "send all sms")):
            date_filter = None
            if "today" in text_low:
                date_filter = "today"
            elif "tomorrow" in text_low:
                date_filter = "tomorrow"
            elif "next week" in text_low:
                date_filter = "next week"
            elif "upcoming" in text_low:
                date_filter = "upcoming"
            cmd = {"action": "show_meetings"}
            if date_filter:
                cmd["date_filter"] = date_filter
            logger.debug("Matched show_meetings: date_filter=%s", date_filter)
            return [cmd]

        # Forwarding setup
        m = re.search(r"(?:set\s*up\s*(?:sms\s*)?forwarding|forward\s*messages)\s*from\s*(\w+)\s*to\s*(\w+)",
                      text_low)
        if m:
            from_contact, to_contact = m.groups()
            logger.debug("Matched forward_messages: from=%s, to=%s", from_contact, to_contact)
            return [{"action": "forward_messages", "from": from_contact, "to": to_contact}]

        # Existing patterns
        if "active" in text_low and "forward" in text_low:
            logger.debug("Matched show_forwarding")
            return [{"action": "show_forwarding"}]

        if "stop all" in text_low and "forward" in text_low:
            logger.debug("Matched sms_forward_stop: all=true")
            return [{"action": "sms_forward_stop", "all": True}]

        m = re.search(r"(?:stop|remove)\s*(?:forward|forwarding)\s*from\s*(\w+)(?:\s*to\s*(\w+))?", text_low)
        if m:
            source, destination = m.groups()
            if destination:
                logger.debug("Matched sms_forward_stop: source=%s, destination=%s", source, destination)
                return [{"action": "sms_forward_stop", "source": source, "destination": destination}]
            logger.debug("Matched sms_forward_stop: source=%s", source)
            return [{"action": "sms_forward_stop", "source": source}]

        if "show" in text_low and "forward" in text_low:
            logger.debug("Matched show_forwarding")
            return [{"action": "show_forwarding"}]

        label_map = {
            "horny": "horny",
            "funny": "funny",
            "romantic": "romantic",
            "scary": "scary",
            "flirty": "flirty",
            "angry": "angry"
        }
        for word, label in label_map.items():
            if word in text_low:
                logger.debug("Matched show_filtered: label=%s", label)
                return [{"action": "show_filtered", "label": label}]

        m = re.search(r"messages from (\w+|\+?\d{10,15})", text_low)
        if m:
            contact = m.group(1)
            logger.debug("Matched show_messages: contact=%s", contact)
            return [{"action": "show_messages", "contact": contact, "sort": "newest", "direction": "all"}]

        m = re.search(r"do i.*have (\w+) as a contact", text_low)
        if m:
            alias = m.group(1)
            logger.debug("Matched contact_exists: alias=%s", alias)
            return [{"action": "contact_exists", "alias": alias}]

        m = re.search(r"(show\s*(me)?\s*(the)?\s*(oldest|earliest)\s*(\d+)?\s*messages?)", text_low)
        if m:
            limit = int(m.group(5)) if m.group(5) else 6
            logger.debug("Matched show_messages: sort=oldest, limit=%d", limit)
            return [{"action": "show_messages", "limit": limit, "sort": "oldest", "direction": "all"}]

        m = re.search(r"(show\s*(me)?\s*(the)?\s*last\s*(\d+)?\s*messages?)", text_low)
        if m:
            limit = int(m.group(4)) if m.group(4) else 6
            logger.debug("Matched show_messages: sort=newest, limit=%d", limit)
            return [{"action": "show_messages", "limit": limit, "sort": "newest", "direction": "all"}]

        logger.debug("No match, defaulting to chat")
        return [{"action": "chat", "message": text}]
class SMSAssistant:
    def __init__(self):
        self.store = SQLiteStore(DB_PATH)
        self.monitor = InboxMonitor(self.store)
        self.parser = CommandParser()
        self.monitor.start()

    def run(self):
        """Run the command-line interface."""
        print("I’m your SMS Assistant, ready to work. How may I help you?")
        while True:
            try:
                user_in = input("\nWhat’s on your mind? ")
            except (EOFError, KeyboardInterrupt):
                break
            if user_in.lower() in {"exit", "quit"}:
                break
            for cmd in self.parser.parse(user_in):
                self._dispatch(cmd)
        self.monitor.stop()
        print("Bye!")

    def _manual_poll(self):
        """Manually poll the inbox."""
        print(" Checking inbox now...")
        try:
            self.monitor._poll()
            print("Inbox updated.")
        except Exception:
            logger.exception("Manual poll failed")
            print(" Polling error.")

    def _list_contacts(self):
        """List all saved contacts."""
        contacts = self.store.list_contacts()
        if not contacts:
            print("No contacts saved.")
            return
        print(" Your contacts:")
        for alias, number in contacts:
            print(f" - {alias}: {number}")

    def _resolve_contact(self, alias: str) -> Optional[str]:
        """Resolve an alias to a phone number."""
        return self.store.resolve(alias.strip().lower())

    def _forward_messages(self, cmd: Dict[str, Any]):
        """Set up real-time SMS forwarding."""
        from_ = cmd.get("from", "").strip().lower()
        to = cmd.get("to", "").strip().lower()

        if not from_ or not to:
            print(" Missing 'from' or 'to' contact.")
            return

        sender_number = self.store.resolve(from_) if not is_valid_number(from_) else normalize_phone_number(from_)
        recipient_number = self.store.resolve(to) if not is_valid_number(to) else normalize_phone_number(to)

        if not sender_number:
            print(f" Contact '{from_}' not found.")
            if is_valid_number(from_):
                save = input(f" Want to save '{from_}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input(" Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(from_))
                        print(f" Saved {alias} as {normalize_phone_number(from_)}")
                        cmd["from"] = alias
                        return self._forward_messages(cmd)
            return
        if not recipient_number:
            print(f" Contact '{to}' not found.")
            if is_valid_number(to):
                save = input(f" Want to save '{to}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input(" Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(to))
                        print(f" Saved {alias} as {normalize_phone_number(to)}")
                        cmd["to"] = alias
                        return self._forward_messages(cmd)
            return

        if not is_valid_number(sender_number):
            print(f" Invalid sender number: {sender_number}")
            return
        if not is_valid_number(recipient_number):
            print(f" Invalid recipient number: {recipient_number}")
            return

        if sender_number == recipient_number:
            print(f" Cannot forward from {from_} to {to} - same number ({sender_number})")
            return

        try:
            callnow_client.sms_forward(
                to_number=sender_number,
                to_number2=recipient_number,
                from_=CALLNOWUSA_NUMBER
            )
            if not hasattr(self.parser, 'forwarding_sessions'):
                self.parser.forwarding_sessions = []
            session = {"from": from_, "to": to, "from_number": sender_number, "to_number": recipient_number}
            self.parser.forwarding_sessions.append(session)
            print(f" Real-time forwarding from {from_} ({sender_number}) to {to} ({recipient_number}) enabled!")
        except Exception as e:
            logger.exception("SMS forwarding failed")
            print(f" Failed to enable forwarding: {str(e)}")

    def _add_contact(self, cmd: Dict[str, Any]):
        """Add a new contact."""
        alias = cmd.get("alias") or cmd.get("name")
        number = cmd.get("number") or cmd.get("phone_number") or cmd.get("phone")
        number = normalize_phone_number(number)
        if not alias or not is_valid_number(number):
            print(" Invalid contact info. Ensure alias and valid phone number are provided.")
            return
        try:
            self.store.add_contact(alias, number)
            print(f" Saved contact {alias} as {number}")
        except Exception as e:
            logger.exception("Failed to add contact: %s", e)
            print(f" Failed to save contact {alias}: {str(e)}")

    def _update_contact(self, cmd: Dict[str, Any]):
        """Update an existing contact's number."""
        alias = cmd.get("alias") or cmd.get("name")
        new_number = cmd.get("number") or cmd.get("phone") or cmd.get("phone_number")
        new_number = normalize_phone_number(new_number)
        if not alias or not is_valid_number(new_number):
            print(" Invalid update info.")
            return
        if self.store.update_contact(alias, new_number):
            print(f" Updated {alias} to {new_number}")
        else:
            print(f" No contact named '{alias}' found.")

    def _delete_contact(self, cmd: Dict[str, Any]):
        """Delete a contact."""
        alias = cmd.get("alias") or cmd.get("name")
        if not alias:
            print(" Missing contact alias.")
            return
        if self.store.delete_contact(alias):
            print(f" Deleted contact '{alias}'.")
        else:
            print(f" No contact named '{alias}' found.")

    def _add_condition(self, cmd: Dict[str, Any]):
        """Add a conditional forwarding rule and schedule SMS forwarding with timers."""
        type_ = cmd.get("type")
        value = cmd.get("value")
        condition_action = cmd.get("condition_action")
        target = cmd.get("target")
        start_time = cmd.get("start_time")
        end_time = cmd.get("end_time")

        if not all([type_, value, condition_action, target]):
            print(" Missing condition parameters (type, value, condition_action, or target).")
            return
        if type_ not in ["priority", "sender", "keyword"]:
            print(" Invalid condition type. Use 'priority', 'sender', or 'keyword'.")
            return
        if condition_action != "forward":
            print(" Invalid condition action. Only 'forward' is supported.")
            return

        value_number = normalize_phone_number(value) if type_ == "sender" and is_valid_number(
            value) else self.store.resolve(value)
        target_number = normalize_phone_number(target) if is_valid_number(target) else self.store.resolve(target)

        if not target_number:
            print(f" Contact or number '{target}' not found.")
            if is_valid_number(target):
                save = input(f" Want to save '{target}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input(" Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(target))
                        print(f" Saved {alias} as {normalize_phone_number(target)}")
                        cmd["target"] = alias
                        return self._add_condition(cmd)
            return
        if not is_valid_number(target_number):
            print(f" Invalid target number: {target_number}")
            return
        if type_ == "sender" and not is_valid_number(value_number):
            print(f" Contact or number '{value}' not found.")
            if is_valid_number(value):
                save = input(f" Want to save '{value}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input(" Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(value))
                        print(f" Saved {alias} as {normalize_phone_number(value)}")
                        cmd["value"] = alias
                        return self._add_condition(cmd)
            return

        now = datetime.now(TZ)
        try:
            if start_time == "now":
                start_dt = now
            else:
                start_dt = datetime.combine(now.date(), datetime.strptime(start_time, "%H:%M").time()).replace(
                    tzinfo=TZ)
                if start_dt < now:
                    start_dt += timedelta(days=1)  # Assume next day if past
            end_dt = None
            store_end_time = None
            if end_time and end_time != "None":
                end_dt = datetime.combine(now.date(), datetime.strptime(end_time, "%H:%M").time()).replace(tzinfo=TZ)
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)  # Handle overnight
                store_end_time = end_dt.strftime("%H:%M")
        except ValueError as e:
            logger.error("Failed to parse times start_time='%s', end_time='%s': %s", start_time, end_time, e)
            print(f" Invalid time format: start_time={start_time}, end_time={end_time}")
            return

        try:
            condition_id = self.store.add_condition(type_, value_number, condition_action, target_number,
                                                    start_dt.strftime("%H:%M"), store_end_time)
            self.parser.last_condition_id = condition_id
            time_str = f" from {start_dt.strftime('%H:%M')} until stopped" if not store_end_time else f" from {start_dt.strftime('%H:%M')} to {store_end_time}"
            print(
                f"Added condition {condition_id}: {type_}={value_number}, {condition_action} to {target_number}{time_str}")
        except Exception as e:
            logger.exception("Failed to add condition: %s", e)
            print(f" Failed to add condition: {str(e)}")
            return

        # Initialize timer storage if not present
        if not hasattr(self, '_condition_timers'):
            self._condition_timers = {}

        # Cancel any existing timers for this condition
        if condition_id in self._condition_timers:
            for timer in self._condition_timers[condition_id]:
                timer.cancel()
                logger.debug("Canceled existing timer for condition %d", condition_id)
            del self._condition_timers[condition_id]

        start_delay = max(0, (start_dt - now).total_seconds())
        self._condition_timers[condition_id] = []

        def start_forwarding():
            try:
                callnow_client.sms_forward(
                    to_number=value_number,
                    to_number2=target_number,
                    from_=CALLNOWUSA_NUMBER
                )
                logger.info("Started forwarding from %s to %s for condition %d", value_number, target_number,
                            condition_id)
                print(f" Started forwarding from {value} to {target} at {start_dt.strftime('%H:%M')}")
            except Exception as e:
                logger.exception("Failed to start forwarding from %s to %s for condition %d: %s", value_number,
                                 target_number, condition_id, e)
                print(f" Failed to start forwarding: {str(e)}")

        def stop_forwarding():
            try:
                callnow_client.sms_forward_stop(
                    to_number=value_number,
                    to_number2=target_number,
                    from_=CALLNOWUSA_NUMBER
                )
                self.store.stop_condition(condition_id)
                logger.info("Stopped forwarding from %s to %s for condition %d", value_number, target_number,
                            condition_id)
                print(f"Stopped forwarding from {value} to {target} at {end_dt.strftime('%H:%M')}")
            except Exception as e:
                logger.exception("Failed to stop forwarding from %s to %s for condition %d: %s", value_number,
                                 target_number, condition_id, e)
                print(f" Failed to stop forwarding: {str(e)}")
            finally:
                if condition_id in self._condition_timers:
                    del self._condition_timers[condition_id]

        # Schedule start timer
        if start_delay > 0:
            start_timer = threading.Timer(start_delay, start_forwarding)
            start_timer.start()
            self._condition_timers[condition_id].append(start_timer)
            logger.info("Scheduled forwarding start for condition %d in %d seconds", condition_id, start_delay)
        else:
            logger.debug("Starting forwarding immediately for condition %d", condition_id)
            start_forwarding()

        # Schedule stop timer
        if end_time and end_dt:
            end_delay = max(0, (end_dt - now).total_seconds())
            if end_delay > start_delay:
                end_timer = threading.Timer(end_delay, stop_forwarding)
                end_timer.start()
                self._condition_timers[condition_id].append(end_timer)
                logger.info("Scheduled forwarding stop for condition %d in %d seconds", condition_id, end_delay)
            else:
                logger.warning("End time %s is before or equal to start time %s for condition %d, skipping stop timer",
                               end_dt.strftime('%H:%M'), start_dt.strftime('%H:%M'), condition_id)
                print(
                    f" End time {end_dt.strftime('%H:%M')} is before or equal to start time {start_dt.strftime('%H:%M')}.")
                return
        else:
            logger.info("No end time specified for condition %d, forwarding will continue until manually stopped",
                        condition_id)
            print(
                f"ℹ️ Forwarding from {value} to {target} will continue until stopped with 'stop condition {condition_id}'")
    def _modify_condition(self, cmd: Dict[str, Any]):
        """Modify an existing conditional forwarding rule."""
        condition_id = cmd.get("id")
        if not condition_id or condition_id == "<last_condition_id>":
            condition_id = self.parser.last_condition_id
        if not condition_id:
            print(" Missing condition ID. Try 'list conditions' to see active conditions.")
            return
        with self.store._lock, self.store.conn:
            cur = self.store.conn.execute("SELECT * FROM conditions WHERE id = ? AND active = 1", (condition_id,))
            row = cur.fetchone()
            if not row:
                print(f" Condition {condition_id} not found or not active.")
                return
            current = {
                "type": row["type"],
                "value": row["value"],
                "action": row["action"],
                "target": row["target"],
                "start_time": row["start_time"],
                "end_time": row["end_time"]
            }
            type_ = cmd.get("type", current["type"])
            value = cmd.get("value", current["value"])
            action = cmd.get("action", current["action"])
            target = cmd.get("target", current["target"])
            start_time = cmd.get("start_time", current["start_time"])
            end_time = cmd.get("end_time", current["end_time"])
            if type_ not in ["priority", "sender", "keyword"]:
                print(" Invalid condition type. Use 'priority', 'sender', or 'keyword'.")
                return
            if action != "forward":
                print(" Invalid action. Only 'forward' is supported.")
                return
            target_number = normalize_phone_number(target) if is_valid_number(target) else self.store.resolve(target)
            if not target_number:
                print(f" Contact '{target}' not found.")
                if is_valid_number(target):
                    save = input(f" Want to save '{target}' as a contact? (y/n): ").strip().lower()
                    if save == "y":
                        alias = input(" Enter a name for this contact: ").strip()
                        if alias:
                            self.store.add_contact(alias, normalize_phone_number(target))
                            print(f" Saved {alias} as {normalize_phone_number(target)}")
                            cmd["target"] = alias
                            return self._modify_condition(cmd)
                return
            if not is_valid_number(target_number):
                print(f" Invalid target number: {target_number}")
                return
            value_number = normalize_phone_number(value) if type_ == "sender" and is_valid_number(value) else value
            if type_ == "sender" and not is_valid_number(value_number):
                print(f" Contact '{value}' not found.")
                if is_valid_number(value):
                    save = input(f" Want to save '{value}' as a contact? (y/n): ").strip().lower()
                    if save == "y":
                        alias = input(" Enter a name for this contact: ").strip()
                        if alias:
                            self.store.add_contact(alias, normalize_phone_number(value))
                            print(f" Saved {alias} as {normalize_phone_number(value)}")
                            cmd["value"] = alias
                            return self._modify_condition(cmd)
                return
            try:
                self.store.conn.execute(
                    """
                    UPDATE conditions SET
                        type = ?, value = ?, action = ?, target = ?, start_time = ?, end_time = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (type_, value_number, action, target_number, start_time, end_time, datetime.now(TZ).isoformat(),
                     condition_id)
                )
                print(f"Modified condition {condition_id}: {type_}={value_number}, {action} to {target_number}" +
                      (f" from {start_time} to {end_time}" if start_time and end_time else ""))
            except Exception as e:
                logger.exception("Failed to modify condition: %s", e)
                print(f" Failed to modify condition: {str(e)}")

    def _list_conditions(self):
        """List all active conditional forwarding rules."""
        conditions = self.store.list_conditions()
        now = datetime.now(TZ)
        active_conditions = []
        for c in conditions:
            if not c.active:
                continue
            if c.end_time:
                try:
                    end_dt = datetime.combine(now.date(), datetime.strptime(c.end_time, "%H:%M").time()).replace(
                        tzinfo=TZ)
                    if end_dt < now:
                        end_dt += timedelta(days=1)  # Handle overnight forwarding
                    if end_dt < now:  # Skip if end_time is still in the past
                        logger.debug("Skipping expired condition %d: end_time=%s", c.id_, c.end_time)
                        continue
                    active_conditions.append(c)
                except ValueError as e:
                    logger.warning("Invalid end_time format for condition %d: %s, skipping", c.id_, c.end_time)
                    continue
            else:
                active_conditions.append(c)
        if not active_conditions:
            print("No active conditions.")
            return
        print(" Active conditions:")
        for cond in active_conditions:
            time_str = f" from {cond.start_time} until stopped" if not cond.end_time else f" from {cond.start_time} to {cond.end_time}"
            print(f" - {cond.id_}: {cond.type_}={cond.value}, {cond.action} to {cond.target}{time_str}")
    def _stop_condition(self, cmd: Dict[str, Any]):
        """Stop a conditional forwarding rule."""
        condition_id = cmd.get("id")
        if not condition_id:
            print(" Missing condition ID. Try 'list conditions' to see active conditions.")
            return
        if self.store.stop_condition(condition_id):
            print(f" Stopped condition {condition_id}")
        else:
            print(f" Condition {condition_id} not found or not active.")

    def _cancel_all_scheduled_sms(self):
        """Cancel all scheduled SMS."""
        try:
            with self.store._lock, self.store.conn:
                cur = self.store.conn.execute("UPDATE scheduled_sms SET active = 0 WHERE active = 1")
                count = cur.rowcount
                if count > 0:
                    print(f" Cancelled {count} scheduled SMS.")
                    # Cancel all timers
                    for timer in self.monitor._scheduled_sms_timers.values():
                        timer.cancel()
                    self.monitor._scheduled_sms_timers.clear()
                else:
                    print("No active scheduled SMS to cancel.")
        except Exception as e:
            logger.exception("Failed to cancel all scheduled SMS: %s", e)
            print(f" Failed to cancel scheduled SMS: {str(e)}")

    def _schedule_sms(self, sms: ScheduledSMS):
        """Schedule an SMS to be sent at the specified time."""
        now = datetime.now(TZ)
        if not sms.active:
            logger.debug("SMS %d not scheduled: inactive", sms.id_)
            return
        if sms.send_time <= now:
            logger.debug("SMS %d not scheduled: past due (%s <= %s)", sms.id_, sms.send_time, now)
            return
        delay = (sms.send_time - now).total_seconds()
        if delay <= 0:
            logger.debug("SMS %d not scheduled: negative or zero delay (%s)", sms.id_, delay)
            return

        def send_sms():
            try:
                callnow_client.messages.create(
                    to=sms.recipient,
                    from_=CALLNOWUSA_NUMBER,
                    body=sms.message,
                )
                self.store.mark_scheduled_sms_sent(sms.id_)
                logger.info("Sent scheduled SMS %d to %s: %s", sms.id_, sms.recipient, sms.message)
            except Exception as e:
                logger.error("Failed to send scheduled SMS %d to %s: %s", sms.id_, sms.recipient, e)
            finally:
                self._scheduled_sms_timers.pop(sms.id_, None)

        logger.debug("Scheduling SMS %d with delay %s seconds", sms.id_, delay)
        timer = threading.Timer(delay, send_sms)
        self._scheduled_sms_timers[sms.id_] = timer
        timer.start()
        logger.info("Scheduled SMS %d to %s at %s", sms.id_, sms.recipient, sms.send_time)

    def _list_scheduled_sms(self):
        """List all active scheduled SMS."""
        scheduled_sms = self.store.list_scheduled_sms()
        if not scheduled_sms:
            print("No active scheduled SMS.")
            return
        print(" Active scheduled SMS:")
        for sms in scheduled_sms:
            print(f" - {sms.id_}: To {sms.recipient} at {sms.send_time.strftime('%Y-%m-%d %H:%M:%S %Z')}: {sms.message}")

    def _fetch_messages(self) -> List[Message]:
        """Fetch cached messages, polling if necessary."""
        if not self.monitor._cache:
            print(" No recent messages. Polling inbox now...")
            try:
                self.monitor._poll()
            except Exception:
                logger.exception("Manual poll failed")
        return self.monitor._cache

    def _filter_by_label(self, label: str) -> List[Message]:
        """Filter cached messages by label."""
        return [m for m in self._fetch_messages() if m.label == label]

    def _show_meetings(self, date_filter: Optional[str] = None):
        """Show messages related to meetings."""
        print(f"Searching for meetings{' ' + date_filter if date_filter else ''}...")
        with self.store._lock, self.store.conn:
            query = "SELECT sender, message, timestamp, direction, label FROM message_log WHERE label IN (?, ?)"
            params = ["meeting", "high priority"]
            if date_filter:
                today = datetime.now(TZ).date()
                if date_filter == "today":
                    query += " AND date(timestamp) = ?"
                    params.append(today.strftime("%Y-%m-%d"))
                elif date_filter == "tomorrow":
                    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                    query += " AND date(timestamp) = ?"
                    params.append(tomorrow)
                elif date_filter == "next week":
                    next_monday = today + timedelta(days=(7 - today.weekday()))
                    next_sunday = next_monday + timedelta(days=6)
                    query += " AND date(timestamp) BETWEEN ? AND ?"
                    params.extend([next_monday.strftime("%Y-%m-%d"), next_sunday.strftime("%Y-%m-%d")])
                elif date_filter == "upcoming":
                    query += " AND date(timestamp) > ?"
                    params.append(today.strftime("%Y-%m-%d"))
            query += " ORDER BY timestamp DESC"
            cur = self.store.conn.execute(query, params)
            messages = []
            for row in cur.fetchall():
                try:
                    ts = parse_datetime(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=TZ)
                    messages.append(
                        Message(
                            sender=row["sender"],
                            body=row["message"],
                            timestamp=ts,
                            direction=row["direction"],
                            label=row["label"]
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to parse message: %s", e)
            if not messages:
                print(f"No meeting messages{' for ' + date_filter if date_filter else ''}.")
                try:
                    self._manual_poll()
                    cur = self.store.conn.execute(query, params)
                    for row in cur.fetchall():
                        ts = parse_datetime(row["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=TZ)
                        messages.append(
                            Message(
                                sender=row["sender"],
                                body=row["message"],
                                timestamp=ts,
                                direction=row["direction"],
                                label=row["label"]
                            )
                        )
                except Exception:
                    logger.exception("Manual poll failed")
                    print(" Polling error.")
            if not messages:
                print(f"Still no meeting messages{' for ' + date_filter if date_filter else ''}.")
            else:
                print(f" Found {len(messages)} meeting messages{' for ' + date_filter if date_filter else ''}:")
                for m in messages:
                    print(f"[{m.timestamp}] {m.sender}: {m.body} ({m.label}, {m.direction})")

    def _show_filtered(self, label: str, date_filter: Optional[str] = None):
        """Show messages filtered by label."""
        print(f"Searching for '{label}' messages{' ' + date_filter if date_filter else ''}...")
        with self.store._lock, self.store.conn:
            query = "SELECT sender, message, timestamp, direction, label FROM message_log WHERE label = ?"
            params = [label]
            if date_filter:
                today = datetime.now(TZ).date()
                if date_filter == "today":
                    query += " AND date(timestamp) = ?"
                    params.append(today.strftime("%Y-%m-%d"))
                elif date_filter == "tomorrow":
                    tomorrow = today + timedelta(days=1)
                    query += " AND date(timestamp) = ?"
                    params.append(tomorrow.strftime("%Y-%m-%d"))
                elif date_filter == "next week":
                    next_monday = today + timedelta(days=(7 - today.weekday()))
                    next_sunday = next_monday + timedelta(days=6)
                    query += " AND date(timestamp) BETWEEN ? AND ?"
                    params.extend([next_monday.strftime("%Y-%m-%d"), next_sunday.strftime("%Y-%m-%d")])
                elif date_filter == "upcoming":
                    query += " AND date(timestamp) > ?"
                    params.append(today.strftime("%Y-%m-%d"))
            query += " ORDER BY timestamp DESC"
            cur = self.store.conn.execute(query, params)
            messages = []
            for row in cur.fetchall():
                try:
                    ts = parse_datetime(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=TZ)
                    messages.append(
                        Message(
                            sender=row["sender"],
                            body=row["message"],
                            timestamp=ts,
                            direction=row["direction"],
                            label=row["label"]
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to parse message: %s", e)
            if not messages:
                print(f"No '{label}' messages{' for ' + date_filter if date_filter else ''}.")
                try:
                    self._manual_poll()
                    cur = self.store.conn.execute(query, params)
                    for row in cur.fetchall():
                        ts = parse_datetime(row["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=TZ)
                        messages.append(
                            Message(
                                sender=row["sender"],
                                body=row["message"],
                                timestamp=ts,
                                direction=row["direction"],
                                label=row["label"]
                            )
                        )
                except Exception:
                    logger.exception("Manual poll failed")
                    print(" Polling error.")
            if not messages:
                print(f"Still no '{label}' messages{' for ' + date_filter if date_filter else ''}")
            else:
                print(f" Found {len(messages)} messages tagged as '{label}'{' for ' + date_filter if date_filter else ''}:")
                for m in messages:
                    print(f"[{m.timestamp}] {m.sender}: {m.body} ({m.label}, {m.direction})")

    def _contact_exists(self, alias: str):
        """Check if a contact exists."""
        num = self.store.resolve(alias)
        if num:
            print(f"Yep, {alias} is {num}.")
        else:
            print(f" Nope, we don't have {alias} saved.")

    def _send_sms(self, cmd: Dict[str, Any]):
        """Send an SMS to a recipient."""
        number = cmd.get("to") or cmd.get("recipient")
        text = cmd.get("message")
        if not number or not text:
            print(" Missing recipient or message.")
            return
        original_input = number.strip()
        number = normalize_phone_number(number) if is_valid_number(number) else number
        if not is_valid_number(number):
            resolved = self.store.resolve(number)
            if resolved:
                number = resolved
            else:
                print(f" No contact '{original_input}' not found.")
                if is_valid_number(original_input):
                    save = input(f" Want to save '{original_input}' as a contact? (y/n): ").strip().lower()
                    if save == "y":
                        alias = input(" Enter a name for this contact: ").strip()
                        if alias:
                            self.store.add_contact(alias, normalize_phone_number(original_input))
                            print(f" Saved {alias} as {normalize_phone_number(original_input)}")
                            self._send_sms({"recipient": alias, "message": text})
                        return
                else:
                    print(" The number wasn't valid")
                return
        if not is_valid_number(number):
            print(" Invalid number.")
            return
        try:
            callnow_client.messages.create(
                to=number,
                from_=CALLNOWUSA_NUMBER,
                body=text,
            )
            print(f"Sent '{text}' to {number}")
        except Exception:
            logger.exception("Send failed")
            print(" Failed to send SMS.")

    def _check_specific_forwarding(self, cmd: Dict[str, Any]):
        """Check if a specific forwarding session exists."""
        from_ = cmd.get("from", "").strip().lower()
        to = cmd.get("to", "").strip().lower()

        if not hasattr(self.parser, 'forwarding_sessions') or not self.parser.forwarding_sessions:
            print(f"No active forwarding sessions found.")
            return

        found = False
        for session in self.parser.forwarding_sessions:
            session_from = session.get("from", "").lower()
            session_to = session.get("to", "").lower()

            if session_from == from_ and session_to == to:
                from_num = session.get("from_number", "")
                to_num = session.get("to_number", "")
                print(f"Yes! Active forwarding: {from_} ({from_num}) → {to} ({to_num})")
                found = True
                break

        if not found:
            print(f"No active forwarding from {from_} to {to}.")
            print("Current active sessions:")
            for session in self.parser.forwarding_sessions:
                session_from = session.get("from")
                session_to = session.get("to")
                from_num = session.get("from_number", "")
                to_num = session.get("to_number", "")
                print(f"   {session_from} ({from_num}) → {session_to} ({to_num})")

    def _show_forwarding(self):
        """Show all active forwarding sessions and conditions."""
        # Display forwarding sessions (from forward_messages)
        if hasattr(self.parser, 'forwarding_sessions') and self.parser.forwarding_sessions:
            print("Active forwarding sessions:")
            for session in self.parser.forwarding_sessions:
                from_ = session.get("from")
                to = session.get("to")
                from_num = session.get("from_number", "")
                to_num = session.get("to_number", "")
                print(f"  {from_} ({from_num}) → {to} ({to_num})")
        else:
            print("No active forwarding sessions (from forward_messages).")

        # Display forwarding conditions (from add_condition)
        conditions = self.store.list_conditions()
        now = datetime.now(TZ)
        forwarding_conditions = []
        for c in conditions:
            if c.action != "forward" or not c.active:
                continue
            if not c.end_time:
                forwarding_conditions.append(c)
                continue
            try:
                end_dt = datetime.combine(now.date(), datetime.strptime(c.end_time, "%H:%M").time()).replace(tzinfo=TZ)
                if end_dt < now:
                    end_dt += timedelta(days=1)  # Handle overnight times
                if end_dt >= now:
                    forwarding_conditions.append(c)
            except ValueError as e:
                logger.warning("Invalid end_time format for condition %d: %s, skipping", c.id_, c.end_time)
                continue
        if forwarding_conditions:
            print("Active forwarding conditions:")
            for cond in forwarding_conditions:
                time_str = f" from {cond.start_time} until stopped" if not cond.end_time else f" from {cond.start_time} to {cond.end_time}"
                print(f" Condition {cond.id_}: {cond.type_}={cond.value} → {cond.target}{time_str}")
        else:
            print("No active forwarding conditions.")
    def _sms_forward(self, cmd: Dict[str, Any]):
        """Set up real-time SMS forwarding (alternative method)."""
        from_ = cmd.get("from", "").strip().lower()
        to = cmd.get("to", "").strip().lower()

        if not from_ or not to:
            print("Missing 'from' or 'to' contact.")
            return

        sender_number = self.store.resolve(from_) if not is_valid_number(from_) else normalize_phone_number(from_)
        recipient_number = self.store.resolve(to) if not is_valid_number(to) else normalize_phone_number(to)

        if not sender_number:
            print(f"Contact '{from_}' not found.")
            if is_valid_number(from_):
                save = input(f" Want to save '{from_}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input("Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(from_))
                        print(f"Saved {alias} as {normalize_phone_number(from_)}")
                        cmd["from"] = alias
                        return self._sms_forward(cmd)
            return
        if not recipient_number:
            print(f"Contact '{to}' not found.")
            if is_valid_number(to):
                save = input(f"Want to save '{to}' as a contact? (y/n): ").strip().lower()
                if save == "y":
                    alias = input("Enter a name for this contact: ").strip()
                    if alias:
                        self.store.add_contact(alias, normalize_phone_number(to))
                        print(f" Saved {alias} as {normalize_phone_number(to)}")
                        cmd["to"] = alias
                        return self._sms_forward(cmd)
            return

        if not is_valid_number(sender_number):
            print(f"Invalid sender number: {sender_number}")
            return
        if not is_valid_number(recipient_number):
            print(f"Invalid recipient number: {recipient_number}")
            return

        if sender_number == recipient_number:
            print(f"Cannot forward from {from_} to {to} - same number ({sender_number})")
            return

        try:
            callnow_client.sms_forward(
                to_number=sender_number,
                to_number2=recipient_number,
                from_=CALLNOWUSA_NUMBER
            )
            if not hasattr(self.parser, 'forwarding_sessions'):
                self.parser.forwarding_sessions = []
            session = {
                "from": from_,
                "to": to,
                "from_number": sender_number,
                "to_number": recipient_number
            }
            self.parser.forwarding_sessions.append(session)
            print(f"Real-time forwarding from {from_} ({sender_number}) to {to} ({recipient_number}) enabled!")
        except Exception as e:
            logger.exception("SMS forwarding failed")
            print(f"Failed to enable forwarding: {str(e)}")

    def _sms_forward_stop(self, cmd: Dict[str, Any]):
        """Stop a forwarding session."""
        if cmd.get("all"):
            if hasattr(self.parser, 'forwarding_sessions') and self.parser.forwarding_sessions:
                print("Stopping all active forwarding sessions...")
                for session in self.parser.forwarding_sessions[:]:
                    try:
                        callnow_client.sms_forward_stop(
                            to_number=session.get("from_number"),
                            to_number2=session.get("to_number"),
                            from_=CALLNOWUSA_NUMBER
                        )
                        print(f"Stopped: {session.get('from')} → {session.get('to')}")
                        self.parser.forwarding_sessions.remove(session)
                    except Exception as e:
                        print(f"Failed to stop: {session.get('from')} → {session.get('to')} - {e}")
                if not self.parser.forwarding_sessions:
                    print("All forwarding sessions stopped!")
            else:
                print("No active forwarding sessions to stop.")
            # Stop all active forwarding conditions
            conditions = self.store.list_conditions()
            forwarding_conditions = [c for c in conditions if c.action == "forward"]
            if forwarding_conditions:
                print("Stopping all active forwarding conditions...")
                for cond in forwarding_conditions:
                    try:
                        callnow_client.sms_forward_stop(
                            to_number=cond.value,
                            to_number2=cond.target,
                            from_=CALLNOWUSA_NUMBER
                        )
                        self.store.stop_condition(cond.id_)
                        print(f"Stopped condition {cond.id_}: {cond.type_}={cond.value} → {cond.target}")
                    except Exception as e:
                        print(f"Failed to stop condition {cond.id_}: {e}")
            return

        if "index" in cmd:
            index = cmd.get("index")
            if not hasattr(self.parser, 'forwarding_sessions') or not self.parser.forwarding_sessions:
                print("No active forwarding sessions to stop.")
                return
            if not isinstance(index, int) or index < 0:
                print("Invalid index for forwarding session.")
                return
            if index >= len(self.parser.forwarding_sessions):
                print(
                    f"No forwarding session at position {index + 1} (only {len(self.parser.forwarding_sessions)} active).")
                return

            session = self.parser.forwarding_sessions[index]
            src_number = session.get("from_number")
            dst_number = session.get("to_number")
            try:
                callnow_client.sms_forward_stop(
                    to_number=src_number,
                    to_number2=dst_number,
                    from_=CALLNOWUSA_NUMBER
                )
                print(
                    f"Stopped forwarding: {session.get('from')} ({src_number}) → {session.get('to')} ({dst_number})")
                self.parser.forwarding_sessions.pop(index)
            except Exception as e:
                print(f"Failed to stop forwarding: {e}")
            return

        src = cmd.get("source")
        dst = cmd.get("destination")
        if not src or not dst:
            print("Missing source or destination. Please specify or use 'stop all forwarding'.")
            return

        src_number = self.store.resolve(src) if not is_valid_number(src) else normalize_phone_number(src)
        if not src_number:
            print(f"Could not resolve source contact '{src}'.")
            return

        dst_number = self.store.resolve(dst) if not is_valid_number(dst) else normalize_phone_number(dst)
        if not dst_number:
            print(f"Could not resolve destination contact '{dst}'.")
            return

        # Check forwarding sessions
        session_to_stop = None
        if hasattr(self.parser, 'forwarding_sessions'):
            for session in self.parser.forwarding_sessions:
                if session.get("from_number", "").lower() == src_number.lower() and session.get("to_number",
                                                                                                "").lower() == dst_number.lower():
                    session_to_stop = session
                    break

        if session_to_stop:
            try:
                callnow_client.sms_forward_stop(
                    to_number=src_number,
                    to_number2=dst_number,
                    from_=CALLNOWUSA_NUMBER
                )
                self.parser.forwarding_sessions.remove(session_to_stop)
                print(f"Stopped forwarding from {src} to {dst}")
            except Exception as e:
                logger.exception("Stop forwarding failed")
                print(f"Failed to stop forwarding: {e}")
            return

        # Check conditions
        conditions = self.store.list_conditions()
        condition_to_stop = None
        for cond in conditions:
            if cond.type_ == "sender" and cond.value == src_number and cond.target == dst_number:
                condition_to_stop = cond
                break

        if condition_to_stop:
            try:
                callnow_client.sms_forward_stop(
                    to_number=src_number,
                    to_number2=dst_number,
                    from_=CALLNOWUSA_NUMBER
                )
                self.store.stop_condition(condition_to_stop.id_)
                print(f"Stopped condition {condition_to_stop.id_}: {src} → {dst}")
            except Exception as e:
                logger.exception("Stop forwarding condition failed")
                print(f"Failed to stop forwarding condition: {e}")
        else:
            print(f"No active forwarding session or condition found from '{src}' to '{dst}'.")
    def _chat(self, text: str):
        """Handle casual chat responses."""
        try:
            rsp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": "Keep it chill and concise."},
                          {"role": "user", "content": text}],
                temperature=0.7,
            )
            print("", rsp.choices[0].message.content.strip())
        except Exception:
            print("LLM glitch.")

    def _show_messages(self, contact: Optional[str] = None, limit: int = 100000, sort: str = "newest",
                       date_filter: Optional[str] = None, direction: str = "all"):
        """Show messages, optionally filtered by contact, direction, or date."""
        with self.store._lock, self.store.conn:
            query = "SELECT sender, message, timestamp, direction, label FROM message_log"
            params = []
            conditions = []

            if contact:
                number = self.store.resolve(contact) if not is_valid_number(contact) else normalize_phone_number(contact)
                if not number:
                    print(f"No contact '{contact}' not found. Did you mean an alias? Try 'list contacts' to check.")
                    all_numbers = {m.sender for m in self._fetch_messages()}
                    similar = [n for n in all_numbers if n.startswith(number[:6])] if number else []
                    if similar:
                        print(f"💡 Did you mean one of these numbers? {', '.join(similar)}")
                    return
                conditions.append("sender = ?")
                params.append(number)

            if direction != "all":
                conditions.append("direction = ?")
                params.append(direction)

            if date_filter:
                today = datetime.now(TZ).date()
                if date_filter == "today":
                    conditions.append("date(timestamp) = ?")
                    params.append(today.strftime("%Y-%m-%d"))
                elif date_filter == "tomorrow":
                    tomorrow = today + timedelta(days=1)
                    conditions.append("date(timestamp) = ?")
                    params.append(tomorrow.strftime("%Y-%m-%d"))
                elif date_filter == "next week":
                    next_monday = today + timedelta(days=(7 - today.weekday()))
                    next_sunday = next_monday + timedelta(days=6)
                    conditions.append("date(timestamp) BETWEEN ? AND ?")
                    params.extend([next_monday.strftime("%Y-%m-%d"), next_sunday.strftime("%Y-%m-%d")])
                elif date_filter == "upcoming":
                    conditions.append("date(timestamp) > ?")
                    params.append(today.strftime("%Y-%m-%d"))

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY timestamp " + ("ASC" if sort == "oldest" else "DESC")

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cur = self.store.conn.execute(query, params)
            messages = []

            for row in cur.fetchall():
                try:
                    ts = parse_datetime(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=TZ)
                    messages.append(
                        Message(
                            sender=row["sender"],
                            body=row["message"],
                            timestamp=ts,
                            direction=row["direction"],
                            label=row["label"]
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to parse message: %s", e)

            if not messages:
                print(f"No {direction} messages found{' for ' + date_filter if date_filter else ''}.")
                try:
                    self._manual_poll()
                    cur = self.store.conn.execute(query, params)
                    for row in cur.fetchall():
                        ts = parse_datetime(row["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=TZ)
                        messages.append(
                            Message(
                                sender=row["sender"],
                                body=row["message"],
                                timestamp=ts,
                                direction=row["direction"],
                                label=row["label"]
                            )
                        )
                except Exception:
                    logger.exception("Manual poll failed")
                    print("Polling error.")

            if not messages:
                print(f"Still no {direction} messages found{' for ' + date_filter if date_filter else ''}.")
                if contact:
                    print(f"Did you mean +14372392447? Check 'list contacts' or try again.")
                return

            print(f"{sort.capitalize()} {min(len(messages), limit)} {direction} messages{' for ' + date_filter if date_filter else ''}:")
            for msg in messages[:limit]:
                print(f"- [{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {msg.sender}: {msg.body} ({msg.label}, {msg.direction})")

    def _dispatch(self, cmd: Dict[str, Any]):
        """Dispatch a parsed command to the appropriate handler."""
        act = cmd.get("action")
        # print("🧠 Parsed Command:", cmd)

        if act == "show_forwarding":
            self._show_forwarding()
        elif act == "show_filtered":
            label = cmd.get("filter") or cmd.get("label")
            if label in {"urgent", "important", "critical", "emergency", "essential"}:
                label = "high priority"
            self._show_filtered(label, cmd.get("date_filter"))
        elif act == "show_messages":
            self._show_messages(
                cmd.get("contact"),
                cmd.get("limit", 100000),
                cmd.get("sort", "newest"),
                cmd.get("date_filter"),
                cmd.get("direction", "all")
            )
        elif act == "list_contacts":
            self._list_contacts()
        elif act == "poll":
            self._manual_poll()
        elif act == "show_meetings":
            self._show_meetings(cmd.get("date_filter"))
        elif act == "update_contact":
            self._update_contact(cmd)
        elif act == "contact_exists":
            alias = cmd.get("alias") or cmd.get("name")
            self._contact_exists(alias)
        elif act == "forward_messages":
            self._forward_messages(cmd)
        elif act == "send_sms":
            self._send_sms(cmd)
        elif act == "add_contact":
            self._add_contact(cmd)
        elif act == "add_condition":
            self._add_condition(cmd)
        elif act == "list_conditions":
            self._list_conditions()
        elif act == "stop_condition":
            self._stop_condition(cmd)
        elif act == "modify_condition":
            self._modify_condition(cmd)
        elif act == "schedule_sms":
            # Handle scheduling an SMS
            recipient = cmd.get("recipient")
            message = cmd.get("message")
            send_time_str = cmd.get("send_time")
            if not recipient or not message or not send_time_str:
                print("Missing recipient, message, or send time.")
                return
            # Normalize recipient
            recipient = normalize_phone_number(recipient) if is_valid_number(recipient) else self.store.resolve(
                recipient)
            if not recipient:
                print(f"Contact '{cmd.get('recipient')}' not found.")
                return
            if not is_valid_number(recipient):
                print(f"Invalid recipient number: {recipient}")
                return
            # Parse send time
            try:
                send_time = parse_time(send_time_str, datetime.now(TZ))
            except ValueError as e:
                logger.error("Failed to parse send time '%s': %s", send_time_str, e)
                print(f"Invalid send time format: {send_time_str}")
                return
            # Create ScheduledSMS object
            try:
                sms_id = self.store.add_scheduled_sms(recipient, message, send_time)
                sms = ScheduledSMS(
                    id_=sms_id,
                    recipient=recipient,
                    message=message,
                    send_time=send_time,
                    active=True,
                    created_at=datetime.now(TZ)
                )
                self.monitor._schedule_sms(sms)
                print(
                    f"Scheduled SMS {sms_id} to {recipient} at {send_time.strftime('%Y-%m-%d %H:%M:%S %Z')}: {message}")
            except Exception as e:
                logger.exception("Failed to schedule SMS: %s", e)
                print(f"Failed to schedule SMS: {str(e)}")
        elif act == "list_scheduled_sms":
            self._list_scheduled_sms()
        elif act == "cancel_all_scheduled_sms":
            self._cancel_all_scheduled_sms()
        elif act == "chat":
            message = cmd.get("message", "").strip().lower()
            if hasattr(self.parser, 'last_cmd') and self.parser.last_cmd:
                last = self.parser.last_cmd
                last_action = last.get("action")
                FOLLOWUP_TRIGGERS = {
                    "create it", "create it then", "add it", "add them", "make it", "make it then",
                    "do it", "ok", "okay", "sure", "go ahead", "fine", "yeah", "yes"
                }
                if any(phrase in message for phrase in FOLLOWUP_TRIGGERS):
                    if last_action == "update_contact":
                        print("💡 Gotcha. Since update failed, I’ll just create the contact instead.")
                        return self._add_contact({
                            "action": "add_contact",
                            "alias": last.get("alias"),
                            "number": last.get("number")
                        })
            if message:
                self._chat(message)
            else:
                print("Missing chat message.")
        elif act == "sms_forward":
            self._sms_forward(cmd)
        elif act == "sms_forward_stop":
            self._sms_forward_stop(cmd)
        elif act == "check_specific_forwarding":
            self._check_specific_forwarding(cmd)
        elif act == "delete_contact":
            self._delete_contact(cmd)
        else:
            print("Unknown action: ", act)
if __name__ == "__main__":
    def _sigint(sig, frame):
        print("\n Exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)
    SMSAssistant().run()

