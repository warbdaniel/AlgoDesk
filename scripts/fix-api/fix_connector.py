"""
Core FIX 4.4 connector with SSL, logon/logout/heartbeat,
sequence number persistence, auto-reconnect, thread-safety,
message validation, sequence gap detection, and connection health metrics.
"""

import os
import ssl
import socket
import time
import json
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

import simplefix

logger = logging.getLogger("fix_connector")

SOH = b"\x01"
FIX_VERSION = b"FIX.4.4"
SEQ_FILE = Path(__file__).parent / ".seq_numbers.json"
HEARTBEAT_INTERVAL = 30
MAX_RECV_BUFFER = 1024 * 1024  # 1 MB max receive buffer to prevent memory exhaustion
MAX_MESSAGE_SIZE = 64 * 1024   # 64 KB max single FIX message


# ── Connection health metrics ────────────────────────────────────

@dataclass
class ConnectionHealth:
    """Tracks connection quality metrics."""
    connected_since: float = 0.0
    last_send_time: float = 0.0
    last_recv_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    reconnect_count: int = 0
    last_reconnect_time: float = 0.0
    send_errors: int = 0
    recv_errors: int = 0
    seq_gaps_detected: int = 0
    invalid_messages: int = 0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def uptime_seconds(self) -> float:
        if self.connected_since <= 0:
            return 0.0
        return time.time() - self.connected_since

    def to_dict(self) -> dict:
        return {
            "connected_since": datetime.fromtimestamp(
                self.connected_since, tz=timezone.utc
            ).isoformat() if self.connected_since > 0 else None,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "reconnect_count": self.reconnect_count,
            "send_errors": self.send_errors,
            "recv_errors": self.recv_errors,
            "seq_gaps_detected": self.seq_gaps_detected,
            "invalid_messages": self.invalid_messages,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class FIXConnector:
    """Thread-safe FIX 4.4 client with SSL, heartbeat, auto-reconnect,
    message validation, sequence gap detection, and health monitoring."""

    def __init__(
        self,
        host: str,
        port: int,
        sender_comp_id: str,
        target_comp_id: str,
        sender_sub_id: str,
        password: str,
        on_message=None,
        reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.host = host
        self.port = port
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.sender_sub_id = sender_sub_id
        self.password = password
        self.on_message = on_message
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._sock = None
        self._ssl_sock = None
        self._connected = False
        self._logged_in = False
        self._should_run = False

        self._send_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._send_seq = 1
        self._recv_seq = 1
        self._last_send_time = 0.0
        self._last_recv_time = 0.0

        self._recv_thread = None
        self._heartbeat_thread = None

        self._parser = simplefix.FixParser()
        self._seq_key = f"{sender_sub_id}_{port}"

        # Connection health metrics
        self.health = ConnectionHealth()

        # Buffer overflow protection
        self._recv_buf_size = 0

        # Pending test requests for latency measurement
        self._pending_test_requests: dict[str, float] = {}

        self._load_sequence_numbers()

    # ── Sequence number persistence ──────────────────────────────

    def _load_sequence_numbers(self):
        try:
            if SEQ_FILE.exists():
                data = json.loads(SEQ_FILE.read_text())
                entry = data.get(self._seq_key, {})
                self._send_seq = entry.get("send", 1)
                self._recv_seq = entry.get("recv", 1)
                logger.info(
                    "Loaded seq numbers: send=%d recv=%d", self._send_seq, self._recv_seq
                )
        except Exception:
            logger.warning("Could not load sequence numbers, starting from 1")
            self._send_seq = 1
            self._recv_seq = 1

    def _save_sequence_numbers(self):
        try:
            data = {}
            if SEQ_FILE.exists():
                data = json.loads(SEQ_FILE.read_text())
            data[self._seq_key] = {"send": self._send_seq, "recv": self._recv_seq}
            SEQ_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.warning("Could not save sequence numbers")

    def reset_sequence_numbers(self):
        with self._seq_lock:
            self._send_seq = 1
            self._recv_seq = 1
            self._save_sequence_numbers()

    # ── Connection ───────────────────────────────────────────────

    def connect(self):
        """Establish SSL connection and start background threads."""
        self._should_run = True
        self._do_connect()

    def _do_connect(self):
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.settimeout(15)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self._ssl_sock = ctx.wrap_socket(raw, server_hostname=self.host)
            self._ssl_sock.connect((self.host, self.port))
            self._ssl_sock.settimeout(1)
            self._connected = True
            self.health.connected_since = time.time()
            self._recv_buf_size = 0
            logger.info("SSL connected to %s:%d", self.host, self.port)

            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True, name=f"fix-recv-{self._seq_key}"
            )
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name=f"fix-hb-{self._seq_key}"
            )
            self._recv_thread.start()
            self._heartbeat_thread.start()

            self._send_logon()
        except Exception as e:
            logger.error("Connection failed: %s", e)
            self._connected = False
            if self._should_run:
                self._schedule_reconnect()

    def disconnect(self):
        """Send logout and close the connection."""
        self._should_run = False
        if self._logged_in:
            self._send_logout()
            time.sleep(1)
        self._cleanup()

    def _cleanup(self):
        with self._state_lock:
            self._logged_in = False
            self._connected = False
        if self._ssl_sock:
            try:
                self._ssl_sock.close()
            except Exception:
                pass
            self._ssl_sock = None
        self._save_sequence_numbers()

    def _schedule_reconnect(self):
        if not self._should_run:
            return
        delay = min(self.reconnect_delay, self.max_reconnect_delay)
        logger.info("Reconnecting in %.1fs ...", delay)
        self.reconnect_delay = min(delay * 2, self.max_reconnect_delay)
        self.health.reconnect_count += 1
        self.health.last_reconnect_time = time.time()
        threading.Timer(delay, self._reconnect).start()

    def _reconnect(self):
        if not self._should_run:
            return
        self._cleanup()
        self._parser = simplefix.FixParser()
        self._do_connect()

    @property
    def is_logged_in(self):
        return self._logged_in

    @property
    def is_connected(self):
        return self._connected

    # ── Message building & sending ───────────────────────────────

    def _next_send_seq(self) -> int:
        with self._seq_lock:
            seq = self._send_seq
            self._send_seq += 1
            return seq

    def build_message(self, msg_type: bytes) -> simplefix.FixMessage:
        msg = simplefix.FixMessage()
        msg.append_pair(8, FIX_VERSION)
        msg.append_pair(35, msg_type)
        msg.append_pair(49, self.sender_comp_id.encode())
        msg.append_pair(56, self.target_comp_id.encode())
        msg.append_pair(50, self.sender_sub_id.encode())
        msg.append_pair(
            52,
            datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3].encode(),
        )
        return msg

    def send_message(self, msg: simplefix.FixMessage):
        seq = self._next_send_seq()
        msg.append_pair(34, str(seq).encode())
        raw = msg.encode()

        if len(raw) > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {len(raw)} bytes (max {MAX_MESSAGE_SIZE})")

        with self._send_lock:
            if not self._connected or not self._ssl_sock:
                raise ConnectionError("Not connected")
            try:
                self._ssl_sock.sendall(raw)
                self._last_send_time = time.monotonic()
                self.health.last_send_time = time.time()
                self.health.messages_sent += 1
                msg_type = _get_field(msg, 35)
                logger.debug("SENT [%s] seq=%d", msg_type, seq)
            except Exception as e:
                logger.error("Send failed: %s", e)
                self.health.send_errors += 1
                self._connected = False
                if self._should_run:
                    self._schedule_reconnect()
                raise
        self._save_sequence_numbers()

    # ── Session messages ─────────────────────────────────────────

    def _send_logon(self):
        msg = self.build_message(b"A")
        msg.append_pair(98, b"0")  # EncryptMethod = None
        msg.append_pair(108, str(HEARTBEAT_INTERVAL).encode())
        msg.append_pair(141, b"Y")  # ResetSeqNumFlag
        msg.append_pair(554, self.password.encode())
        self.reset_sequence_numbers()
        self.send_message(msg)
        logger.info("Logon sent to %s:%d", self.host, self.port)

    def _send_logout(self):
        try:
            msg = self.build_message(b"5")
            self.send_message(msg)
            logger.info("Logout sent")
        except Exception:
            pass

    def _send_heartbeat(self, test_req_id: str = None):
        msg = self.build_message(b"0")
        if test_req_id:
            msg.append_pair(112, test_req_id.encode())
        self.send_message(msg)

    def _send_test_request(self):
        msg = self.build_message(b"1")
        req_id = str(int(time.time()))
        msg.append_pair(112, req_id.encode())
        self._pending_test_requests[req_id] = time.monotonic()
        # Prune old pending requests (older than 60s)
        cutoff = time.monotonic() - 60
        stale = [k for k, v in self._pending_test_requests.items() if v < cutoff]
        for k in stale:
            del self._pending_test_requests[k]
        self.send_message(msg)

    # ── Receive loop ─────────────────────────────────────────────

    def _recv_loop(self):
        while self._should_run and self._connected:
            try:
                data = self._ssl_sock.recv(4096)
                if not data:
                    logger.warning("Connection closed by remote")
                    self._connected = False
                    break
                self._recv_buf_size += len(data)
                if self._recv_buf_size > MAX_RECV_BUFFER:
                    logger.error(
                        "Receive buffer overflow (%d bytes), reconnecting",
                        self._recv_buf_size,
                    )
                    self.health.recv_errors += 1
                    self._connected = False
                    break
                self._parser.append_buffer(data)
                while True:
                    msg = self._parser.get_message()
                    if msg is None:
                        break
                    self._recv_buf_size = 0  # Reset after successful parse
                    self._last_recv_time = time.monotonic()
                    self.health.last_recv_time = time.time()
                    self.health.messages_received += 1
                    self._handle_message(msg)
            except socket.timeout:
                continue
            except Exception as e:
                if self._should_run:
                    logger.error("Recv error: %s", e)
                    self.health.recv_errors += 1
                    self._connected = False
                break

        if self._should_run:
            self._schedule_reconnect()

    def _validate_message(self, msg: simplefix.FixMessage) -> bool:
        """Validate essential FIX message fields. Returns True if valid."""
        msg_type = _get_field(msg, 35)
        if not msg_type:
            logger.warning("Received message with no MsgType (35)")
            self.health.invalid_messages += 1
            return False

        # Validate SenderCompID matches expected TargetCompID
        sender = _get_field(msg, 49)
        if sender and sender != self.target_comp_id:
            logger.warning(
                "Unexpected SenderCompID: got %s, expected %s", sender, self.target_comp_id
            )
            self.health.invalid_messages += 1
            return False

        # Validate TargetCompID matches our SenderCompID
        target = _get_field(msg, 56)
        if target and target != self.sender_comp_id:
            logger.warning(
                "Unexpected TargetCompID: got %s, expected %s", target, self.sender_comp_id
            )
            self.health.invalid_messages += 1
            return False

        return True

    def _check_sequence_gap(self, seq_num_str: str | None) -> bool:
        """Check for sequence number gaps. Returns True if gap detected."""
        if not seq_num_str:
            return False
        try:
            seq_num = int(seq_num_str)
        except ValueError:
            return False

        with self._seq_lock:
            expected = self._recv_seq
            if seq_num > expected:
                gap_size = seq_num - expected
                logger.warning(
                    "Sequence gap detected: expected %d, got %d (gap=%d)",
                    expected, seq_num, gap_size,
                )
                self.health.seq_gaps_detected += 1
                # Send Resend Request (35=2) for missing messages
                self._request_resend(expected, seq_num - 1)
                self._recv_seq = seq_num + 1
                return True
            elif seq_num < expected:
                # Possible duplicate or resend — log but don't reject
                logger.debug("Received seq %d, expected %d (possible resend)", seq_num, expected)
            else:
                self._recv_seq = seq_num + 1
        return False

    def _request_resend(self, begin_seq: int, end_seq: int):
        """Send a Resend Request (35=2) for missing sequence range."""
        try:
            msg = self.build_message(b"2")
            msg.append_pair(7, str(begin_seq).encode())   # BeginSeqNo
            msg.append_pair(16, str(end_seq).encode())     # EndSeqNo
            self.send_message(msg)
            logger.info("Resend request sent for seq %d-%d", begin_seq, end_seq)
        except Exception:
            logger.exception("Failed to send resend request")

    def _handle_message(self, msg: simplefix.FixMessage):
        # Validate message
        if not self._validate_message(msg):
            return

        msg_type = _get_field(msg, 35)
        seq_num = _get_field(msg, 34)

        # Check for sequence gaps (skip for logon/logout which may reset seq)
        if msg_type not in ("A", "5"):
            self._check_sequence_gap(seq_num)
        elif seq_num:
            with self._seq_lock:
                self._recv_seq = int(seq_num) + 1

        if msg_type == "A":  # Logon
            logger.info("Logon confirmed")
            with self._state_lock:
                self._logged_in = True
            self.reconnect_delay = 5.0
        elif msg_type == "5":  # Logout
            logger.info("Logout received")
            with self._state_lock:
                self._logged_in = False
            if self._should_run:
                self._schedule_reconnect()
            return
        elif msg_type == "0":  # Heartbeat
            # Measure latency from test request -> heartbeat response
            test_id = _get_field(msg, 112)
            if test_id and test_id in self._pending_test_requests:
                sent_at = self._pending_test_requests.pop(test_id)
                latency_ms = (time.monotonic() - sent_at) * 1000
                self.health.latency_samples.append(latency_ms)
                logger.debug("Heartbeat latency: %.1fms", latency_ms)
            else:
                logger.debug("Heartbeat received")
        elif msg_type == "1":  # Test Request
            test_id = _get_field(msg, 112)
            self._send_heartbeat(test_id)
        elif msg_type == "3":  # Reject
            reason = _get_field(msg, 58)
            ref_seq = _get_field(msg, 45)
            logger.warning("Session reject: seq=%s reason=%s", ref_seq, reason)
        elif msg_type == "j":  # Business reject
            reason = _get_field(msg, 58)
            ref_id = _get_field(msg, 372)
            logger.warning("Business reject: refMsgType=%s reason=%s", ref_id, reason)

        if self.on_message:
            try:
                self.on_message(msg)
            except Exception:
                logger.exception("Error in message callback")

        self._save_sequence_numbers()

    # ── Heartbeat loop ───────────────────────────────────────────

    def _heartbeat_loop(self):
        while self._should_run and self._connected:
            time.sleep(1)
            now = time.monotonic()
            if self._logged_in and (now - self._last_send_time) > HEARTBEAT_INTERVAL:
                try:
                    self._send_heartbeat()
                except Exception:
                    break
            if self._logged_in and (now - self._last_recv_time) > HEARTBEAT_INTERVAL * 2:
                logger.warning("No data received for %ds, sending test request",
                               HEARTBEAT_INTERVAL * 2)
                try:
                    self._send_test_request()
                except Exception:
                    break
            # Force reconnect if no data for 3x heartbeat interval
            if self._logged_in and (now - self._last_recv_time) > HEARTBEAT_INTERVAL * 3:
                logger.error(
                    "Connection appears dead (no data for %ds), forcing reconnect",
                    HEARTBEAT_INTERVAL * 3,
                )
                self._connected = False
                break


def _get_field(msg: simplefix.FixMessage, tag: int) -> str | None:
    """Extract a FIX field value as string."""
    val = msg.get(tag)
    if val is None:
        return None
    return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)


def get_all_fields(msg: simplefix.FixMessage, tag: int) -> list[str]:
    """Extract all values for a repeating FIX tag."""
    results = []
    for t, v in msg.pairs:
        if int(t) == tag:
            results.append(v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v))
    return results


def msg_to_dict(msg: simplefix.FixMessage) -> dict:
    """Convert a FIX message to a flat dict (last value wins for dups)."""
    d = {}
    for t, v in msg.pairs:
        key = int(t)
        d[key] = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
    return d
