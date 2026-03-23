"""
Core FIX 4.4 connector with SSL, logon/logout/heartbeat,
sequence number persistence, auto-reconnect, and thread-safety.
"""

import os
import ssl
import socket
import time
import json
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone

import simplefix

logger = logging.getLogger("fix_connector")

SOH = b"\x01"
FIX_VERSION = b"FIX.4.4"
SEQ_FILE = Path(__file__).parent / ".seq_numbers.json"
HEARTBEAT_INTERVAL = 30


class FIXConnector:
    """Thread-safe FIX 4.4 client with SSL, heartbeat, and auto-reconnect."""

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
        with self._send_lock:
            if not self._connected or not self._ssl_sock:
                raise ConnectionError("Not connected")
            try:
                self._ssl_sock.sendall(raw)
                self._last_send_time = time.monotonic()
                msg_type = _get_field(msg, 35)
                logger.debug("SENT [%s] seq=%d", msg_type, seq)
            except Exception as e:
                logger.error("Send failed: %s", e)
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
        self.send_message(msg)

    # ── Receive loop ─────────────────────────────────────────────

    def _recv_loop(self):
        buf = b""
        while self._should_run and self._connected:
            try:
                data = self._ssl_sock.recv(4096)
                if not data:
                    logger.warning("Connection closed by remote")
                    self._connected = False
                    break
                buf += data
                self._parser.append_buffer(data)
                while True:
                    msg = self._parser.get_message()
                    if msg is None:
                        break
                    self._last_recv_time = time.monotonic()
                    self._handle_message(msg)
            except socket.timeout:
                continue
            except Exception as e:
                if self._should_run:
                    logger.error("Recv error: %s", e)
                    self._connected = False
                break

        if self._should_run:
            self._schedule_reconnect()

    def _handle_message(self, msg: simplefix.FixMessage):
        msg_type = _get_field(msg, 35)
        seq_num = _get_field(msg, 34)
        if seq_num:
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
            logger.debug("Heartbeat received")
        elif msg_type == "1":  # Test Request
            test_id = _get_field(msg, 112)
            self._send_heartbeat(test_id)
        elif msg_type == "3":  # Reject
            reason = _get_field(msg, 58)
            logger.warning("Session reject: %s", reason)
        elif msg_type == "j":  # Business reject
            reason = _get_field(msg, 58)
            logger.warning("Business reject: %s", reason)

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
                logger.warning("No data received, sending test request")
                try:
                    self._send_test_request()
                except Exception:
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
