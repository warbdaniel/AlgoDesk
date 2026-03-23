#!/usr/bin/env python3
"""
Trading Desk Guardian - System Watchdog & Protector
====================================================
Monitors system health, services, security, and file integrity.
Automatically recovers from failures and alerts on anomalies.
"""

import os
import sys
import json
import time
import signal
import hashlib
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler
from collections import defaultdict

import psutil
import yaml


class AlertManager:
    """Manages alerts with rate limiting and severity tracking."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.alert_log = config['general'].get('alert_log', '/opt/trading-desk/logs/guardian/alerts.log')
        self.rate_limit = config['alerting'].get('rate_limit', 60)
        self.last_alerts = {}  # key -> timestamp
        self.alert_counts = defaultdict(int)
        self.daily_alerts = []

        # Setup alert file logger
        os.makedirs(os.path.dirname(self.alert_log), exist_ok=True)
        self.alert_handler = RotatingFileHandler(
            self.alert_log, maxBytes=10*1024*1024, backupCount=3
        )
        self.alert_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(message)s')
        )
        self.alert_logger = logging.getLogger('guardian.alerts')
        self.alert_logger.addHandler(self.alert_handler)
        self.alert_logger.setLevel(logging.INFO)

    def send(self, severity, category, message, key=None):
        """Send an alert with rate limiting."""
        alert_key = key or f"{category}:{message[:50]}"
        now = time.time()

        # Rate limiting
        if alert_key in self.last_alerts:
            elapsed = now - self.last_alerts[alert_key]
            if elapsed < self.rate_limit:
                return False

        self.last_alerts[alert_key] = now
        self.alert_counts[severity] += 1

        alert_line = f"[{severity}] [{category}] {message}"
        self.alert_logger.info(alert_line)
        self.daily_alerts.append({
            'time': datetime.now().isoformat(),
            'severity': severity,
            'category': category,
            'message': message
        })

        # Also log to main logger with appropriate level
        if severity == 'EMERGENCY':
            self.logger.critical(f"🚨 {alert_line}")
        elif severity == 'CRITICAL':
            self.logger.error(f"🔴 {alert_line}")
        elif severity == 'WARNING':
            self.logger.warning(f"🟡 {alert_line}")
        else:
            self.logger.info(f"🟢 {alert_line}")

        return True

    def get_daily_summary(self):
        """Return and reset daily alerts."""
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_alerts': len(self.daily_alerts),
            'by_severity': dict(self.alert_counts),
            'alerts': self.daily_alerts[-50:]  # last 50
        }
        self.daily_alerts = []
        self.alert_counts = defaultdict(int)
        return summary


class SystemMonitor:
    """Monitors system resources: CPU, RAM, disk, swap, load."""

    def __init__(self, config, alert_mgr, logger):
        self.config = config.get('resources', {})
        self.alert = alert_mgr
        self.logger = logger
        self.cpu_history = []
        self.cpu_cores = psutil.cpu_count()

    def check_all(self):
        """Run all system resource checks."""
        results = {
            'cpu': self.check_cpu(),
            'memory': self.check_memory(),
            'disk': self.check_disk(),
            'swap': self.check_swap(),
            'load': self.check_load(),
            'uptime': self.get_uptime(),
        }
        return results

    def check_cpu(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        cfg = self.config.get('cpu', {})
        self.cpu_history.append((time.time(), cpu_percent))
        # Keep only last 5 minutes
        cutoff = time.time() - 300
        self.cpu_history = [(t, v) for t, v in self.cpu_history if t > cutoff]

        status = 'OK'
        sustained = cfg.get('sustained_seconds', 60)
        # Check sustained high CPU
        if len(self.cpu_history) > 1:
            recent = [v for t, v in self.cpu_history if t > time.time() - sustained]
            if recent and all(v > cfg.get('critical_percent', 95) for v in recent):
                self.alert.send('CRITICAL', 'CPU', f'CPU sustained at {cpu_percent:.1f}% for {sustained}s')
                status = 'CRITICAL'
            elif recent and all(v > cfg.get('warning_percent', 80) for v in recent):
                self.alert.send('WARNING', 'CPU', f'CPU sustained at {cpu_percent:.1f}% for {sustained}s')
                status = 'WARNING'

        return {'percent': cpu_percent, 'cores': self.cpu_cores, 'status': status}

    def check_memory(self):
        mem = psutil.virtual_memory()
        cfg = self.config.get('memory', {})
        status = 'OK'

        if mem.percent > cfg.get('critical_percent', 95):
            self.alert.send('CRITICAL', 'MEMORY',
                f'Memory at {mem.percent:.1f}% ({mem.used // (1024**3):.1f}GB / {mem.total // (1024**3):.1f}GB)')
            status = 'CRITICAL'
        elif mem.percent > cfg.get('warning_percent', 80):
            self.alert.send('WARNING', 'MEMORY',
                f'Memory at {mem.percent:.1f}% ({mem.used // (1024**3):.1f}GB / {mem.total // (1024**3):.1f}GB)')
            status = 'WARNING'

        return {
            'percent': mem.percent,
            'used_gb': round(mem.used / (1024**3), 2),
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'status': status
        }

    def check_disk(self):
        cfg = self.config.get('disk', {})
        paths = cfg.get('check_paths', ['/'])
        results = {}

        for path in paths:
            try:
                usage = psutil.disk_usage(path)
                status = 'OK'
                if usage.percent > cfg.get('critical_percent', 90):
                    self.alert.send('CRITICAL', 'DISK',
                        f'Disk {path} at {usage.percent:.1f}% ({usage.free // (1024**3)}GB free)')
                    status = 'CRITICAL'
                elif usage.percent > cfg.get('warning_percent', 80):
                    self.alert.send('WARNING', 'DISK',
                        f'Disk {path} at {usage.percent:.1f}% ({usage.free // (1024**3)}GB free)')
                    status = 'WARNING'

                results[path] = {
                    'percent': usage.percent,
                    'free_gb': round(usage.free / (1024**3), 2),
                    'total_gb': round(usage.total / (1024**3), 2),
                    'status': status
                }
            except Exception as e:
                results[path] = {'status': 'ERROR', 'error': str(e)}

        return results

    def check_swap(self):
        swap = psutil.swap_memory()
        cfg = self.config.get('swap', {})
        status = 'OK'

        if swap.total > 0:
            if swap.percent > cfg.get('critical_percent', 80):
                self.alert.send('CRITICAL', 'SWAP',
                    f'Swap at {swap.percent:.1f}% - trading latency affected!')
                status = 'CRITICAL'
            elif swap.percent > cfg.get('warning_percent', 50):
                self.alert.send('WARNING', 'SWAP',
                    f'Swap at {swap.percent:.1f}% - may affect trading performance')
                status = 'WARNING'

        return {'percent': swap.percent, 'used_mb': round(swap.used / (1024**2), 1), 'status': status}

    def check_load(self):
        load1, load5, load15 = psutil.getloadavg()
        cfg = self.config.get('load_average', {})
        warn_mult = cfg.get('warning_multiplier', 2.0)
        crit_mult = cfg.get('critical_multiplier', 4.0)
        status = 'OK'

        if load5 > self.cpu_cores * crit_mult:
            self.alert.send('CRITICAL', 'LOAD', f'Load avg {load1:.1f}/{load5:.1f}/{load15:.1f} (CPUs: {self.cpu_cores})')
            status = 'CRITICAL'
        elif load5 > self.cpu_cores * warn_mult:
            self.alert.send('WARNING', 'LOAD', f'Load avg {load1:.1f}/{load5:.1f}/{load15:.1f} (CPUs: {self.cpu_cores})')
            status = 'WARNING'

        return {'load1': load1, 'load5': load5, 'load15': load15, 'status': status}

    def get_uptime(self):
        boot = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot
        return {'boot_time': boot.isoformat(), 'uptime_hours': round(uptime.total_seconds() / 3600, 1)}


class ServiceMonitor:
    """Monitors Docker containers, ports, and processes."""

    def __init__(self, config, alert_mgr, logger):
        self.config = config.get('services', {})
        self.recovery_config = config.get('recovery', {})
        self.alert = alert_mgr
        self.logger = logger
        self.restart_counts = defaultdict(int)
        self.last_restart = defaultdict(float)

    def check_all(self):
        results = {
            'docker_containers': self.check_docker_containers(),
            'ports': self.check_ports(),
            'processes': self.check_processes(),
        }
        return results

    def check_docker_containers(self):
        results = {}
        for container_cfg in self.config.get('docker_containers', []):
            name = container_cfg['name']
            try:
                result = subprocess.run(
                    ['docker', 'inspect', '--format', '{{.State.Status}}', name],
                    capture_output=True, text=True, timeout=10
                )
                status = result.stdout.strip()
                if status != 'running':
                    self.alert.send('CRITICAL', 'SERVICE',
                        f'Docker container "{name}" is {status or "not found"}!')
                    if container_cfg.get('restart_on_failure') and self.recovery_config.get('auto_restart_services'):
                        self._restart_container(name, container_cfg)
                    results[name] = {'status': status or 'not_found', 'healthy': False}
                else:
                    results[name] = {'status': 'running', 'healthy': True}
            except subprocess.TimeoutExpired:
                self.alert.send('CRITICAL', 'SERVICE', f'Docker inspect timed out for "{name}"')
                results[name] = {'status': 'timeout', 'healthy': False}
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e), 'healthy': False}
        return results

    def _restart_container(self, name, cfg):
        max_restarts = cfg.get('max_restarts', 3)
        cooldown = cfg.get('restart_cooldown', 300)
        now = time.time()

        if self.restart_counts[name] >= max_restarts:
            self.alert.send('EMERGENCY', 'SERVICE',
                f'Container "{name}" exceeded max restarts ({max_restarts}). Manual intervention needed!')
            return

        if now - self.last_restart[name] < cooldown:
            return

        self.logger.info(f'Attempting to restart container "{name}" (attempt {self.restart_counts[name] + 1}/{max_restarts})')
        try:
            subprocess.run(['docker', 'restart', name], capture_output=True, timeout=60)
            self.restart_counts[name] += 1
            self.last_restart[name] = now
            time.sleep(5)
            # Verify it came back
            result = subprocess.run(
                ['docker', 'inspect', '--format', '{{.State.Status}}', name],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip() == 'running':
                self.alert.send('INFO', 'RECOVERY', f'Container "{name}" successfully restarted')
            else:
                self.alert.send('CRITICAL', 'RECOVERY', f'Container "{name}" failed to restart')
        except Exception as e:
            self.alert.send('CRITICAL', 'RECOVERY', f'Failed to restart container "{name}": {e}')

    def check_ports(self):
        results = {}
        connections = psutil.net_connections(kind='inet')
        listening = {c.laddr.port for c in connections if c.status == 'LISTEN'}

        for port_cfg in self.config.get('ports', []):
            port = port_cfg['port']
            name = port_cfg.get('name', f'port-{port}')
            is_listening = port in listening
            if not is_listening:
                self.alert.send('CRITICAL', 'SERVICE', f'Port {port} ({name}) is NOT listening!')
            results[name] = {'port': port, 'listening': is_listening, 'status': 'OK' if is_listening else 'DOWN'}
        return results

    def check_processes(self):
        results = {}
        running_names = set()
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                running_names.add(proc.info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        for proc_cfg in self.config.get('processes', []):
            name = proc_cfg['name']
            found = name in running_names
            if not found and proc_cfg.get('required', False):
                self.alert.send('CRITICAL', 'SERVICE', f'Required process "{name}" is not running!')
            results[name] = {'running': found, 'required': proc_cfg.get('required', False)}
        return results


class NetworkSecurity:
    """Monitors network for suspicious activity."""

    def __init__(self, config, alert_mgr, logger):
        self.config = config.get('network', {})
        self.alert = alert_mgr
        self.logger = logger
        self.connection_history = defaultdict(list)  # IP -> [timestamps]

    def check_all(self):
        results = {
            'unexpected_ports': self.check_unexpected_ports(),
            'connection_count': self.check_connection_count(),
            'suspicious_activity': self.check_suspicious_activity(),
        }
        return results

    def check_unexpected_ports(self):
        expected = set(self.config.get('expected_listening_ports', []))
        connections = psutil.net_connections(kind='inet')
        listening = set()
        for c in connections:
            if c.status == 'LISTEN':
                listening.add(c.laddr.port)

        unexpected = listening - expected
        if unexpected:
            self.alert.send('WARNING', 'NETWORK',
                f'Unexpected ports listening: {sorted(unexpected)}')

        return {
            'expected': sorted(expected),
            'actual': sorted(listening),
            'unexpected': sorted(unexpected),
            'status': 'WARNING' if unexpected else 'OK'
        }

    def check_connection_count(self):
        connections = psutil.net_connections(kind='inet')
        established = [c for c in connections if c.status == 'ESTABLISHED']
        count = len(established)
        cfg_warn = self.config.get('max_connections_warning', 500)
        cfg_crit = self.config.get('max_connections_critical', 1000)
        status = 'OK'

        if count > cfg_crit:
            self.alert.send('CRITICAL', 'NETWORK', f'{count} established connections (threshold: {cfg_crit})')
            status = 'CRITICAL'
        elif count > cfg_warn:
            self.alert.send('WARNING', 'NETWORK', f'{count} established connections (threshold: {cfg_warn})')
            status = 'WARNING'

        return {'established': count, 'status': status}

    def check_suspicious_activity(self):
        findings = []
        if not self.config.get('detect_port_scans', True):
            return {'findings': findings, 'status': 'OK'}

        connections = psutil.net_connections(kind='inet')
        now = time.time()
        threshold = self.config.get('port_scan_threshold', 20)

        # Track connections per remote IP
        ip_ports = defaultdict(set)
        for c in connections:
            if c.raddr and c.status in ('ESTABLISHED', 'SYN_RECV', 'SYN_SENT'):
                ip_ports[c.raddr.ip].add(c.raddr.port if c.raddr else 0)
                self.connection_history[c.raddr.ip].append(now)

        # Check for port scan patterns
        for ip, ports in ip_ports.items():
            if len(ports) > threshold:
                self.alert.send('CRITICAL', 'SECURITY',
                    f'Possible port scan from {ip}: {len(ports)} different ports')
                findings.append({'type': 'port_scan', 'source_ip': ip, 'port_count': len(ports)})

        # Clean old history (>60s)
        for ip in list(self.connection_history.keys()):
            self.connection_history[ip] = [t for t in self.connection_history[ip] if t > now - 60]
            if not self.connection_history[ip]:
                del self.connection_history[ip]

        # Check rapid connections from same IP
        for ip, timestamps in self.connection_history.items():
            if len(timestamps) > threshold:
                self.alert.send('WARNING', 'SECURITY',
                    f'Rapid connections from {ip}: {len(timestamps)} in 60s')
                findings.append({'type': 'rapid_connections', 'source_ip': ip, 'count': len(timestamps)})

        return {'findings': findings, 'status': 'WARNING' if findings else 'OK'}


class FileIntegrityMonitor:
    """Monitors critical files for unexpected changes."""

    def __init__(self, config, alert_mgr, logger, state_file):
        self.config = config.get('file_integrity', {})
        self.alert = alert_mgr
        self.logger = logger
        self.state_file = state_file
        self.hashes = self._load_hashes()
        self.last_check = 0

    def _load_hashes(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('file_hashes', {})
        except Exception:
            pass
        return {}

    def _save_hashes(self):
        state = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
            except Exception:
                pass
        state['file_hashes'] = self.hashes
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _hash_file(self, filepath):
        try:
            h = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def check_all(self):
        interval = self.config.get('check_interval', 300)
        now = time.time()
        if now - self.last_check < interval and self.hashes:
            return {'status': 'SKIPPED', 'reason': 'within check interval'}

        self.last_check = now
        results = {'changed': [], 'new': [], 'missing': [], 'ok': []}

        # Check individual files
        for filepath in self.config.get('watch_files', []):
            self._check_file(filepath, results)

        # Check directories
        for dir_cfg in self.config.get('watch_directories', []):
            dirpath = dir_cfg.get('path', dir_cfg) if isinstance(dir_cfg, dict) else dir_cfg
            recursive = dir_cfg.get('recursive', False) if isinstance(dir_cfg, dict) else False
            if os.path.isdir(dirpath):
                for entry in os.scandir(dirpath):
                    if entry.is_file():
                        self._check_file(entry.path, results)

        self._save_hashes()
        status = 'OK'
        if results['changed']:
            status = 'WARNING'
        if results['missing']:
            status = 'CRITICAL'

        return {
            'status': status,
            'changed_count': len(results['changed']),
            'new_count': len(results['new']),
            'missing_count': len(results['missing']),
            'ok_count': len(results['ok']),
            'details': results
        }

    def _check_file(self, filepath, results):
        current_hash = self._hash_file(filepath)
        if current_hash is None:
            if filepath in self.hashes:
                self.alert.send('CRITICAL', 'FILE_INTEGRITY',
                    f'Watched file MISSING: {filepath}')
                results['missing'].append(filepath)
            return

        if filepath in self.hashes:
            if self.hashes[filepath] != current_hash:
                self.alert.send('WARNING', 'FILE_INTEGRITY',
                    f'File CHANGED: {filepath}')
                results['changed'].append(filepath)
                self.hashes[filepath] = current_hash
            else:
                results['ok'].append(filepath)
        else:
            self.logger.info(f'New file registered for integrity monitoring: {filepath}')
            results['new'].append(filepath)
            self.hashes[filepath] = current_hash


class LogWatcher:
    """Monitors log files for error patterns and brute force attempts."""

    def __init__(self, config, alert_mgr, logger):
        self.config = config.get('log_monitoring', {})
        self.alert = alert_mgr
        self.logger = logger
        self.file_positions = {}  # file -> last read position
        self.pattern_counts = defaultdict(lambda: defaultdict(int))
        self.last_reset = time.time()
        self.failed_login_tracker = defaultdict(list)

    def check_all(self):
        now = time.time()
        # Reset counters every 5 minutes
        if now - self.last_reset > 300:
            self.pattern_counts = defaultdict(lambda: defaultdict(int))
            self.last_reset = now

        results = {'files_checked': 0, 'matches_found': 0, 'brute_force_detected': False}

        for watch_cfg in self.config.get('watch_files', []):
            path = watch_cfg['path']
            patterns = watch_cfg.get('patterns', [])

            if os.path.isdir(path):
                for entry in os.scandir(path):
                    if entry.is_file() and entry.name.endswith('.log'):
                        matches = self._scan_file(entry.path, patterns)
                        results['files_checked'] += 1
                        results['matches_found'] += matches
            elif os.path.isfile(path):
                matches = self._scan_file(path, patterns)
                results['files_checked'] += 1
                results['matches_found'] += matches

        # Check for brute force
        results['brute_force_detected'] = self._check_brute_force()

        return results

    def _scan_file(self, filepath, patterns):
        try:
            file_size = os.path.getsize(filepath)
            last_pos = self.file_positions.get(filepath, 0)

            if file_size < last_pos:
                last_pos = 0  # File was rotated

            if file_size == last_pos:
                return 0

            matches = 0
            with open(filepath, 'r', errors='ignore') as f:
                f.seek(last_pos)
                for line in f:
                    for pattern in patterns:
                        if pattern.lower() in line.lower():
                            self.pattern_counts[filepath][pattern] += 1
                            matches += 1

                            # Check if this is an auth failure
                            if any(p in pattern.lower() for p in ['failed password', 'invalid user', 'authentication failure']):
                                self._track_failed_login(line)

                self.file_positions[filepath] = f.tell()

            # Alert if threshold exceeded
            threshold = self.config.get('alert_threshold', 10)
            for pattern, count in self.pattern_counts[filepath].items():
                if count >= threshold:
                    self.alert.send('WARNING', 'LOGS',
                        f'Pattern "{pattern}" matched {count} times in {os.path.basename(filepath)}')

            return matches
        except Exception as e:
            return 0

    def _track_failed_login(self, line):
        now = time.time()
        # Extract IP if possible
        parts = line.split()
        for i, part in enumerate(parts):
            if part == 'from' and i + 1 < len(parts):
                ip = parts[i + 1]
                self.failed_login_tracker[ip].append(now)
                break

    def _check_brute_force(self):
        threshold = self.config.get('failed_login_threshold', 5)
        now = time.time()
        detected = False

        for ip in list(self.failed_login_tracker.keys()):
            # Keep only last 5 minutes
            self.failed_login_tracker[ip] = [t for t in self.failed_login_tracker[ip] if t > now - 300]
            if not self.failed_login_tracker[ip]:
                del self.failed_login_tracker[ip]
                continue

            if len(self.failed_login_tracker[ip]) >= threshold:
                self.alert.send('CRITICAL', 'SECURITY',
                    f'Brute force attempt detected from {ip}: {len(self.failed_login_tracker[ip])} failed logins in 5min')
                detected = True

        return detected


class Guardian:
    """Main Guardian daemon - orchestrates all monitors."""

    VERSION = "1.0.0"
    BANNER = r"""
    ╔══════════════════════════════════════════════════════╗
    ║         TRADING DESK GUARDIAN v{version}              ║
    ║         Watching over your algo desk 24/7            ║
    ╚══════════════════════════════════════════════════════╝
    """

    def __init__(self, config_path='/opt/trading-desk/guardian/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.check_count = 0
        self.start_time = None

        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info(self.BANNER.format(version=self.VERSION))

        # Setup state file
        self.state_file = self.config['general'].get('state_file', '/opt/trading-desk/guardian/state.json')

        # Initialize components
        self.alert_mgr = AlertManager(self.config, self.logger)
        self.sys_monitor = SystemMonitor(self.config, self.alert_mgr, self.logger)
        self.svc_monitor = ServiceMonitor(self.config, self.alert_mgr, self.logger)
        self.net_security = NetworkSecurity(self.config, self.alert_mgr, self.logger)
        self.file_integrity = FileIntegrityMonitor(self.config, self.alert_mgr, self.logger, self.state_file)
        self.log_watcher = LogWatcher(self.config, self.alert_mgr, self.logger)

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGHUP, self._handle_reload)

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_file = self.config['general'].get('log_file', '/opt/trading-desk/logs/guardian/guardian.log')
        max_bytes = self.config['general'].get('max_log_size_mb', 50) * 1024 * 1024
        backup_count = self.config['general'].get('log_backup_count', 5)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger = logging.getLogger('guardian')
        logger.setLevel(logging.DEBUG)

        # File handler with rotation
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _handle_shutdown(self, signum, frame):
        self.logger.info(f'Received signal {signum}, shutting down gracefully...')
        self.running = False

    def _handle_reload(self, signum, frame):
        self.logger.info('Received SIGHUP, reloading configuration...')
        try:
            self.config = self._load_config()
            self.alert_mgr.send('INFO', 'GUARDIAN', 'Configuration reloaded successfully')
        except Exception as e:
            self.alert_mgr.send('CRITICAL', 'GUARDIAN', f'Failed to reload config: {e}')

    def run_single_check(self):
        """Run a single health check cycle. Used by CLI tool."""
        results = {}
        results['timestamp'] = datetime.now().isoformat()
        results['guardian_version'] = self.VERSION

        try:
            results['system'] = self.sys_monitor.check_all()
        except Exception as e:
            results['system'] = {'error': str(e)}

        try:
            results['services'] = self.svc_monitor.check_all()
        except Exception as e:
            results['services'] = {'error': str(e)}

        try:
            results['network'] = self.net_security.check_all()
        except Exception as e:
            results['network'] = {'error': str(e)}

        try:
            results['file_integrity'] = self.file_integrity.check_all()
        except Exception as e:
            results['file_integrity'] = {'error': str(e)}

        try:
            results['logs'] = self.log_watcher.check_all()
        except Exception as e:
            results['logs'] = {'error': str(e)}

        # Compute overall health
        results['overall_health'] = self._compute_health(results)
        return results

    def _compute_health(self, results):
        """Determine overall system health from all checks."""
        issues = []

        # Check system resources
        sys_data = results.get('system', {})
        for key in ['cpu', 'memory', 'swap', 'load']:
            item = sys_data.get(key, {})
            if isinstance(item, dict) and item.get('status') in ('WARNING', 'CRITICAL'):
                issues.append(f"{key}: {item['status']}")

        # Check disk
        disk = sys_data.get('disk', {})
        if isinstance(disk, dict):
            for path, info in disk.items():
                if isinstance(info, dict) and info.get('status') in ('WARNING', 'CRITICAL'):
                    issues.append(f"disk({path}): {info['status']}")

        # Check services
        svc_data = results.get('services', {})
        for category in ['docker_containers', 'ports', 'processes']:
            items = svc_data.get(category, {})
            if isinstance(items, dict):
                for name, info in items.items():
                    if isinstance(info, dict):
                        if info.get('status') in ('DOWN', 'CRITICAL', 'error', 'timeout'):
                            issues.append(f"service({name}): DOWN")
                        elif info.get('healthy') is False:
                            issues.append(f"service({name}): UNHEALTHY")
                        elif info.get('running') is False and info.get('required'):
                            issues.append(f"process({name}): NOT RUNNING")

        # Check network
        net_data = results.get('network', {})
        for key in ['unexpected_ports', 'connection_count', 'suspicious_activity']:
            item = net_data.get(key, {})
            if isinstance(item, dict) and item.get('status') in ('WARNING', 'CRITICAL'):
                issues.append(f"network({key}): {item['status']}")

        # Check file integrity
        fi_data = results.get('file_integrity', {})
        if isinstance(fi_data, dict) and fi_data.get('status') in ('WARNING', 'CRITICAL'):
            issues.append(f"file_integrity: {fi_data['status']}")

        if not issues:
            return {'status': 'HEALTHY', 'issues': []}
        elif any('CRITICAL' in i or 'DOWN' in i or 'NOT RUNNING' in i for i in issues):
            return {'status': 'CRITICAL', 'issues': issues}
        else:
            return {'status': 'DEGRADED', 'issues': issues}

    def _save_state(self, results):
        """Save current state for CLI tool to read."""
        state = {
            'last_check': results['timestamp'],
            'check_count': self.check_count,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'overall_health': results['overall_health'],
            'last_results': results,
        }
        # Preserve file hashes
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    existing = json.load(f)
                    if 'file_hashes' in existing:
                        state['file_hashes'] = existing['file_hashes']
        except Exception:
            pass

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def run(self):
        """Main daemon loop."""
        self.running = True
        self.start_time = datetime.now()
        interval = self.config['general'].get('check_interval', 30)

        self.logger.info(f'Guardian started. Check interval: {interval}s')
        self.alert_mgr.send('INFO', 'GUARDIAN', 'Guardian daemon started successfully')

        # Initial file integrity baseline
        self.logger.info('Building initial file integrity baseline...')
        self.file_integrity.check_all()

        while self.running:
            try:
                self.check_count += 1
                self.logger.debug(f'--- Health check #{self.check_count} ---')

                results = self.run_single_check()
                self._save_state(results)

                health = results['overall_health']
                if health['status'] == 'HEALTHY':
                    if self.check_count % 60 == 1:  # Log healthy status every ~30 min
                        self.logger.info(f'System HEALTHY (check #{self.check_count})')
                elif health['status'] == 'DEGRADED':
                    self.logger.warning(f'System DEGRADED: {", ".join(health["issues"])}')
                else:
                    self.logger.error(f'System CRITICAL: {", ".join(health["issues"])}')

                # Daily summary
                if self.config['alerting'].get('daily_summary', True):
                    now = datetime.now()
                    summary_hour = self.config['alerting'].get('summary_hour', 8)
                    if now.hour == summary_hour and now.minute < 1:
                        summary = self.alert_mgr.get_daily_summary()
                        self.logger.info(f'Daily summary: {summary["total_alerts"]} alerts, severity: {summary["by_severity"]}')

            except Exception as e:
                self.logger.error(f'Error in health check cycle: {e}', exc_info=True)

            # Sleep in small increments for responsive shutdown
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)

        self.logger.info('Guardian shutdown complete.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trading Desk Guardian')
    parser.add_argument('--config', default='/opt/trading-desk/guardian/config.yaml',
                        help='Path to config file')
    parser.add_argument('--check', action='store_true',
                        help='Run a single check and exit')
    args = parser.parse_args()

    guardian = Guardian(config_path=args.config)

    if args.check:
        results = guardian.run_single_check()
        print(json.dumps(results, indent=2, default=str))
        health = results['overall_health']
        sys.exit(0 if health['status'] == 'HEALTHY' else 1)
    else:
        guardian.run()


if __name__ == '__main__':
    main()
