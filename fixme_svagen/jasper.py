"""Remote JasperGold checks with per-assertion related-cover evidence."""

from __future__ import annotations

import os
import re
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .sva import Assertion, insert_in_module, named_assertion


@dataclass
class JGResult:
    assertion_status: str
    precondition_status: str | None
    contrapositive_status: str | None
    log: str
    exit_code: int | None

    def syntax_ok(self) -> bool:
        return self.assertion_status in ("proven", "cex", "bounded_proven", "undetermined") or "Starting proof on task" in self.log

    def label(self, implication: bool) -> tuple[bool | None, str]:
        if not implication:
            return None, "outside_scope_non_implication"
        if self.assertion_status == "cex":
            return False, "assertion_falsified"
        if self.assertion_status != "proven":
            return None, "formal_inconclusive_or_error"
        reasons = []
        if self.precondition_status == "unreachable":
            reasons.append("antecedent_never_true")
        if self.contrapositive_status == "unreachable":
            reasons.append("consequent_always_true")
        if reasons:
            return True, "; ".join(reasons)
        if self.precondition_status == "covered" and self.contrapositive_status == "covered":
            return False, "neither_targeted_vacuity_reason"
        return None, "related_cover_inconclusive_or_unsupported"


_STATUS = r"proven|cex|covered|unreachable|undetermined|unknown|error|bounded_proven|bounded_unreachable"


def parse_log(log: str, exit_code: int | None) -> JGResult:
    """Read named rows from `report -summary -force -result`, never aggregate counts."""
    rows = re.findall(r"(?m)^\[\d+\]\s+(\S+)\s+(" + _STATUS + r")\b", log)
    statuses = {name: status for name, status in rows}
    main = next((status for name, status in statuses.items() if name.endswith(".sangam_assert")), None)
    pre = next((status for name, status in statuses.items() if re.search(r"\.sangam_assert:precondition\d+$", name)), None)
    contra = next((status for name, status in statuses.items() if re.search(r"\.sangam_assert:contrapositive\d+$", name)), None)
    if main is None:
        # Jasper may omit report rows when elaboration or proof failed.
        main = "error" if exit_code != 0 or re.search(r"(?m)^ERROR\b", log) else "unknown"
    return JGResult(main, pre, contra, log, exit_code)


def license_connection_error(console_log: str) -> bool:
    """Ignore failed feature attempts when a later license checkout succeeds."""
    return bool(re.search(r"FLEXnet ERROR\(-15\b|Cannot connect to license server", console_log, re.I))


def load_connection(env_file: Path) -> dict[str, str]:
    values = dict(os.environ)
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), value.strip().strip("\"'"))
    missing = [x for x in ("HOST_NAME", "USERNAME", "PASSWORD") if not values.get(x)]
    if missing:
        raise RuntimeError("Missing JasperGold connection fields: " + ", ".join(missing))
    return values


class JasperRunner:
    def __init__(self, env_file: Path, timeout: int = 180):
        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("Install paramiko to use the JasperGold SSH runner") from exc
        values = load_connection(env_file)
        self.paramiko = paramiko
        self.values = values
        self.timeout = timeout
        self._connect()

    def _connect(self) -> None:
        self.ssh = self.paramiko.SSHClient()
        # Matches SANGAM's SSH behavior. Pin the host key in production if available.
        self.ssh.set_missing_host_key_policy(self.paramiko.AutoAddPolicy())
        self.ssh.connect(self.values["HOST_NAME"], username=self.values["USERNAME"],
                         password=self.values["PASSWORD"], timeout=12)
        self.sftp = self.ssh.open_sftp()

    def _ensure_connection(self) -> None:
        transport = self.ssh.get_transport()
        if transport is None or not transport.is_active():
            self.ssh.close()
            self._connect()

    def close(self) -> None:
        try:
            self.sftp.close()
        finally:
            self.ssh.close()

    def _remove_scratch(self, path: str) -> None:
        if not path.startswith("/tmp/sangam_fixme_"):
            return
        for item in self.sftp.listdir_attr(path):
            child = path + "/" + item.filename
            if stat.S_ISDIR(item.st_mode):
                self._remove_scratch(child)
            else:
                self.sftp.remove(child)
        self.sftp.rmdir(path)

    def _license_error(self, scratch: str) -> bool:
        try:
            with self.sftp.open(scratch + "/jgproject/jg_console.log", "r") as handle:
                text = handle.read().decode(errors="replace")
            return license_connection_error(text)
        except (OSError, EOFError):
            return False

    def preflight(self) -> None:
        probe = Assertion("assert property (@(posedge clk) a |-> 1'b1);", "", True)
        regular_timeout = self.timeout
        self.timeout = min(self.timeout, 90)
        try:
            result = self.check("module sangam_probe(input logic clk, input logic a); endmodule", "sangam_probe", probe, "clk", None)
        finally:
            self.timeout = regular_timeout
        if (result.assertion_status != "proven" or result.precondition_status != "covered"
                or result.contrapositive_status != "unreachable"):
            raise RuntimeError(
                "JasperGold preflight failed: "
                f"assertion={result.assertion_status}, precondition={result.precondition_status}, "
                f"contrapositive={result.contrapositive_status}. {result.log[-400:]}"
            )

    def __enter__(self) -> "JasperRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def check(self, rtl: str, top: str, assertion: Assertion, clock: str | None, reset: str | None) -> JGResult:
        self._ensure_connection()
        property_clocks = list(dict.fromkeys(re.findall(
            r"@\s*\(\s*posedge\s+([A-Za-z_][A-Za-z_0-9]*)",
            assertion.support + "\n" + assertion.text,
        )))
        clocks = property_clocks or ([clock] if clock else [])
        for name in (top, *clocks, reset):
            if name and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                return JGResult("error", None, None, f"Unsafe identifier: {name!r}", None)
        scratch = "/tmp/sangam_fixme_" + uuid4().hex
        self.sftp.mkdir(scratch)
        try:
            source = insert_in_module(rtl, top, named_assertion(assertion))
            reset_expr = "-none"
            if reset:
                active_low = reset.lower().endswith(("_n", "_b")) or reset.lower().startswith(("nreset", "nrst"))
                reset_expr = "{~" + reset + "}" if active_low else reset
            script = "\n".join([
                "analyze -sv12 dut.sv",
                f"elaborate -top {top} -create_related_covers {{precondition contrapositive}}",
                *((f"clock {item}" for item in clocks) if clocks else ("clock -none",)),
                f"reset {reset_expr}",
                "prove -all",
                "report -summary -force -result",
                "exit -force",
                "",
            ])
            for filename, content in (("dut.sv", source), ("run.tcl", script)):
                with self.sftp.open(scratch + "/" + filename, "w") as handle:
                    handle.write(content)
            command = (
                "csh -c 'source cad_cshrc; cd " + scratch
                + "; timeout --signal=TERM --kill-after=5s "
                + str(self.timeout) + "s jg -batch run.tcl'"
            )
            _, stdout, stderr = self.ssh.exec_command(command, timeout=self.timeout)
            stdout.channel.set_combine_stderr(True)
            stdout.channel.settimeout(self.timeout + 20)
            try:
                log = stdout.read().decode(errors="replace")
                code = stdout.channel.recv_exit_status()
                # License checkout may fail for one attempt and succeed later.
                # A completed named proof report is stronger evidence than an
                # earlier transient FlexNet error in the console log.
                parsed = parse_log(log, code)
                if code == 0 and parsed.assertion_status in (
                    "proven", "cex", "bounded_proven", "undetermined"
                ):
                    return parsed
                if self._license_error(scratch):
                    return JGResult("license_unavailable", None, None,
                                    log + "\nJasperGold license checkout failed", code)
                if code == 124:
                    return JGResult("timeout", None, None, log + "\nJasperGold hard timeout", code)
            except socket.timeout:
                stdout.channel.close()
                return JGResult("timeout", None, None, "JasperGold timed out", None)
            return parsed
        except Exception as exc:
            if isinstance(exc, (self.paramiko.SSHException, EOFError, ConnectionError)):
                raise RuntimeError("JasperGold SSH connection lost during check") from exc
            return JGResult("error", None, None, f"JasperGold runner error: {exc}", None)
        finally:
            try:
                self._remove_scratch(scratch)
            except (OSError, EOFError, self.paramiko.SSHException):
                # The raw tool output is saved locally; failure to clean /tmp
                # must not turn a valid proof into an error label.
                pass
