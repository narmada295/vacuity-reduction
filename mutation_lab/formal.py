"""Isolated lint and JasperGold runs for golden, mutant, and frozen SVAs."""

from __future__ import annotations

import re
import shlex
import shutil
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .core import Config, resolve_config_path
from .planner import _env_values


STATUS = r"proven|cex|covered|unreachable|undetermined|unknown|error|bounded_proven|bounded_unreachable"


def parse_statuses(log: str) -> dict[str, str]:
    """Parse named JasperGold report rows; summaries are not enough for grading."""
    return dict(re.findall(r"(?m)^\[\d+\]\s+(\S+)\s+(" + STATUS + r")\b", log))


def named_status(statuses: dict[str, str], label: str) -> str:
    matches = [status for name, status in statuses.items()
               if name == label or name.endswith("." + label)]
    return matches[0] if len(matches) == 1 else "missing_or_ambiguous"


def _tcl_word(value: str) -> str:
    if "}" in value or "{" in value or "\n" in value:
        raise ValueError(f"Unsupported path in Jasper Tcl: {value!r}")
    return "{" + value + "}"


def prepare_job(config: Config, directory: Path, checker_file: Path) -> None:
    """The checker is a separate bind/checker module; RTL remains untouched here."""
    shutil.copy2(checker_file, directory / "checker.sv")
    environment = ""
    if config.environment_file:
        shutil.copy2(config.environment_file, directory / "environment.sv")
        environment = "{environment.sv}"
    include = " ".join("+incdir+" + str(Path("rtl") / item)
                       for item in config.include_dirs)
    files = " ".join(_tcl_word(str(Path("rtl") / item)) for item in config.sources)
    analyze = f"analyze -sv12 {include} {files} {environment} {{checker.sv}}"
    template = config.jasper.get("tcl_template")
    if template:
        source = resolve_config_path(config.path.parent, template).read_text()
        replacements = {
            "{{ANALYZE_COMMAND}}": analyze,
            "{{RTL_SOURCES}}": files,
            "{{INCLUDE_FLAGS}}": include,
            "{{CHECKER_FILE}}": "{checker.sv}",
            "{{ENVIRONMENT_FILE}}": environment,
            "{{TOP}}": config.top,
        }
        for placeholder, value in replacements.items():
            source = source.replace(placeholder, value)
        if "{{" in source or "}}" in source:
            raise ValueError("Unknown placeholder in Jasper Tcl template")
        (directory / "run.tcl").write_text(source)
        return
    script = [analyze, f"elaborate -top {config.top}"]
    script.append(f"clock {config.clock}" if config.clock else "clock -none")
    if config.reset:
        reset = config.reset
        active_low = reset.lower().endswith(("_n", "_b")) or reset.lower().startswith(("nreset", "nrst"))
        script.append("reset {~" + reset + "}" if active_low else "reset " + reset)
    else:
        script.append("reset -none")
    script.extend(["prove -all", "report -summary -force -result", "exit -force", ""])
    (directory / "run.tcl").write_text("\n".join(script))


def lint(config: Config, directory: Path) -> dict:
    settings = config.lint_settings
    if not settings.get("enabled", True):
        return {"status": "skipped", "detail": "lint disabled by config"}
    executable = settings.get("executable", "verilator")
    if not shutil.which(executable):
        return {"status": "unavailable", "detail": "verilator not found; JasperGold elaboration remains the gate"}
    command = [executable, "--lint-only", "-Wno-fatal", "--top-module", config.top]
    command.extend(settings.get("extra_args", []))
    command.extend("-I" + str(Path("rtl") / item) for item in config.include_dirs)
    command.extend(str(Path("rtl") / item) for item in config.sources)
    try:
        process = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": "Verilator lint timed out"}
    return {"status": "passed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "detail": (process.stdout + process.stderr)[-4000:]}


@dataclass
class ToolResult:
    status: str
    exit_code: int | None
    log: str


class JasperRunner:
    def __init__(self, config: Config):
        self.settings = config.jasper
        self.timeout = int(self.settings.get("timeout_seconds", 180))
        self.mode = self.settings.get("mode", "local")
        if self.mode not in ("local", "ssh"):
            raise ValueError("jasper.mode must be local or ssh")
        self.config = config

    def run(self, directory: Path) -> ToolResult:
        if self.mode == "ssh":
            return self._ssh_run(directory)
        executable = self.settings.get("executable", "jg")
        if not shutil.which(executable):
            return ToolResult("unavailable", None, f"JasperGold executable not found: {executable}")
        try:
            process = subprocess.run([executable, "-batch", "run.tcl"], cwd=directory,
                                     capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or b""
            return ToolResult("timeout", None, partial.decode(errors="replace")
                              if isinstance(partial, bytes) else partial)
        log = process.stdout + process.stderr
        return ToolResult("completed" if process.returncode == 0 else "error", process.returncode, log)

    def _ssh_run(self, directory: Path) -> ToolResult:
        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError("Install paramiko for JasperGold SSH mode") from exc
        env_file = self.settings.get("env_file")
        env_path = (self.config.path.parent / env_file).resolve() if env_file else None
        values = _env_values(env_path)
        missing = [name for name in ("HOST_NAME", "USERNAME", "PASSWORD") if not values.get(name)]
        if missing:
            raise RuntimeError("Missing SSH settings: " + ", ".join(missing))
        remote = "/tmp/sangam_mutlab_" + uuid4().hex
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(values["HOST_NAME"], username=values["USERNAME"],
                    password=values["PASSWORD"], timeout=15)
        sftp = ssh.open_sftp()
        created = False
        try:
            sftp.mkdir(remote)
            created = True
            for path in directory.rglob("*"):
                relative = path.relative_to(directory).as_posix()
                target = remote + "/" + relative
                if path.is_dir():
                    sftp.mkdir(target)
                elif path.is_file():
                    sftp.put(str(path), target)
            setup = self.settings.get("setup", "source cad_cshrc")
            if not re.fullmatch(r"[A-Za-z0-9_./ ;=-]+", setup):
                raise ValueError("jasper.setup contains unsupported shell characters")
            command = (f"{setup}; cd {remote}; timeout --signal=TERM --kill-after=5s "
                       f"{self.timeout}s jg -batch run.tcl")
            _, stdout, stderr = ssh.exec_command("csh -c " + shlex.quote(command), timeout=self.timeout + 30)
            stdout.channel.set_combine_stderr(True)
            stdout.channel.settimeout(self.timeout + 40)
            try:
                log = stdout.read().decode(errors="replace")
                code = stdout.channel.recv_exit_status()
            except socket.timeout:
                stdout.channel.close()
                return ToolResult("timeout", None, "Remote JasperGold timed out")
            status = ("completed" if code == 0 else
                      "license_unavailable" if code == 124 and "Waiting for license" in log else
                      "timeout" if code == 124 else "error")
            return ToolResult(status, code, log)
        finally:
            try:
                if created:
                    _remove_remote_tree(sftp, remote)
            finally:
                sftp.close()
                ssh.close()


def _remove_remote_tree(sftp: object, path: str) -> None:
    if not path.startswith("/tmp/sangam_mutlab_"):
        raise ValueError("Refusing to remove a non-mutation-lab remote directory")
    for entry in sftp.listdir_attr(path):
        child = path + "/" + entry.filename
        if stat.S_ISDIR(entry.st_mode):
            _remove_remote_tree(sftp, child)
        else:
            sftp.remove(child)
    sftp.rmdir(path)


def run_check(config: Config, directory: Path, checker_file: Path) -> dict:
    prepare_job(config, directory, checker_file)
    result = JasperRunner(config).run(directory)
    (directory / "jasper.log").write_text(result.log)
    return {"status": result.status, "exit_code": result.exit_code,
            "properties": parse_statuses(result.log), "log": str(directory / "jasper.log")}
