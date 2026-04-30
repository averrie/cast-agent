"""Thin wrapper around mini-swe-agent's DockerEnvironment for our ToolSet/EvidenceLayer."""

import logging
import os
import platform
import shlex
import subprocess

# Silence mini-swe-agent's startup banner (emoji breaks Windows terminals)
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent.environments.docker import DockerEnvironment  # noqa: E402

logger = logging.getLogger(__name__)


def get_swebench_image(instance: dict) -> str:
    """Get the Docker image name for a SWE-bench instance.

    Replicates mini-swe-agent's image naming convention.
    """
    image_name = instance.get("image_name") or instance.get("docker_image")
    if image_name is None:
        iid = instance["instance_id"]
        id_docker = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
    return image_name


class SWEBenchEnv:
    """Simplified Docker environment for SWE-bench instances.

    Wraps :class:`DockerEnvironment` from mini-swe-agent with a minimal
    interface suitable for our ToolSet / EvidenceLayer implementations.
    """

    def __init__(self, instance: dict, timeout: int = 60):
        image = get_swebench_image(instance)
        self.instance_id = instance["instance_id"]
        logger.info(
            "Starting Docker container for %s (image: %s)", self.instance_id, image
        )
        try:
            self.docker = DockerEnvironment(
                image=image,
                cwd="/testbed",
                timeout=timeout,
                interpreter=["bash", "-c"],
                env={
                    "PAGER": "cat",
                    "MANPAGER": "cat",
                    "LESS": "-R",
                    "PIP_PROGRESS_BAR": "off",
                    "TQDM_DISABLE": "1",
                },
            )
        except subprocess.CalledProcessError as e:
            # Surface the actual Docker error (stderr) instead of just the return code
            stderr = (e.stderr or "").strip()
            raise RuntimeError(
                f"Docker failed for {self.instance_id}: {stderr or e}"
            ) from e

    def run(self, command: str, timeout: int | None = None) -> tuple[str, int]:
        """Execute a bash command in the container.

        Returns
        -------
        (output, return_code) : tuple[str, int]
        """
        result = self.docker.execute({"command": command}, timeout=timeout)
        output = result.get("output", "")
        if result.get("exception_info"):
            output += f"\n{result['exception_info']}"
        return output.strip(), result.get("returncode", 1)

    def write_file(self, path: str, content: str, timeout: int = 60) -> tuple[str, int]:
        """Write *content* to *path* inside the container via stdin pipe.

        Avoids the Windows 8191-char command-line length limit that breaks
        heredoc-based writes for large files.
        """
        container_id = self.docker.container_id
        exe = self.docker.config.executable
        cwd = self.docker.config.cwd or "/testbed"
        cmd = [
            exe,
            "exec",
            "-i",
            "-w",
            cwd,
            container_id,
            "bash",
            "-c",
            f"cat > {shlex.quote(path)}",
        ]
        # Use binary mode to avoid Windows text-mode \n → \r\n conversion
        # which would corrupt every line in the file inside the Linux container.
        result = subprocess.run(
            cmd,
            input=content.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        output = (result.stdout or b"").decode("utf-8", errors="replace") + (
            result.stderr or b""
        ).decode("utf-8", errors="replace")
        return output.strip(), result.returncode

    def get_patch(self) -> str:
        """Capture current changes as a unified diff (tracked files only)."""
        out, _ = self.run("git diff")
        # run() strips trailing newline; git apply requires it
        if out and not out.endswith("\n"):
            out += "\n"
        return out

    def cleanup(self) -> None:
        """Stop and remove the Docker container.

        Overrides mini-swe-agent's cleanup which uses Unix shell syntax
        (``>/dev/null``, ``timeout``) that fails on Windows.
        """
        container_id = getattr(self.docker, "container_id", None)
        if container_id is None:
            return
        exe = self.docker.config.executable
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    [exe, "stop", container_id],
                    capture_output=True,
                    timeout=60,
                )
                subprocess.run(
                    [exe, "rm", "-f", container_id],
                    capture_output=True,
                    timeout=10,
                )
            else:
                self.docker.cleanup()
        except Exception:
            pass
        # Prevent the __del__ finalizer from running the broken Unix cleanup
        self.docker.container_id = None
