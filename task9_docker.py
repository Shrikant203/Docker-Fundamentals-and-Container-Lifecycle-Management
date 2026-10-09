#!/usr/bin/env python3
"""
Task 9: Docker Fundamentals and Container Lifecycle Management
---------------------------------------------------------------
Runs every Docker step required by the task on YOUR machine, saves the real
command log, and builds the PDF report (source code + outputs + observations).

Usage:
    1. Start Docker Desktop and wait until it says "Engine running".
    2. python task9_docker.py            -> runs all steps, writes log + PDF
       python task9_docker.py --pdf-only -> rebuild PDF from an existing log
    3. (Optional) put Docker Desktop screenshots in ./screenshots/ BEFORE
       building the PDF; they are appended automatically.

Outputs:
    docker_command_log.txt   plain-text log of every command + output
    docker_results.json      structured results (used for the PDF)
    Task9_Docker_Report.pdf  report to upload on the L&T Edutech LMS
"""
import glob
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

LOG_TXT = "docker_command_log.txt"
LOG_JSON = "docker_results.json"
PDF_OUT = "Task9_Docker_Report.pdf"

# ----------------------------------------------------------------------------
# Files used by the custom image (a tiny "deep learning" inference app)
# ----------------------------------------------------------------------------
APP_PY = '''import numpy as np

# Tiny 2-layer neural network forward pass (stand-in for a DL inference service)
rng = np.random.default_rng(42)
x = rng.normal(size=(4, 8))                  # batch of 4 samples, 8 features
W1, b1 = rng.normal(size=(8, 16)), np.zeros(16)
W2, b2 = rng.normal(size=(16, 3)), np.zeros(3)

h = np.maximum(0, x @ W1 + b1)               # ReLU hidden layer
logits = h @ W2 + b2
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

print("Containerized DL inference app")
print("NumPy version :", np.__version__)
print("Output shape  :", probs.shape)
print("Predicted classes:", probs.argmax(axis=1).tolist())
'''

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY app.py .
CMD ["python", "app.py"]
"""


def write_build_context():
    os.makedirs("dl_app", exist_ok=True)
    with open("dl_app/app.py", "w") as f:
        f.write(APP_PY)
    with open("dl_app/Dockerfile", "w") as f:
        f.write(DOCKERFILE)


# ----------------------------------------------------------------------------
# Steps: (section, title, command, observation)
# ----------------------------------------------------------------------------
S1 = "1. Install Docker and verify the setup"
S2 = "2. Create and manage containers"
S3 = "3. Explore Docker images and registries"
S4 = "4. Container lifecycle operations"
S5 = "5. Networking concepts"
S6 = "6. Storage concepts (volumes and bind mounts)"
S7 = "7. Cleanup"

STEPS = [
    # ---- 1 ----
    (S1, "Docker client/server version", "docker --version && docker version --format 'Client: {{.Client.Version}} | Server: {{.Server.Version}}'",
     "The client (CLI) talks to the Docker daemon (server) over a socket. Both versions confirm the installation works."),
    (S1, "Engine information", "docker info --format 'Containers: {{.Containers}} | Images: {{.Images}} | Storage driver: {{.Driver}} | OS: {{.OperatingSystem}}'",
     "Shows the daemon's state, the storage driver (overlay2 is typical) and the host OS the engine runs on."),
    (S1, "Run the hello-world test container", "docker run --rm hello-world",
     "The daemon did not have the image locally, so it pulled it from Docker Hub, created a container, ran it, and streamed the output back to the CLI."),

    # ---- 2 ----
    (S2, "Run a detached web container with port mapping", "docker run -d --name dl-web -p 8080:80 nginx:alpine",
     "-d runs in the background, --name gives a friendly name, -p maps host port 8080 to container port 80."),
    (S2, "List running containers", "docker ps",
     "Shows container ID, image, command, status, port mapping and name of running containers."),
    (S2, "Call the container from the host", "curl -s -o /dev/null -w 'HTTP status: %{http_code}\\n' http://localhost:8080",
     "HTTP 200 proves the published port forwards host traffic into the container."),
    (S2, "Run a command inside the running container", "docker exec dl-web sh -c 'echo Hello from inside the container && nginx -v && hostname'",
     "docker exec starts an extra process inside an existing container; the hostname equals the container ID."),
    (S2, "View container logs", "docker logs --tail 5 dl-web",
     "Logs capture the container's stdout/stderr (nginx access logs here)."),
    (S2, "Resource usage snapshot", "docker stats --no-stream dl-web",
     "Shows live CPU, memory, network and block I/O of the container."),
    (S2, "Inspect container metadata", "docker inspect --format 'Name={{.Name}} State={{.State.Status}} IP={{.NetworkSettings.IPAddress}} Image={{.Config.Image}}' dl-web",
     "docker inspect returns the full JSON configuration; --format extracts specific fields."),

    # ---- 3 ----
    (S3, "Pull an image from the registry (Docker Hub)", "docker pull python:3.11-slim",
     "Images are made of read-only layers identified by digests; layers already present are not downloaded again."),
    (S3, "Search the registry", "docker search tensorflow --limit 5",
     "Docker Hub is the default public registry; search lists matching repositories with stars and official flags."),
    (S3, "List local images", "docker images",
     "Shows repository, tag, image ID, creation date and size of each local image."),
    (S3, "Image layers (history)", "docker history python:3.11-slim",
     "Every Dockerfile instruction creates a layer; this lists them with their sizes."),
    (S3, "Build a custom DL-style image from a Dockerfile", "docker build -t dl-app:1.0 ./dl_app",
     "docker build reads the Dockerfile, executes each instruction as a cached layer, and tags the result dl-app:1.0."),
    (S3, "Run the custom image", "docker run --rm dl-app:1.0",
     "Runs the containerized NumPy neural-network inference; the container exits when the process finishes."),
    (S3, "Tag the image for a registry", "docker tag dl-app:1.0 yourdockerhubuser/dl-app:1.0 && docker images yourdockerhubuser/dl-app",
     "Tagging prepares the image to be pushed with 'docker push' to a registry (replace yourdockerhubuser with your account)."),

    # ---- 4 ----
    (S4, "Create a container without starting it", "docker create --name lifecycle-demo python:3.11-slim sleep 600 && docker ps -a --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "State: Created. The writable layer exists but no process is running."),
    (S4, "Start", "docker start lifecycle-demo && docker ps --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "State: Running."),
    (S4, "Pause", "docker pause lifecycle-demo && docker ps --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "State: Paused. Processes are frozen using the cgroup freezer (memory is kept)."),
    (S4, "Unpause", "docker unpause lifecycle-demo && docker ps --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "Processes resume exactly where they stopped."),
    (S4, "Stop (graceful, SIGTERM then SIGKILL)", "docker stop -t 2 lifecycle-demo && docker ps -a --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "State: Exited. Docker sends SIGTERM, waits, then SIGKILL; the filesystem is kept."),
    (S4, "Restart", "docker restart lifecycle-demo && docker ps --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "Stop followed by start; the container returns to Running."),
    (S4, "Kill (immediate SIGKILL)", "docker kill lifecycle-demo && docker ps -a --filter name=lifecycle-demo --format 'table {{.Names}}\\t{{.Status}}'",
     "Forces immediate termination, exit code 137 (128 + 9)."),
    (S4, "Restart policy demo", "docker run -d --name policy-demo --restart unless-stopped nginx:alpine && docker inspect --format 'RestartPolicy={{.HostConfig.RestartPolicy.Name}}' policy-demo",
     "Restart policies (no, on-failure, always, unless-stopped) let the daemon restart containers automatically."),
    (S4, "Remove containers", "docker rm -f lifecycle-demo policy-demo && docker ps -a --format 'table {{.Names}}\\t{{.Status}}'",
     "docker rm deletes the container and its writable layer; the image remains."),

    # ---- 5 ----
    (S5, "List default networks", "docker network ls",
     "Default networks: bridge (default for containers), host and none."),
    (S5, "Create a user-defined bridge network", "docker network create dl-net",
     "User-defined bridges provide automatic DNS-based service discovery between containers."),
    (S5, "Start two containers on the network", "docker run -d --name backend --network dl-net nginx:alpine && docker run -d --name client --network dl-net alpine sleep 300",
     "Both containers join dl-net and get private IP addresses."),
    (S5, "Container-to-container communication by name", "docker exec client sh -c 'ping -c 2 backend && wget -qO- http://backend | head -n 4'",
     "The name 'backend' resolves through Docker's embedded DNS, proving container-to-container networking."),
    (S5, "Inspect the network", "docker network inspect dl-net --format '{{range .Containers}}{{.Name}} -> {{.IPv4Address}}{{println}}{{end}}'",
     "Lists the containers attached to the network with their IP addresses."),

    # ---- 6 ----
    (S6, "Create a named volume", "docker volume create dl-data && docker volume ls --filter name=dl-data",
     "Volumes are managed by Docker and live outside the container's writable layer."),
    (S6, "Write data from one container", "docker run --rm -v dl-data:/data alpine sh -c \"echo 'model checkpoint v1' > /data/checkpoint.txt && ls -l /data\"",
     "The container writes into the volume and is then deleted."),
    (S6, "Read the data from a NEW container (persistence)", "docker run --rm -v dl-data:/data alpine cat /data/checkpoint.txt",
     "The data survived container deletion, proving volume persistence (key for datasets and trained models)."),
    (S6, "Bind mount a host folder", "mkdir -p host_data && echo 'dataset sample from host' > host_data/sample.txt && docker run --rm -v \"$(pwd)/host_data:/mnt/host\" alpine cat /mnt/host/sample.txt",
     "A bind mount maps a host directory into the container; changes are visible on both sides."),
    (S6, "Inspect the volume", "docker volume inspect dl-data --format 'Name={{.Name}} Driver={{.Driver}} Mountpoint={{.Mountpoint}}'",
     "Shows where Docker stores the volume on the host."),

    # ---- 7 ----
    (S7, "Remove all demo resources", "docker rm -f dl-web backend client 2>&1; docker network rm dl-net; docker volume rm dl-data; docker ps -a --format 'table {{.Names}}\\t{{.Status}}'",
     "Cleaning up avoids leftover containers, networks and volumes."),
]

SECTION_THEORY = {
    S1: "Docker uses a client-server architecture: the Docker CLI (client) sends REST API calls to the Docker daemon (dockerd), which manages images, containers, networks and volumes and delegates container execution to containerd and runc. Docker Desktop bundles the engine, CLI, a Linux VM (on Windows/macOS), and a GUI.",
    S2: "A container is a running instance of an image: an isolated process using Linux namespaces (isolation) and cgroups (resource limits), with a thin writable layer on top of read-only image layers.",
    S3: "An image is an immutable, layered template. A registry (Docker Hub, GHCR, ECR, private registries) stores and distributes images by repository:tag and digest. A Dockerfile describes how to build an image reproducibly, which is critical for deep learning environments with pinned library versions.",
    S4: "Container states: Created -> Running -> Paused -> Exited -> Removed. Transitions are driven by create, start, pause/unpause, stop/kill, restart and rm.",
    S5: "Docker network drivers: bridge (default, single host), host (shares host network stack), none (no networking), overlay (multi-host swarm). User-defined bridge networks add DNS resolution by container name and better isolation.",
    S6: "Container filesystems are ephemeral. Volumes (Docker-managed) and bind mounts (host paths) persist data beyond the container lifecycle. tmpfs mounts keep data in memory only.",
    S7: "",
}


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def run_cmd(cmd, timeout=900):
    """Run a shell command and return a result dict with REAL output."""
    start = datetime.now()
    try:
        import shutil
        bash = shutil.which("bash")  # Git Bash / WSL on Windows; /bin/bash on Linux/macOS
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                           executable=bash if (bash and os.name != "nt") else None) if os.name != "nt" or not bash \
            else subprocess.run([bash, "-c", cmd], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "Command timed out.", 124
    return {"command": cmd, "output": out.rstrip(), "returncode": rc,
            "time": start.strftime("%Y-%m-%d %H:%M:%S")}


def run_step(i, results=None):
    section, title, cmd, obs = STEPS[i]
    write_build_context()
    r = run_cmd(cmd)
    r.update({"section": section, "title": title, "observation": obs})
    print(f"\n### [{i + 1}/{len(STEPS)}] {title}\n$ {cmd}\n{r['output']}\n(exit code {r['returncode']})")
    if results is not None:
        results.append(r)
    return r


def docker_ready():
    r = run_cmd("docker info", timeout=60)
    return r["returncode"] == 0


def save_log(results):
    with open(LOG_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(LOG_TXT, "w") as f:
        f.write(f"Docker command log - generated {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 78 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"\n[{i}] {r['section']} - {r['title']}  ({r['time']})\n")
            f.write(f"$ {r['command']}\n{r['output']}\n(exit code {r['returncode']})\n")
            f.write("-" * 78 + "\n")


# ----------------------------------------------------------------------------
# PDF report
# ----------------------------------------------------------------------------
def build_pdf(results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                    Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)
    from xml.sax.saxutils import escape

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1F3A6E"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#1F3A6E"))
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], spaceBefore=8, spaceAfter=3)
    BODY = ParagraphStyle("B", parent=ss["BodyText"], leading=14)
    CMD = ParagraphStyle("CMD", fontName="Courier-Bold", fontSize=8, leading=10, backColor=colors.HexColor("#E8EEF7"),
                         borderPadding=4, leftIndent=2)
    OUT = ParagraphStyle("OUT", fontName="Courier", fontSize=7.2, leading=9, backColor=colors.HexColor("#F4F4F4"),
                         borderPadding=4, leftIndent=2)

    def wrap(text, width=105, max_lines=45):
        lines = []
        for ln in text.splitlines():
            lines.extend(textwrap.wrap(ln, width, replace_whitespace=False, drop_whitespace=False) or [""])
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines - see docker_command_log.txt)"]
        return "\n".join(lines)

    doc = SimpleDocTemplate(PDF_OUT, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Task 9: Docker Fundamentals and Container Lifecycle Management")
    s = []
    s.append(Paragraph("Task 9: Docker Fundamentals and Container Lifecycle Management", H1))
    s.append(Paragraph("Practical Implementation Report", H2))
    s.append(Paragraph(f"Generated: {datetime.now():%d %B %Y, %H:%M}", BODY))
    s.append(Spacer(1, 10))

    s.append(Paragraph("1. Objective", H2))
    s.append(Paragraph("To understand Docker architecture and manage containerized deep learning applications.", BODY))
    s.append(Paragraph("2. Reference Tools", H2))
    s.append(Paragraph("Docker Desktop and Docker CLI.", BODY))

    # Environment
    env = next((r for r in results if r["title"].startswith("Docker client")), None)
    if env:
        s.append(Paragraph("3. Environment", H2))
        s.append(Preformatted(wrap(env["output"]), OUT))

    s.append(Paragraph("4. Summary of work (mapped to the required approach)", H2))
    ok = sum(1 for r in results if r["returncode"] == 0)
    rows = [["Required approach", "Steps", "Succeeded"]]
    for sec in [S1, S2, S3, S4, S5, S6]:
        rs = [r for r in results if r["section"] == sec]
        rows.append([sec, str(len(rs)), f"{sum(1 for r in rs if r['returncode'] == 0)}/{len(rs)}"])
    t = Table(rows, colWidths=[11 * cm, 2.2 * cm, 3 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A6E")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    s.append(t)
    s.append(Paragraph(f"Total: {ok}/{len(results)} commands completed with exit code 0.", BODY))

    s.append(Paragraph("5. Source code", H2))
    s.append(Paragraph("<b>dl_app/app.py</b>", BODY))
    s.append(Preformatted(APP_PY, OUT))
    s.append(Paragraph("<b>dl_app/Dockerfile</b>", BODY))
    s.append(Preformatted(DOCKERFILE, OUT))

    s.append(PageBreak())
    s.append(Paragraph("6. Command execution, outputs and observations", H1))
    cur, n = None, 0
    for r in results:
        if r["section"] != cur:
            cur = r["section"]
            s.append(Paragraph(cur, H2))
            if SECTION_THEORY.get(cur):
                s.append(Paragraph(f"<b>Concept:</b> {escape(SECTION_THEORY[cur])}", BODY))
        n += 1
        block = [Paragraph(f"Step {n}: {escape(r['title'])}", H3),
                 Preformatted(wrap("$ " + r["command"], 105, 12), CMD),
                 Spacer(1, 3),
                 Preformatted(wrap(r["output"] or "(no output)") + f"\n[exit code {r['returncode']}]", OUT),
                 Spacer(1, 3),
                 Paragraph(f"<b>Observation:</b> {escape(r['observation'])}", BODY)]
        s.append(KeepTogether(block[:4]))
        s.extend(block[4:])

    shots = sorted(glob.glob("screenshots/*.png") + glob.glob("screenshots/*.jpg") + glob.glob("screenshots/*.jpeg"))
    s.append(PageBreak())
    s.append(Paragraph("7. Screenshots (Docker Desktop)", H1))
    if shots:
        for p in shots:
            s.append(Paragraph(escape(os.path.basename(p)), H3))
            img = Image(p)
            max_w, max_h = 16.5 * cm, 20 * cm
            ratio = min(max_w / img.imageWidth, max_h / img.imageHeight, 1)
            img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
            s.append(img)
            s.append(Spacer(1, 8))
    else:
        s.append(Paragraph("No screenshots found. Add PNG/JPG files to the 'screenshots' folder and re-run "
                           "<font face='Courier'>python task9_docker.py --pdf-only</font>.", BODY))

    s.append(Paragraph("8. Overall observations and conclusion", H1))
    for b in [
        "Docker's client-server design lets the same CLI command work against any engine; the daemon performs all work.",
        "Images are layered and cached, which made repeated pulls/builds fast and gave a reproducible environment for the DL app.",
        "Containers are lightweight, isolated processes; the full lifecycle (create, start, pause, stop, kill, restart, remove) was exercised and verified through status output.",
        "User-defined bridge networks provide name-based discovery; published ports (-p) expose services to the host.",
        "Container storage is ephemeral; named volumes and bind mounts persist datasets and model checkpoints independently of container lifetime.",
        "Taken together, these capabilities make Docker well suited for packaging, shipping and running deep learning workloads consistently across machines.",
    ]:
        s.append(Paragraph("&bull; " + escape(b), BODY))
    doc.build(s)
    print(f"\nPDF written: {PDF_OUT}")


def main():
    if "--pdf-only" in sys.argv:
        with open(LOG_JSON) as f:
            build_pdf(json.load(f))
        return
    write_build_context()
    if not docker_ready():
        sys.exit("Docker is not running or not installed. Start Docker Desktop, wait for 'Engine running', "
                 "then run this script again.")
    results = []
    for i in range(len(STEPS)):
        run_step(i, results)
    save_log(results)
    build_pdf(results)
    print(f"Log saved: {LOG_TXT}")


if __name__ == "__main__":
    main()
