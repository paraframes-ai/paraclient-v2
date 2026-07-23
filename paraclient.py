#!/usr/bin/env python3
"""
paraclient.py — a Claude Code-style terminal chat for the ParaFrames tutor.

Talks to any OpenAI-compatible endpoint (your vLLM server from the RUNBOOK).
Pick the SUBJECT via the `model` field (the vLLM lora-module name, e.g. `math`)
and the MODE via the system prompt (`socratic` / `graduated_hint`) — the two
behaviors each adapter is trained on.

    # endpoint is required — pass it, or set PARACLIENT_BASE_URL, or use config
    python paraclient.py --base-url http://localhost:8000/v1 --subject math

Inside the chat, everything after `/` is a command. Try `/help`. Plain text is
sent to the tutor and streamed back with live markdown rendering. Ctrl-C stops a
running generation (keeping the partial reply); Ctrl-D or `/exit` quits.

Deps (light — you do NOT need torch/vllm just to run the client):
    pip install openai rich prompt_toolkit
Config file (optional, JSON): ~/.config/paraclient/config.json
    {"base_url": "http://localhost:8000/v1", "subject": "math", "mode": "socratic"}
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Modes -> system prompts. These mirror the two behaviors the adapters are
# trained + eval'd on (see eval_adapter.py). Kept subject-agnostic so the same
# client works for math, civics, language_arts, general, ...
# ---------------------------------------------------------------------------
MODES = {
    "socratic": (
        "You are a Socratic tutor for a K-12 student. Never state the final "
        "answer. Guide the student with one question at a time, helping them "
        "reason their way to it. Keep replies short and encouraging."
    ),
    "graduated_hint": (
        "You are a homework tutor for a K-12 student. Guide with questions "
        "first, but if the student stays stuck, escalate to hints and then "
        "worked steps so they can finish. Be patient and encouraging."
    ),
}
DEFAULT_MODE = "socratic"
DEFAULT_SUBJECT = "math"

CONFIG_DIR = Path(os.environ.get("PARACLIENT_HOME",
                                 Path.home() / ".config" / "paraclient"))
CONFIG_PATH = CONFIG_DIR / "config.json"
HISTORY_PATH = CONFIG_DIR / "history"


def load_config_file() -> dict:
    """Optional JSON config; silently ignored if absent, warned if malformed."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[paraclient] warning: could not read {CONFIG_PATH}: {e}",
              file=sys.stderr)
        return {}


def resolve(args, cfg, key, env, default=None):
    """Precedence: CLI flag > env var > config file > default."""
    val = getattr(args, key.replace("-", "_"), None)
    if val is not None:
        return val
    if env and os.environ.get(env):
        return os.environ[env]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


# ---------------------------------------------------------------------------
# The chat session
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, client, console, subject, mode, temperature,
                 max_tokens, base_url):
        self.client = client
        self.console = console
        self.subject = subject
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.custom_system = None          # overrides mode prompt if set
        self.history = []                  # list of {role, content} (no system)

    # ---- prompt assembly ----
    def system_prompt(self) -> str:
        return self.custom_system or MODES.get(self.mode, MODES[DEFAULT_MODE])

    def messages(self):
        return [{"role": "system", "content": self.system_prompt()}] + self.history

    # ---- networking ----
    def list_models(self, quiet=False):
        """Return the server's model/subject ids, or None on failure."""
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception as e:                       # noqa: BLE001
            if not quiet:
                self._api_error(e)
            return None

    def stream_reply(self):
        """Stream one assistant turn. Returns the full text (may be partial if
        interrupted). Appends the reply to history on the caller's behalf."""
        from rich.live import Live
        from rich.markdown import Markdown

        acc = ""
        interrupted = False
        try:
            stream = self.client.chat.completions.create(
                model=self.subject,
                messages=self.messages(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except Exception as e:                       # noqa: BLE001
            self._api_error(e)
            return None

        try:
            with Live(console=self.console, refresh_per_second=12,
                      vertical_overflow="visible") as live:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        acc += delta
                        live.update(Markdown(acc))
                if not acc:
                    live.update(Markdown("*(empty reply)*"))
        except KeyboardInterrupt:
            interrupted = True
            try:
                stream.close()
            except Exception:                        # noqa: BLE001
                pass
        except Exception as e:                       # noqa: BLE001
            self._api_error(e)
            return None

        if interrupted:
            self.console.print("[yellow]— stopped —[/yellow]")
        if acc:
            self.history.append({"role": "assistant", "content": acc})
        return acc

    def _api_error(self, e):
        from openai import APIConnectionError
        if isinstance(e, APIConnectionError):
            self.console.print(
                f"[red]Could not reach the server at[/red] "
                f"[bold]{self.base_url}[/bold].\n"
                "[dim]Is vLLM running? Start it per the RUNBOOK, or point "
                "elsewhere with /url.[/dim]")
        else:
            self.console.print(f"[red]API error:[/red] {e}")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------
HELP = """\
[bold]Commands[/bold]
  [cyan]/help[/cyan]                 show this help
  [cyan]/subject[/cyan] <name>       switch adapter/subject (the vLLM `model`), alias [cyan]/model[/cyan]
  [cyan]/models[/cyan]               list the subjects the server actually serves
  [cyan]/mode[/cyan] [name]          show or set mode: socratic | graduated_hint
  [cyan]/system[/cyan] [text]        show / override / clear the system prompt ([cyan]/system reset[/cyan])
  [cyan]/temp[/cyan] [0-2]           show or set sampling temperature
  [cyan]/tokens[/cyan] [n]           show or set max output tokens
  [cyan]/url[/cyan] [base_url]       show or switch the endpoint
  [cyan]/info[/cyan]                 show current settings
  [cyan]/retry[/cyan]                re-run the last user turn
  [cyan]/undo[/cyan]                 drop the last user+assistant exchange
  [cyan]/clear[/cyan]                clear the conversation (keeps settings)
  [cyan]/save[/cyan] [path]          save the transcript as markdown
  [cyan]/exit[/cyan]                 quit (also Ctrl-D)

Plain text is sent to the tutor. Ctrl-C stops a running reply.\
"""


def cmd_info(sess):
    sys_preview = sess.system_prompt().replace("\n", " ")
    if len(sys_preview) > 70:
        sys_preview = sys_preview[:67] + "..."
    sess.console.print(
        f"[bold]subject[/bold]  {sess.subject}\n"
        f"[bold]mode[/bold]     {sess.mode}"
        f"{' [dim](overridden by /system)[/dim]' if sess.custom_system else ''}\n"
        f"[bold]url[/bold]      {sess.base_url}\n"
        f"[bold]temp[/bold]     {sess.temperature}\n"
        f"[bold]tokens[/bold]   {sess.max_tokens}\n"
        f"[bold]turns[/bold]    {len(sess.history)}\n"
        f"[bold]system[/bold]   [dim]{sys_preview}[/dim]")


def save_transcript(sess, path):
    lines = [f"# ParaClient transcript — {sess.subject} / {sess.mode}",
             f"_{datetime.now().isoformat(timespec='seconds')}_", "",
             f"**System:** {sess.system_prompt()}", ""]
    for m in sess.history:
        who = "You" if m["role"] == "user" else "Tutor"
        lines.append(f"### {who}\n\n{m['content']}\n")
    Path(path).write_text("\n".join(lines))
    return path


def handle_command(sess, line) -> bool:
    """Return True to keep looping, False to quit."""
    c = sess.console
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q"):
        return False
    elif cmd in ("help", "h", "?"):
        c.print(HELP)
    elif cmd == "info":
        cmd_info(sess)
    elif cmd in ("subject", "model"):
        if arg:
            sess.subject = arg
            c.print(f"[green]subject → {arg}[/green]")
            avail = sess.list_models(quiet=True)
            if avail is not None and arg not in avail:
                c.print(f"[yellow]note: '{arg}' isn't in the server's list "
                        f"({', '.join(avail) or 'none'}) — /models to check[/yellow]")
        else:
            c.print(f"subject: [bold]{sess.subject}[/bold]")
    elif cmd == "models":
        ids = sess.list_models()
        if ids is not None:
            if not ids:
                c.print("[yellow](server returned no models)[/yellow]")
            for mid in ids:
                mark = "  [green]← current[/green]" if mid == sess.subject else ""
                c.print(f"  [bold]{mid}[/bold]{mark}")
            if sess.subject not in ids:
                c.print(f"[yellow]current subject '{sess.subject}' is not "
                        f"served — pick one above with /subject[/yellow]")
    elif cmd == "mode":
        if not arg:
            c.print(f"mode: [bold]{sess.mode}[/bold]  "
                    f"[dim](options: {', '.join(MODES)})[/dim]")
        elif arg in MODES:
            sess.mode, sess.custom_system = arg, None
            c.print(f"[green]mode → {arg}[/green]")
        else:
            c.print(f"[red]unknown mode '{arg}'[/red] — options: "
                    f"{', '.join(MODES)}")
    elif cmd == "system":
        if not arg:
            c.print(f"[dim]{sess.system_prompt()}[/dim]")
        elif arg.lower() == "reset":
            sess.custom_system = None
            c.print("[green]system prompt reset to mode default[/green]")
        else:
            sess.custom_system = arg
            c.print("[green]custom system prompt set[/green]")
    elif cmd == "temp":
        if not arg:
            c.print(f"temp: [bold]{sess.temperature}[/bold]")
        else:
            try:
                sess.temperature = max(0.0, min(2.0, float(arg)))
                c.print(f"[green]temp → {sess.temperature}[/green]")
            except ValueError:
                c.print("[red]temp must be a number 0–2[/red]")
    elif cmd == "tokens":
        if not arg:
            c.print(f"max tokens: [bold]{sess.max_tokens}[/bold]")
        else:
            try:
                sess.max_tokens = max(1, int(arg))
                c.print(f"[green]max tokens → {sess.max_tokens}[/green]")
            except ValueError:
                c.print("[red]tokens must be an integer[/red]")
    elif cmd == "url":
        if not arg:
            c.print(f"url: [bold]{sess.base_url}[/bold]")
        else:
            from openai import OpenAI
            sess.base_url = arg
            sess.client = OpenAI(base_url=arg,
                                 api_key=sess.client.api_key)
            c.print(f"[green]url → {arg}[/green]")
    elif cmd == "clear":
        sess.history.clear()
        c.print("[green]conversation cleared[/green]")
    elif cmd == "undo":
        # drop trailing assistant, then trailing user
        if sess.history and sess.history[-1]["role"] == "assistant":
            sess.history.pop()
        if sess.history and sess.history[-1]["role"] == "user":
            sess.history.pop()
        c.print("[green]last exchange removed[/green]")
    elif cmd == "retry":
        if sess.history and sess.history[-1]["role"] == "assistant":
            sess.history.pop()
        if sess.history and sess.history[-1]["role"] == "user":
            c.print(f"[dim]retrying: {sess.history[-1]['content'][:60]}[/dim]")
            _turn_header(sess, "Tutor")
            sess.stream_reply()
        else:
            c.print("[yellow]nothing to retry[/yellow]")
    elif cmd == "save":
        path = arg or f"transcript-{sess.subject}-{datetime.now():%Y%m%d-%H%M%S}.md"
        try:
            save_transcript(sess, path)
            c.print(f"[green]saved → {path}[/green]")
        except OSError as e:
            c.print(f"[red]could not save: {e}[/red]")
    else:
        c.print(f"[red]unknown command '/{cmd}'[/red] — try /help")
    return True


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _turn_header(sess, who):
    color = "cyan" if who == "You" else "green"
    sess.console.print(f"\n[bold {color}]{who}[/bold {color}] "
                       f"[dim]· {sess.subject} · {sess.mode}[/dim]")


def banner(console, sess):
    console.rule("[bold]ParaClient[/bold] · ParaFrames tutor")
    console.print(f"[dim]{sess.base_url}  ·  subject=[/dim][bold]{sess.subject}"
                  f"[/bold][dim]  ·  mode=[/dim][bold]{sess.mode}[/bold]")
    console.print("[dim]Type to chat · /help for commands · Ctrl-C stops a "
                  "reply · Ctrl-D quits[/dim]\n")


def build_prompt_session(sess, subjects=()):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import HTML

    cmds = ["/help", "/subject ", "/model ", "/models", "/mode ", "/system ",
            "/temp ", "/tokens ", "/url ", "/info", "/retry", "/undo",
            "/clear", "/save ", "/exit"]
    # offer live subject names as completions of /subject
    cmds += [f"/subject {s}" for s in subjects]
    completer = WordCompleter(cmds, sentence=True, ignore_case=True)

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        hist = FileHistory(str(HISTORY_PATH))
    except OSError:
        hist = None

    def toolbar():
        return HTML(f" <b>{sess.subject}</b> · {sess.mode} · "
                    f"t={sess.temperature} · {sess.base_url}")

    return PromptSession(history=hist, completer=completer,
                         complete_while_typing=False,
                         bottom_toolbar=toolbar)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Claude Code-style terminal chat for the ParaFrames tutor.")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint "
                    "(required; or set PARACLIENT_BASE_URL / config file)")
    ap.add_argument("--api-key", help="API key (default: env or 'EMPTY')")
    ap.add_argument("--subject", "--model", dest="subject",
                    help=f"adapter/subject = vLLM `model` (default {DEFAULT_SUBJECT})")
    ap.add_argument("--mode", choices=list(MODES),
                    help=f"tutor mode (default {DEFAULT_MODE})")
    ap.add_argument("--temp", dest="temp", type=float,
                    help="sampling temperature (default 0.3)")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    help="max output tokens (default 800)")
    args = ap.parse_args()

    cfg = load_config_file()

    base_url = resolve(args, cfg, "base-url", "PARACLIENT_BASE_URL")
    if not base_url:
        print("error: no endpoint configured.\n"
              "  pass  --base-url http://localhost:8000/v1\n"
              "  or    export PARACLIENT_BASE_URL=http://localhost:8000/v1\n"
              f"  or    write {CONFIG_PATH}  with  "
              '{"base_url": "http://localhost:8000/v1"}',
              file=sys.stderr)
        sys.exit(2)

    api_key = (args.api_key or os.environ.get("PARACLIENT_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
               or "EMPTY")
    subject = resolve(args, cfg, "subject", "PARACLIENT_SUBJECT", DEFAULT_SUBJECT)
    mode = resolve(args, cfg, "mode", "PARACLIENT_MODE", DEFAULT_MODE)
    if mode not in MODES:
        print(f"error: unknown mode '{mode}' (options: {', '.join(MODES)})",
              file=sys.stderr)
        sys.exit(2)
    temperature = args.temp if args.temp is not None else cfg.get("temp", 0.3)
    max_tokens = args.max_tokens or cfg.get("max_tokens", 800)

    from openai import OpenAI
    from rich.console import Console

    console = Console()
    client = OpenAI(base_url=base_url, api_key=api_key)
    sess = Session(client, console, subject, mode, temperature, max_tokens,
                   base_url)

    banner(console, sess)

    # best-effort startup probe: confirm we can reach the server and show what
    # subjects it serves (also feeds tab-completion). Never fatal.
    console.print("[dim]connecting…[/dim]")
    subjects = sess.list_models(quiet=True)
    if subjects:
        console.print(f"[dim]subjects available: [/dim]{', '.join(subjects)}")
        if sess.subject not in subjects:
            console.print(f"[yellow]heads up: '{sess.subject}' isn't served — "
                          f"switch with /subject or /models[/yellow]")
    else:
        console.print("[yellow]couldn't list models (server unreachable or key "
                      "rejected). You can still chat, or fix with /url.[/yellow]")
    console.print()

    psession = build_prompt_session(sess, subjects or ())

    while True:
        try:
            line = psession.prompt("› ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(sess, line):
                console.print("[dim]bye[/dim]")
                break
            continue

        sess.history.append({"role": "user", "content": line})
        _turn_header(sess, "Tutor")
        sess.stream_reply()


if __name__ == "__main__":
    main()
