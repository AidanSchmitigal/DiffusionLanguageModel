"""
diffusion_view.py -- a rich terminal viewer for masked-diffusion decoding.

Every token is a fixed-width, 3-line "stack":

    ┌────────┐
    │  the   │  row 1: token currently in the sequence (░░░░ = still masked)
    │  cat   │  row 2: what the model predicts for this position at this step
    │ ██████ │  row 3: confidence of that prediction
    └────────┘

When a ground-truth sample is supplied (`truth=`), a 4th row is added:

    ┌────────┐
    │  the   │  row 1: token currently in the sequence
    │  cat   │  row 2: prediction
    │ ██████ │  row 3: confidence
    │  the   │  row 4: the original sample token (green = match, red = mismatch)
    └────────┘

The header then reports how much of the original sample the model reproduced.

Cell colour is the diff against the previous step:
    MASK     still masked
    NEW      mask -> token on this step
    CHANGED  a committed token was replaced
    KEPT     identical to the previous step

A yellow prediction row means the model disagrees with a committed token.

Every step is recorded, so you can step backwards and forwards. The bottom
panel shows the step summary and the top-k distribution at the cursor.

Keys
    ← / →      previous / next step         (→ on the last step continues generation)
    ↑ / ↓      move cursor one row          , / .   move cursor one token
    f          jump cursor to next token filled this step
    g / G      first / last step            q       quit

Run `python diffusion_view.py` for a demo with fake data (add --live to skip pausing).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:  # keeps torch / transformers optional at import time
    import torch
    from transformers import PreTrainedTokenizerBase


# --------------------------------------------------------------------------- #
# Keyboard input (POSIX + Windows)
# --------------------------------------------------------------------------- #
class _KeyReader:
    _CSI = {
        b"[A": "up",
        b"[B": "down",
        b"[C": "right",
        b"[D": "left",
        b"[H": "home",
        b"[F": "end",
        b"OH": "home",
        b"OF": "end",
        b"[1~": "home",
        b"[4~": "end",
    }
    _WIN = {"H": "up", "P": "down", "M": "right", "K": "left", "G": "home", "O": "end"}

    def __enter__(self) -> "_KeyReader":
        if os.name != "nt":
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # keeps Ctrl-C working
        return self

    def __exit__(self, *exc) -> None:
        if os.name != "nt":
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self) -> str:
        if os.name == "nt":
            import msvcrt

            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                return self._WIN.get(msvcrt.getwch(), "")
            return ch

        import select

        first = os.read(self._fd, 1)
        if first != b"\x1b":
            return first.decode(errors="ignore")
        seq = b""
        while select.select([self._fd], [], [], 0.02)[0]:
            seq += os.read(self._fd, 1)
            if len(seq) >= 2 and (seq[-1:].isalpha() or seq.endswith(b"~")):
                break
        return self._CSI.get(seq, "esc")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
class State(Enum):
    MASKED = "MASK"
    NEW = "NEW"
    CHANGED = "CHANGED"
    KEPT = "KEPT"


@dataclass(frozen=True)
class Look:
    bg: tuple[str, str]  # alternating backgrounds so neighbouring tokens stay distinct
    fg: str
    sub_bg: str  # background for prediction + confidence rows
    sub_fg: str


PALETTE = {
    State.MASKED: Look(("#34373d", "#3b3f46"), "#8b929c", "#26282d", "#7fb2e5"),
    State.NEW: Look(("#1f7a43", "#25904f"), "bold #ffffff", "#15502d", "#b7f0c8"),
    State.CHANGED: Look(("#a33636", "#b83f3f"), "bold #ffffff", "#6e2323", "#ffd0d0"),
    State.KEPT: Look(("#27456b", "#2d5180"), "#e6edf3", "#1b3350", "#8fb0d6"),
}
DISAGREE_FG = "#ffd166"
MASK_GLYPH = "░"
_SPECIAL = {"\n": "⏎", "\t": "⇥", "\r": "␍", " ": "·"}
_EIGHTHS = " ▏▎▍▌▋▊▉█"


def _fit(s: str, width: int, align: str = "center") -> str:
    """Truncate with an ellipsis and pad so the result is exactly `width` cells."""
    if cell_len(s) > width:
        out, used = "", 0
        for ch in s:
            w = cell_len(ch)
            if used + w > width - 1:
                break
            out += ch
            used += w
        s = out + "…"
    pad = width - cell_len(s)
    if align == "left":
        return s + " " * pad
    left = pad // 2
    return " " * left + s + " " * (pad - left)


def _bar(frac: float, width: int) -> str:
    frac = 0.0 if frac != frac else min(max(frac, 0.0), 1.0)
    full, part = divmod(round(frac * width * 8), 8)
    s = "█" * full + (_EIGHTHS[part] if part and full < width else "")
    return s.ljust(width)


def _conf_color(c: float) -> str:
    return "#e5534b" if c < 0.34 else "#e3b341" if c < 0.67 else "#3fb950"


# --------------------------------------------------------------------------- #
# Recorded data
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    step: int
    before: list[int]
    after: list[int]
    pred: list[int]  # model prediction at EVERY position
    conf: list[float]  # confidence of that prediction at EVERY position
    committed: list[int]  # positions filled on this step, most confident first
    topk_ids: list[list[int]] | None
    topk_probs: list[list[float]] | None
    masked_before: int
    masked_after: int


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #
class DiffusionViewer:
    """
    Use as a context manager around the sampling loop:

        with DiffusionViewer(tokenizer, mask_id) as viewer:
            for step in ...:
                ...
                if not viewer.record(...):
                    break
            viewer.finish(x[0])
            viewer.browse()

    Pass `truth=<token ids>` to also show the original sample text (row 4 of
    each cell) and the percentage of it the model reproduced.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        mask_id: int,
        *,
        cell_width: int = 8,
        topk: int = 5,
        truth: "torch.Tensor | list[int] | None" = None,
        console: Console | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.cell_width = max(4, cell_width)
        self.topk = max(2, topk)
        self.console = console or Console()

        # Ground truth: the original (un-noised) sample text, one id per position.
        # None means there is nothing to compare against (prefix/suffix mode).
        if truth is None:
            self.truth: list[int] | None = None
        else:
            raw = truth.tolist() if hasattr(truth, "tolist") else list(truth)
            flat: list[int] = []
            for item in raw:  # accept [L] and [1, L]
                if isinstance(item, (list, tuple)):
                    flat.extend(int(t) for t in item)
                else:
                    flat.append(int(item))
            self.truth = flat

        self.frames: list[Frame] = []
        self.index = 0
        self.cursor = 0
        self._scroll = 0
        self._cols = 1
        self._done = False
        self._quit = False
        self._cache: dict[int, str] = {}
        self._live: Live | None = None
        self._keys: _KeyReader | None = None

    # ---- lifecycle -------------------------------------------------------- #
    def __enter__(self) -> "DiffusionViewer":
        if sys.stdin.isatty():
            self._keys = _KeyReader().__enter__()
        self._live = Live(
            Text("Waiting for first step…"),
            console=self.console,
            screen=True,
            auto_refresh=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.__exit__(*exc)
        if self._keys:
            self._keys.__exit__(*exc)

    # ---- public API ------------------------------------------------------- #
    def record(
        self,
        *,
        step: int,
        before: "torch.Tensor",  # [L] tokens before this step
        after: "torch.Tensor",  # [L] tokens after this step
        pred: "torch.Tensor",  # [L] prediction at every position
        conf: "torch.Tensor",  # [L] confidence at every position (NOT masked to -inf)
        probs: "torch.Tensor",  # [L, V] distribution the prediction was drawn from
        committed: "torch.Tensor | list[int]",
    ) -> bool:
        """Store a step and render it. Returns False if the user quit."""
        k = min(self.topk, probs.shape[-1])
        top = probs.float().topk(k, dim=-1)
        before_l, after_l = before.tolist(), after.tolist()
        frame = Frame(
            step=step,
            before=before_l,
            after=after_l,
            pred=pred.tolist(),
            conf=conf.float().clamp(0, 1).tolist(),
            committed=list(committed.tolist() if hasattr(committed, "tolist") else committed),
            topk_ids=top.indices.cpu().tolist(),
            topk_probs=top.values.cpu().tolist(),
            masked_before=sum(t == self.mask_id for t in before_l),
            masked_after=sum(t == self.mask_id for t in after_l),
        )
        self.frames.append(frame)
        self.index = len(self.frames) - 1
        if frame.committed:
            self.cursor = frame.committed[0]
        self._refresh()
        return not self._quit

    def finish(self, final: "torch.Tensor") -> None:
        """Append a clean final frame (no diff, no prediction noise)."""
        toks = final.tolist()
        self.frames.append(
            Frame(
                step=(self.frames[-1].step + 1) if self.frames else 0,
                before=toks,
                after=toks,
                pred=toks,
                conf=[1.0] * len(toks),
                committed=[],
                topk_ids=None,
                topk_probs=None,
                masked_before=0,
                masked_after=sum(t == self.mask_id for t in toks),
            )
        )
        self.index = len(self.frames) - 1
        self._done = True
        self._refresh()

    def browse(self, index: int | None = None) -> None:
        """Interactive review of all recorded steps until the user presses q."""
        if index is not None:
            self.index = index
        if not self._quit:
            self._interact(until_advance=False)

    # ---- input loop ------------------------------------------------------- #
    def _interact(self, until_advance: bool) -> bool:
        if self._keys is None:  # not a TTY: never block
            return True
        self._refresh()
        while True:
            key = self._keys.read()
            last = len(self.frames) - 1
            L = len(self.frames[self.index].after)
            if key in (".", " ", "\n", "\r", "n"):
                if self.index < last:
                    self.index += 1
                elif until_advance:
                    return True
            elif key in (",", "p"):
                self.index = max(0, self.index - 1)
            elif key in ("home", "g"):
                self.index = 0
            elif key in ("end", "G"):
                self.index = last
            elif key == "up":
                self.cursor = max(0, self.cursor - self._cols)
            elif key == "down":
                self.cursor = min(L - 1, self.cursor + self._cols)
            elif key == "left":
                self.cursor = max(0, self.cursor - 1)
            elif key == "right":
                self.cursor = min(L - 1, self.cursor + 1)
            elif key == "f":
                filled = sorted(self.frames[self.index].committed)
                nxt = [p for p in filled if p > self.cursor] or filled
                if nxt:
                    self.cursor = nxt[0]
            elif key in ("q", "Q"):
                self._quit = True
                return False
            self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render(), refresh=True)

    # ---- token text ------------------------------------------------------- #
    def _esc(self, tid: int) -> str:
        s = self._cache.get(tid)
        if s is None:
            raw = self.tokenizer.decode([tid])
            s = "".join(_SPECIAL.get(c, c if c.isprintable() else "�") for c in raw) or "∅"
            self._cache[tid] = s
        return s

    def _state(self, f: Frame, i: int) -> State:
        if f.after[i] == self.mask_id:
            return State.MASKED
        if f.before[i] == self.mask_id:
            return State.NEW
        return State.CHANGED if f.before[i] != f.after[i] else State.KEPT

    # ---- ground truth ----------------------------------------------------- #
    def _score(self, f: Frame) -> tuple[int, int, int]:
        """(correct, resolved, total) of the committed tokens against the original sample."""
        if self.truth is None:
            return (0, 0, 0)
        total = min(len(self.truth), len(f.after))
        correct = resolved = 0
        for i in range(total):
            if f.after[i] == self.mask_id:
                continue
            resolved += 1
            if f.after[i] == self.truth[i]:
                correct += 1
        return (correct, resolved, total)

    def _truth_cell(self, f: Frame, i: int, look: Look) -> tuple[str, str]:
        """The 4th row: the original sample token at position i, plus its verdict."""
        assert self.truth is not None
        if i >= len(self.truth):
            return ("", f"dim on {look.sub_bg}")
        truth = self.truth[i]
        cur = f.after[i]
        if cur == self.mask_id:
            return (self._esc(truth), f"dim on {look.sub_bg}")
        if cur == truth:
            return (self._esc(truth), f"bold #3fb950 on {look.sub_bg}")
        return (self._esc(truth), f"bold #e5534b on {look.sub_bg}")

    # ---- rendering -------------------------------------------------------- #
    def _render(self):
        if not self.frames:
            return Text("Waiting for first step…")

        size = self.console.size
        f = self.frames[self.index]
        L, W = len(f.after), self.cell_width
        gutter = len(str(L - 1)) + 1
        cols = max(1, (size.width - gutter) // (W + 1))
        self._cols = cols

        chrome = self.topk + 8 + (2 if self.truth is not None else 0)  # header, legend, match, spacer, panel, footer
        H = 5 if self.truth is not None else 4  # visual lines per cell
        rows_total = math.ceil(L / cols)
        rows_vis = max(1, min(rows_total, (size.height - chrome) // H))

        cur_row = self.cursor // cols
        self._scroll = min(self._scroll, max(0, rows_total - rows_vis))
        if cur_row < self._scroll:
            self._scroll = cur_row
        elif cur_row >= self._scroll + rows_vis:
            self._scroll = cur_row - rows_vis + 1

        lines: list[Text] = []
        for r in range(rows_vis):
            start = (self._scroll + r) * cols
            row_lines = [Text(no_wrap=True, overflow="crop") for _ in range(H)]
            row_lines[0].append(" " * gutter)
            row_lines[1].append(str(start).rjust(gutter - 1) + " ", style="dim")
            for k in range(2, H):
                row_lines[k].append(" " * gutter)

            for col in range(cols):
                i = start + col
                if i >= L:
                    break
                look = PALETTE[self._state(f, i)]
                bg = look.bg[i % 2]
                cur, prd = f.after[i], f.pred[i]
                state = self._state(f, i)

                top = MASK_GLYPH * (W - 2) if cur == self.mask_id else self._esc(cur)
                disagree = state in (State.KEPT, State.CHANGED) and prd != cur
                pred_fg = DISAGREE_FG if disagree else look.sub_fg

                cells: list[tuple[str, str]] = [
                    ("▼" if i == self.cursor else "", "bold #ffd166"),
                    (top, f"{look.fg} on {bg}"),
                    (self._esc(prd), f"{'italic ' if disagree else ''}{pred_fg} on {look.sub_bg}"),
                    (_bar(f.conf[i], W), f"{_conf_color(f.conf[i])} on {look.sub_bg}"),
                ]
                if self.truth is not None:
                    cells.append(self._truth_cell(f, i, look))
                for line, (txt, style) in zip(row_lines, cells):
                    line.append(_fit(txt, W), style=style)
                    line.append(" ")
            lines.extend(row_lines)

        return Group(
            self._header(f),
            self._legend(),
            *([self._match_line(f)] if self.truth is not None else []),
            Text(""),
            Text("\n").join(lines),
            self._detail(f),
            self._footer(),
        )

    def _header(self, f: Frame) -> Text:
        n, prog = len(self.frames), 24
        filled = round(prog * (self.index + 1) / n)
        return Text.assemble(
            ("◆ diffusion trace  ", "bold"),
            (f"step {f.step}", "bold cyan"),
            f"  frame {self.index + 1}/{n}  ",
            ("━" * filled, "cyan"),
            ("━" * (prog - filled), "grey30"),
            f"  {f.masked_after}/{len(f.after)} masked  ",
            ("■ done", "green") if self._done else ("▶ generating", "yellow"),
        )

    def _match_line(self, f: Frame) -> Text | None:
        """How much of the original sample the model has reproduced so far."""
        if self.truth is None:
            return None
        correct, resolved, total = self._score(f)
        pct = (correct / resolved * 100) if resolved else 0.0
        t = Text.assemble(
            ("match ", "dim"),
            (f"{pct:.1f}%", "bold cyan"),
            (f"   {correct}/{total} correct", "dim"),
        )
        if total - resolved:
            t.append(f"   {total - resolved} still masked", "dim yellow")
        t.append("   row 4 = original sample: ", "dim")
        t.append("green = match", "#3fb950")
        t.append(", ", "dim")
        t.append("red = mismatch", "#e5534b")
        t.no_wrap = True
        t.overflow = "crop"
        return t

    def _legend(self) -> Text:
        t = Text(no_wrap=True, overflow="crop")
        for st in State:
            look = PALETTE[st]
            t.append(f" {st.value} ", style=f"{look.fg} on {look.bg[0]}")
            t.append(" ")
        t.append("  1 token · 2 pred (", style="dim")
        t.append("yellow", style=DISAGREE_FG)
        t.append(" = disagrees) · 3 conf", style="dim")
        if self.truth is not None:
            t.append(" · 4 sample", style="dim")
        return t

    def _footer(self) -> Text:
        return Text(
            "←/→ step  ↑/↓ row  ,/. token  f next filled  g/G first/last  q quit",
            style="dim",
            no_wrap=True,
            overflow="crop",
        )

    def _detail(self, f: Frame) -> Panel:
        k, i = self.topk, self.cursor
        state = self._state(f, i)
        look = PALETTE[state]
        height = k + 2 + (1 if self.truth is not None else 0)

        # left: step summary -------------------------------------------------
        left: list[Text] = [
            Text.assemble(("step ", "dim"), (str(f.step), "bold"), f"   masked {f.masked_before} → {f.masked_after}"),
            Text.assemble(("filled ", "dim"), (str(len(f.committed)), "bold"), (" this step", "dim")),
        ]
        room = height - 3
        left.append(Text("pos   token       conf", style="dim"))
        shown = f.committed if len(f.committed) <= room else f.committed[: room - 1]
        for pos in shown:
            left.append(
                Text.assemble(
                    f"{pos:<5} ",
                    (_fit(self._esc(f.after[pos]), 10, "left"), f"{PALETTE[State.NEW].fg} on {PALETTE[State.NEW].bg[0]}"),
                    f" {f.conf[pos]:.2f}",
                )
            )
        if len(shown) < len(f.committed):
            left.append(Text(f"+{len(f.committed) - len(shown)} more", style="dim"))

        # right: cursor detail + top-k --------------------------------------
        cur_txt = MASK_GLYPH * 3 if f.after[i] == self.mask_id else self._esc(f.after[i])
        prev_txt = MASK_GLYPH * 3 if f.before[i] == self.mask_id else self._esc(f.before[i])
        head = Text.assemble((f"pos {i}", "bold"), "  ", (f" {state.value} ", f"{look.fg} on {look.bg[0]}"), "  ")
        if state in (State.NEW, State.CHANGED):
            head.append(prev_txt, style="dim")
            head.append(" → ")
        head.append(cur_txt, style="bold")

        right: list[Text] = [
            head,
            Text.assemble(("pred ", "dim"), (self._esc(f.pred[i]), "bold"), (f"   p={f.conf[i]:.3f}", _conf_color(f.conf[i]))),
        ]
        if self.truth is not None and i < len(self.truth):
            truth = self.truth[i]
            if f.after[i] == self.mask_id:
                verdict = ("  pending", "dim")
            elif f.after[i] == truth:
                verdict = ("  ✓ match", "bold #3fb950")
            else:
                verdict = ("  ✗ mismatch", "bold #e5534b")
            right.append(Text.assemble(("truth ", "dim"), (self._esc(truth), "bold"), verdict))
        if f.topk_ids is None or f.topk_probs is None:
            right.append(Text("no distribution for the final state", style="dim"))
        else:
            for rank, (tid, p) in enumerate(zip(f.topk_ids[i], f.topk_probs[i]), 1):
                mark = "✓" if tid == f.after[i] and f.after[i] != self.mask_id else " "
                right.append(
                    Text.assemble(
                        (f"{rank:>2} ", "dim"),
                        _fit(self._esc(tid), 14, "left"),
                        " ",
                        (_bar(p, 24), _conf_color(p)),
                        f" {p * 100:5.1f}% ",
                        (mark, "green"),
                    )
                )

        def pad(rows: list[Text]) -> Text:
            rows = (rows + [Text("")] * height)[:height]
            return Text("\n").join(rows)

        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=2, no_wrap=True, overflow="crop")
        grid.add_column(width=1)
        grid.add_column(ratio=3, no_wrap=True, overflow="crop")
        grid.add_row(pad(left), Text("\n".join(["│"] * height), style="dim"), pad(right))
        return Panel(grid, title=f"position {i} · top-{k}", title_align="left", border_style="grey37", height=height + 2)


# --------------------------------------------------------------------------- #
# Demo with fake data:  python diffusion_view.py [--live]
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import torch

    words = [
        "<mask>",
        "the",
        " cat",
        " dog",
        "\n",
        " sat",
        " on",
        " mat",
        " ran",
        " quickly",
        " and",
        " jumped",
        " over",
        " lazy",
        " fox",
        " hello",
        " world",
        " antidisestablishmentarianism",
        ".",
        " ",
        "\t",
    ]

    class Tok:
        def decode(self, ids):
            return "".join(words[i % len(words)] for i in ids)

    L, V, steps, mask_id = 150, 64, 12, 0
    # a fake "original sample" to compare against (row 4 of each cell)
    truth = [(i * 7 + 3) % 21 + 1 for i in range(L)]
    x = torch.full((L,), mask_id)
    pause = "--live" not in sys.argv
    with DiffusionViewer(Tok(), mask_id, truth=truth) as viewer:
        for step in range(steps):
            probs = torch.rand(L, V) ** 6
            probs[:, mask_id] = 0
            probs /= probs.sum(-1, keepdim=True)
            conf, pred = probs.max(-1)
            is_mask = x == mask_id
            n_mask = int(is_mask.sum())
            k = max(1, math.ceil(n_mask / (steps - step)))
            _, idx = conf.masked_fill(~is_mask, -1).topk(k)
            before = x.clone()
            x[idx] = pred[idx]
            # let a few positions land on the truth so the demo shows both verdicts
            x[::11] = torch.tensor(truth[::11])
            if not viewer.record(step=step, before=before, after=x, pred=pred, conf=conf, probs=probs, committed=idx):
                break
            if pause:
                import time

                time.sleep(0.3)
        viewer.finish(x)
        viewer.browse()


if __name__ == "__main__":
    _demo()
