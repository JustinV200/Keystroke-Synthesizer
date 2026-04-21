"""Tkinter UI for capturing a user's real keystrokes for fine-tuning.

Records per-key DownTime, UpTime, ActionTime, and DownEvent (matching the
schema produced by the KLiCKe dataset and consumed by dataPipeline.dataPrepper).
Saves each capture under src/FineTuner/users/<name>/ as a paired
texts/<id>.txt and csv/<id>.csv so it can be fed straight into dataLoader.
"""
from __future__ import annotations

import csv
import os
import random
import re
import time
import tkinter as tk
from tkinter import ttk, messagebox


# Prompt pool — one is chosen at random each time the UI opens / New Prompt is
# pressed. Variety matters so char_embed learns the user's rhythm across a
# wide character distribution, not just a handful of memorized bigrams.
PROMPTS = [
    # 1. Pangrams + keystroke-dynamics blurb (original default).
    "The quick brown fox jumps over the lazy dog. Amazingly few discotheques "
    "provide jukeboxes. Pack my box with five dozen liquor jugs. How "
    "vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow.\n\n"
    "Keystroke dynamics describes the way a particular person types on a "
    "keyboard; dwell time and flight time are the two features most often "
    "used to characterize an individual's rhythm.",

    # 2. Conversational prose with common bigrams.
    "When I was younger, my grandfather used to tell me that the secret to a "
    "good story is knowing which details to leave out. He would sit on the "
    "porch in the evenings, watching the sun slip behind the hills, and "
    "describe places he had only read about in books. I never understood "
    "why he cared so much about accuracy until I started writing myself.",

    # 3. Technical/code-adjacent text with symbols and numbers.
    "The function returns 0 on success and -1 on failure. If the buffer is "
    "null or size == 0, it short-circuits immediately. Remember: index 42 is "
    "exclusive, and the caller owns the pointer. Always check errno before "
    "assuming the syscall failed for the reason you expected — EINTR is "
    "sneaky and shows up in roughly 2% of real-world workloads.",

    # 4. News-style text with proper nouns and punctuation.
    "Researchers at the university announced a new method for measuring "
    "atmospheric carbon that costs roughly one-tenth of existing techniques. "
    "The team, led by Dr. Ellen Park, published their findings in Nature on "
    "Tuesday. \"It's not a silver bullet,\" Park said, \"but it will let "
    "smaller labs contribute to climate monitoring in a serious way.\"",

    # 5. Lyrical/varied punctuation to exercise shift-layer characters.
    "It's strange how a song you haven't heard in years can pull you back to "
    "a single afternoon — the smell of the kitchen, the slant of the light, "
    "the hum of the refrigerator. Memory doesn't work in straight lines; it "
    "loops, skips, and sometimes (when you least expect it) hands you "
    "something you thought you'd lost forever. Isn't that odd?",
]

# Where captures are written (sibling of this file).
USERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users")


def _keysym_to_down_event(event) -> str:
    """Translate a Tk KeyPress event into the DownEvent string dataPrepper expects.

    dataPrepper treats single-character strings plus "Space" and "Enter" as
    character-producing; anything else (Shift, Control, arrow keys, etc.) is
    filtered out by _is_char_producing. "Backspace" is handled specially by
    the edit-replay logic.
    """
    keysym = event.keysym
    char = event.char

    if keysym in ("space", "Space"):
        return "Space"
    if keysym in ("Return", "KP_Enter"):
        return "Enter"
    if keysym == "BackSpace":
        return "Backspace"

    # Printable single character (letters, digits, punctuation) — keep the
    # actual character so case/shift-layer information is preserved.
    if char and len(char) == 1 and char.isprintable():
        return char

    # Modifiers, arrows, F-keys, etc. Kept verbatim so they appear in the
    # raw CSV; the pipeline will drop them as non-producing.
    return keysym


class KeystrokeCapture(tk.Tk):
    """Minimal capture UI — user types a prompt, we save the raw keystroke CSV."""

    def __init__(self):
        super().__init__()
        self.title("KeyForge — Keystroke Capture")
        self.geometry("720x560")

        # Per-key state: DownEvent string -> down_time_ms, closed out on release.
        self._open_keys: dict[str, float] = {}
        self._rows: list[dict] = []
        self._t0: float | None = None
        self._action_idx = 0
        self._recording = False

        self._build_ui()

    # UI construction
    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        name_frame = ttk.Frame(self)
        name_frame.pack(fill="x", **pad)
        ttk.Label(name_frame, text="User name:").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=24)
        self.name_entry.pack(side="left", padx=8)

        prompt_header = ttk.Frame(self)
        prompt_header.pack(fill="x", **pad)
        ttk.Label(prompt_header, text="Prompt (copy this as you type below):").pack(side="left")
        self.new_prompt_btn = ttk.Button(prompt_header, text="New prompt", command=self._pick_new_prompt, width=12)
        self.new_prompt_btn.pack(side="right")

        self.prompt_box = tk.Text(self, height=7, wrap="word", bg="#f4f1ea", fg="#1a1a1a")
        self._set_prompt(random.choice(PROMPTS))
        # Keep it editable-but-readonly: block input via key/paste bindings so
        # the fg color is respected (tk.Text dims text when state=disabled and
        # has no disabledforeground option).
        self.prompt_box.bind("<Key>", lambda e: "break")
        self.prompt_box.bind("<<Paste>>", lambda e: "break")
        self.prompt_box.pack(fill="x", padx=12)

        ttk.Label(self, text="Type here while recording:").pack(anchor="w", **pad)
        self.typing_box = tk.Text(self, height=10, wrap="word", bg="#ffffff", fg="#1a1a1a", insertbackground="#1a1a1a")
        self.typing_box.pack(fill="both", expand=True, padx=12)
        # Capture via Tk events rather than pynput so focus scoping is trivial.
        self.typing_box.bind("<KeyPress>", self._on_key_press)
        self.typing_box.bind("<KeyRelease>", self._on_key_release)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btn_frame, text="Start recording", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btn_frame, text="Stop & save", command=self._stop_and_save, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self._clear)
        self.clear_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready. Enter a name, then Start recording.")
        ttk.Label(self, textvariable=self.status_var, foreground="#555").pack(anchor="w", **pad)

    def _set_prompt(self, text: str):
        """Replace the prompt box contents (bypassing the read-only key binding)."""
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", text)

    def _pick_new_prompt(self):
        """Swap in a different random prompt. Disabled mid-recording."""
        if self._recording:
            return
        current = self.prompt_box.get("1.0", "end-1c")
        # Avoid picking the same prompt twice in a row.
        choices = [p for p in PROMPTS if p != current] or PROMPTS
        self._set_prompt(random.choice(choices))

    # Recording control
    def _start(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a user name first.")
            return
        if not re.match(r"^[A-Za-z0-9_\- ]+$", name):
            messagebox.showwarning(
                "Invalid name",
                "Use letters, numbers, spaces, hyphens, or underscores only.",
            )
            return

        self._rows.clear()
        self._open_keys.clear()
        self._action_idx = 0
        self._t0 = time.perf_counter()
        self._recording = True

        self.typing_box.config(state="normal")
        self.typing_box.delete("1.0", "end")
        self.typing_box.focus_set()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.name_entry.config(state="disabled")
        self.status_var.set(f"Recording… type the prompt. ({name})")

    def _clear(self):
        # Full reset: stop any in-progress recording, drop all buffered rows,
        # and put the UI back to its initial state.
        self._recording = False
        self._rows.clear()
        self._open_keys.clear()
        self._action_idx = 0
        self._t0 = None

        self.typing_box.config(state="normal")
        self.typing_box.delete("1.0", "end")
        self._reset_controls()
        self.status_var.set("Cleared. Ready to record.")

    def _stop_and_save(self):
        if not self._recording:
            return
        self._recording = False
        # Close out any keys held at stop time so no row has a missing UpTime.
        now_ms = self._ms_since_start()
        for evt, down_ms in list(self._open_keys.items()):
            self._rows.append(self._row(down_ms, now_ms, evt))
        self._open_keys.clear()

        if not self._rows:
            messagebox.showinfo("Nothing to save", "No keystrokes were recorded.")
            self._reset_controls()
            return

        typed_text = self.typing_box.get("1.0", "end-1c")
        name = self.name_var.get().strip()
        try:
            csv_path, txt_path = self._write_capture(name, typed_text)
        except OSError as e:
            messagebox.showerror("Save failed", f"Could not save capture:\n{e}")
            self._reset_controls()
            return

        messagebox.showinfo(
            "Saved",
            f"Saved {len(self._rows)} keystroke rows.\n\nCSV: {csv_path}\nText: {txt_path}",
        )
        self._reset_controls()
        self.status_var.set(f"Saved {len(self._rows)} rows for {name}.")

    def _reset_controls(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.name_entry.config(state="normal")

    # Event handlers
    def _ms_since_start(self) -> float:
        assert self._t0 is not None
        return (time.perf_counter() - self._t0) * 1000.0

    def _on_key_press(self, event):
        if not self._recording:
            return
        evt = _keysym_to_down_event(event)
        # Tk fires auto-repeat press events when a key is held. Treat a repeat
        # as a new press: close the previous one before opening a fresh entry.
        if evt in self._open_keys:
            prev_down = self._open_keys.pop(evt)
            self._rows.append(self._row(prev_down, self._ms_since_start(), evt))
        self._open_keys[evt] = self._ms_since_start()

    def _on_key_release(self, event):
        if not self._recording:
            return
        evt = _keysym_to_down_event(event)
        down_ms = self._open_keys.pop(evt, None)
        if down_ms is None:
            # Release without a matching press (e.g. key was held before Start).
            return
        self._rows.append(self._row(down_ms, self._ms_since_start(), evt))

    # Row / file helpers
    def _row(self, down_ms: float, up_ms: float, evt: str) -> dict:
        self._action_idx += 1
        # dataPrepper requires ActionTime non-null; using dwell is close enough
        # to the KLiCKe convention for downstream cleaning to accept it.
        action_time = max(up_ms - down_ms, 0.0)
        return {
            "event_id": self._action_idx,
            "DownTime": round(down_ms, 3),
            "UpTime": round(up_ms, 3),
            "ActionTime": round(action_time, 3),
            "DownEvent": evt,
        }

    def _write_capture(self, name: str, typed_text: str) -> tuple[str, str]:
        safe_name = re.sub(r"\s+", "_", name.strip())
        user_dir = os.path.join(USERS_DIR, safe_name)
        csv_dir = os.path.join(user_dir, "csv")
        txt_dir = os.path.join(user_dir, "texts")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        capture_id = time.strftime("capture_%Y%m%d_%H%M%S")
        csv_path = os.path.join(csv_dir, f"{capture_id}.csv")
        txt_path = os.path.join(txt_dir, f"{capture_id}.txt")

        # Sort by DownTime so the CSV is in the order dataPrepper expects.
        rows = sorted(self._rows, key=lambda r: r["DownTime"])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["event_id", "DownTime", "UpTime", "ActionTime", "DownEvent"]
            )
            writer.writeheader()
            writer.writerows(rows)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(typed_text)

        return csv_path, txt_path


if __name__ == "__main__":
    app = KeystrokeCapture()
    app.mainloop()
