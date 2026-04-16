import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from pynput.keyboard import Controller, Key

from Synthesize.synthesize import predict_keystrokes, DEFAULT_OUTPUT_DIR


# Characters that need a pynput Key enum instead of a raw string
_SPECIAL_KEYS = {
    " ": Key.space,
    "\n": Key.enter,
    "\t": Key.tab,
}


class KeyForgeApp(tk.Tk):
    ##application class for the keyforge app
    #Will contain a input box for the user to type in, and a button to generate the keystrokes
    # Then allows the user to either download the csv file, or actually type out the
    # keystrokes in real time, with the app showing the timing of each keystroke as they type
    def __init__(self):
        super().__init__()
        self.title("KeyForge")
        self.geometry("420x460")

        # Create input box
        self.input_label = ttk.Label(self, text="Enter text to generate keystrokes:")
        self.input_label.pack(pady=10)

        self.input_entry = tk.Text(self, width=50, height=10)
        self.input_entry.pack(pady=5)

        # Create generate button
        self.generate_button = ttk.Button(self, text="Generate Keystrokes", command=self.generate_keystrokes)
        self.generate_button.pack(pady=10)

        # Download button — created once, disabled until keystrokes exist
        self.download_button = ttk.Button(self, text="Download CSV", command=self.download_csv, state="disabled")
        self.download_button.pack(pady=5)

        # TypeIt button — created once, disabled until keystrokes exist
        self.typeit_button = ttk.Button(self, text="TypeIt", command=self.typeit, state="disabled")
        self.typeit_button.pack(pady=5)

        # Typing speed slider (WPM). Scales replay only — the downloaded CSV
        # always reflects the raw model prediction. 40 WPM is the avg typist.
        self.BASELINE_WPM = 40.0
        self.wpm_var = tk.IntVar(value=int(self.BASELINE_WPM))
        wpm_frame = ttk.Frame(self)
        wpm_frame.pack(pady=(10, 2), fill="x", padx=20)
        self.wpm_label = ttk.Label(wpm_frame, text=f"Typing speed: {self.wpm_var.get()} WPM")
        self.wpm_label.pack(anchor="w")
        self.wpm_slider = ttk.Scale(
            wpm_frame, from_=20, to=120, orient="horizontal",
            variable=self.wpm_var, command=self._on_wpm_change,
        )
        self.wpm_slider.pack(fill="x")

        # Status label + indeterminate progress bar for long-running work
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack(pady=(8, 2))
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=200)
        self.progress.pack(pady=2)

        # Placeholder for generated keystrokes
        self.keystrokes = None
        self.keyboard = Controller()

    def _on_wpm_change(self, _value):
        # Scale is a float coming in; keep IntVar display clean
        self.wpm_label.config(text=f"Typing speed: {self.wpm_var.get()} WPM")

    # -- Generate 
    def generate_keystrokes(self):
        # Get input text and generate the keystroke dataframe using synthesize.py's predict_keystrokes function.  This runs on a background thread to keep the UI responsive, with a busy spinner and status message while it runs.
        input_text = self.input_entry.get("1.0", "end-1c").strip()
        if not input_text:
            print("No text entered.")
            return

        # Disable controls, show spinner
        self._set_busy(True, "Loading model & synthesizing...")

        # Run inference on a background thread so the Tk event loop stays alive
        threading.Thread(
            target=self._generate_worker, args=(input_text,), daemon=True
        ).start()

    def _generate_worker(self, input_text):
        # Generate keystrokes in a background thread
        try:
            df = predict_keystrokes(input_text)
        except Exception as e:       # surface errors back to the UI thread
            self.after(0, self._generate_done, None, e)
            return
        self.after(0, self._generate_done, df, None)

    def _generate_done(self, df, err):
        # Update UI with results from generation thread
        self._set_busy(False, "")
        if err is not None:
            messagebox.showerror("Error", f"Failed to generate keystrokes:\n{err}")
            return
        self.keystrokes = df
        self.download_button.config(state="normal")
        self.typeit_button.config(state="normal")
        self.status_var.set(f"Ready — {len(df)} keystrokes generated.")

    #  Download 
    def download_csv(self):
        #convert self.keystrokes to a csv file and prompt the user to download it.  Use filedialog.asksaveasfilename to get the destination path, with a default filename of "keystrokes.csv" and default directory of DEFAULT_OUTPUT_DIR.  Show a success messagebox after saving.
        if self.keystrokes is None:
            messagebox.showwarning("No data", "Generate keystrokes first.")
            return
        path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_OUTPUT_DIR,
            initialfile="keystrokes.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.keystrokes.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    # Type it out
    def typeit(self):
        #use pynput to simulate typing out the keystrokes in real time, with the specified DwellTime and FlightTime.  This runs on a background thread so the UI stays responsive, with a countdown in the status message to give the user time to click into the target window.
        if self.keystrokes is None:
            messagebox.showwarning("No data", "Generate keystrokes first.")
            return
        self.typeit_button.config(state="disabled")
        self.generate_button.config(state="disabled")
        self._countdown(3)

    def _countdown(self, n):
        # Countdown before starting to type, giving the user time to focus the target window
        if n > 0:
            self.status_var.set(f"Typing in {n}... (click target window now)")
            self.after(1000, self._countdown, n - 1)
        else:
            self.status_var.set("Typing...")
            threading.Thread(target=self._replay_worker, daemon=True).start()

    def _replay_worker(self):
        # Replay the keystrokes using pynput, respecting the DwellTime and FlightTime.  This runs in a background thread to keep the UI responsive.  After replaying, it updates the status message and re-enables the buttons.

        # Scale replay times by slider: higher target WPM => smaller scale =>
        # shorter Dwell/Flight sleeps. Baseline is the average typist (~40 WPM).
        target_wpm = max(1, self.wpm_var.get())
        scale = self.BASELINE_WPM / target_wpm

        try:
            for _, row in self.keystrokes.iterrows():
                ft = row["FlightTime"]
                if pd.notna(ft) and ft > 0:
                    time.sleep((ft * scale) / 1000.0)
                ch = row["char"]
                key = _SPECIAL_KEYS.get(ch, ch)
                try:
                    self.keyboard.press(key)
                    time.sleep(max(row["DwellTime"] * scale, 1) / 1000.0)
                    self.keyboard.release(key)
                except Exception:
                    # Fall back to typing the raw character (handles unicode)
                    self.keyboard.type(ch)
        finally:
            self.after(0, self._replay_done)

    def _replay_done(self):
        # when your done typing, update the status and re-enable buttons
        self.status_var.set("Done typing.")
        self.typeit_button.config(state="normal")
        self.generate_button.config(state="normal")

    # -- Helpers 
    def _set_busy(self, busy, message=""):
        # Update the UI to reflect a busy state (e.g. during generation or typing).  Shows a message and starts/stops the progress spinner, and disables/enables buttons accordingly.
        self.status_var.set(message)
        if busy:
            self.progress.start(10)
            self.generate_button.config(state="disabled")
            self.download_button.config(state="disabled")
            self.typeit_button.config(state="disabled")
        else:
            self.progress.stop()
            self.generate_button.config(state="normal")


if __name__ == "__main__":
    app = KeyForgeApp()
    app.mainloop()