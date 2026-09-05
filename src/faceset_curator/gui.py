# Copyright © 2026 Hanafi Mohd Radi. All rights reserved.

from __future__ import annotations

import copy
import logging
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from . import __copyright__
from .analyzer import BaselineAnalyzer
from .app_logging import configure_logging
from .cuda_analyzer import InsightFaceAnalyzer
from .identity import identity_statistics
from .models import CuratorConfig, ImageAnalysis
from .pipeline import CurationCancelled, curate
from .preflight import InsufficientDiskSpace, check_disk_space
from .review import (CATEGORY_FILTERS, apply_manual_selection, duplicate_peers,
                     filter_items, sync_review_output, why_rejected)
from .telemetry import SystemMonitor

LOGGER = logging.getLogger(__name__)


def friendly_failure(message: str) -> str:
    lowered = message.lower()
    if "download" in lowered or "connectionreset" in lowered or "connection reset" in lowered:
        return ("The face-analysis models could not be downloaded completely. FSC kept the partial "
                "download and will resume it when you start again.")
    if "cuda" in lowered or "cudnn" in lowered or "executionprovider" in lowered:
        return ("CUDA could not start, so FSC stopped instead of silently using the CPU. Open diagnostics "
                "for details, then run 'fsc doctor --device cuda'.")
    if "reference" in lowered:
        return "The target reference images could not be enrolled. Use clear images containing exactly one face."
    if "free space" in lowered or "not enough" in lowered:
        return message
    return "Curation stopped because an unexpected error occurred. Open diagnostics for the technical details."


class FaceSetCuratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceSet Curator")
        self.geometry("1120x760")
        self.minsize(900, 640)
        self.configure(bg="#101720")
        self.log_path = configure_logging()
        LOGGER.info("Desktop application started")
        self.events: queue.Queue[tuple] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.active_config: CuratorConfig | None = None
        self.run_dir: Path | None = None
        self.reference_photo_refs: list[ImageTk.PhotoImage] = []
        self.review_photo_refs: list[ImageTk.PhotoImage] = []
        self.items: list[ImageAnalysis] = []
        self.review_current: ImageAnalysis | None = None
        self.review_compare: ImageAnalysis | None = None
        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.references_var = tk.StringVar(value="No reference images selected")
        self.backend_var = tk.StringVar(value="insightface")
        self.device_var = tk.StringVar(value="auto")
        self.profile_var = tk.StringVar(value="Balanced")
        self.duplicates_var = tk.StringVar(value="Strong")
        self.identity_var = tk.StringVar(value="High")
        self.identity_threshold_var = tk.DoubleVar(value=0.72)
        self.count_var = tk.IntVar(value=100)
        self.copy_rejected_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.eta_var = tk.StringVar(value="Estimated time remaining: —")
        self.hardware_var = tk.StringVar(value="CPU —   •   RAM —   •   GPU —   •   VRAM —")
        self.provider_var = tk.StringVar(value="AI engine: not started")
        self.result_category_var = tk.StringVar(value="All")
        self.result_identity_var = tk.DoubleVar(value=0.0)
        self.result_quality_var = tk.DoubleVar(value=0.0)
        self.result_count_var = tk.StringVar(value="0 images")
        self.review_primary_var = tk.StringVar(value="Select an image to review.")
        self.review_compare_var = tk.StringVar(value="Choose a comparison image.")
        self.run_started_at: float | None = None
        self.smoothed_analysis_rate: float | None = None
        self.references: list[Path] = []
        self._style()
        self._build()
        self.after(100, self._drain_events)
        self.metrics_thread = threading.Thread(target=self._monitor_hardware, daemon=True)
        self.metrics_thread.start()

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", "#0d141c")
        self.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        self.option_add("*TCombobox*Listbox.selectBackground", "#32d3a2")
        self.option_add("*TCombobox*Listbox.selectForeground", "#07120f")
        style.configure("TFrame", background="#101720")
        style.configure("Card.TFrame", background="#18232f")
        style.configure("TLabel", background="#101720", foreground="#eaf2f8", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#9fb1c1")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 24), foreground="#ffffff")
        style.configure("Card.TLabel", background="#18232f")
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=(12, 8))
        style.configure("Accent.TButton", background="#32d3a2", foreground="#07120f")
        style.map("Accent.TButton", background=[("active", "#5ae3bb"), ("disabled", "#45665c")])
        style.configure("TEntry", fieldbackground="#0d141c", foreground="#ffffff", insertcolor="#ffffff", padding=8)
        style.configure("TCombobox", fieldbackground="#0d141c", background="#253443",
                        foreground="#ffffff", arrowcolor="#ffffff", padding=6)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#0d141c"), ("disabled", "#25313b")],
            foreground=[("readonly", "#ffffff"), ("disabled", "#91a0ac")],
            selectbackground=[("readonly", "#0d141c")],
            selectforeground=[("readonly", "#ffffff")],
            background=[("readonly", "#253443"), ("active", "#31475a")],
            arrowcolor=[("readonly", "#ffffff"), ("disabled", "#91a0ac")],
        )
        style.configure("Horizontal.TProgressbar", troughcolor="#0d141c", background="#32d3a2")
        style.configure("TCheckbutton", background="#18232f", foreground="#eaf2f8")
        style.configure("Treeview", background="#0d141c", fieldbackground="#0d141c",
                        foreground="#eaf2f8", rowheight=25)
        style.configure("Treeview.Heading", background="#253443", foreground="#ffffff")
        style.map("Treeview", background=[("selected", "#32d3a2")],
                  foreground=[("selected", "#07120f")])

    def _build(self) -> None:
        shell = ttk.Frame(self, padding=24)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(header, text="FaceSet Curator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Build the best collective face dataset—not merely the highest individual scores.", style="Muted.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(header, text=__copyright__, style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill="both", expand=True)
        setup = ttk.Frame(self.notebook, padding=22, style="Card.TFrame")
        self.results = ttk.Frame(self.notebook, padding=18, style="Card.TFrame")
        self.notebook.add(setup, text="  Setup & Run  ")
        self.notebook.add(self.results, text="  Results  ")
        self._build_setup(setup)
        self._build_results(self.results)

    def _row(self, parent, row: int, label: str, variable: tk.Variable, command, button: str) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=14, pady=8)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, pady=8)

    def _build_setup(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Input", font=("Segoe UI Semibold", 15), style="Card.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._row(parent, 1, "Source images", self.source_var, self._choose_source, "Choose folder")
        self._row(parent, 2, "Output location", self.output_var, self._choose_output, "Choose folder")
        ttk.Label(parent, text="Target references", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=8)
        ttk.Label(parent, textvariable=self.references_var, style="Card.TLabel").grid(row=3, column=1, sticky="w", padx=14)
        ttk.Button(parent, text="Choose images", command=self._choose_references).grid(row=3, column=2)
        self.reference_strip = ttk.Frame(parent, style="Card.TFrame")
        self.reference_strip.grid(row=4, column=1, columnspan=2, sticky="w", padx=14, pady=(0, 8))
        ttk.Label(self.reference_strip, text="Recommended: 2–5 clear, varied, single-face images",
                  style="Card.TLabel").pack(anchor="w")

        ttk.Separator(parent).grid(row=5, column=0, columnspan=3, sticky="ew", pady=14)
        ttk.Label(parent, text="Analysis", font=("Segoe UI Semibold", 15), style="Card.TLabel").grid(row=6, column=0, columnspan=3, sticky="w")
        controls = ttk.Frame(parent, style="Card.TFrame")
        controls.grid(row=7, column=0, columnspan=3, sticky="ew", pady=12)
        for index in range(8): controls.columnconfigure(index, weight=1)
        ttk.Label(controls, text="Backend", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.backend_var, values=("insightface", "baseline"), state="readonly", width=14).grid(row=1, column=0, sticky="w")
        ttk.Label(controls, text="Device", style="Card.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Combobox(controls, textvariable=self.device_var, values=("auto", "cuda", "cpu"), state="readonly", width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(controls, text="Best images", style="Card.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(controls, from_=1, to=1000, textvariable=self.count_var, width=10).grid(row=1, column=2, sticky="w")
        ttk.Label(controls, text="Profile", style="Card.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Combobox(controls, textvariable=self.profile_var,
                     values=("Balanced", "Quality first", "Diversity first"),
                     state="readonly", width=16).grid(row=1, column=3, sticky="w")
        ttk.Label(controls, text="Duplicates", style="Card.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Combobox(controls, textvariable=self.duplicates_var,
                     values=("Strong", "Normal", "Exact only"),
                     state="readonly", width=14).grid(row=1, column=4, sticky="w")
        ttk.Label(controls, text="Identity", style="Card.TLabel").grid(row=0, column=5, sticky="w")
        identity_control = ttk.Combobox(controls, textvariable=self.identity_var,
                                        values=("High", "Normal", "Custom"), state="readonly", width=14)
        identity_control.grid(row=1, column=5, sticky="w")
        identity_control.bind("<<ComboboxSelected>>", self._identity_mode_changed)
        ttk.Label(controls, text="Threshold", style="Card.TLabel").grid(row=0, column=6, sticky="w")
        self.identity_threshold_control = ttk.Spinbox(
            controls, from_=0.0, to=1.0, increment=0.01,
            textvariable=self.identity_threshold_var, width=10, state="disabled"
        )
        self.identity_threshold_control.grid(row=1, column=6, sticky="w")
        ttk.Checkbutton(controls, text="Copy rejected", variable=self.copy_rejected_var).grid(row=1, column=7, sticky="w")

        ttk.Separator(parent).grid(row=8, column=0, columnspan=3, sticky="ew", pady=16)
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew")
        ttk.Label(parent, textvariable=self.status_var, style="Card.TLabel").grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(parent, textvariable=self.eta_var, style="Card.TLabel").grid(row=11, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(parent, textvariable=self.hardware_var, style="Card.TLabel").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        diagnostic_header = ttk.Frame(parent, style="Card.TFrame")
        diagnostic_header.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Label(diagnostic_header, textvariable=self.provider_var, style="Card.TLabel").pack(side="left")
        self.diagnostics_button = ttk.Button(diagnostic_header, text="Show diagnostics",
                                             command=self._toggle_diagnostics)
        self.diagnostics_button.pack(side="right")
        self.diagnostics_frame = ttk.Frame(parent, style="Card.TFrame")
        self.diagnostics_frame.grid(row=14, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        self.diagnostics_text = tk.Text(
            self.diagnostics_frame, height=7, wrap="word", bg="#0d141c", fg="#d8e4ed",
            insertbackground="#ffffff", relief="flat", padx=8, pady=8, state="disabled"
        )
        diagnostic_scroll = ttk.Scrollbar(self.diagnostics_frame, orient="vertical",
                                          command=self.diagnostics_text.yview)
        self.diagnostics_text.configure(yscrollcommand=diagnostic_scroll.set)
        self.diagnostics_text.pack(side="left", fill="both", expand=True)
        diagnostic_scroll.pack(side="right", fill="y")
        self.diagnostics_frame.grid_remove()
        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=15, column=0, columnspan=3, sticky="e", pady=(18, 0))
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.run_button = ttk.Button(actions, text="Start curation", style="Accent.TButton", command=self._start)
        self.run_button.pack(side="left")

    def _build_results(self, parent) -> None:
        top = ttk.Frame(parent, style="Card.TFrame")
        top.pack(fill="x")
        summary = ttk.Frame(top, style="Card.TFrame")
        summary.pack(side="left")
        self.summary_label = ttk.Label(summary, text="Run FSC to see the selected set.", font=("Segoe UI Semibold", 14), style="Card.TLabel")
        self.summary_label.pack(anchor="w")
        self.identity_summary_label = ttk.Label(summary, text="Identity distribution will appear here.", style="Card.TLabel")
        self.identity_summary_label.pack(anchor="w", pady=(4, 0))
        ttk.Button(top, text="Open report", command=self._open_report).pack(side="right", padx=4)
        ttk.Button(top, text="Open output", command=self._open_output).pack(side="right", padx=4)

        filters = ttk.Frame(parent, style="Card.TFrame")
        filters.pack(fill="x", pady=(14, 8))
        ttk.Label(filters, text="Show", style="Card.TLabel").pack(side="left")
        category = ttk.Combobox(filters, textvariable=self.result_category_var,
                                values=tuple(CATEGORY_FILTERS), state="readonly", width=18)
        category.pack(side="left", padx=(6, 14))
        category.bind("<<ComboboxSelected>>", lambda _event: self._refresh_review_list())
        ttk.Label(filters, text="Min identity", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(filters, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.result_identity_var, width=7).pack(side="left", padx=(6, 14))
        ttk.Label(filters, text="Min quality", style="Card.TLabel").pack(side="left")
        ttk.Spinbox(filters, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.result_quality_var, width=7).pack(side="left", padx=(6, 14))
        ttk.Button(filters, text="Apply filters", command=self._refresh_review_list).pack(side="left")
        ttk.Label(filters, textvariable=self.result_count_var, style="Card.TLabel").pack(side="right")

        split = ttk.Panedwindow(parent, orient="horizontal")
        split.pack(fill="both", expand=True)
        listing = ttk.Frame(split, style="Card.TFrame")
        review = ttk.Frame(split, padding=(14, 0, 0, 0), style="Card.TFrame")
        split.add(listing, weight=3)
        split.add(review, weight=2)
        columns = ("name", "category", "identity", "quality", "value", "group")
        self.review_tree = ttk.Treeview(listing, columns=columns, show="headings", selectmode="browse")
        widths = {"name": 180, "category": 135, "identity": 65, "quality": 65,
                  "value": 65, "group": 90}
        for column in columns:
            self.review_tree.heading(column, text=column.replace("_", " ").title())
            self.review_tree.column(column, width=widths[column], anchor="w")
        review_scroll = ttk.Scrollbar(listing, orient="vertical", command=self.review_tree.yview)
        self.review_tree.configure(yscrollcommand=review_scroll.set)
        self.review_tree.pack(side="left", fill="both", expand=True)
        review_scroll.pack(side="right", fill="y")
        self.review_tree.bind("<<TreeviewSelect>>", self._review_selection_changed)

        previews = ttk.Frame(review, style="Card.TFrame")
        previews.pack(fill="x")
        left_preview = ttk.Frame(previews, style="Card.TFrame")
        right_preview = ttk.Frame(previews, style="Card.TFrame")
        left_preview.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right_preview.pack(side="left", fill="both", expand=True, padx=(5, 0))
        ttk.Label(left_preview, text="Review", style="Card.TLabel").pack(anchor="w")
        self.review_primary_image = ttk.Label(left_preview, text="No image", style="Card.TLabel")
        self.review_primary_image.pack(pady=5)
        ttk.Label(right_preview, text="Compare", style="Card.TLabel").pack(anchor="w")
        self.review_compare_image = ttk.Label(right_preview, text="No comparison", style="Card.TLabel")
        self.review_compare_image.pack(pady=5)
        ttk.Label(review, textvariable=self.review_primary_var, style="Card.TLabel",
                  wraplength=420, justify="left").pack(fill="x", anchor="w", pady=(8, 4))
        ttk.Label(review, textvariable=self.review_compare_var, style="Card.TLabel",
                  wraplength=420, justify="left").pack(fill="x", anchor="w", pady=4)
        review_actions = ttk.Frame(review, style="Card.TFrame")
        review_actions.pack(fill="x", pady=(10, 0))
        self.compare_button = ttk.Button(review_actions, text="Set comparison",
                                         command=self._set_comparison, state="disabled")
        self.compare_button.pack(side="left", padx=(0, 5))
        self.include_button = ttk.Button(review_actions, text="Include / swap in",
                                         command=self._manual_include, state="disabled")
        self.include_button.pack(side="left", padx=5)
        self.exclude_button = ttk.Button(review_actions, text="Exclude / replace",
                                         command=self._manual_exclude, state="disabled")
        self.exclude_button.pack(side="left", padx=5)

    def _choose_source(self) -> None:
        if value := filedialog.askdirectory(title="Choose source image folder"):
            self.source_var.set(value)

    def _choose_output(self) -> None:
        if value := filedialog.askdirectory(title="Choose output location"):
            self.output_var.set(value)

    def _choose_references(self) -> None:
        values = filedialog.askopenfilenames(title="Choose clear target-face references", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if values:
            self.references = [Path(value) for value in values]
            self.references_var.set(f"{len(values)} reference image(s) selected")
            self._show_reference_previews()

    def _show_reference_previews(self) -> None:
        for child in self.reference_strip.winfo_children():
            child.destroy()
        self.reference_photo_refs.clear()
        for path in self.references[:5]:
            card = ttk.Frame(self.reference_strip, padding=(0, 4, 8, 0), style="Card.TFrame")
            card.pack(side="left")
            try:
                with Image.open(path) as opened:
                    preview = ImageOps.fit(opened.convert("RGB"), (48, 48))
                photo = ImageTk.PhotoImage(preview)
                self.reference_photo_refs.append(photo)
                ttk.Label(card, image=photo, style="Card.TLabel").pack()
            except Exception:
                ttk.Label(card, text="Preview unavailable", style="Card.TLabel").pack()
            ttk.Label(card, text=path.name[:12], style="Card.TLabel").pack()
        if len(self.references) > 5:
            ttk.Label(self.reference_strip, text=f"+{len(self.references) - 5} more",
                      style="Card.TLabel").pack(side="left", padx=4)

    def _identity_mode_changed(self, _event=None) -> None:
        mode = self.identity_var.get()
        if mode == "High":
            self.identity_threshold_var.set(0.72)
            self.identity_threshold_control.configure(state="disabled")
        elif mode == "Normal":
            self.identity_threshold_var.set(0.65)
            self.identity_threshold_control.configure(state="disabled")
        else:
            self.identity_threshold_control.configure(state="normal")

    def _validate(self) -> tuple[Path, Path] | None:
        source, output = Path(self.source_var.get()), Path(self.output_var.get())
        if not source.is_dir() or not output.is_dir():
            messagebox.showerror("Folders required", "Choose valid source and output folders.")
            return None
        if source.resolve() == output.resolve() or output.resolve().is_relative_to(source.resolve()):
            messagebox.showerror("Unsafe output", "Output must be outside the source folder so originals remain untouched.")
            return None
        if self.backend_var.get() == "insightface" and not self.references:
            messagebox.showerror("References required", "Choose at least one clear, single-face image of the target person.")
            return None
        if self.backend_var.get() == "insightface" and not 2 <= len(self.references) <= 5:
            if not messagebox.askyesno(
                "Reference recommendation",
                f"You selected {len(self.references)} reference image(s). FSC recommends 2–5 clear, varied, "
                "single-face images for reliable identity enrollment. Continue anyway?",
            ):
                return None
        try:
            threshold = self.identity_threshold_var.get()
        except tk.TclError:
            messagebox.showerror("Invalid identity threshold", "Enter a number between 0.00 and 1.00.")
            return None
        if not 0.0 <= threshold <= 1.0:
            messagebox.showerror("Invalid identity threshold", "Identity threshold must be between 0.00 and 1.00.")
            return None
        return source, output

    def _start(self) -> None:
        validated = self._validate()
        if not validated: return
        source, output = validated
        self.stop_event.clear()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress["value"] = 0
        self.status_var.set("Loading face-analysis models…")
        self.eta_var.set("Estimated time remaining: calculating…")
        self.run_started_at = time.monotonic()
        self.smoothed_analysis_rate = None
        self.provider_var.set("AI engine: checking…")
        self._clear_diagnostics()
        profile = {"Balanced": "balanced", "Quality first": "quality", "Diversity first": "diversity"}[self.profile_var.get()]
        duplicates = {"Strong": "strong", "Normal": "normal", "Exact only": "exact"}[self.duplicates_var.get()]
        identity = self.identity_var.get().lower()
        config = CuratorConfig(target_count=self.count_var.get(), profile=profile,
                               duplicate_strength=duplicates, identity_verification=identity,
                               identity_threshold=self.identity_threshold_var.get(),
                               copy_rejected=self.copy_rejected_var.get())
        self.active_config = config
        try:
            disk = check_disk_space(source, output, config)
        except (InsufficientDiskSpace, OSError, ValueError) as exc:
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.status_var.set("Ready")
            self.eta_var.set("Estimated time remaining: —")
            messagebox.showerror("Output space check", str(exc))
            LOGGER.warning("Disk preflight failed: %s", exc)
            return
        self._append_diagnostic(
            f"Log: {self.log_path}\nDisk preflight: {disk.required_bytes / 1024**3:.2f} GiB required; "
            f"{disk.available_bytes / 1024**3:.2f} GiB available\n"
        )
        LOGGER.info("Starting curation source=%s output=%s images=%s", source, output, disk.image_count)
        backend, device, references = self.backend_var.get(), self.device_var.get(), list(self.references)
        self.worker = threading.Thread(target=self._work, args=(source, output, config, backend, device, references), daemon=True)
        self.worker.start()

    def _work(self, source: Path, output: Path, config: CuratorConfig, backend: str,
              device: str, references: list[Path]) -> None:
        try:
            self.events.put(("progress_details", {"stage": "model_loading", "current": 0, "total": 0,
                                                   "message": "Loading face-analysis models"}))
            analyzer = BaselineAnalyzer() if backend == "baseline" else InsightFaceAnalyzer(
                references, device, batch_size=config.gpu_batch_size, cpu_workers=config.cpu_workers,
                status_callback=lambda details: self.events.put(("progress_details", details)),
                cancelled=self.stop_event.is_set,
            )
            performance = getattr(analyzer, "performance_summary", {})
            if performance:
                self.events.put(("performance", performance))
            else:
                self.events.put(("provider", "Baseline CPU (face identity unavailable)"))
            warning = getattr(analyzer, "enrollment_summary", {}).get("warning")
            if warning:
                self.events.put(("reference_warning", warning))
            run_dir, items = curate(source, output, config, analyzer,
                                    progress_details=lambda details: self.events.put(("progress_details", details)),
                                    cancelled=self.stop_event.is_set)
            self.events.put(("complete", run_dir, items))
        except (CurationCancelled, InterruptedError):
            LOGGER.info("Curation cancelled")
            self.events.put(("cancelled",))
        except Exception as exc:
            LOGGER.exception("Curation failed")
            diagnostics = (f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}\n"
                           f"Requested backend: {backend}\nRequested device: {device}\n")
            self.events.put(("error", friendly_failure(str(exc)), diagnostics))

    def _cancel(self) -> None:
        self.stop_event.set()
        self.status_var.set("Cancelling after the current image…")
        self.eta_var.set("Estimated time remaining: —")

    def _monitor_hardware(self) -> None:
        monitor = SystemMonitor()
        while True:
            self.events.put(("metrics", monitor.sample()))
            time.sleep(1.0)

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, text, current, total = event
                    self.status_var.set(f"{text}  ({current}/{total})")
                    self.progress["maximum"] = max(1, total)
                    self.progress["value"] = current
                    self._update_eta(text, current, total)
                elif event[0] == "complete":
                    self._complete(event[1], event[2])
                elif event[0] == "error":
                    self._failed(event[1], event[2])
                elif event[0] == "cancelled":
                    self.run_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.status_var.set("Cancelled")
                    self.eta_var.set("Estimated time remaining: —")
                elif event[0] == "metrics":
                    self.hardware_var.set(event[1])
                elif event[0] == "reference_warning":
                    messagebox.showwarning("Reference enrollment", event[1])
                elif event[0] == "performance":
                    details = event[1]
                    provider = details.get("provider", "unknown")
                    self.provider_var.set(
                        "AI engine: CUDA active" if provider == "CUDAExecutionProvider" else
                        "AI engine: CPU mode" if provider == "CPUExecutionProvider" else
                        f"AI engine: {provider}"
                    )
                    self._append_diagnostic(
                        f"Provider: {provider}\nGPU batch: {details.get('gpu_batch_size')}\n"
                        f"CPU decode workers: {details.get('cpu_decode_workers')}\n"
                    )
                    self.status_var.set(
                        f"GPU batch {details['gpu_batch_size']} · CPU decode workers "
                        f"{details['cpu_decode_workers']} · warming up models…"
                    )
                elif event[0] == "provider":
                    self.provider_var.set(f"AI engine: {event[1]}")
                elif event[0] == "review_complete":
                    self.items = event[1]
                    summary = event[2]
                    self._finish_review_update(summary)
                elif event[0] == "review_error":
                    self._set_review_buttons()
                    self.run_button.configure(state="normal")
                    messagebox.showerror("Manual review", event[1])
                elif event[0] == "progress_details":
                    self._update_progress_details(event[1])
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _update_progress_details(self, details: dict) -> None:
        labels = {
            "model_loading": "Models",
            "model_download": "Model download",
            "model_verification": "Model verification",
            "preflight": "Storage check",
            "resuming": "Recovery",
            "gpu_warmup": "GPU warm-up",
            "scanning": "Scanning",
            "fingerprinting": "Fingerprinting",
            "cache_lookup": "Cache",
            "analyzing": "Face analysis",
            "identity_filtering": "Identity filtering",
            "duplicate_filtering": "Duplicate filtering",
            "clustering": "Diversity clustering",
            "optimizing": "Set optimization",
            "copying": "Copying",
            "reporting": "Reporting",
            "complete": "Complete",
        }
        stage = details.get("stage", "")
        current, total = details.get("current", 0), details.get("total", 0)
        message = details.get("message", labels.get(stage, stage))
        if stage == "model_download" and total:
            counter = f" ({current / 1024**2:.1f}/{total / 1024**2:.1f} MB)"
        else:
            counter = f" ({current}/{total})" if total else ""
        self.status_var.set(f"{labels.get(stage, stage)} — {message}{counter}")
        self.progress["maximum"] = max(1, total)
        self.progress["value"] = min(current, max(1, total)) if total else 0

        metrics = []
        if stage == "analyzing" and details.get("rate") is not None:
            observed = float(details["rate"])
            self.smoothed_analysis_rate = (observed if self.smoothed_analysis_rate is None else
                                           0.25 * observed + 0.75 * self.smoothed_analysis_rate)
            metrics.append(f"Speed {self.smoothed_analysis_rate:.2f} images/s")
            remaining = max(0, details.get("analysis_total", 0) - details.get("analyzed_new", 0))
            if self.smoothed_analysis_rate and details.get("analyzed_new", 0) >= 3:
                metrics.append(f"ETA {self._format_duration(remaining / self.smoothed_analysis_rate)}")
            else:
                metrics.append("ETA calculating…")
        elif details.get("rate") is not None:
            metrics.append(f"Speed {float(details['rate']):.2f} {details.get('rate_label', 'items/s')}")
            if current >= 3 and details.get("eta_seconds") is not None:
                metrics.append(f"ETA {self._format_duration(float(details['eta_seconds']))}")
            else:
                metrics.append("ETA calculating…")
        elif stage in {"model_loading", "model_download", "model_verification", "gpu_warmup",
                       "scanning", "fingerprinting", "cache_lookup"}:
            metrics.append("ETA calculating…")
        elif stage == "complete":
            metrics.append("Completed")
        else:
            metrics.append("ETA finishing…")
        if "cache_hits" in details:
            metrics.append(f"Cache hits {details['cache_hits']}")
        if "eligible_count" in details:
            metrics.append(f"Eligible {details['eligible_count']}")
        if "selected_count" in details:
            metrics.append(f"Selected {details['selected_count']}")
        self.eta_var.set("   ·   ".join(metrics))

    def _complete(self, run_dir: Path, items: list[ImageAnalysis]) -> None:
        self.run_dir = run_dir
        self.items = items
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        selected = sorted((item for item in items if item.category == "selected"), key=lambda item: -item.dataset_value)
        self.status_var.set(f"Complete — selected {len(selected)} from {len(items)} images")
        self.eta_var.set("Completed")
        self.summary_label.configure(text=f"Best set: {len(selected)} selected from {len(items)} analyzed")
        threshold = float(self.active_config.identity_threshold) if self.active_config else float(self.identity_threshold_var.get())
        identity = identity_statistics(items, threshold)
        if identity["minimum"] is None:
            identity_text = "Identity: no faces were scored."
        else:
            identity_text = (f"Identity: {identity['minimum']:.3f}–{identity['maximum']:.3f}, "
                             f"median {identity['median']:.3f}; {identity['passed_threshold']} passed "
                             f"threshold {identity['threshold']:.2f}")
        self.identity_summary_label.configure(text=identity_text)
        self.result_category_var.set("Selected")
        self._refresh_review_list()
        self.notebook.select(self.results)
        if not selected:
            counts = {category: sum(item.category == category for item in items) for category in {
                "rejected/identity", "rejected/multiple_faces", "rejected/no_face_or_invalid", "rejected/quality"
            }}
            messagebox.showwarning(
                "No images qualified",
                f"FSC selected 0 images. Identity rejected {counts['rejected/identity']}; "
                f"multiple faces {counts['rejected/multiple_faces']}; no face/invalid "
                f"{counts['rejected/no_face_or_invalid']}; quality {counts['rejected/quality']}.\n\n"
                f"{identity_text}\n\nReview the reference images or choose a less strict identity threshold.",
            )

    def _refresh_review_list(self) -> None:
        try:
            minimum_identity = float(self.result_identity_var.get())
            minimum_quality = float(self.result_quality_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid filter", "Identity and quality filters must be numbers from 0 to 1.")
            return
        visible = filter_items(self.items, self.result_category_var.get(), minimum_identity, minimum_quality)
        self.review_tree.delete(*self.review_tree.get_children())
        item_indices = {id(item): index for index, item in enumerate(self.items)}
        for item in visible:
            index = item_indices[id(item)]
            self.review_tree.insert("", "end", iid=str(index), values=(
                item.source.name, item.category.replace("rejected/", ""),
                f"{item.identity_score:.3f}", f"{item.quality_score:.3f}",
                f"{item.dataset_value:.1f}", item.duplicate_group or "—",
            ))
        self.result_count_var.set(f"{len(visible)} of {len(self.items)} images")

    def _review_selection_changed(self, _event=None) -> None:
        selected = self.review_tree.selection()
        if not selected:
            return
        index = int(selected[0])
        if index >= len(self.items):
            return
        self.review_current = self.items[index]
        peers = duplicate_peers(self.review_current, self.items)
        if peers:
            self.review_compare = peers[0]
        self._show_review_pair()

    def _set_comparison(self) -> None:
        if self.review_current:
            self.review_compare = self.review_current
            self.review_compare_var.set(self._review_description(self.review_compare))
            self._show_review_pair()

    def _show_review_pair(self) -> None:
        self.review_photo_refs.clear()
        self._set_review_preview(self.review_primary_image, self.review_current)
        self._set_review_preview(self.review_compare_image, self.review_compare)
        self.review_primary_var.set(self._review_description(self.review_current))
        self.review_compare_var.set(self._review_description(self.review_compare))
        self._set_review_buttons()

    def _set_review_preview(self, label: ttk.Label, item: ImageAnalysis | None) -> None:
        if item is None:
            label.configure(image="", text="No image")
            return
        try:
            with Image.open(item.source) as opened:
                preview = ImageOps.fit(opened.convert("RGB"), (190, 190))
            photo = ImageTk.PhotoImage(preview)
            self.review_photo_refs.append(photo)
            label.configure(image=photo, text="")
        except Exception:
            label.configure(image="", text="Preview unavailable")

    @staticmethod
    def _review_description(item: ImageAnalysis | None) -> str:
        if item is None:
            return "Choose an image for side-by-side comparison."
        return (f"{item.source.name}\nCategory: {item.category}\nIdentity {item.identity_score:.3f} · "
                f"Quality {item.quality_score:.3f} · Dataset value {item.dataset_value:.1f}\n"
                f"Expression {item.expression} · Blur {item.blur_score:.2f} · "
                f"Occlusion {item.occlusion_score:.2f} · Compression {item.compression_artifact_score:.2f}\n"
                f"Why: {why_rejected(item)}")

    def _set_review_buttons(self, busy: bool = False) -> None:
        current = self.review_current
        self.compare_button.configure(state="disabled" if busy or current is None else "normal")
        include_allowed = current is not None and current.category in {"rejected/redundant", "rejected/manual"}
        self.include_button.configure(state="normal" if include_allowed and not busy else "disabled")
        self.exclude_button.configure(state="normal" if current and current.category == "selected" and not busy else "disabled")

    def _manual_include(self) -> None:
        if not self.review_current:
            return
        exclude = self.review_compare if self.review_compare and self.review_compare.category == "selected" else None
        detail = (f" and replace {exclude.source.name}" if exclude else
                  "; FSC will automatically remove the least-useful selected image")
        if messagebox.askyesno(
            "Include image",
            f"Include {self.review_current.source.name}{detail}?\n\n"
            "All collective dataset values will be recalculated.",
        ):
            self._run_manual_review(self.review_current.path, exclude.path if exclude else None)

    def _manual_exclude(self) -> None:
        if not self.review_current:
            return
        if messagebox.askyesno(
            "Exclude image",
            f"Exclude {self.review_current.source.name}? FSC will choose the best replacement and "
            "recalculate all collective dataset values.",
        ):
            self._run_manual_review(None, self.review_current.path)

    def _run_manual_review(self, include_path: str | None, exclude_path: str | None) -> None:
        if not self.active_config or not self.run_dir:
            return
        self._set_review_buttons(busy=True)
        self.run_button.configure(state="disabled")
        self.status_var.set("Applying manual review and recalculating the selected set…")
        source_items = copy.deepcopy(self.items)
        previous = {item.path: item.category for item in source_items}
        config, run_dir = self.active_config, self.run_dir

        def work() -> None:
            try:
                summary = apply_manual_selection(source_items, config, include_path, exclude_path)
                sync_review_output(run_dir, source_items, config, previous, summary)
                self.events.put(("review_complete", source_items, summary))
            except Exception as exc:
                self.events.put(("review_error", f"The review change could not be applied: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _finish_review_update(self, summary: dict) -> None:
        selected = [item for item in self.items if item.category == "selected"]
        self.summary_label.configure(text=f"Best set: {len(selected)} selected from {len(self.items)} analyzed")
        self.status_var.set(f"Manual review saved — {len(selected)} selected; reports refreshed")
        self.run_button.configure(state="normal")
        self.review_current = None
        self.review_compare = None
        self.review_primary_var.set("Select an image to review.")
        self.review_compare_var.set("Choose a comparison image.")
        self.review_primary_image.configure(image="", text="No image")
        self.review_compare_image.configure(image="", text="No comparison")
        self.review_photo_refs.clear()
        self._refresh_review_list()
        self._set_review_buttons()
        messagebox.showinfo(
            "Manual review saved",
            f"Selected set recalculated. Added {len(summary['added'])}; removed {len(summary['removed'])}. "
            "Output copies and JSON, CSV, and HTML reports were updated.",
        )

    def _toggle_diagnostics(self) -> None:
        if self.diagnostics_frame.winfo_viewable():
            self.diagnostics_frame.grid_remove()
            self.diagnostics_button.configure(text="Show diagnostics")
        else:
            self.diagnostics_frame.grid()
            self.diagnostics_button.configure(text="Hide diagnostics")

    def _clear_diagnostics(self) -> None:
        self.diagnostics_text.configure(state="normal")
        self.diagnostics_text.delete("1.0", "end")
        self.diagnostics_text.configure(state="disabled")

    def _append_diagnostic(self, message: str) -> None:
        self.diagnostics_text.configure(state="normal")
        self.diagnostics_text.insert("end", message.rstrip() + "\n\n")
        self.diagnostics_text.see("end")
        self.diagnostics_text.configure(state="disabled")

    def _failed(self, message: str, diagnostics: str) -> None:
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Stopped")
        self.eta_var.set("Estimated time remaining: —")
        self.provider_var.set("AI engine: startup failed")
        self._append_diagnostic(diagnostics)
        if not self.diagnostics_frame.winfo_viewable():
            self._toggle_diagnostics()
        messagebox.showerror("FaceSet Curator", f"{message}\n\nTechnical details are shown in Diagnostics.")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _update_eta(self, stage: str, current: int, total: int) -> None:
        if stage in {"Optimizing the best collective set", "Complete"} or (total and current >= total):
            self.eta_var.set("Estimated time remaining: finishing…" if stage != "Complete" else "Completed")
            return
        if not self.run_started_at or current < 3 or total <= 0:
            self.eta_var.set("Estimated time remaining: calculating…")
            return
        elapsed = time.monotonic() - self.run_started_at
        remaining = elapsed / current * (total - current)
        self.eta_var.set(f"Estimated time remaining: {self._format_duration(remaining)}")

    def _open_output(self) -> None:
        if self.run_dir and self.run_dir.exists(): os.startfile(self.run_dir)

    def _open_report(self) -> None:
        if self.run_dir and (self.run_dir / "report.html").exists(): os.startfile(self.run_dir / "report.html")


def main() -> None:
    FaceSetCuratorApp().mainloop()


if __name__ == "__main__":
    main()
