#!/usr/bin/env python3
"""
Copilot Expert Studio
A PySide6 desktop GUI that:
- uses GitHub Copilot SDK for chat
- shells out to GitHub CLI / Copilot CLI for auth verification
- lets the user create/load expert Markdown files
- can materialize expert Markdown as .agent.md or instruction files
- supports copy/paste friendly code editing panes
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QAction, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qasync import QEventLoop, asyncSlot

# GitHub Copilot Python SDK
# Package: `pip install github-copilot-sdk` — imported as `copilot`.
# Docs: https://github.com/github/copilot-sdk/tree/main/python
from copilot import CopilotClient
from copilot.generated.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionIdleData,
)
from copilot.session import PermissionRequestResult


APP_TITLE = "Copilot Expert Studio"
DEFAULT_MODELS = [
    "gpt-5",
    "gpt-4.1",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    "auto",
]

HOME = Path.home()
APP_DIR = HOME / ".copilot-expert-studio"
APP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_AGENTS_DIR = HOME / ".copilot" / "agents"
LOCAL_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_INSTRUCTIONS_FILE = HOME / ".copilot" / "copilot-instructions.md"


@dataclass
class ExpertProfile:
    name: str
    description: str
    markdown: str

    @staticmethod
    def _yaml_escape(value: str) -> str:
        """Escape a string for safe inclusion in a double-quoted YAML value."""
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    def to_agent_markdown(self, model: str = "gpt-5") -> str:
        safe_name = self._yaml_escape(self.name.strip() or "expert-agent")
        safe_desc = self._yaml_escape(self.description.strip() or "Specialist coding agent")
        safe_model = self._yaml_escape(model.strip() or "gpt-5")
        return f"""---
name: "{safe_name}"
description: "{safe_desc}"
model: "{safe_model}"
user-invocable: true
disable-model-invocation: false
---

{self.markdown.rstrip()}
"""

    def to_instruction_markdown(self) -> str:
        header = f"# {self.name}\n\n" if self.name.strip() else ""
        desc = f"> {self.description.strip()}\n\n" if self.description.strip() else ""
        return f"{header}{desc}{self.markdown.rstrip()}\n"


class ChatWorker(QObject):
    chunk = Signal(str)
    status = Signal(str)
    done = Signal()
    error = Signal(str)

    def __init__(self, prompt: str, system_message: str, model: str, cwd: str):
        super().__init__()
        self.prompt = prompt
        self.system_message = system_message
        self.model = model if model != "auto" else "gpt-5"
        self.cwd = cwd

    @staticmethod
    def _permission_handler(request, invocation) -> PermissionRequestResult:
        """Allow reads and custom tools; deny shell commands and file writes."""
        kind = getattr(request, "kind", None)
        kind_value = getattr(kind, "value", str(kind)) if kind else ""
        if kind_value in ("shell", "write"):
            return PermissionRequestResult(kind="denied-interactively-by-user")
        return PermissionRequestResult(kind="approved")

    async def run(self) -> None:
        client = None
        try:
            self.status.emit("Starting Copilot SDK session...")
            client = CopilotClient()
            await client.start()

            session_kwargs: dict = {
                "model": self.model,
                "on_permission_request": self._permission_handler,
                "streaming": True,
            }
            if self.system_message.strip():
                session_kwargs["system_message"] = {
                    "mode": "replace",
                    "content": self.system_message,
                }

            session = await client.create_session(**session_kwargs)

            idle_event = asyncio.Event()

            def on_event(event):
                try:
                    match event.data:
                        case AssistantMessageDeltaData() as data:
                            delta = data.delta_content or ""
                            if delta:
                                self.chunk.emit(delta)
                        case AssistantMessageData() as data:
                            pass  # final message; deltas already streamed
                        case SessionIdleData():
                            idle_event.set()
                except Exception as exc:
                    self.error.emit(f"Event handling error: {exc}")

            session.on(on_event)

            self.status.emit("Sending prompt to Copilot...")
            await session.send(self.prompt)
            try:
                await asyncio.wait_for(idle_event.wait(), timeout=180)
            except asyncio.TimeoutError:
                self.status.emit("Response timeout reached; stopping wait.")

            await session.disconnect()

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if client is not None:
                try:
                    await client.stop()
                except Exception:
                    pass
            self.done.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 980)
        self.current_worker: Optional[ChatWorker] = None
        self.current_task: Optional[asyncio.Task] = None

        self._build_ui()
        self._apply_styles()
        self._load_saved_state()
        self.refresh_auth_status()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        toolbar = self._build_top_bar()
        outer.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        self.setStatusBar(QStatusBar(self))

    def _build_top_bar(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(APP_TITLE)
        title.setObjectName("TitleLabel")
        subtitle = QLabel("GitHub Copilot SDK + CLI workbench for expert agents and code-heavy prompting.")
        subtitle.setObjectName("SubtitleLabel")

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.auth_label = QLabel("Auth: unknown")
        self.auth_label.setObjectName("AuthPill")

        refresh_btn = QPushButton("Refresh auth")
        refresh_btn.clicked.connect(self.refresh_auth_status)

        login_btn = QPushButton("Login via gh")
        login_btn.clicked.connect(self.login_via_gh)

        install_btn = QPushButton("Install help")
        install_btn.clicked.connect(self.show_install_help)

        layout.addLayout(title_box, 1)
        layout.addWidget(self.auth_label)
        layout.addWidget(refresh_btn)
        layout.addWidget(login_btn)
        layout.addWidget(install_btn)
        return w

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_runtime_box())
        layout.addWidget(self._build_profile_box(), 1)
        layout.addWidget(self._build_file_box())
        return panel

    def _build_runtime_box(self) -> QWidget:
        box = QGroupBox("Runtime")
        form = QFormLayout(box)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_MODELS)

        self.cwd_edit = QLineEdit(str(Path.cwd()))
        cwd_browse = QPushButton("Browse")
        cwd_browse.clicked.connect(self.choose_cwd)
        cwd_row = QHBoxLayout()
        cwd_row.addWidget(self.cwd_edit, 1)
        cwd_row.addWidget(cwd_browse)

        self.materialize_agent_checkbox = QCheckBox("Write expert as ~/.copilot/agents/<name>.agent.md for CLI use")
        self.materialize_instructions_checkbox = QCheckBox("Write merged instructions to ~/.copilot/copilot-instructions.md")

        form.addRow("Model", self.model_combo)
        form.addRow("Working directory", self._wrap_layout(cwd_row))
        form.addRow("", self.materialize_agent_checkbox)
        form.addRow("", self.materialize_instructions_checkbox)
        return box

    def _build_profile_box(self) -> QWidget:
        box = QGroupBox("Expert profile")
        layout = QVBoxLayout(box)

        form = QFormLayout()
        self.profile_name = QLineEdit("nodejs-gds-expert")
        self.profile_description = QLineEdit("Senior Node.js expert aligned to UK government/GDS engineering standards.")
        form.addRow("Name", self.profile_name)
        form.addRow("Description", self.profile_description)
        layout.addLayout(form)

        self.profile_tabs = QTabWidget()

        self.markdown_editor = QPlainTextEdit()
        self.markdown_editor.setPlaceholderText(
            "Paste your expert markdown here.\n\n"
            "This can include coding standards, framework guidance, architecture rules, review heuristics, "
            "team conventions, and examples."
        )

        self.system_prompt_preview = QPlainTextEdit()
        self.system_prompt_preview.setReadOnly(True)

        self.profile_tabs.addTab(self.markdown_editor, "Expert Markdown")
        self.profile_tabs.addTab(self.system_prompt_preview, "Effective System Prompt")
        layout.addWidget(self.profile_tabs, 1)

        button_row = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_profile)

        load_btn = QPushButton("Load .md")
        load_btn.clicked.connect(self.load_profile_markdown)

        save_btn = QPushButton("Save .md")
        save_btn.clicked.connect(self.save_profile_markdown)

        materialize_btn = QPushButton("Write agent/instructions")
        materialize_btn.clicked.connect(self.materialize_profile)

        preview_btn = QPushButton("Refresh preview")
        preview_btn.clicked.connect(self.refresh_prompt_preview)

        button_row.addWidget(new_btn)
        button_row.addWidget(load_btn)
        button_row.addWidget(save_btn)
        button_row.addWidget(materialize_btn)
        button_row.addWidget(preview_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        return box

    def _build_file_box(self) -> QWidget:
        box = QGroupBox("Clipboard and code")
        layout = QVBoxLayout(box)

        self.code_context = QPlainTextEdit()
        self.code_context.setPlaceholderText(
            "Paste code, logs, stack traces, or config here.\n\n"
            "This box is appended to the chat prompt so the expert agent has grounded context."
        )
        self.code_context.setTabStopDistance(32)

        row = QHBoxLayout()
        paste_btn = QPushButton("Paste")
        paste_btn.clicked.connect(self.code_context.paste)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self.code_context.copy)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.code_context.clear)
        load_btn = QPushButton("Load file")
        load_btn.clicked.connect(self.load_code_file)

        row.addWidget(paste_btn)
        row.addWidget(copy_btn)
        row.addWidget(clear_btn)
        row.addWidget(load_btn)
        row.addStretch(1)

        layout.addWidget(self.code_context, 1)
        layout.addLayout(row)
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        prompt_box = QGroupBox("Chat")
        prompt_layout = QVBoxLayout(prompt_box)

        self.user_prompt = QPlainTextEdit()
        self.user_prompt.setPlaceholderText("Ask the expert agent something concrete.")
        self.user_prompt.setMaximumHeight(180)

        prompt_buttons = QHBoxLayout()
        send_btn = QPushButton("Send to Copilot")
        send_btn.clicked.connect(self.send_prompt)

        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_request)

        copy_prompt_btn = QPushButton("Copy merged prompt")
        copy_prompt_btn.clicked.connect(self.copy_merged_prompt)

        prompt_buttons.addWidget(send_btn)
        prompt_buttons.addWidget(stop_btn)
        prompt_buttons.addWidget(copy_prompt_btn)
        prompt_buttons.addStretch(1)

        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.response_view.setAcceptRichText(False)

        response_buttons = QHBoxLayout()
        copy_resp_btn = QPushButton("Copy response")
        copy_resp_btn.clicked.connect(self.copy_response)
        clear_resp_btn = QPushButton("Clear response")
        clear_resp_btn.clicked.connect(self.response_view.clear)
        response_buttons.addWidget(copy_resp_btn)
        response_buttons.addWidget(clear_resp_btn)
        response_buttons.addStretch(1)

        prompt_layout.addWidget(self.user_prompt)
        prompt_layout.addLayout(prompt_buttons)
        prompt_layout.addWidget(self.response_view, 1)
        prompt_layout.addLayout(response_buttons)

        layout.addWidget(prompt_box, 1)
        return panel

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _apply_styles(self):
        self.setStyleSheet("""
        QWidget {
            background: #0f172a;
            color: #e5e7eb;
            font-size: 13px;
        }
        QMainWindow, QGroupBox {
            background: #0f172a;
        }
        QGroupBox {
            border: 1px solid #334155;
            border-radius: 14px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: 600;
            background: #111827;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 4px 0 4px;
            color: #cbd5e1;
        }
        QPushButton {
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 10px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            background: #273449;
        }
        QPushButton:pressed {
            background: #334155;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QTabWidget::pane {
            background: #020617;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 6px;
        }
        QComboBox QAbstractItemView {
            background: #020617;
            color: #e5e7eb;
            selection-background-color: #1d4ed8;
        }
        QLabel#TitleLabel {
            font-size: 24px;
            font-weight: 700;
            color: #f8fafc;
        }
        QLabel#SubtitleLabel {
            color: #94a3b8;
        }
        QLabel#AuthPill {
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 8px 12px;
            color: #cbd5e1;
            font-weight: 600;
        }
        QStatusBar {
            background: #111827;
            color: #cbd5e1;
        }
        """)

        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(11)

        for editor in [self.markdown_editor, self.system_prompt_preview, self.code_context, self.user_prompt, self.response_view]:
            editor.setFont(mono)

    def _load_saved_state(self):
        state_file = APP_DIR / "state.json"
        if not state_file.exists():
            self.refresh_prompt_preview()
            return

        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            self.profile_name.setText(data.get("profile_name", self.profile_name.text()))
            self.profile_description.setText(data.get("profile_description", self.profile_description.text()))
            self.markdown_editor.setPlainText(data.get("markdown", self.default_markdown()))
            self.code_context.setPlainText(data.get("code_context", ""))
            self.user_prompt.setPlainText(data.get("user_prompt", ""))
            self.cwd_edit.setText(data.get("cwd", str(Path.cwd())))
            saved_model = data.get("model", "gpt-5")
            index = self.model_combo.findText(saved_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.setEditText(saved_model)
        except Exception:
            self.markdown_editor.setPlainText(self.default_markdown())

        self.refresh_prompt_preview()

    def closeEvent(self, event):
        self._save_state()
        super().closeEvent(event)

    def _save_state(self):
        data = {
            "profile_name": self.profile_name.text(),
            "profile_description": self.profile_description.text(),
            "markdown": self.markdown_editor.toPlainText(),
            "code_context": self.code_context.toPlainText(),
            "user_prompt": self.user_prompt.toPlainText(),
            "cwd": self.cwd_edit.text(),
            "model": self.model_combo.currentText(),
        }
        (APP_DIR / "state.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def default_markdown(self) -> str:
        return """# Role

You are a senior Node.js engineer and technical reviewer.

# Priorities

- Prefer maintainable, testable, production-grade solutions.
- Follow UK government / GDS style expectations where applicable.
- Be explicit about tradeoffs, security, performance, and operability.
- Avoid magical abstractions when a simpler implementation is clearer.

# Node.js guidance

- Use current stable Node.js patterns.
- Prefer clear module boundaries and typed interfaces where possible.
- Keep async error handling explicit.
- Validate external inputs.
- Explain runtime, deployment, and observability implications.

# Response format

- Start with the concrete answer or implementation.
- Call out risks and edge cases.
- For code changes, explain why each change exists.
"""

    def build_profile(self) -> ExpertProfile:
        return ExpertProfile(
            name=self.profile_name.text().strip(),
            description=self.profile_description.text().strip(),
            markdown=self.markdown_editor.toPlainText(),
        )

    def build_system_prompt(self) -> str:
        profile = self.build_profile()
        parts = [
            f"You are acting as the expert profile '{profile.name}'.",
            f"Profile description: {profile.description}" if profile.description else "",
            "",
            "Apply the following expert markdown exactly as durable guidance:",
            profile.markdown.strip(),
        ]
        return "\n".join(p for p in parts if p)

    def build_chat_prompt(self) -> str:
        user_prompt = self.user_prompt.toPlainText().strip()
        code_ctx = self.code_context.toPlainText().strip()
        if code_ctx:
            return f"{user_prompt}\n\n## Code / logs / config context\n\n```text\n{code_ctx}\n```"
        return user_prompt

    def refresh_prompt_preview(self):
        self.system_prompt_preview.setPlainText(self.build_system_prompt())

    def set_status(self, text: str):
        self.statusBar().showMessage(text)

    def append_response(self, text: str):
        cursor = self.response_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.response_view.setTextCursor(cursor)
        self.response_view.ensureCursorVisible()

    def refresh_auth_status(self):
        gh_ok = self._command_ok(["gh", "auth", "status"])
        copilot_ok = self._command_ok(["copilot", "--version"])
        if gh_ok and copilot_ok:
            self.auth_label.setText("Auth/runtime: gh OK, copilot CLI present")
        elif gh_ok:
            self.auth_label.setText("Auth/runtime: gh OK, copilot CLI missing")
        elif copilot_ok:
            self.auth_label.setText("Auth/runtime: copilot CLI present, gh not authenticated")
        else:
            self.auth_label.setText("Auth/runtime: not ready")
        self.set_status("Authentication/runtime check complete.")

    def _command_ok(self, cmd: List[str]) -> bool:
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return True
        except Exception:
            return False

    def login_via_gh(self):
        if shutil.which("gh") is None:
            QMessageBox.warning(self, "GitHub CLI missing", "Install GitHub CLI first, then run login again.")
            return
        try:
            subprocess.Popen(["gh", "auth", "login", "-w"])
            self.set_status("Opened GitHub CLI login flow in the browser.")
        except Exception as exc:
            QMessageBox.critical(self, "Login failed", str(exc))

    def show_install_help(self):
        msg = (
            "Required local dependencies:\n\n"
            "1) Python 3.11+\n"
            "2) GitHub CLI (`gh`) authenticated with your GitHub account\n"
            "3) GitHub Copilot CLI installed and working\n"
            "4) Python packages:\n"
            "   pip install PySide6 qasync github-copilot-sdk\n\n"
            "Suggested checks:\n"
            "   gh auth status\n"
            "   copilot --version\n"
        )
        QMessageBox.information(self, "Install help", msg)

    def choose_cwd(self):
        path = QFileDialog.getExistingDirectory(self, "Select working directory", self.cwd_edit.text())
        if path:
            self.cwd_edit.setText(path)

    def new_profile(self):
        self.profile_name.setText("expert-agent")
        self.profile_description.clear()
        self.markdown_editor.setPlainText(self.default_markdown())
        self.refresh_prompt_preview()

    def load_profile_markdown(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load markdown", str(Path.cwd()), "Markdown (*.md)")
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8")
        self.markdown_editor.setPlainText(text)
        inferred = Path(path).stem.replace(".agent", "")
        if inferred:
            self.profile_name.setText(inferred)
        self.refresh_prompt_preview()
        self.set_status(f"Loaded markdown: {path}")

    def save_profile_markdown(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save markdown", str(Path.cwd() / f"{self.profile_name.text() or 'expert'}.md"), "Markdown (*.md)")
        if not path:
            return
        Path(path).write_text(self.markdown_editor.toPlainText(), encoding="utf-8")
        self.set_status(f"Saved markdown: {path}")

    def load_code_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load code or text file", str(Path.cwd()), "Text files (*.py *.js *.ts *.tsx *.jsx *.json *.md *.txt *.log *.yaml *.yml *.cs *.go *.java *.sql);;All files (*)")
        if not path:
            return
        try:
            self.code_context.setPlainText(Path(path).read_text(encoding="utf-8"))
            self.set_status(f"Loaded context file: {path}")
        except UnicodeDecodeError:
            QMessageBox.warning(self, "Unsupported file", "This file does not look like UTF-8 text.")

    def materialize_profile(self):
        self.refresh_prompt_preview()
        profile = self.build_profile()
        written = []

        if self.materialize_agent_checkbox.isChecked():
            safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in (profile.name or "expert-agent").lower()).strip("-_") or "expert-agent"
            agent_path = LOCAL_AGENTS_DIR / f"{safe_name}.agent.md"
            agent_path.write_text(profile.to_agent_markdown(self.model_combo.currentText()), encoding="utf-8")
            written.append(str(agent_path))

        if self.materialize_instructions_checkbox.isChecked():
            LOCAL_INSTRUCTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_INSTRUCTIONS_FILE.write_text(profile.to_instruction_markdown(), encoding="utf-8")
            written.append(str(LOCAL_INSTRUCTIONS_FILE))

        if written:
            QMessageBox.information(self, "Profile written", "Wrote:\n\n" + "\n".join(written))
            self.set_status("Expert files materialized for local Copilot CLI use.")
        else:
            QMessageBox.information(self, "Nothing written", "Enable at least one write target first.")

    def copy_response(self):
        QApplication.clipboard().setText(self.response_view.toPlainText())
        self.set_status("Response copied to clipboard.")

    def copy_merged_prompt(self):
        merged = (
            "=== SYSTEM PROMPT ===\n"
            + self.build_system_prompt()
            + "\n\n=== USER PROMPT ===\n"
            + self.build_chat_prompt()
        )
        QApplication.clipboard().setText(merged)
        self.set_status("Merged prompt copied to clipboard.")

    def stop_request(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.set_status("Cancellation requested.")
        else:
            self.set_status("No active request.")

    @asyncSlot()
    async def send_prompt(self):
        self.refresh_prompt_preview()

        if not self.user_prompt.toPlainText().strip():
            QMessageBox.warning(self, "Missing prompt", "Enter a user prompt first.")
            return

        if shutil.which("copilot") is None:
            QMessageBox.critical(self, "Copilot CLI missing", "The Python SDK depends on the Copilot CLI runtime/auth path. Install Copilot CLI first.")
            return

        if self.materialize_agent_checkbox.isChecked() or self.materialize_instructions_checkbox.isChecked():
            self.materialize_profile()

        self.response_view.clear()
        self.append_response("")
        self.set_status("Preparing request...")

        worker = ChatWorker(
            prompt=self.build_chat_prompt(),
            system_message=self.build_system_prompt(),
            model=self.model_combo.currentText().strip(),
            cwd=self.cwd_edit.text().strip(),
        )
        self.current_worker = worker

        worker.chunk.connect(self.append_response)
        worker.status.connect(self.set_status)
        worker.error.connect(lambda msg: QMessageBox.critical(self, "Copilot error", msg))
        worker.done.connect(lambda: self.set_status("Request finished."))

        self.current_task = asyncio.create_task(worker.run())
        try:
            await self.current_task
        except asyncio.CancelledError:
            self.set_status("Request cancelled.")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    win = MainWindow()
    win.show()

    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
