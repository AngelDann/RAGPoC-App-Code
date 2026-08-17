from __future__ import annotations

import hashlib
import uuid

from django.db import models
from django.utils import timezone


def default_uuid() -> str:
    return str(uuid.uuid4())


def default_tiptap_json() -> dict:
    return {"type": "doc", "content": []}


def calculate_content_hash(title: str, plain_text: str) -> str:
    return hashlib.sha256(f"{title}\n{plain_text}".encode()).hexdigest()


class Workspace(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "workspaces"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return self.name


class Notebook(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, related_name="notebooks")
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    position = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "notebooks"
        ordering = ["position", "name"]

    def __str__(self) -> str:
        return f"{self.workspace.name} / {self.name}"


class Page(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    notebook = models.ForeignKey(Notebook, on_delete=models.CASCADE, related_name="pages")
    title = models.CharField(max_length=300, default="Untitled")
    content_json = models.JSONField(default=default_tiptap_json)
    plain_text = models.TextField(blank=True, default="")
    content_hash = models.CharField(max_length=64, blank=True, default="")
    position = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "pages"
        ordering = ["position", "title"]

    def save(self, *args, **kwargs):
        if not self.content_hash:
            self.content_hash = calculate_content_hash(self.title, self.plain_text)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.notebook.name} / {self.title}"


class Document(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    source_path = models.CharField(max_length=1000, unique=True)
    original_filename = models.CharField(max_length=500)
    media_type = models.CharField(max_length=50)
    content_hash = models.CharField(max_length=64, db_index=True)
    byte_size = models.BigIntegerField(default=0)
    status = models.CharField(max_length=50, default="indexed")
    created_at = models.DateTimeField(default=timezone.now)
    indexed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "documents"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.original_filename


class NotebookDocument(models.Model):
    notebook = models.ForeignKey(Notebook, on_delete=models.CASCADE, related_name="notebook_documents", db_column="notebook_id")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="notebook_documents", db_column="document_id")
    attached_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "notebook_documents"
        unique_together = ("notebook", "document")
        ordering = ["-attached_at"]

    def __str__(self) -> str:
        return f"{self.notebook.name} ↔ {self.document.original_filename}"


class ChatThread(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, related_name="chat_threads", null=True, blank=True)
    notebook = models.ForeignKey(Notebook, on_delete=models.CASCADE, related_name="chat_threads", null=True, blank=True)
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name="chat_threads", null=True, blank=True)
    scope = models.CharField(max_length=20, default="workspace")
    title = models.CharField(max_length=250, default="Nueva conversación")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "chat_threads"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title


class ChatMessage(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    thread = models.ForeignKey(ChatThread, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20)  # user, assistant, system
    content = models.TextField()
    sources_json = models.JSONField(default=list, blank=True)
    attachments_json = models.JSONField(default=list, blank=True)  # attached files in prompt
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "chat_messages"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:40]}"


class NotebookArtifact(models.Model):
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    notebook = models.ForeignKey(Notebook, on_delete=models.CASCADE, related_name="artifacts", db_column="notebook_id")
    artifact_type = models.CharField(max_length=50)  # diagram, quiz, study_guide, flashcards, summary
    title = models.CharField(max_length=250)
    content = models.TextField()
    metadata_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "notebook_artifacts"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.artifact_type}] {self.title} ({self.notebook.name})"


class AgentSkill(models.Model):
    """Procedural Memory: Habilidades operativas y flujos de trabajo reutilizables (estilo Hermes)."""
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    name = models.CharField(max_length=100, unique=True)
    description = models.CharField(max_length=300)
    instructions = models.TextField()  # Pasos, directivas y comandos
    category = models.CharField(max_length=50, default="general")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "agent_skills"
        ordering = ["name"]

    def __str__(self) -> str:
        return f"Skill: {self.name} ({self.category})"


class AppConfig(models.Model):
    """Singleton row holding BYOK overrides and agent settings set from the Ajustes UI."""
    id = models.PositiveSmallIntegerField(primary_key=True, default=1, editable=False)
    openrouter_api_key = models.CharField(max_length=200, blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "app_config"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "AppConfig":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self) -> str:
        return "Configuración de la aplicación"


class AgentMemory(models.Model):
    """Declarative Memory: Hechos persistentes, preferencias del usuario y lecciones aprendidas."""
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    category = models.CharField(max_length=50, default="user_preference")  # user_preference, domain_fact, procedure
    content = models.TextField()
    source = models.CharField(max_length=100, default="chat_auto")  # chat_auto, user_explicit, system
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "agent_memories"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"Memory [{self.category}]: {self.content[:50]}"


class ApiUsageLog(models.Model):
    """One row per outbound call to the LLM/embedding provider, for usage monitoring."""
    id = models.CharField(primary_key=True, max_length=64, default=default_uuid)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    category = models.CharField(max_length=30, db_index=True)  # chat, embedding, inline_ai, artifact
    action = models.CharField(max_length=100, blank=True, default="")
    provider = models.CharField(max_length=50, default="openrouter")
    model = models.CharField(max_length=150, blank=True, default="")
    status = models.CharField(max_length=20, default="success")  # success, error
    error_message = models.TextField(blank=True, default="")
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    cost_usd = models.FloatField(null=True, blank=True)
    request_count = models.IntegerField(default=1)
    duration_ms = models.IntegerField(null=True, blank=True)
    metadata_json = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "api_usage_logs"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.category}] {self.action} ({self.total_tokens} tok)"



