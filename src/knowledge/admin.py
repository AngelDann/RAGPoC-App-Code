from django.contrib import admin

from .models import (
    AgentMemory,
    AgentSkill,
    ApiUsageLog,
    ChatMessage,
    ChatThread,
    Document,
    Notebook,
    NotebookArtifact,
    NotebookDocument,
    Page,
    Workspace,
)


@admin.register(ApiUsageLog)
class ApiUsageLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "category", "action", "model", "status", "total_tokens", "cost_usd", "duration_ms")
    list_filter = ("category", "status", "model")
    search_fields = ("action", "model", "error_message")
    date_hierarchy = "created_at"


@admin.register(AgentSkill)
class AgentSkillAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "is_active", "updated_at")
    list_filter = ("category", "is_active")
    search_fields = ("name", "description", "instructions")


@admin.register(AgentMemory)
class AgentMemoryAdmin(admin.ModelAdmin):
    list_display = ("category", "source", "content_preview", "updated_at")
    list_filter = ("category", "source")
    search_fields = ("content",)

    def content_preview(self, obj: AgentMemory) -> str:
        return obj.content[:60]


@admin.register(NotebookArtifact)
class NotebookArtifactAdmin(admin.ModelAdmin):
    list_display = ("title", "artifact_type", "notebook", "created_at")
    list_filter = ("artifact_type", "created_at")
    search_fields = ("title", "content")


@admin.register(NotebookDocument)
class NotebookDocumentAdmin(admin.ModelAdmin):
    list_display = ("notebook", "document", "attached_at")
    list_filter = ("attached_at",)


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0


@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ("title", "scope", "page", "created_at", "updated_at")
    list_filter = ("scope",)
    search_fields = ("title", "id")
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("thread", "role", "content", "created_at")
    list_filter = ("role",)
    search_fields = ("content",)


class NotebookInline(admin.TabularInline):
    model = Notebook
    extra = 0


class PageInline(admin.TabularInline):
    model = Page
    extra = 0


class NotebookDocumentInline(admin.TabularInline):
    model = NotebookDocument
    extra = 0


@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = ("name", "id", "created_at", "updated_at")
    search_fields = ("name", "id")
    inlines = [NotebookInline]


@admin.register(Notebook)
class NotebookAdmin(admin.ModelAdmin):
    list_display = ("name", "workspace", "position", "created_at", "updated_at")
    list_filter = ("workspace",)
    search_fields = ("name", "description")
    inlines = [PageInline, NotebookDocumentInline]


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
    list_display = ("title", "notebook", "position", "updated_at")
    list_filter = ("notebook__workspace", "notebook")
    search_fields = ("title", "plain_text")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "media_type", "status", "byte_size", "created_at")
    list_filter = ("media_type", "status")
    search_fields = ("original_filename", "source_path", "content_hash")
    inlines = [NotebookDocumentInline]
