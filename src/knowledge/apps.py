from django.apps import AppConfig

SKILL_CREATOR_INSTRUCTIONS = """\
Objetivo: ayudar a crear nuevas skills (memoria procedimental) de alta calidad para este agente, mediante `manage_skill`.

Cuándo usarla:
- El usuario pide explícitamente crear/guardar una skill o "enseñarle" un procedimiento al agente.
- Detectas un flujo de trabajo repetible (formato de minutas, checklist de revisión, plantilla de análisis, etc.) que valdría la pena reutilizar.

Procedimiento:
1. Antes de crear, llama `manage_skill(action='list')` para revisar si ya existe una skill equivalente; si existe, actualízala con `action='update'` en vez de duplicarla.
2. Define un `name` corto en kebab-case (minúsculas, guiones, sin espacios ni tildes), descriptivo del procedimiento (ej. `resumir-reuniones`, `analizar-logs`, `redactar-issue`).
3. Escribe una `description` de una sola frase que indique claramente CUÁNDO debe activarse esta skill (para que el propio agente sepa reconocer la situación en el futuro).
4. Escribe `instructions` como pasos numerados, concretos y accionables — no expliques el "qué" sino el "cómo": formato esperado, orden de pasos, criterios de calidad, ejemplos breves si ayudan.
5. Asigna una `category` coherente con el dominio (ej. `writing`, `research`, `coding`, `general`) para mantener el catálogo organizado.
6. Guarda con `manage_skill(action='create', name=..., description=..., instructions=..., category=...)`.
7. Confirma al usuario en una línea qué skill quedó guardada y cuándo se activará.

Reglas:
- Nunca guardes secretos, credenciales ni datos sensibles dentro de una skill.
- Si el usuario pide corregir o mejorar una skill existente, usa `action='update'` con el mismo `name` (no crees una copia).
- Sé conciso: una skill no es una nota larga, es un procedimiento reutilizable.
"""


class KnowledgeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "knowledge"

    def ready(self):
        self._seed_default_skills()

    def _seed_default_skills(self):
        """Ensure baseline procedural skills exist every time the project boots."""
        from django.db.utils import OperationalError, ProgrammingError

        try:
            from knowledge.models import AgentSkill

            AgentSkill.objects.update_or_create(
                name="skill-creator",
                defaults={
                    "description": (
                        "Meta-skill: úsala cuando el usuario pida crear, actualizar o "
                        "enseñarle al agente un nuevo procedimiento/skill reutilizable."
                    ),
                    "instructions": SKILL_CREATOR_INSTRUCTIONS,
                    "category": "meta",
                    "is_active": True,
                },
            )
        except (OperationalError, ProgrammingError):
            # DB/tabla aún no disponible (p. ej. antes de aplicar migraciones por primera vez).
            # Se sembrará en el próximo arranque una vez migrada la base de datos.
            pass
