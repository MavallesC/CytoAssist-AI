from django.apps import AppConfig
from django.db.backends.signals import connection_created
from django.dispatch import receiver


class WsiAnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'wsi_analysis'

    def ready(self):
        @receiver(connection_created)
        def configure_sqlite(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                with connection.cursor() as cursor:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA synchronous=NORMAL;')
