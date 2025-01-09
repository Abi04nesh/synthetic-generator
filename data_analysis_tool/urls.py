from django.contrib import admin
from django.urls import path
from analyzer import views
from django.conf import settings  # Import settings
from django.conf.urls.static import static  # Import static to serve media files

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.upload_dataset, name="upload_dataset"),
]

# Serve media files during development (only if DEBUG is True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
