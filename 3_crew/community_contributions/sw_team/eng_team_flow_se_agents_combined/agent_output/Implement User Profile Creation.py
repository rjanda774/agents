```python
# settings.py
INSTALLED_APPS = [
    ...
    'rest_framework',
    'trading_profiles',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.TokenAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
}

# models.py
from django.db import models
from django.contrib.auth.models import User

class TradingProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    preferences = models.JSONField(default=dict)
    statistics = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.user.username}'s Trading Profile"

# serializers.py
from rest_framework import serializers
from .models import TradingProfile

class TradingProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradingProfile
        fields = ['user', 'preferences', 'statistics']

    def validate_preferences(self, value):
        # Add custom validation for preferences
        if not isinstance(value, dict):
            raise serializers.ValidationError("Preferences must be a dictionary.")
        return value

    def validate_statistics(self, value):
        # Add custom validation for statistics
        if not isinstance(value, dict):
            raise serializers.ValidationError("Statistics must be a dictionary.")
        return value

# views.py
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from .models import TradingProfile
from .serializers import TradingProfileSerializer

class TradingProfileViewSet(viewsets.ModelViewSet):
    queryset = TradingProfile.objects.all()
    serializer_class = TradingProfileSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TradingProfileViewSet

router = DefaultRouter()
router.register(r'trading-profiles', TradingProfileViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import TradingProfile

@receiver(post_save, sender=User)
def create_trading_profile(sender, instance, created, **kwargs):
    if created:
        TradingProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_trading_profile(sender, instance, **kwargs):
    instance.tradingprofile.save()

# admin.py
from django.contrib import admin
from .models import TradingProfile

admin.site.register(TradingProfile)

# Ensure to run migrations after creating models
# python manage.py makemigrations
# python manage.py migrate
```

This code implements a Django application with Django REST Framework to manage user trading profiles. It includes authentication, permissions, model validation, serializer security, error handling, and signal handlers for creating and saving trading profiles. The code is modular and testable, following best practices in software design.