import pytest
from celery import Celery
from app.core.celery_app import celery_app

def test_celery_app_is_instance_of_celery():
    """Tests that celery_app is an instance of Celery."""
    assert isinstance(celery_app, Celery)

def test_celery_app_main_name():
    """Tests that the main name is set to 'worker'."""
    assert celery_app.main == 'worker'

def test_celery_app_broker_url():
    """Tests that the broker URL is correctly set."""
    assert celery_app.conf.broker_url == 'redis://redis:6379/0'

def test_celery_app_result_backend():
    """Tests that the result backend is correctly set."""
    assert celery_app.conf.result_backend == 'redis://redis:6379/0'

def test_celery_app_include_tasks():
    """Tests that the tasks module is included."""
    assert 'app.tasks.tasks' in celery_app.conf.include

def test_celery_app_task_serializer():
    """Tests that task_serializer is set to 'json'."""
    assert celery_app.conf.task_serializer == 'json'

def test_celery_app_accept_content():
    """Tests that accept_content includes 'json'."""
    assert 'json' in celery_app.conf.accept_content

def test_celery_app_result_serializer():
    """Tests that result_serializer is set to 'json'."""
    assert celery_app.conf.result_serializer == 'json'

def test_celery_app_timezone():
    """Tests that timezone is set to 'UTC'."""
    assert celery_app.conf.timezone == 'UTC'

def test_celery_app_enable_utc():
    """Tests that enable_utc is True."""
    assert celery_app.conf.enable_utc is True

def test_celery_app_task_track_started():
    """Tests that task_track_started is True."""
    assert celery_app.conf.task_track_started is True