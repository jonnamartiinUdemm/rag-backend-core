from app.core.celery_app import celery_app
import time


@celery_app.task
def test_task(name: str):
    print(f'Processing task for {name}')
    time.sleep(5)
    return f"Hello {name}, this task was processed by the Celery Worker"