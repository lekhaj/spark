# celeryconfig.py
# This configuration file is for the Celery worker and should be run on a separate machine.

# The broker is the message queue that Celery uses.
# Replace with your actual Redis server URL.
# If running Redis locally, this URL should be fine.
broker_url = 'redis://localhost:6379/0'

# The result backend stores the state and return values of tasks.
result_backend = 'redis://localhost:6379/0'

# Set the result serializer to json for safe and easy serialization.
result_serializer = 'json'

# A list of modules to import when the Celery worker starts.
imports = ('tasks',)

# You can configure task queues here, for example, to send different types of tasks to different workers.
task_queues = {
    'default': {
        'exchange': 'default',
        'binding_key': 'default',
    },
}
