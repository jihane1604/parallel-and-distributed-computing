from celery import Celery

app = Celery("tasks", broker = "pyamqp://guest@localhost//", backend = "rpc://")

@app.task
def power(number, exponent):
    print("I am here")
    return number ** exponent