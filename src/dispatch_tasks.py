from src.tasks import power

def dispatch():
    results_objs = [power.apply_async((number, 2)) for number in range(1, 10001)]
    results = [result.get() for result in results_objs]

    return results