from typeguard import typechecked

@typechecked
def add(a: (int, float), b: (int, float))-> (int, float):
    return a+b
    
print(add(4, 2))
print(add('f', 2))
print(add(7.0, 2))