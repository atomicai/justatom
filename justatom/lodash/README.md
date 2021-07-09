### <p align="center">_class Lodash_</p>
#### _Example_
```python
from src.lodash import _

@_.benchmark
def rock_n_roll(arg, sec=2):
    import time
    time.sleep(sec)
    return f'It took me {sec} to complete your job'
```