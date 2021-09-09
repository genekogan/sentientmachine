# Sentient Machine

```
from eden.client import Client
from eden.datatypes import Image

c = Client(url = 'http://127.0.0.1:5656', username= 'sentient_machine')

config = {
    'question': 'when will I be free?'
}


run_response = c.run(config)
print(run_response)

```