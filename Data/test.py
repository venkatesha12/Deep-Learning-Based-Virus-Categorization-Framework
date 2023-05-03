
import json

with open('it.json') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
for d in data.values():
	print(d)