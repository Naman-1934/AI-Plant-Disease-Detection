from pathlib import Path
from dotenv import dotenv_values

env_path = Path(".env").resolve()

print("Env path:", env_path)
print("Exists:", env_path.exists())
print()

values = dotenv_values(env_path)

print("Variables loaded:")
for key, value in values.items():
    print(f"{key} = {value}")