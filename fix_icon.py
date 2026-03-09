import re

with open("app.py", "r") as f:
    text = f.read()

# Replace the swords icon with a universally supported emoji
old_icons = '''icons = [
        "bar-chart-line",
        "bullseye",
        "search",
        "trophy",
        "diagram-3",
        "joystick",
        "people",
        "swords",
        "folder2-open"
    ]'''

new_icons = '''icons = [
        "bar-chart-line",
        "bullseye",
        "search",
        "trophy",
        "diagram-3",
        "joystick",
        "people",
        "shield-sword",
        "folder2-open"
    ]'''

text = text.replace(old_icons, new_icons)

with open("app.py", "w") as f:
    f.write(text)
