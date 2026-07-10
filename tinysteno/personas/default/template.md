---
created: {{ date }}
type: meeting
tags: [{{ tags | join(', ') }}]
duration: {{ duration }}
participants: {{ (participants or []) | join(', ') }}
---

# {{ title }}

{% if overview %}
## Overview
{{ overview }}

{% endif %}
{% if key_points %}
## Key Points
{% for point in key_points %}{{ loop.index }}. {{ point }}
{% endfor %}

{% endif %}
{% if action_items %}
## Action Items
{% for item in action_items %}- [ ] {{ item }}
{% endfor %}

{% endif %}
{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
