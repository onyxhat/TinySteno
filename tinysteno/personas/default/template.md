---
created: {{ date }}
type: meeting
tags: [meeting]
duration: {{ duration }}
participants: {{ (participants or []) | join(', ') }}
---

# {{ title }}

{% if overview %}
## Overview
{{ overview }}

{% endif %}
{% if participants %}
## Participants
{% for p in participants %}- {{ p }}
{% endfor %}

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
