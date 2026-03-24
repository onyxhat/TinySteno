---
created: {{ date }}
type: 1on1
tags: [1on1{% for tag in generated_tags %}, {{ tag }}{% endfor %}]
duration: {{ duration }}
participants: {{ (participants or []) | join(', ') }}
---

# {{ title }}

{% if goals %}
## Goals
{% for item in goals %}- {{ item }}
{% endfor %}

{% endif %}
{% if needs %}
## Needs
{% for item in needs %}- {{ item }}
{% endfor %}

{% endif %}
{% if struggles %}
## Struggles
{% for item in struggles %}- {{ item }}
{% endfor %}

{% endif %}
{% if recent_wins %}
## Recent Wins
{% for item in recent_wins %}- {{ item }}
{% endfor %}

{% endif %}
{% if action_items_employee %}
## Action Items for Employee
{% for item in action_items_employee %}- [ ] {{ item }}
{% endfor %}

{% endif %}
{% if action_items_manager %}
## Action Items for Manager
{% for item in action_items_manager %}- [ ] {{ item }}
{% endfor %}

{% endif %}
{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
