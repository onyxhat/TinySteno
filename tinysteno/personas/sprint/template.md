---
created: {{ date }}
type: sprint
tags: [sprint, agile{% for tag in generated_tags %}, {{ tag }}{% endfor %}]
duration: {{ duration }}
ceremony: {{ ceremony_type }}
---

# {{ title }}

**Ceremony:** {{ ceremony_type }}

## Overview
{{ overview }}

{% if completed_items %}
## Completed
{% for item in completed_items %}- {{ item }}
{% endfor %}
{% endif %}

{% if incomplete_items %}
## Not Completed
{% for item in incomplete_items %}- {{ item }}
{% endfor %}
{% endif %}

{% if blockers %}
## Blockers
{% for b in blockers %}- {{ b }}
{% endfor %}
{% endif %}

{% if retrospective_notes %}
## Retrospective Notes
{% for note in retrospective_notes %}- {{ note }}
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
