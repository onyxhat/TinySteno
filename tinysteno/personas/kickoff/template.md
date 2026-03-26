---
created: {{ date }}
type: kickoff
tags: [{{ tags | join(', ') }}]
duration: {{ duration }}
---

# {{ title }}

## Overview
{{ overview }}

{% if objectives %}
## Objectives
{% for obj in objectives %}- {{ obj }}
{% endfor %}
{% endif %}

{% if stakeholders %}
## Stakeholders
{% for s in stakeholders %}- {{ s }}
{% endfor %}
{% endif %}

## Scope
{{ scope }}

{% if risks %}
## Risks
{% for r in risks %}- {{ r }}
{% endfor %}
{% endif %}

{% if decisions %}
## Decisions
{% for d in decisions %}- {{ d }}
{% endfor %}
{% endif %}

{% if next_steps %}
## Next Steps
{% for step in next_steps %}- [ ] {{ step }}
{% endfor %}
{% endif %}

{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
