---
created: {{ date }}
type: executive-summary
tags: [executive, summary{% for tag in generated_tags %}, {{ tag }}{% endfor %}]
duration: {{ duration }}
---

# {{ title }}

## Summary
{{ summary }}

{% if key_decisions %}
## Key Decisions
{% for d in key_decisions %}- {{ d }}
{% endfor %}
{% endif %}

{% if risks %}
## Risks
{% for r in risks %}- {{ r }}
{% endfor %}
{% endif %}

{% if asks %}
## Asks
{% for ask in asks %}- {{ ask }}
{% endfor %}
{% endif %}

{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
