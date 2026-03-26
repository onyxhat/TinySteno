---
created: {{ date }}
type: executive-summary
tags: [{{ tags | join(', ') }}]
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
