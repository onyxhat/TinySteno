---
created: {{ date }}
type: irm
tags: [incident, response{% for tag in generated_tags %}, {{ tag }}{% endfor %}]
duration: {{ duration }}
severity: {{ severity }}
---

# {{ title }}

## Overview
{{ overview }}

**Severity:** {{ severity }}
**Impact:** {{ impact }}

{% if responders %}
## Responders
{% for r in responders %}- {{ r }}
{% endfor %}
{% endif %}

{% if timeline %}
## Timeline
{% for event in timeline %}- {{ event }}
{% endfor %}
{% endif %}

{% if mitigations %}
## Mitigations Applied
{% for m in mitigations %}- {{ m }}
{% endfor %}
{% endif %}

{% if follow_ups %}
## Follow-Ups
{% for item in follow_ups %}- [ ] {{ item }}
{% endfor %}
{% endif %}

{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
