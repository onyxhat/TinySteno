---
created: {{ date }}
type: irm
tags: [{{ tags | join(', ') }}]
duration: {{ duration }}
severity: {{ severity }}
responders: {{ (responders or []) | join(', ') }}
---

# {{ title }}

## Overview
{{ overview }}

**Severity:** {{ severity }}
**Impact:** {{ impact }}

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
