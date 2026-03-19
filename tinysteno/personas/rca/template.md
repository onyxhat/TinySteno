---
created: {{ date }}
type: rca
tags: [incident, rca]
duration: {{ duration }}
---

# {{ title }}

## Overview
{{ overview }}

{% if timeline %}
## Timeline
{% for event in timeline %}- {{ event }}
{% endfor %}
{% endif %}

## Root Cause
{{ root_cause }}

{% if contributing_factors %}
## Contributing Factors
{% for factor in contributing_factors %}- {{ factor }}
{% endfor %}
{% endif %}

{% if corrective_actions %}
## Corrective Actions
{% for item in corrective_actions %}- [ ] {{ item }}
{% endfor %}
{% endif %}

{% if transcript %}
## Transcript
```
{{ transcript }}
```
{% endif %}
