---
created: {{ date }}
type: rca
tags: [incident, rca]
duration: {{ duration }}
---

# {{ title }}

## 1. Incident Summary

{{ overview }}

{% if timeline %}
## 2. Timeline

| Time | Event | Source |
|------|-------|--------|
{% for event in timeline %}| {{ event }} |
{% endfor %}
{% endif %}

## 3. Root Cause

{{ root_cause }}

{% if contributing_factors %}
## 4. Contributing Factors

{% for factor in contributing_factors %}- {{ factor }}
{% endfor %}
{% endif %}

## 5. Resolution

{{ resolution }}

{% if action_items %}
## 6. Action Items

| # | Action | Owner (Role) | Priority |
|---|--------|--------------|----------|
{% for item in action_items %}| {{ loop.index }} | {{ item }} |
{% endfor %}
{% endif %}

{% if open_questions %}
## 7. Open Questions

{% for question in open_questions %}- {{ question }}
{% endfor %}
{% endif %}

{% if transcript %}
## Transcript

```
{{ transcript }}
```
{% endif %}
