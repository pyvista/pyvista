---
title: 🐛 Weekly linkcheck failed
about: Use this template for checking broken links
name: Weekly linkcheck failed
labels: maintenance, bug
---

Links are checked weekly to make sure the documentation links areg working. Last check on {{ date | date('dddd, MMM Do YYYY, hh:mm A') }} failed. 

Partial logs output can be found here:

{{ env.OUTPUT }}